import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core.document_manager import DocumentManager
from core.knowledge_base_sync import KnowledgeBaseSyncService
from core.rag_system import RAGSystem


def _print_json(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _bootstrap(args):
    rag = RAGSystem()
    rag.initialize()
    document_manager = DocumentManager(rag)
    sync_service = KnowledgeBaseSyncService(rag, document_manager.markdown_dir)
    lock_conn = sync_service._try_advisory_lock("knowledge_base_sync")
    if lock_conn is None:
        _print_json(
            {
                "status": "skipped_locked",
                "message": "已有其他知识库任务在执行，本轮 bootstrap 跳过。",
            }
        )
        return 2
    try:
        result = document_manager.index_existing_markdowns(skip_existing=not args.reindex_all)
        rag.refresh_knowledge_base_status()
        _print_json(
            {
                "status": "completed",
                "action": "bootstrap",
                "result": result,
                "knowledge_base": rag.get_knowledge_base_status(),
            }
        )
        return 0
    finally:
        sync_service._release_advisory_lock(lock_conn, "knowledge_base_sync")


def _sync_local(args):
    rag = RAGSystem()
    rag.initialize()
    result = DocumentManager(rag).sync_local_documents(
        trigger_type="job",
        soft_delete_missing=args.soft_delete_missing,
    )
    rag.record_import_event(result.to_event())
    rag.refresh_knowledge_base_status()
    _print_json(
        {
            "status": result.status,
            "action": "sync-local",
            "result": result.to_event(),
            "knowledge_base": rag.get_knowledge_base_status(),
        }
    )
    return 0 if result.status != "skipped_locked" else 2


def _sync_official(args):
    rag = RAGSystem()
    rag.initialize()
    result = DocumentManager(rag).sync_official_source(
        source=args.source,
        limit=args.limit,
        trigger_type="job",
    )
    rag.record_import_event(result.to_event())
    rag.refresh_knowledge_base_status()
    _print_json(
        {
            "status": result.status,
            "action": "sync-official",
            "result": result.to_event(),
            "knowledge_base": rag.get_knowledge_base_status(),
        }
    )
    return 0 if result.status != "skipped_locked" else 2


def _sync_all(args):
    del args
    rag = RAGSystem()
    rag.initialize()
    results = DocumentManager(rag).sync_all_sources(trigger_type="job")
    for result in results:
        rag.record_import_event(result.to_event())
    rag.refresh_knowledge_base_status()
    _print_json(
        {
            "action": "sync-all",
            "results": [result.to_event() for result in results],
            "knowledge_base": rag.get_knowledge_base_status(),
        }
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Knowledge base maintenance jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Index existing local markdowns.")
    bootstrap.add_argument("--reindex-all", action="store_true", help="Reindex all markdown files, not only missing ones.")
    bootstrap.set_defaults(func=_bootstrap)

    sync_local = subparsers.add_parser("sync-local", help="Sync local markdown files into the knowledge base.")
    sync_local.add_argument("--soft-delete-missing", action="store_true", help="Mark missing local documents as inactive.")
    sync_local.set_defaults(func=_sync_local)

    sync_official = subparsers.add_parser("sync-official", help="Sync one official source.")
    sync_official.add_argument("source", choices=["medlineplus", "nhc", "who"])
    sync_official.add_argument("--limit", type=int, default=10)
    sync_official.set_defaults(func=_sync_official)

    sync_all = subparsers.add_parser("sync-all", help="Run local and official source sync jobs.")
    sync_all.set_defaults(func=_sync_all)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    parsed = parser.parse_args()
    raise SystemExit(parsed.func(parsed))
