"""Knowledge-base status, bootstrap, and sync supervision."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import config


logger = logging.getLogger(__name__)


def _default_knowledge_base_status() -> dict:
    return {
        "status": "not_checked",
        "message": "尚未检查知识库状态。",
        "last_error": "",
        "stats": {
            "local_markdown_files": 0,
            "documents": 0,
            "inactive_documents": 0,
            "parent_chunks": 0,
            "child_chunks": 0,
            "last_bootstrap_result": "",
            "last_sync_result": "",
            "recent_imports": [],
        },
    }


class KnowledgeBaseSupervisor:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.status = getattr(rag_system, "_knowledge_base_status", None) or _default_knowledge_base_status()
        self._bootstrap_lock = threading.Lock()
        self._bootstrap_thread = None
        self._sync_thread = None

    def get_status(self):
        recent_imports = list(self.status["stats"].get("recent_imports", []))
        if not recent_imports:
            import_task_store = getattr(self.rag_system, "import_task_store", None)
            if import_task_store is None:
                return {
                    "status": self.status["status"],
                    "message": self.status["message"],
                    "last_error": self.status["last_error"],
                    "stats": {
                        **dict(self.status["stats"]),
                        "recent_imports": [],
                    },
                }
            try:
                recent_imports = import_task_store.list_recent(config.RECENT_IMPORT_TASK_LIMIT)
                self.status["stats"]["recent_imports"] = recent_imports
            except Exception:
                logger.warning("Failed to load recent import tasks", exc_info=True)
                recent_imports = []
        return {
            "status": self.status["status"],
            "message": self.status["message"],
            "last_error": self.status["last_error"],
            "stats": {
                **dict(self.status["stats"]),
                "recent_imports": recent_imports,
            },
        }

    def update_status(self, status, message, *, last_error=None, stats=None):
        self.status["status"] = status
        self.status["message"] = message
        if last_error is not None:
            self.status["last_error"] = last_error
        if stats:
            if "recent_imports" not in stats:
                stats["recent_imports"] = list(self.status["stats"].get("recent_imports", []))
            self.status["stats"].update(stats)

    def record_import_event(self, event: dict):
        history = list(self.status["stats"].get("recent_imports", []))
        payload = dict(event)
        payload.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        try:
            import_task_store = getattr(self.rag_system, "import_task_store", None)
            if import_task_store is not None:
                import_task_store.save_event(payload)
        except Exception:
            logger.warning("Failed to persist import event", exc_info=True)
        history.insert(0, payload)
        self.status["stats"]["recent_imports"] = history[:config.RECENT_IMPORT_TASK_LIMIT]
        scope = payload.get("scope") or payload.get("source") or ""
        self.status["stats"]["last_sync_result"] = (
            f"{payload.get('label', scope)} | 新增 {payload.get('written', 0)} | "
            f"更新 {payload.get('updated', 0)} | 下线 {payload.get('deactivated', 0)} | "
            f"未变化 {payload.get('unchanged', 0)}"
        )

    def refresh_status(self):
        doc_manager = self._get_document_manager()
        local_stats = doc_manager.get_local_document_stats()
        db_stats = self.rag_system.vector_db.get_collection_stats()
        stats = {
            **local_stats,
            **db_stats,
            "last_bootstrap_result": self.status["stats"].get("last_bootstrap_result", ""),
            "last_sync_result": self.status["stats"].get("last_sync_result", ""),
            "recent_imports": list(self.status["stats"].get("recent_imports", [])),
        }
        import_task_store = getattr(self.rag_system, "import_task_store", None)
        if import_task_store is not None:
            try:
                stats["recent_imports"] = import_task_store.list_recent(config.RECENT_IMPORT_TASK_LIMIT)
            except Exception:
                logger.warning("Failed to refresh recent import tasks", exc_info=True)

        current_status = self.status["status"]
        if current_status == "building":
            self.status["stats"].update(stats)
            return self.get_status()

        if stats["local_markdown_files"] == 0 and stats["child_chunks"] == 0:
            self.update_status("no_documents", "尚无可索引文档。", stats=stats, last_error="")
        elif stats["documents"] < stats["local_markdown_files"]:
            self.update_status("pending_rebuild", "检测到本地文档未完成索引，等待后台补建。", stats=stats, last_error="")
        elif stats["child_chunks"] > 0:
            self.update_status("ready", "知识库可检索。", stats=stats, last_error="")
        else:
            self.update_status("pending_rebuild", "知识库缺少可检索内容，等待后台补建。", stats=stats, last_error="")
        return self.get_status()

    def start_bootstrap(self):
        if not config.AUTO_BOOTSTRAP_KNOWLEDGE_BASE:
            return
        status = self.refresh_status()
        if status["status"] not in {"pending_rebuild"}:
            return
        if self._bootstrap_thread and self._bootstrap_thread.is_alive():
            return
        self._bootstrap_thread = threading.Thread(
            target=self._bootstrap_knowledge_base,
            name="kb-bootstrap",
            daemon=True,
        )
        self._bootstrap_thread.start()

    def start_sync_scheduler(self):
        if not config.ENABLE_KB_SYNC_SCHEDULER:
            return
        if not self.rag_system.is_ready():
            return
        if self._sync_thread and self._sync_thread.is_alive():
            return
        self._sync_thread = threading.Thread(
            target=self._knowledge_base_sync_loop,
            name="kb-sync-scheduler",
            daemon=True,
        )
        self._sync_thread.start()

    def _bootstrap_knowledge_base(self):
        with self._bootstrap_lock:
            doc_manager = self._get_document_manager()
            local_stats = doc_manager.get_local_document_stats()
            if local_stats["local_markdown_files"] == 0:
                self.rag_system._set_startup_step("knowledge_base_bootstrap", "completed", "当前没有本地文档，无需补建。")
                self.update_status(
                    "no_documents",
                    "尚无可索引文档。",
                    stats={**local_stats, **self.rag_system.vector_db.get_collection_stats()},
                    last_error="",
                )
                return

            self.update_status(
                "building",
                "正在后台补建知识库索引，请稍候。",
                stats={**local_stats, **self.rag_system.vector_db.get_collection_stats()},
                last_error="",
            )
            self.rag_system._set_startup_step("knowledge_base_bootstrap", "running", "正在后台补建知识库。")
            try:
                result = doc_manager.index_existing_markdowns(skip_existing=True)
                self.refresh_status()
                self.status["stats"]["last_bootstrap_result"] = (
                    f"processed={result['processed']}, added={result['added']}, skipped={result['skipped']}"
                )
                if self.status["status"] == "ready":
                    self.status["message"] = "知识库已完成后台补建，可正常检索。"
                self.rag_system._set_startup_step("knowledge_base_bootstrap", "completed", "知识库后台补建完成。")
            except Exception as exc:
                logger.exception("Knowledge base bootstrap failed")
                self.rag_system._set_startup_step("knowledge_base_bootstrap", "failed", f"知识库后台补建失败：{exc}")
                self.update_status(
                    "failed",
                    "知识库后台补建失败。",
                    stats={**local_stats, **self.rag_system.vector_db.get_collection_stats()},
                    last_error=str(exc),
                )

    def _knowledge_base_sync_loop(self):
        interval_seconds = max(int(config.KB_SYNC_INTERVAL_HOURS), 1) * 3600
        while True:
            time.sleep(interval_seconds)
            try:
                doc_manager = self._get_document_manager()
                results = doc_manager.sync_all_sources(trigger_type="scheduler")
                for result in results:
                    self.record_import_event(result.to_event())
                self.refresh_status()
            except Exception as exc:
                logger.exception("Scheduled knowledge base sync failed")
                self.status["last_error"] = str(exc)
                self.status["stats"]["last_sync_result"] = f"后台同步失败：{exc}"

    def _get_document_manager(self):
        doc_manager = getattr(self.rag_system, "document_manager", None)
        if doc_manager is not None:
            return doc_manager
        from core.document_manager import DocumentManager

        return DocumentManager(self.rag_system)
