"""DocumentManager — CRUD facade for the knowledge base.

Handles:
    - Upload (PDF/MD/HTML via unstructured) → markdown conversion → chunking → indexing
    - List / delete documents with vector store cleanup
    - Official source sync (MedlinePlus, NHC, WHO) with content-hash dedup
    - Knowledge base status reporting (document count, chunk count, recent imports)
"""

from pathlib import Path
import logging
import shutil
import config
from core.document_parsers import supported_upload_extensions, unstructured_to_markdown
from core.knowledge_base_sync import KnowledgeBaseSyncService
from core.medical_source_ingest import (
    NhcPdfWhitelistImporter,
    WhoHtmlWhitelistImporter,
)
from core.official_source_profiles import list_official_source_profiles
from utils import clear_directory_contents, pdf_to_markdown


logger = logging.getLogger(__name__)

class DocumentManager:

    def __init__(self, rag_system=None, *, vector_db=None, parent_store=None,
                 chunker=None, collection_name=None):
        # New explicit dependency injection — preferred
        if rag_system is not None:
            # Backward compatibility: unwrap from rag_system
            self.vector_db = rag_system.vector_db
            self.parent_store = rag_system.parent_store
            self.chunker = rag_system.chunker
            self.collection_name = rag_system.collection_name
        else:
            self.vector_db = vector_db
            self.parent_store = parent_store
            self.chunker = chunker
            self.collection_name = collection_name or config.CHILD_COLLECTION
        self.rag_system = rag_system  # kept for legacy callers, but new code should not use it
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        # NOTE: no longer sets rag_system.document_manager = self
        # The caller (ApiContainer or RAGSystem) is responsible for that assignment.

    def get_markdown_paths(self):
        if not self.markdown_dir.exists():
            return []
        return sorted(self.markdown_dir.glob("*.md"))

    def get_local_document_stats(self):
        markdown_paths = self.get_markdown_paths()
        return {
            "local_markdown_files": len(markdown_paths),
            "local_markdown_names": [path.name for path in markdown_paths],
        }

    def get_official_source_coverage(self):
        """Return user-facing coverage notes for official source sync.

        WHO/NHC are intentionally curated manifests in this demo project; this
        makes that scope visible instead of looking like the sync button is
        broken or empty.
        """
        markdown_names = [path.name for path in self.get_markdown_paths()]

        def count_prefix(prefix: str) -> int:
            return sum(1 for name in markdown_names if name.startswith(prefix))

        manifest_counts = {"medlineplus": None}
        try:
            manifest_counts["nhc"] = len(NhcPdfWhitelistImporter().load_manifest())
        except Exception:
            logger.warning("Failed to load NHC whitelist manifest", exc_info=True)
            manifest_counts["nhc"] = 0
        try:
            manifest_counts["who"] = len(WhoHtmlWhitelistImporter().load_manifest())
        except Exception:
            logger.warning("Failed to load WHO whitelist manifest", exc_info=True)
            manifest_counts["who"] = 0

        coverage = []
        for profile in list_official_source_profiles():
            data = profile.to_dict(
                manifest_count=manifest_counts.get(profile.source),
                local_file_count=count_prefix(profile.source_prefix),
            )
            if data["manifest_count"] is not None:
                data["coverage_note"] = f"当前内置 {data['manifest_count']} 条。{data['coverage_note']}"
            coverage.append(data)
        return coverage

    def _index_markdown_paths(self, markdown_paths, progress_callback=None, skip_existing=True):
        markdown_paths = [Path(path) for path in markdown_paths if path]
        if not markdown_paths:
            return {"processed": 0, "added": 0, "skipped": 0}

        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)
        indexed_document_nos = self.vector_db.get_indexed_document_nos() if skip_existing else set()

        added = 0
        skipped = 0
        processed = 0

        for index, md_path in enumerate(markdown_paths):
            if progress_callback:
                progress_callback((index + 1) / len(markdown_paths), f"Processing {md_path.name}")

            processed += 1
            document_no = md_path.stem
            if skip_existing and document_no in indexed_document_nos:
                skipped += 1
                continue

            try:
                parent_chunks, child_chunks = self.chunker.create_chunks_single(md_path)
                if not child_chunks:
                    skipped += 1
                    continue

                self.parent_store.save_many(parent_chunks)
                collection.add_documents(child_chunks)
                indexed_document_nos.add(document_no)
                added += 1
            except Exception as e:
                logger.exception("Failed to index markdown document %s", md_path)
                skipped += 1

        return {"processed": processed, "added": added, "skipped": skipped}

    def index_existing_markdowns(self, progress_callback=None, skip_existing=True):
        return self._index_markdown_paths(
            self.get_markdown_paths(),
            progress_callback=progress_callback,
            skip_existing=skip_existing,
        )

    def add_documents_with_report(self, document_paths, progress_callback=None):
        if not document_paths:
            return {"processed": 0, "added": 0, "updated": 0, "unchanged": 0, "deactivated": 0, "skipped": 0, "failed": 0, "skipped_details": [], "failure_details": [], "conversion_details": []}

        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        allowed_extensions = supported_upload_extensions()
        document_paths = [p for p in document_paths if p and Path(p).suffix.lower() in allowed_extensions]

        if not document_paths:
            return {"processed": 0, "added": 0, "updated": 0, "unchanged": 0, "deactivated": 0, "skipped": 0, "failed": 0, "skipped_details": [], "failure_details": [], "conversion_details": []}

        prepared_markdowns = []
        conversion_details = []
        failure_details = []
        sync_service = KnowledgeBaseSyncService(self.rag_system, self.markdown_dir)

        for i, doc_path in enumerate(document_paths):
            source_path = Path(doc_path)
            if progress_callback:
                progress_callback((i + 1) / len(document_paths), f"Processing {source_path.name}")

            try:
                suffix = source_path.suffix.lower()
                if suffix == ".md":
                    md_path = self.markdown_dir / source_path.name
                    if source_path.resolve() != md_path.resolve():
                        if md_path.exists() and not config.KB_REPLACE_LOCAL_DUPLICATES:
                            failure_details.append(f"{source_path.name}: 检测到同名 Markdown，当前配置禁止自动替换。")
                            continue
                        shutil.copyfile(source_path, md_path)
                elif suffix == ".pdf":
                    md_target = self.markdown_dir / f"{source_path.stem}.md"
                    if md_target.exists() and not config.KB_REPLACE_LOCAL_DUPLICATES:
                        failure_details.append(f"{source_path.name}: 检测到同名 Markdown，当前配置禁止自动替换。")
                        continue
                    conversion_result = pdf_to_markdown(str(source_path), self.markdown_dir)
                    md_path = conversion_result.output_path
                    detail = (
                        f"{source_path.name}: method={conversion_result.method_used} "
                        f"chars={conversion_result.extracted_char_count} "
                        f"scan_like={'yes' if conversion_result.scan_like else 'no'}"
                    )
                    if conversion_result.warnings:
                        detail += f" | warnings={' ; '.join(conversion_result.warnings[:2])}"
                    conversion_details.append(detail)
                else:
                    md_target = self.markdown_dir / f"{source_path.stem}.md"
                    if md_target.exists() and not config.KB_REPLACE_LOCAL_DUPLICATES:
                        failure_details.append(f"{source_path.name}: 检测到同名 Markdown，当前配置禁止自动替换。")
                        continue
                    conversion_result = unstructured_to_markdown(str(source_path), self.markdown_dir)
                    md_path = conversion_result.output_path
                    detail = (
                        f"{source_path.name}: method={conversion_result.method_used} "
                        f"chars={conversion_result.extracted_char_count}"
                    )
                    if conversion_result.warnings:
                        detail += f" | warnings={' ; '.join(conversion_result.warnings[:2])}"
                    conversion_details.append(detail)
                prepared_markdowns.append(md_path)
            except Exception as e:
                logger.exception("Failed to prepare uploaded document %s", doc_path)
                failure_details.append(f"{source_path.name}: {e}")

        sync_result = sync_service.sync_local_documents(
            prepared_markdowns,
            trigger_type="manual",
            progress_callback=progress_callback,
            soft_delete_missing=False,
        )
        combined_failures = [*failure_details, *list(sync_result.failure_details)]
        sync_event = sync_result.to_event()
        sync_event["failure_details"] = combined_failures
        sync_event["failed"] = len(combined_failures)
        sync_event["conversion_details"] = [*conversion_details, *list(sync_result.conversion_details)]
        return {
            "processed": len(document_paths),
            "added": sync_result.added,
            "updated": sync_result.updated,
            "unchanged": sync_result.unchanged,
            "deactivated": sync_result.deactivated,
            "skipped": sync_result.skipped,
            "failed": len(combined_failures),
            "skipped_details": [],
            "failure_details": combined_failures,
            "conversion_details": sync_event["conversion_details"],
            "sync_event": sync_event,
        }

    def add_documents(self, document_paths, progress_callback=None):
        report = self.add_documents_with_report(document_paths, progress_callback=progress_callback)
        return report["added"], report["skipped"]

    def get_markdown_files(self):
        return sorted([p.name for p in self.get_markdown_paths()])

    def get_document_inventory(self):
        items = []
        for path in self.get_markdown_paths():
            stat = path.stat()
            metadata = {}
            title = path.stem
            try:
                raw_text = path.read_text(encoding="utf-8")
                metadata = KnowledgeBaseSyncService._extract_front_matter_metadata(raw_text)
                body = KnowledgeBaseSyncService._strip_front_matter(raw_text)
                title = metadata.get("title") or KnowledgeBaseSyncService._first_heading(body) or title
                origin, source_key = KnowledgeBaseSyncService._classify_existing_markdown(path, metadata)
            except Exception:
                logger.warning("Failed to read document inventory metadata for %s", path, exc_info=True)
                origin, source_key = "local", f"local:{path.name}"

            items.append(
                {
                    "name": path.name,
                    "file_type": metadata.get("file_type") or path.suffix.lstrip(".") or "md",
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                    "title": title,
                    "source_name": metadata.get("source_name") or metadata.get("source") or ("本地文档" if origin == "local" else ""),
                    "source_type": metadata.get("source_type") or ("local_document" if origin == "local" else ""),
                    "source_key": metadata.get("source_key") or source_key,
                    "sync_status": metadata.get("sync_status") or "active",
                    "is_active": str(metadata.get("is_active") or "true").lower() != "false",
                    "freshness_bucket": metadata.get("freshness_bucket") or "",
                    "original_url": metadata.get("original_url") or metadata.get("source_url") or "",
                }
            )
        return items

    def clear_all(self):
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(self.markdown_dir)

        self.vector_db.delete_collection(self.collection_name)
        self.parent_store.clear_store()
        self.vector_db.create_collection(self.collection_name)

    def sync_local_documents(self, markdown_paths=None, progress_callback=None, trigger_type: str = "manual", soft_delete_missing: bool = False):
        sync_service = KnowledgeBaseSyncService(self.rag_system, self.markdown_dir)
        return sync_service.sync_local_documents(
            markdown_paths=markdown_paths,
            trigger_type=trigger_type,
            progress_callback=progress_callback,
            soft_delete_missing=soft_delete_missing,
        )

    def sync_official_source(self, source: str, limit: int = 10, progress_callback=None, trigger_type: str = "manual"):
        sync_service = KnowledgeBaseSyncService(self.rag_system, self.markdown_dir)
        return sync_service.sync_official_source(
            source=source,
            limit=int(limit) if limit is not None else None,
            trigger_type=trigger_type,
            progress_callback=progress_callback,
            soft_delete_missing=True,
        )

    def sync_all_sources(self, trigger_type: str = "scheduler"):
        sync_service = KnowledgeBaseSyncService(self.rag_system, self.markdown_dir)
        return sync_service.sync_all(trigger_type=trigger_type)

    def import_official_source(self, source: str, limit: int = 10, overwrite: bool = False, index_after_import: bool = True):
        return self.sync_official_source(source=source, limit=limit, trigger_type="manual")
