from __future__ import annotations
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psycopg

import config
from core.medical_source_ingest import (
    MedlinePlusXmlImporter,
    NhcPdfWhitelistImporter,
    StandardDocumentRecord,
    WhoHtmlWhitelistImporter,
)
from db.document_ids import build_document_no


logger = logging.getLogger(__name__)


_STANDARD_METADATA_FIELDS = {
    "source",
    "source_name",
    "source_key",
    "source_type",
    "file_type",
    "title",
    "original_url",
    "source_url",
    "published_at",
    "fetched_at",
    "freshness_bucket",
    "content_hash",
    "sync_status",
    "is_active",
    "local_markdown_name",
    "entry_id",
}
_OFFICIAL_SOURCES = {
    "medlineplus": "MedlinePlus",
    "nhc": "国家卫生健康委员会",
    "who": "World Health Organization",
}


@dataclass
class SyncRunResult:
    source: str
    label: str
    status: str = "completed"
    downloaded: int = 0
    processed: int = 0
    written: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deactivated: int = 0
    skipped: int = 0
    failed: int = 0
    index_added: int = 0
    index_skipped: int = 0
    duration_ms: float = 0.0
    trigger_type: str = "manual"
    scope: str = ""
    note: str = ""
    conversion_details: list[str] = field(default_factory=list)
    failure_details: list[str] = field(default_factory=list)

    def to_event(self) -> dict:
        return {
            "source": self.source,
            "label": self.label,
            "status": self.status,
            "downloaded": self.downloaded,
            "written": self.written,
            "updated": self.updated,
            "deactivated": self.deactivated,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "failed": self.failed,
            "index_added": self.index_added,
            "index_skipped": self.index_skipped,
            "duration_ms": self.duration_ms,
            "note": self.note,
            "conversion_details": list(self.conversion_details),
            "failure_details": list(self.failure_details),
            "trigger_type": self.trigger_type,
            "scope": self.scope,
        }


class KnowledgeBaseSyncService:
    def __init__(self, rag_system, markdown_dir: str | Path | None = None):
        self.rag_system = rag_system
        self.markdown_dir = Path(markdown_dir or config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        return psycopg.connect(self.rag_system.vector_db.conninfo)

    @staticmethod
    def _extract_front_matter_metadata(raw_text: str) -> dict:
        lines = str(raw_text or "").splitlines()
        metadata = {}
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if metadata:
                    break
                continue
            match = re.match(r"^([A-Za-z][A-Za-z0-9 _-]*):\s*(.+?)\s*$", stripped)
            if not match:
                if metadata:
                    break
                continue
            key = match.group(1).strip().lower().replace(" ", "_")
            metadata[key] = match.group(2).strip()
        return metadata

    @staticmethod
    def _strip_front_matter(raw_text: str) -> str:
        lines = str(raw_text or "").splitlines()
        stripped_lines = []
        seen_metadata = False
        metadata_done = False
        for line in lines:
            stripped = line.strip()
            is_metadata_line = bool(re.match(r"^([A-Za-z][A-Za-z0-9 _-]*):\s*(.+?)\s*$", stripped))
            if not metadata_done and is_metadata_line:
                seen_metadata = True
                continue
            if seen_metadata and not metadata_done:
                if stripped:
                    metadata_done = True
                    stripped_lines.append(line)
                continue
            stripped_lines.append(line)
        return "\n".join(stripped_lines).strip()

    @staticmethod
    def _collapse_text(value: str) -> str:
        lines = [line.rstrip() for line in str(value or "").replace("\r\n", "\n").splitlines()]
        cleaned = []
        blank_count = 0
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 1:
                    cleaned.append("")
                continue
            blank_count = 0
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    @staticmethod
    def _first_heading(text: str) -> str:
        match = re.search(r"^\s*#\s+(.+?)\s*$", str(text or ""), re.MULTILINE)
        return match.group(1).strip() if match else ""

    @classmethod
    def _classify_existing_markdown(cls, path: Path, metadata: dict) -> tuple[str, str]:
        source_key = str(metadata.get("source_key") or "").strip()
        if source_key.startswith("official:"):
            return "official", source_key
        stem = path.stem
        source_name = str(metadata.get("source") or metadata.get("source_name") or "").strip().lower()
        original_url = str(metadata.get("original_url") or metadata.get("source_url") or "").strip().lower()
        if "medlineplus" in source_name:
            return "official", f"official:medlineplus:{stem}"
        if "world health organization" in source_name or "who.int" in original_url:
            return "official", f"official:who:{stem}"
        if "卫生健康委员会" in source_name or "gov.cn" in original_url:
            return "official", f"official:nhc:{stem}"
        return "local", f"local:{path.name}"

    @staticmethod
    def _normalize_markdown_for_hash(value: str) -> str:
        normalized = str(value or "").replace("\r\n", "\n")
        normalized = re.sub(r"[ \t]+\n", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @classmethod
    def _content_hash(cls, value: str) -> str:
        normalized = cls._normalize_markdown_for_hash(value)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _lock_id(name: str) -> int:
        return int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:15], 16)

    def _try_advisory_lock(self, name: str):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (self._lock_id(name),))
            locked = bool(cur.fetchone()[0])
        if not locked:
            conn.close()
            return None
        return conn

    def _release_advisory_lock(self, conn, name: str):
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (self._lock_id(name),))
            conn.commit()
        finally:
            conn.close()

    def _record_metadata(self, record: StandardDocumentRecord, content_hash: str, sync_status: str, is_active: bool) -> dict:
        metadata = {
            "source": record.output_filename,
            "source_name": record.source_name,
            "source_key": record.source_key,
            "source_type": record.source_type,
            "file_type": record.file_type,
            "title": record.title,
            "original_url": record.source_url,
            "source_url": record.source_url,
            "published_at": record.published_at,
            "fetched_at": record.fetched_at,
            "freshness_bucket": record.freshness_bucket,
            "content_hash": content_hash,
            "sync_status": sync_status,
            "is_active": "true" if is_active else "false",
            "local_markdown_name": record.output_filename,
            "entry_id": record.entry_id,
        }
        for key, value in dict(record.metadata or {}).items():
            if value in (None, "", [], {}):
                continue
            metadata[key] = value
        return metadata

    def _render_standard_markdown(self, record: StandardDocumentRecord, content_hash: str, sync_status: str = "active", is_active: bool = True) -> str:
        metadata = self._record_metadata(record, content_hash, sync_status, is_active)
        title = record.title.strip() or Path(record.output_filename).stem
        metadata_lines = [
            f"Source: {record.source_name}",
            f"Source Key: {record.source_key}",
            f"Source type: {record.source_type}",
            f"File type: {record.file_type}",
            f"Title: {title}",
            f"Original URL: {record.source_url}",
            f"Published At: {record.published_at}",
            f"Fetched At: {record.fetched_at}",
            f"Freshness Bucket: {record.freshness_bucket}",
            f"Content Hash: {content_hash}",
            f"Sync Status: {sync_status}",
            f"Is Active: {'true' if is_active else 'false'}",
            f"Local Markdown Name: {record.output_filename}",
            f"Entry ID: {record.entry_id}",
        ]
        for key, value in metadata.items():
            if key in _STANDARD_METADATA_FIELDS:
                continue
            if isinstance(value, list):
                if value:
                    metadata_lines.append(f"{key.replace('_', ' ').title()}: {', '.join(str(item) for item in value)}")
                continue
            metadata_lines.append(f"{key.replace('_', ' ').title()}: {value}")

        body = self._collapse_text(record.markdown_body)
        if not re.match(r"^\s*#\s+", body):
            body = f"# {title}\n\n{body}".strip()
        return "\n".join(metadata_lines).strip() + "\n\n" + body.strip() + "\n"

    def _list_documents(self, scope_prefix: str | None = None) -> dict:
        clauses = []
        params = []
        if scope_prefix:
            clauses.append("source_key LIKE %s")
            params.append(f"{scope_prefix}%")
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        d.id,
                        d.document_no,
                        d.source_key,
                        d.content_hash,
                        coalesce(d.is_active, true) AS is_active,
                        d.metadata,
                        EXISTS (SELECT 1 FROM child_chunks c WHERE c.document_id = d.id LIMIT 1) AS has_chunks
                    FROM documents d
                    {where_sql}
                    """,
                    params,
                )
                rows = cur.fetchall()
        return {
            row[2]: {
                "id": row[0],
                "document_no": row[1],
                "source_key": row[2],
                "content_hash": row[3] or "",
                "is_active": bool(row[4]),
                "metadata": dict(row[5] or {}),
                "has_chunks": bool(row[6]),
            }
            for row in rows
            if row[2]
        }

    def _upsert_document_row(self, record: StandardDocumentRecord, content_hash: str, sync_status: str, is_active: bool, deleted_at=None):
        metadata = self._record_metadata(record, content_hash, sync_status, is_active)
        document_no = build_document_no(record.source_key)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM documents
                    WHERE source_key = %s OR document_no = %s
                    ORDER BY CASE WHEN source_key = %s THEN 0 ELSE 1 END
                    LIMIT 1
                    """,
                    (record.source_key, document_no, record.source_key),
                )
                row = cur.fetchone()
                if row:
                    cur.execute(
                        """
                        UPDATE documents
                        SET title = %s,
                            source_name = %s,
                            source_key = %s,
                            file_type = %s,
                            doc_type = %s,
                            department = %s,
                            source_url = %s,
                            content_hash = %s,
                            sync_status = %s,
                            is_active = %s,
                            last_synced_at = NOW(),
                            deleted_at = %s,
                            metadata = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (
                            record.title,
                            record.source_name,
                            record.source_key,
                            record.file_type,
                            record.source_type,
                            metadata.get("department") or None,
                            record.source_url or None,
                            content_hash,
                            sync_status,
                            is_active,
                            deleted_at,
                            json.dumps(metadata, ensure_ascii=False),
                            row[0],
                        ),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO documents (
                            document_no, title, source_name, source_key, file_type, doc_type, department,
                            source_url, content_hash, sync_status, is_active, last_synced_at, deleted_at, metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s::jsonb)
                        """,
                        (
                            document_no,
                            record.title,
                            record.source_name,
                            record.source_key,
                            record.file_type,
                            record.source_type,
                            metadata.get("department") or None,
                            record.source_url or None,
                            content_hash,
                            sync_status,
                            is_active,
                            deleted_at,
                            json.dumps(metadata, ensure_ascii=False),
                        ),
                    )
            conn.commit()

    def _clear_chunks(self, document_id: int):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM child_chunks WHERE document_id = %s", (document_id,))
                cur.execute("DELETE FROM parent_chunks WHERE document_id = %s", (document_id,))
            conn.commit()

    def _remove_markdown_file(self, file_name: str):
        target = (self.markdown_dir / str(file_name or "").strip()).resolve()
        try:
            if target.is_file() and self.markdown_dir.resolve() in target.parents:
                target.unlink()
        except Exception:
            logger.warning("Failed to remove markdown file %s", target, exc_info=True)
            return

    def _deactivate_missing(self, scope_prefix: str, current_source_keys: set[str]) -> int:
        existing = self._list_documents(scope_prefix)
        deactivated = 0
        now = datetime.now()
        for source_key, item in existing.items():
            if source_key in current_source_keys or not item["is_active"]:
                continue
            metadata = dict(item["metadata"] or {})
            self._clear_chunks(item["id"])
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE documents
                        SET is_active = FALSE,
                            sync_status = 'deleted',
                            deleted_at = %s,
                            last_synced_at = %s,
                            updated_at = NOW(),
                            metadata = jsonb_set(coalesce(metadata, '{}'::jsonb), '{sync_status}', to_jsonb('deleted'::text), true)
                        WHERE id = %s
                        """,
                        (now, now, item["id"]),
                    )
                conn.commit()
            self._remove_markdown_file(metadata.get("local_markdown_name") or metadata.get("source"))
            deactivated += 1
        return deactivated

    def _build_local_record_from_file(self, path: Path, *, force_local: bool = False) -> tuple[str, StandardDocumentRecord | None]:
        raw_text = path.read_text(encoding="utf-8")
        metadata = self._extract_front_matter_metadata(raw_text)
        origin, derived_source_key = self._classify_existing_markdown(path, metadata)
        if origin != "local" and not force_local:
            return origin, None
        body = self._strip_front_matter(raw_text)
        title = metadata.get("title") or self._first_heading(body) or path.stem
        source_key = f"local:{path.name}" if force_local else derived_source_key
        source_type = metadata.get("source_type") or "local_document"
        source_url = metadata.get("original_url") or metadata.get("source_url") or ""
        published_at = metadata.get("published_at") or ""
        fetched_at = datetime.now().strftime("%Y-%m-%d")
        freshness_bucket = metadata.get("freshness_bucket") or ("current" if not published_at else metadata.get("freshness_bucket", "unknown"))
        extra_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in _STANDARD_METADATA_FIELDS
        }
        record = StandardDocumentRecord(
            source_key=source_key,
            entry_id=path.stem,
            output_filename=path.name,
            title=title,
            source_name=metadata.get("source_name") or path.name,
            source_url=source_url,
            markdown_body=self._collapse_text(body),
            published_at=published_at,
            fetched_at=fetched_at,
            freshness_bucket=freshness_bucket or "unknown",
            source_type=source_type,
            file_type="md",
            metadata=extra_metadata,
        )
        return "local", record

    def _sync_records(
        self,
        records: list[StandardDocumentRecord],
        *,
        source: str,
        label: str,
        scope_prefix: str,
        downloaded: int,
        trigger_type: str,
        note: str = "",
        conversion_details: list[str] | None = None,
        failure_details: list[str] | None = None,
        progress_callback=None,
        soft_delete_missing: bool = True,
    ) -> SyncRunResult:
        started_at = time.perf_counter()
        existing = self._list_documents(scope_prefix)
        result = SyncRunResult(
            source=source,
            label=label,
            downloaded=downloaded,
            processed=len(records),
            trigger_type=trigger_type,
            scope=scope_prefix.rstrip(":"),
            note=note,
            conversion_details=list(conversion_details or []),
            failure_details=list(failure_details or []),
        )
        result.failed = len(result.failure_details)

        current_source_keys = set()
        for index, record in enumerate(records):
            if progress_callback:
                progress_callback((index + 1) / max(len(records), 1), desc=f"Syncing {record.output_filename}")
            rendered = self._render_standard_markdown(record, content_hash="")
            content_hash = self._content_hash(rendered)
            rendered = self._render_standard_markdown(record, content_hash=content_hash)
            current_source_keys.add(record.source_key)
            existing_item = existing.get(record.source_key)
            if existing_item and existing_item["content_hash"] == content_hash and existing_item["is_active"] and existing_item["has_chunks"]:
                target_path = self.markdown_dir / record.output_filename
                if not target_path.exists():
                    target_path.write_text(rendered, encoding="utf-8")
                self._upsert_document_row(record, content_hash, sync_status="unchanged", is_active=True)
                result.unchanged += 1
                continue

            target_path = self.markdown_dir / record.output_filename
            target_path.write_text(rendered, encoding="utf-8")
            sync_status = "updated" if existing_item else "active"
            self._upsert_document_row(record, content_hash, sync_status=sync_status, is_active=True)
            if existing_item:
                self._clear_chunks(existing_item["id"])
                result.updated += 1
            else:
                result.added += 1

            index_result = self.rag_system.document_manager._index_markdown_paths(
                [target_path],
                progress_callback=None,
                skip_existing=False,
            )
            result.index_added += int(index_result.get("added") or 0)
            result.index_skipped += int(index_result.get("skipped") or 0)

        if soft_delete_missing:
            result.deactivated += self._deactivate_missing(scope_prefix, current_source_keys)

        result.written = result.added
        result.skipped = result.unchanged
        result.status = "completed" if result.failed == 0 else "completed_with_failures"
        result.duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return result

    def sync_local_documents(self, markdown_paths=None, *, trigger_type: str = "manual", progress_callback=None, soft_delete_missing: bool = False, use_lock: bool = True) -> SyncRunResult:
        lock_conn = None
        if use_lock:
            lock_conn = self._try_advisory_lock("knowledge_base_sync")
            if lock_conn is None:
                return SyncRunResult(
                    source="local",
                    label="本地文档同步",
                    status="skipped_locked",
                    trigger_type=trigger_type,
                    scope="local",
                    note="已有其他同步任务正在执行，本轮本地同步跳过。",
                )
        explicit_paths = markdown_paths is not None
        try:
            if markdown_paths is None:
                markdown_paths = sorted(self.markdown_dir.glob("*.md"))
            else:
                markdown_paths = [Path(path) for path in markdown_paths if path]

            records = []
            for path in markdown_paths:
                origin, record = self._build_local_record_from_file(Path(path), force_local=explicit_paths)
                if origin != "local" or record is None:
                    continue
                records.append(record)

            return self._sync_records(
                records,
                source="local",
                label="本地文档同步",
                scope_prefix="local:",
                downloaded=len(records),
                trigger_type=trigger_type,
                note="扫描本地 Markdown 文档并按 source_key 做新增、更新或下线。",
                progress_callback=progress_callback,
                soft_delete_missing=soft_delete_missing and config.KB_SOFT_DELETE_MISSING,
            )
        finally:
            if use_lock:
                self._release_advisory_lock(lock_conn, "knowledge_base_sync")

    def _official_records(self, source: str, limit: int | None = None):
        source = str(source or "").strip().lower()
        if source == "medlineplus":
            discovered_url, records = MedlinePlusXmlImporter().build_sync_records(limit=limit)
            return discovered_url, records, [], []
        if source == "nhc":
            discovered_url, records, conversion_details, failure_details = NhcPdfWhitelistImporter().build_sync_records(limit=limit)
            return discovered_url, records, conversion_details, failure_details
        if source == "who":
            discovered_url, records, failure_details = WhoHtmlWhitelistImporter().build_sync_records(limit=limit)
            return discovered_url, records, [], failure_details
        raise ValueError(f"Unsupported official source: {source}")

    def sync_official_source(self, source: str, *, limit: int | None = None, trigger_type: str = "manual", progress_callback=None, soft_delete_missing: bool = True, use_lock: bool = True) -> SyncRunResult:
        lock_conn = None
        if use_lock:
            lock_conn = self._try_advisory_lock("knowledge_base_sync")
            if lock_conn is None:
                return SyncRunResult(
                    source=source,
                    label=f"{source} 同步",
                    status="skipped_locked",
                    trigger_type=trigger_type,
                    scope=f"official:{source}",
                    note="已有其他同步任务正在执行，本轮官方同步跳过。",
                )
        try:
            discovered_url, records, conversion_details, failure_details = self._official_records(source, limit=limit)
            label = {
                "medlineplus": "MedlinePlus 同步",
                "nhc": "国家卫健委同步",
                "who": "WHO 同步",
            }.get(source, source)
            return self._sync_records(
                records,
                source=source,
                label=label,
                scope_prefix=f"official:{source}:",
                downloaded=len(records) + len(failure_details),
                trigger_type=trigger_type,
                note=f"来源地址：{discovered_url}",
                conversion_details=conversion_details,
                failure_details=failure_details,
                progress_callback=progress_callback,
                soft_delete_missing=soft_delete_missing and config.KB_SOFT_DELETE_MISSING and not failure_details,
            )
        finally:
            if use_lock:
                self._release_advisory_lock(lock_conn, "knowledge_base_sync")

    def sync_all(self, *, trigger_type: str = "scheduler") -> list[SyncRunResult]:
        lock_conn = self._try_advisory_lock("knowledge_base_sync")
        if lock_conn is None:
            return [
                SyncRunResult(
                    source="scheduler",
                    label="知识库定时同步",
                    status="skipped_locked",
                    trigger_type=trigger_type,
                    scope="knowledge_base",
                    note="已有其他同步任务正在执行，本轮跳过。",
                )
            ]
        try:
            results = [
                self.sync_local_documents(trigger_type=trigger_type, soft_delete_missing=True, use_lock=False),
            ]
            for source in config.KB_SYNC_OFFICIAL_SOURCES:
                results.append(self.sync_official_source(source, trigger_type=trigger_type, soft_delete_missing=True, use_lock=False))
            return results
        finally:
            self._release_advisory_lock(lock_conn, "knowledge_base_sync")
