import json
import config
from pathlib import Path
from typing import List, Dict
import psycopg
from db.document_ids import build_document_no


class ParentStoreManager:
    def __init__(self):
        self._conninfo = (
            f"host={config.POSTGRES_HOST} "
            f"port={config.POSTGRES_PORT} "
            f"dbname={config.POSTGRES_DB} "
            f"user={config.POSTGRES_USER} "
            f"password={config.POSTGRES_PASSWORD}"
        )

    def _connect(self):
        from db.connection import connect; return connect()

    @staticmethod
    def _document_info_from_metadata(metadata: Dict) -> Dict:
        metadata = dict(metadata or {})
        source_name = metadata.get("source", "unknown.md")
        source_path = Path(source_name)
        source_key = str(metadata.get("source_key") or f"local:{source_name}").strip()
        document_no = str(metadata.get("document_no") or build_document_no(source_key)).strip()
        return {
            "document_no": document_no,
            "title": metadata.get("title") or source_name,
            "source_name": metadata.get("source_name") or source_name,
            "source_key": source_key,
            "file_type": metadata.get("file_type") or source_path.suffix.lstrip(".") or "md",
            "doc_type": metadata.get("doc_type") or metadata.get("source_type") or "",
            "department": metadata.get("department") or "",
            "authority_level": metadata.get("authority_level") or "",
            "source_url": metadata.get("source_url") or metadata.get("original_url") or "",
            "content_hash": metadata.get("content_hash") or "",
            "sync_status": metadata.get("sync_status") or "active",
            "is_active": str(metadata.get("is_active", "true")).strip().lower() not in {"false", "0", "no"},
            "last_synced_at": metadata.get("last_synced_at"),
            "deleted_at": metadata.get("deleted_at"),
            "metadata": metadata,
        }

    def _ensure_document(self, conn, metadata: Dict) -> int:
        info = self._document_info_from_metadata(metadata)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM documents
                WHERE source_key = %s OR document_no = %s
                ORDER BY CASE WHEN source_key = %s THEN 0 ELSE 1 END
                LIMIT 1
                """,
                (
                    info["source_key"],
                    info["document_no"],
                    info["source_key"],
                ),
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
                        authority_level = %s,
                        source_url = %s,
                        content_hash = %s,
                        sync_status = %s,
                        is_active = %s,
                        last_synced_at = COALESCE(%s, last_synced_at, NOW()),
                        deleted_at = %s,
                        metadata = %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id
                    """,
                    (
                        info["title"],
                        info["source_name"],
                        info["source_key"],
                        info["file_type"],
                        info["doc_type"] or None,
                        info["department"] or None,
                        info["authority_level"] or None,
                        info["source_url"] or None,
                        info["content_hash"] or None,
                        info["sync_status"],
                        info["is_active"],
                        info["last_synced_at"],
                        info["deleted_at"],
                        json.dumps(info["metadata"], ensure_ascii=False),
                        row[0],
                    ),
                )
                row = cur.fetchone()
            else:
                cur.execute(
                    """
                    INSERT INTO documents (
                        document_no, title, source_name, source_key, file_type, doc_type, department,
                        authority_level, source_url, content_hash, sync_status, is_active,
                        last_synced_at, deleted_at, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()), %s, %s::jsonb)
                    RETURNING id
                    """,
                    (
                        info["document_no"],
                        info["title"],
                        info["source_name"],
                        info["source_key"],
                        info["file_type"],
                        info["doc_type"] or None,
                        info["department"] or None,
                        info["authority_level"] or None,
                        info["source_url"] or None,
                        info["content_hash"] or None,
                        info["sync_status"],
                        info["is_active"],
                        info["last_synced_at"],
                        info["deleted_at"],
                        json.dumps(info["metadata"], ensure_ascii=False),
                    ),
                )
                row = cur.fetchone()
        return row[0]

    def save(self, parent_id: str, content: str, metadata: Dict) -> None:
        with self._connect() as conn:
            document_id = self._ensure_document(conn, metadata)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO parent_chunks (parent_id, document_id, title, department, content, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (parent_id)
                    DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        title = EXCLUDED.title,
                        department = EXCLUDED.department,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        parent_id,
                        document_id,
                        metadata.get("H1") or metadata.get("H2") or metadata.get("H3") or metadata.get("source"),
                        metadata.get("department"),
                        content,
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )
            conn.commit()

    def save_many(self, parents: List) -> None:
        if not parents:
            return
        with self._connect() as conn:
            document_cache = {}
            with conn.cursor() as cur:
                for parent_id, doc in parents:
                    document_no = self._document_info_from_metadata(doc.metadata)["document_no"]
                    document_id = document_cache.get(document_no)
                    if document_id is None:
                        document_id = self._ensure_document(conn, doc.metadata)
                        document_cache[document_no] = document_id
                    cur.execute(
                        """
                        INSERT INTO parent_chunks (parent_id, document_id, title, department, content, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (parent_id)
                        DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            title = EXCLUDED.title,
                            department = EXCLUDED.department,
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata
                        """,
                        (
                            parent_id,
                            document_id,
                            doc.metadata.get("H1") or doc.metadata.get("H2") or doc.metadata.get("H3") or doc.metadata.get("source"),
                            doc.metadata.get("department"),
                            doc.page_content,
                            json.dumps(doc.metadata, ensure_ascii=False),
                        ),
                    )
            conn.commit()

    def load(self, parent_id: str) -> Dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content, metadata
                    FROM parent_chunks
                    WHERE parent_id = %s
                    """,
                    (parent_id,),
                )
                row = cur.fetchone()
        if not row:
            raise FileNotFoundError(f"Parent chunk not found: {parent_id}")
        return {"page_content": row[0], "metadata": row[1] or {}}

    def load_content(self, parent_id: str) -> Dict:
        data = self.load(parent_id)
        return {
            "content": data["page_content"],
            "parent_id": parent_id,
            "metadata": data["metadata"],
        }

    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        unique_ids = list(dict.fromkeys(parent_ids))
        if not unique_ids:
            return []

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT parent_id, content, metadata
                    FROM parent_chunks
                    WHERE parent_id = ANY(%s)
                    """,
                    (unique_ids,),
                )
                rows = cur.fetchall()

        row_map = {
            row[0]: {
                "content": row[1],
                "parent_id": row[0],
                "metadata": row[2] or {},
            }
            for row in rows
        }
        return [row_map[parent_id] for parent_id in unique_ids if parent_id in row_map]

    def clear_store(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE parent_chunks, documents RESTART IDENTITY CASCADE")
            conn.commit()
