from __future__ import annotations
import json

import psycopg

import config


class ImportTaskStore:
    def __init__(self):
        self._conninfo = (
            f"host={config.POSTGRES_HOST} "
            f"port={config.POSTGRES_PORT} "
            f"dbname={config.POSTGRES_DB} "
            f"user={config.POSTGRES_USER} "
            f"password={config.POSTGRES_PASSWORD}"
        )

    def _connect(self):
        return psycopg.connect(self._conninfo)

    def save_event(self, event: dict):
        payload = dict(event or {})
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO import_task_logs (
                        source,
                        label,
                        status,
                        downloaded,
                        written,
                        updated,
                        deactivated,
                        unchanged,
                        skipped,
                        failed,
                        index_added,
                        index_skipped,
                        duration_ms,
                        note,
                        trigger_type,
                        scope,
                        conversion_details,
                        failure_details
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        payload.get("source", ""),
                        payload.get("label", ""),
                        payload.get("status", "completed"),
                        int(payload.get("downloaded") or 0),
                        int(payload.get("written") or 0),
                        int(payload.get("updated") or 0),
                        int(payload.get("deactivated") or 0),
                        int(payload.get("unchanged") or 0),
                        int(payload.get("skipped") or 0),
                        int(payload.get("failed") or 0),
                        int(payload.get("index_added") or 0),
                        int(payload.get("index_skipped") or 0),
                        float(payload.get("duration_ms") or 0),
                        payload.get("note", ""),
                        payload.get("trigger_type", "manual"),
                        payload.get("scope", ""),
                        json.dumps(payload.get("conversion_details") or [], ensure_ascii=False),
                        json.dumps(payload.get("failure_details") or [], ensure_ascii=False),
                    ),
                )
            conn.commit()

    def list_recent(self, limit: int | None = None) -> list[dict]:
        effective_limit = int(limit or config.RECENT_IMPORT_TASK_LIMIT)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        source,
                        label,
                        status,
                        downloaded,
                        written,
                        updated,
                        deactivated,
                        unchanged,
                        skipped,
                        failed,
                        index_added,
                        index_skipped,
                        duration_ms,
                        note,
                        trigger_type,
                        scope,
                        conversion_details,
                        failure_details,
                        created_at
                    FROM import_task_logs
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (effective_limit,),
                )
                rows = cur.fetchall()

        events = []
        for row in rows:
            events.append(
                {
                    "source": row[0],
                    "label": row[1],
                    "status": row[2],
                    "downloaded": int(row[3] or 0),
                    "written": int(row[4] or 0),
                    "updated": int(row[5] or 0),
                    "deactivated": int(row[6] or 0),
                    "unchanged": int(row[7] or 0),
                    "skipped": int(row[8] or 0),
                    "failed": int(row[9] or 0),
                    "index_added": int(row[10] or 0),
                    "index_skipped": int(row[11] or 0),
                    "duration_ms": float(row[12] or 0),
                    "note": row[13] or "",
                    "trigger_type": row[14] or "manual",
                    "scope": row[15] or "",
                    "conversion_details": list(row[16] or []),
                    "failure_details": list(row[17] or []),
                    "timestamp": row[18].strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        return events
