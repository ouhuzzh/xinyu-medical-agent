"""Append-only audit log store.

Records security-relevant operations so they can be reviewed after the fact:
who did what, when, from where, and whether it succeeded.

Design choices:
  - Append-only by convention (no DELETE/UPDATE methods exposed).  If you
    need to purge for retention compliance, do it via a scheduled SQL job.
  - Best-effort: failure to write an audit row must NEVER break the action
    being audited.  All public methods catch and log internally.
  - Structured `detail` field (JSONB) carries action-specific metadata
    without proliferating columns.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)


class AuditLogStore:
    def __init__(self):
        self._conninfo = (
            f"host={config.POSTGRES_HOST} "
            f"port={config.POSTGRES_PORT} "
            f"dbname={config.POSTGRES_DB} "
            f"user={config.POSTGRES_USER} "
            f"password={config.POSTGRES_PASSWORD}"
        )

    def _connect(self):
        from db.connection import connect
        return connect()

    def record(
        self,
        *,
        action: str,
        actor_user_id: str = "",
        actor_username: str = "",
        target_type: str = "",
        target_id: str = "",
        client_ip: str = "",
        request_id: str = "",
        success: bool = True,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert an audit row.  Swallows all DB errors — auditing must never
        break the operation being audited."""
        try:
            payload = json.dumps(detail or {}, ensure_ascii=False)
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO audit_log
                            (actor_user_id, actor_username, action,
                             target_type, target_id, client_ip, request_id,
                             success, detail)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            actor_user_id or "",
                            actor_username or "",
                            action,
                            target_type or "",
                            target_id or "",
                            client_ip or "",
                            request_id or "",
                            bool(success),
                            payload,
                        ),
                    )
                conn.commit()
        except Exception:
            # Last-resort log so the event isn't completely lost.
            logger.warning(
                "audit_log_write_failed action=%s actor=%s success=%s",
                action, actor_user_id, success, exc_info=True,
            )

    def list_recent(
        self,
        *,
        actor_user_id: str = "",
        action: str = "",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return recent audit rows, newest first.  For admin/debug use."""
        clauses = []
        params: list[Any] = []
        if actor_user_id:
            clauses.append("actor_user_id = %s")
            params.append(actor_user_id)
        if action:
            clauses.append("action = %s")
            params.append(action)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        sql = f"""
            SELECT id, actor_user_id, actor_username, action, target_type,
                   target_id, client_ip, request_id, success, detail, created_at
            FROM audit_log
            {where}
            ORDER BY id DESC
            LIMIT %s
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        except Exception:
            logger.warning("audit_log_list_failed", exc_info=True)
            return []
        cols = [
            "id", "actor_user_id", "actor_username", "action", "target_type",
            "target_id", "client_ip", "request_id", "success", "detail", "created_at",
        ]
        return [dict(zip(cols, r)) for r in rows]
