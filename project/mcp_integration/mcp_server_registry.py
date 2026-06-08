"""Platform-curated MCP server registry.

Read-only catalog of MCP servers exposing MCP servers. Maintained by platform
operators (rows are seeded via SQL or admin tooling, not user-facing CRUD).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config
import psycopg

logger = logging.getLogger(__name__)


class MCPServerRegistry:
    """Read-only access to the hospitals table."""

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

    def list_active(self) -> List[Dict[str, Any]]:
        """List all active hospitals (for the binding UI)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, code, name, description, mcp_url, auth_type, is_active, created_at
                    FROM hospitals
                    WHERE is_active = TRUE
                    ORDER BY name
                    """
                )
                rows = cur.fetchall()
        columns = ["id", "code", "name", "description", "mcp_url", "auth_type", "is_active", "created_at"]
        return [dict(zip(columns, row)) for row in rows]

    def get_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Look up a hospital by its short code (e.g., 'xiehe')."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, code, name, description, mcp_url, auth_type, is_active, created_at
                    FROM hospitals
                    WHERE code = %s
                    """,
                    (code.strip().lower(),),
                )
                row = cur.fetchone()
        if not row:
            return None
        columns = ["id", "code", "name", "description", "mcp_url", "auth_type", "is_active", "created_at"]
        return dict(zip(columns, row))

    def upsert_hospital(
        self,
        code: str,
        name: str,
        mcp_url: str,
        description: str = "",
        auth_type: str = "bearer",
        is_active: bool = True,
    ) -> int:
        """Admin function to seed/update the registry."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hospitals (code, name, description, mcp_url, auth_type, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (code) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        mcp_url = EXCLUDED.mcp_url,
                        auth_type = EXCLUDED.auth_type,
                        is_active = EXCLUDED.is_active,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (code.strip().lower(), name, description, mcp_url, auth_type, is_active),
                )
                row_id = cur.fetchone()[0]
            conn.commit()
        return row_id

    def check_reachability(self, timeout: float = 2.0) -> List[Dict[str, Any]]:
        """Best-effort TCP ping each active hospital's mcp_url.

        Returns a list of {code, name, mcp_url, reachable, error} dicts.  Pure
        TCP connect — does NOT speak the MCP protocol, just confirms there is
        something listening on the target host:port.  Cheap (~10ms per host)
        and safe to run at boot.
        """
        import socket
        from urllib.parse import urlparse

        results = []
        for hospital in self.list_active():
            url = hospital.get("mcp_url", "")
            parsed = urlparse(url)
            host = parsed.hostname or ""
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            result = {
                "code": hospital["code"],
                "name": hospital["name"],
                "mcp_url": url,
                "reachable": False,
                "error": "",
            }
            if not host:
                result["error"] = "invalid mcp_url (no host)"
                results.append(result)
                continue
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                sock.connect((host, int(port)))
                result["reachable"] = True
            except (socket.timeout, OSError) as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
            results.append(result)
        return results
