"""Per-user MCP server credential store.

Stores encrypted tokens for each (user_id, hospital_code) pair. Tokens are
encrypted at rest with Fernet (see token_crypto.py).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config
import psycopg

from .token_crypto import decrypt_token, encrypt_token

logger = logging.getLogger(__name__)


class UserMCPCredentialStore:
    """CRUD for user_hospital_credentials, with transparent encryption."""

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

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_credential(
        self,
        user_id: str,
        hospital_code: str,
        plain_token: str,
        label: str = "",
    ) -> int:
        """Insert or replace a user's token for a hospital.

        Token is encrypted before persistence.  Replaces any existing entry
        for the same (user_id, hospital_code).
        """
        if not plain_token.strip():
            raise ValueError("token 不能为空")
        encrypted = encrypt_token(plain_token.strip())
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_hospital_credentials
                        (user_id, hospital_code, token_encrypted, label)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, hospital_code) DO UPDATE SET
                        token_encrypted = EXCLUDED.token_encrypted,
                        label = EXCLUDED.label,
                        updated_at = NOW(),
                        last_health_status = 'unknown',
                        last_health_at = NULL
                    RETURNING id
                    """,
                    (user_id, hospital_code.strip().lower(), encrypted, label.strip()),
                )
                row_id = cur.fetchone()[0]
            conn.commit()
        return row_id

    def delete_credential(self, user_id: str, hospital_code: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM user_hospital_credentials
                    WHERE user_id = %s AND hospital_code = %s
                    """,
                    (user_id, hospital_code.strip().lower()),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def update_health(
        self,
        user_id: str,
        hospital_code: str,
        status: str,
    ):
        """Update last health check result (called by UserMCPPool)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE user_hospital_credentials
                    SET last_health_status = %s,
                        last_health_at = NOW()
                    WHERE user_id = %s AND hospital_code = %s
                    """,
                    (status, user_id, hospital_code.strip().lower()),
                )
            conn.commit()

    def mark_used(self, user_id: str, hospital_code: str):
        """Update last_used_at after a successful tool call."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE user_hospital_credentials
                    SET last_used_at = NOW()
                    WHERE user_id = %s AND hospital_code = %s
                    """,
                    (user_id, hospital_code.strip().lower()),
                )
            conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """List user's hospital bindings (without decrypted tokens)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, hospital_code, label,
                           last_used_at, last_health_status, last_health_at,
                           created_at, updated_at
                    FROM user_hospital_credentials
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()
        columns = [
            "id", "user_id", "hospital_code", "label",
            "last_used_at", "last_health_status", "last_health_at",
            "created_at", "updated_at",
        ]
        return [dict(zip(columns, row)) for row in rows]

    def get_decrypted_token(self, user_id: str, hospital_code: str) -> Optional[str]:
        """Fetch and decrypt a single credential's token.  None if not found."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT token_encrypted
                    FROM user_hospital_credentials
                    WHERE user_id = %s AND hospital_code = %s
                    """,
                    (user_id, hospital_code.strip().lower()),
                )
                row = cur.fetchone()
        if not row:
            return None
        return decrypt_token(row[0])

    def get_all_decrypted(self, user_id: str) -> Dict[str, str]:
        """Return {hospital_code: plain_token} for all of user's bindings."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT hospital_code, token_encrypted
                    FROM user_hospital_credentials
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()
        result = {}
        for code, encrypted in rows:
            decrypted = decrypt_token(encrypted)
            if decrypted:
                result[code] = decrypted
        return result
