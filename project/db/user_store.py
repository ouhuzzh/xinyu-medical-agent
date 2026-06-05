"""User account store — registration, login, and user management.

Stores user accounts in PostgreSQL with bcrypt-hashed passwords.
Supports username/password registration and lookup.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config
import psycopg

logger = logging.getLogger(__name__)


class UserStore:
    """PostgreSQL-backed user account store."""

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

    def create_user(
        self,
        username: str,
        password: str,
        display_name: str = "",
        role: str = "user",
    ) -> Dict[str, Any]:
        """Create a new user.  Raises ValueError if username exists or validation fails."""
        username = username.strip().lower()
        if not username or len(username) < 2:
            raise ValueError("用户名至少需要 2 个字符。")
        if len(password) < config.PASSWORD_MIN_LENGTH:
            raise ValueError(f"密码至少需要 {config.PASSWORD_MIN_LENGTH} 个字符。")
        if role not in ("user", "admin"):
            raise ValueError("角色必须是 user 或 admin。")

        password_hash = self._hash_password(password)
        display_name = display_name.strip() or username

        with self._connect() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO users (username, display_name, password_hash, role)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, username, display_name, role, is_active, created_at
                        """,
                        (username, display_name, password_hash, role),
                    )
                    row = cur.fetchone()
                except psycopg.errors.UniqueViolation:
                    raise ValueError(f"用户名 '{username}' 已存在。")
            conn.commit()

        columns = ["id", "username", "display_name", "role", "is_active", "created_at"]
        return dict(zip(columns, row))

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Look up a user by username.  Returns None if not found."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username, display_name, password_hash, role, is_active, created_at, updated_at
                    FROM users
                    WHERE username = %s
                    """,
                    (username.strip().lower(),),
                )
                row = cur.fetchone()
        if not row:
            return None
        columns = ["id", "username", "display_name", "password_hash", "role", "is_active", "created_at", "updated_at"]
        return dict(zip(columns, row))

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Look up a user by id."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username, display_name, role, is_active, created_at
                    FROM users
                    WHERE id = %s
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        columns = ["id", "username", "display_name", "role", "is_active", "created_at"]
        return dict(zip(columns, row))

    def verify_password(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify username/password.  Returns user dict (without hash) on success, None on failure."""
        user = self.get_user_by_username(username)
        if user is None:
            return None
        if not user.get("is_active", True):
            return None
        if not self._check_password(password, user["password_hash"]):
            return None
        # Return without password_hash
        return {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
        }

    def update_display_name(self, user_id: int, display_name: str):
        """Update a user's display name."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users SET display_name = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (display_name.strip(), user_id),
                )
            conn.commit()

    def change_password(self, user_id: int, old_password: str, new_password: str):
        """Change password after verifying old password."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT password_hash FROM users WHERE id = %s",
                    (user_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError("用户不存在。")
                if not self._check_password(old_password, row[0]):
                    raise ValueError("旧密码不正确。")
                if len(new_password) < config.PASSWORD_MIN_LENGTH:
                    raise ValueError(f"新密码至少需要 {config.PASSWORD_MIN_LENGTH} 个字符。")
                new_hash = self._hash_password(new_password)
                cur.execute(
                    """
                    UPDATE users SET password_hash = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (new_hash, user_id),
                )
            conn.commit()

    def list_users(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all users (admin function)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username, display_name, role, is_active, created_at
                    FROM users
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        columns = ["id", "username", "display_name", "role", "is_active", "created_at"]
        return [dict(zip(columns, row)) for row in rows]

    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        import bcrypt
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def _check_password(password: str, password_hash: str) -> bool:
        """Verify a password against a bcrypt hash."""
        import bcrypt
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except Exception:
            return False
