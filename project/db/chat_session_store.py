import uuid

import config
import psycopg


class ChatSessionStore:
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

    def create_session(self, owner_user_id: str) -> str:
        thread_id = uuid.uuid4().hex
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (thread_id, owner_user_id)
                    VALUES (%s, %s)
                    """,
                    (thread_id, owner_user_id),
                )
            conn.commit()
        return thread_id

    def get_session(self, thread_id: str):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT thread_id, owner_user_id, status, created_at, updated_at
                    FROM chat_sessions
                    WHERE thread_id = %s
                    LIMIT 1
                    """,
                    (thread_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "thread_id": row[0],
            "owner_user_id": row[1] or "",
            "status": row[2] or "",
            "created_at": row[3],
            "updated_at": row[4],
        }

    def assign_owner_if_missing(self, thread_id: str, owner_user_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET owner_user_id = %s,
                        updated_at = NOW()
                    WHERE thread_id = %s
                      AND coalesce(owner_user_id, '') = ''
                    """,
                    (owner_user_id, thread_id),
                )
                changed = cur.rowcount > 0
            conn.commit()
        return changed
