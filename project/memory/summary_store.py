import config
import psycopg


class SummaryStore:
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

    def ensure_session(self, thread_id: str, conn=None):
        owns_connection = conn is None
        connection = conn or self._connect()
        try:
            with connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (thread_id)
                    VALUES (%s)
                    ON CONFLICT (thread_id) DO NOTHING
                    """,
                    (thread_id,),
                )
            if owns_connection:
                connection.commit()
        finally:
            if owns_connection:
                connection.close()

    def get_summary(self, thread_id: str) -> str:
        with self._connect() as conn:
            self.ensure_session(thread_id, conn=conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT summary_content
                    FROM chat_session_summaries
                    WHERE thread_id = %s AND summary_type = 'long_term'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (thread_id,),
                )
                row = cur.fetchone()
        return row[0] if row else ""

    def save_summary(self, thread_id: str, summary: str, last_message_index: int = 0):
        if not summary.strip():
            return
        with self._connect() as conn:
            self.ensure_session(thread_id, conn=conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_session_summaries (
                        thread_id, summary_type, summary_content, last_message_index
                    )
                    VALUES (%s, 'long_term', %s, %s)
                    ON CONFLICT (thread_id, summary_type)
                    DO UPDATE SET
                        summary_content = EXCLUDED.summary_content,
                        last_message_index = EXCLUDED.last_message_index,
                        updated_at = NOW()
                    """,
                    (thread_id, summary, last_message_index),
                )
            conn.commit()

    def clear_session(self, thread_id: str):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_session_summaries WHERE thread_id = %s", (thread_id,))
            conn.commit()
