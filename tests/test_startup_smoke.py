import asyncio
import contextlib
import io
import sys
import unittest
import warnings

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

warnings.filterwarnings("ignore", category=ResourceWarning)

import psycopg  # noqa: E402
import config  # noqa: E402


def _db_available() -> bool:
    try:
        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            connect_timeout=1,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("select 1")
                cur.fetchone()
        return True
    except Exception:
        return False


@unittest.skipUnless(_db_available(), "PostgreSQL is unavailable for startup smoke tests.")
class StartupSmokeTests(unittest.TestCase):
    def _close_event_loop(self):
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            return
        if loop and not loop.is_closed():
            loop.close()
            asyncio.set_event_loop(None)

    def test_rag_system_initialize(self):
        from core.rag_system import RAGSystem

        rag = RAGSystem()
        rag.initialize()

        self.assertIsNotNone(rag.agent_graph)

    def test_gradio_ui_creation(self):
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed event loop.*")
            warnings.filterwarnings("ignore", category=ResourceWarning)
            from ui.gradio_app import create_gradio_ui
            demo = create_gradio_ui(start_background_tasks=False)
        try:
            self.assertIsNotNone(demo)
        finally:
            close = getattr(demo, "close", None)
            if callable(close):
                close()
            self._close_event_loop()


if __name__ == "__main__":
    unittest.main()
