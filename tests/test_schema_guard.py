import sys
import unittest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from db.schema_guard import EmbeddingSchemaGuard  # noqa: E402


class FakeCursor:
    def __init__(self, rows=None, exc=None):
        self.rows = rows or []
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, params=None):
        if self.exc:
            raise self.exc

    def fetchall(self):
        return list(self.rows)


class FakeConnection:
    def __init__(self, rows=None, exc=None):
        self.rows = rows or []
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return FakeCursor(self.rows, self.exc)


def fake_connect(rows=None, exc=None):
    return lambda: FakeConnection(rows=rows, exc=exc)


class EmbeddingSchemaGuardTests(unittest.TestCase):
    def test_check_ok_when_all_vector_dimensions_match(self):
        rows = [
            ("child_chunks", "embedding", "vector(1024)"),
            ("user_memories", "embedding", "vector(1024)"),
            ("episodic_memories", "embedding", "vector(1024)"),
            ("reflection_memories", "embedding", "vector(1024)"),
        ]
        guard = EmbeddingSchemaGuard(
            expected_dimension=1024,
            app_env="development",
            connect_fn=fake_connect(rows),
        )

        result = guard.check()

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.actual_dimensions["child_chunks.embedding"], 1024)
        self.assertEqual(result.errors, [])

    def test_check_degraded_in_development_when_dimension_mismatches(self):
        rows = [
            ("child_chunks", "embedding", "vector(768)"),
            ("user_memories", "embedding", "vector(1024)"),
            ("episodic_memories", "embedding", "vector(1024)"),
            ("reflection_memories", "embedding", "vector(1024)"),
        ]
        guard = EmbeddingSchemaGuard(
            expected_dimension=1024,
            app_env="development",
            connect_fn=fake_connect(rows),
        )

        result = guard.check()

        self.assertEqual(result.status, "degraded")
        self.assertIn("child_chunks.embedding is vector(768)", result.errors[0])

    def test_check_failed_in_production_when_database_unavailable(self):
        guard = EmbeddingSchemaGuard(
            expected_dimension=1024,
            app_env="production",
            connect_fn=fake_connect(exc=RuntimeError("database unavailable")),
        )

        result = guard.check()

        self.assertEqual(result.status, "failed")
        self.assertIn("database unavailable", result.errors[0])
        with self.assertRaises(RuntimeError):
            guard.assert_compatible()


if __name__ == "__main__":
    unittest.main()
