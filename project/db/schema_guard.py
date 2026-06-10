"""Runtime schema checks that protect model/schema compatibility."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

import config
from db.connection import connect


logger = logging.getLogger(__name__)

EMBEDDING_VECTOR_COLUMNS = (
    ("child_chunks", "embedding"),
    ("user_memories", "embedding"),
    ("episodic_memories", "embedding"),
    ("reflection_memories", "embedding"),
)


@dataclass
class SchemaGuardResult:
    status: str
    message: str
    expected_dimension: int
    actual_dimensions: dict[str, int | None]
    errors: list[str]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "expected_dimension": self.expected_dimension,
            "actual_dimensions": dict(self.actual_dimensions),
            "errors": list(self.errors),
        }


class EmbeddingSchemaGuard:
    def __init__(
        self,
        *,
        expected_dimension: int | None = None,
        app_env: str | None = None,
        connect_fn: Callable | None = None,
    ):
        self.expected_dimension = int(expected_dimension or config.VECTOR_DIMENSION)
        self.app_env = (app_env or config.APP_ENV or "development").strip().lower()
        self._connect = connect_fn or connect
        self._last_result: SchemaGuardResult | None = None

    def backend_name(self) -> str:
        return "postgres"

    def check(self) -> SchemaGuardResult:
        try:
            actual_dimensions = self._load_vector_dimensions()
        except Exception as exc:
            result = SchemaGuardResult(
                status="failed" if self.app_env == "production" else "degraded",
                message="Embedding schema check could not query PostgreSQL.",
                expected_dimension=self.expected_dimension,
                actual_dimensions={},
                errors=[str(exc)],
            )
            self._last_result = result
            logger.warning("Embedding schema guard failed to inspect PostgreSQL", exc_info=True)
            return result

        errors = []
        for table_name, column_name in EMBEDDING_VECTOR_COLUMNS:
            key = f"{table_name}.{column_name}"
            actual = actual_dimensions.get(key)
            if actual is None:
                errors.append(f"{key} is missing or is not declared as vector({self.expected_dimension}).")
            elif actual != self.expected_dimension:
                errors.append(f"{key} is vector({actual}), expected vector({self.expected_dimension}).")

        status = "ok" if not errors else ("failed" if self.app_env == "production" else "degraded")
        message = (
            "Embedding vector dimensions match configuration."
            if status == "ok"
            else "Embedding vector dimensions do not match configuration."
        )
        result = SchemaGuardResult(
            status=status,
            message=message,
            expected_dimension=self.expected_dimension,
            actual_dimensions=actual_dimensions,
            errors=errors,
        )
        self._last_result = result
        return result

    def get_health(self, *, refresh: bool = False) -> dict:
        if refresh or self._last_result is None:
            return self.check().to_dict()
        return self._last_result.to_dict()

    def assert_compatible(self):
        result = self.check()
        if result.status == "failed":
            raise RuntimeError(result.message + " " + "; ".join(result.errors))
        return result

    def _load_vector_dimensions(self) -> dict[str, int | None]:
        table_names = sorted({table_name for table_name, _ in EMBEDDING_VECTOR_COLUMNS})
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = current_schema()
                      AND c.relname = ANY(%s)
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped
                    """,
                    (table_names,),
                )
                rows = cur.fetchall()

        actual = {f"{table}.{column}": None for table, column in EMBEDDING_VECTOR_COLUMNS}
        for table_name, column_name, data_type in rows:
            key = f"{table_name}.{column_name}"
            if key not in actual:
                continue
            actual[key] = self._parse_vector_dimension(str(data_type or ""))
        return actual

    @staticmethod
    def _parse_vector_dimension(data_type: str) -> int | None:
        match = re.fullmatch(r"vector\((\d+)\)", data_type.strip())
        if not match:
            return None
        return int(match.group(1))
