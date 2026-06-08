import json
from collections import Counter

import psycopg

import config


class RouteLogStore:
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

    def save_log(self, payload: dict):
        item = dict(payload or {})
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO route_logs (
                        request_id,
                        thread_id,
                        user_query,
                        primary_intent,
                        secondary_intent,
                        decision_source,
                        route_reason,
                        had_pending_state,
                        extra_metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        item.get("request_id") or None,
                        item.get("thread_id") or None,
                        item.get("user_query") or "",
                        item.get("primary_intent") or "",
                        item.get("secondary_intent") or "",
                        item.get("decision_source") or "",
                        item.get("route_reason") or "",
                        bool(item.get("had_pending_state")),
                        json.dumps(item.get("extra_metadata") or {}, ensure_ascii=False),
                    ),
                )
            conn.commit()

    def list_recent(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        request_id,
                        thread_id,
                        user_query,
                        primary_intent,
                        secondary_intent,
                        decision_source,
                        route_reason,
                        had_pending_state,
                        extra_metadata,
                        created_at
                    FROM route_logs
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (int(limit or 20),),
                )
                rows = cur.fetchall()

        return [
            {
                "request_id": row[0] or "",
                "thread_id": row[1] or "",
                "user_query": row[2] or "",
                "primary_intent": row[3] or "",
                "secondary_intent": row[4] or "",
                "decision_source": row[5] or "",
                "route_reason": row[6] or "",
                "had_pending_state": bool(row[7]),
                "extra_metadata": dict(row[8] or {}),
                "timestamp": row[9].strftime("%Y-%m-%d %H:%M:%S"),
            }
            for row in rows
        ]

    def summarize_recent(self, limit: int = 200) -> dict:
        events = self.list_recent(limit=limit)
        total = len(events)
        if total == 0:
            return {
                "sample_count": 0,
                "compound_request_rate": 0.0,
                "pending_resume_rate": 0.0,
                "checkpoint_resume_rate": 0.0,
                "secondary_turn_completion_rate": 0.0,
                "deferred_question_rate": 0.0,
                "intent_distribution": {},
                "secondary_intent_distribution": {},
                "decision_source_distribution": {},
                "route_reason_distribution": {},
            }

        def _ratio(count: int) -> float:
            return round(count / total, 4)

        intent_counter = Counter(item.get("primary_intent") or "unspecified" for item in events)
        secondary_counter = Counter(item.get("secondary_intent") or "none" for item in events)
        source_counter = Counter(item.get("decision_source") or "unspecified" for item in events)
        reason_counter = Counter(item.get("route_reason") or "unspecified" for item in events)

        return {
            "sample_count": total,
            "compound_request_rate": _ratio(sum(1 for item in events if item.get("secondary_intent"))),
            "pending_resume_rate": _ratio(sum(1 for item in events if item.get("had_pending_state"))),
            "checkpoint_resume_rate": _ratio(
                sum(1 for item in events if (item.get("extra_metadata") or {}).get("checkpoint_resumed"))
            ),
            "secondary_turn_completion_rate": _ratio(
                sum(1 for item in events if (item.get("extra_metadata") or {}).get("secondary_turn_executed"))
            ),
            "deferred_question_rate": _ratio(
                sum(1 for item in events if (item.get("extra_metadata") or {}).get("deferred_user_question"))
            ),
            "intent_distribution": dict(intent_counter),
            "secondary_intent_distribution": dict(secondary_counter),
            "decision_source_distribution": dict(source_counter),
            "route_reason_distribution": dict(reason_counter),
        }

    def build_recent_report(self, limit: int = 50) -> dict:
        events = self.list_recent(limit=limit)
        return {
            "summary": self.summarize_recent(limit=limit),
            "events": events,
        }
