from collections import Counter

import psycopg

import config


class RetrievalLogStore:
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

    def list_recent(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        request_id,
                        thread_id,
                        query_text,
                        rewritten_query,
                        retrieval_mode,
                        top_k,
                        result_count,
                        selected_parent_ids,
                        query_plan,
                        graded_doc_count,
                        sufficiency_result,
                        retry_count,
                        final_confidence_bucket,
                        created_at
                    FROM retrieval_logs
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
                "query_text": row[2] or "",
                "rewritten_query": row[3] or "",
                "retrieval_mode": row[4] or "",
                "top_k": int(row[5] or 0),
                "result_count": int(row[6] or 0),
                "selected_parent_ids": list(row[7] or []),
                "query_plan": list(row[8] or []),
                "graded_doc_count": int(row[9] or 0),
                "sufficiency_result": row[10] or "",
                "retry_count": int(row[11] or 0),
                "final_confidence_bucket": row[12] or "",
                "timestamp": row[13].strftime("%Y-%m-%d %H:%M:%S"),
            }
            for row in rows
        ]

    def summarize_recent(self, limit: int = 200) -> dict:
        events = self.list_recent(limit=limit)
        total = len(events)
        if total == 0:
            return {
                "sample_count": 0,
                "retry_rate": 0.0,
                "multi_query_rate": 0.0,
                "no_evidence_rate": 0.0,
                "low_confidence_rate": 0.0,
                "avg_query_plan_size": 0.0,
                "avg_result_count": 0.0,
                "avg_retry_count": 0.0,
                "confidence_distribution": {},
                "sufficiency_distribution": {},
            }

        def _ratio(count: int) -> float:
            return round(count / total, 4)

        confidence_counter = Counter(item.get("final_confidence_bucket") or "unspecified" for item in events)
        sufficiency_counter = Counter(item.get("sufficiency_result") or "unspecified" for item in events)
        low_confidence = {"low", "no_evidence"}
        return {
            "sample_count": total,
            "retry_rate": _ratio(sum(1 for item in events if int(item.get("retry_count") or 0) > 0)),
            "multi_query_rate": _ratio(sum(1 for item in events if len(item.get("query_plan") or []) > 1)),
            "no_evidence_rate": _ratio(sum(1 for item in events if item.get("final_confidence_bucket") == "no_evidence")),
            "low_confidence_rate": _ratio(sum(1 for item in events if item.get("final_confidence_bucket") in low_confidence)),
            "avg_query_plan_size": round(sum(len(item.get("query_plan") or []) for item in events) / total, 3),
            "avg_result_count": round(sum(int(item.get("result_count") or 0) for item in events) / total, 3),
            "avg_retry_count": round(sum(int(item.get("retry_count") or 0) for item in events) / total, 3),
            "confidence_distribution": dict(confidence_counter),
            "sufficiency_distribution": dict(sufficiency_counter),
        }

    def build_recent_report(self, limit: int = 50) -> dict:
        events = self.list_recent(limit=limit)
        return {
            "summary": self.summarize_recent(limit=limit),
            "events": events,
        }
