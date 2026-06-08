import json

import psycopg

import config


class AppointmentSkillLogStore:
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
                    INSERT INTO appointment_skill_logs (
                        thread_id,
                        skill_mode,
                        request_type,
                        selected_candidate_count,
                        required_confirmation,
                        final_action,
                        extra_metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        item.get("thread_id") or None,
                        item.get("skill_mode") or "",
                        item.get("request_type") or "",
                        int(item.get("selected_candidate_count") or 0),
                        bool(item.get("required_confirmation")),
                        item.get("final_action") or "",
                        json.dumps(item.get("extra_metadata") or {}, ensure_ascii=False),
                    ),
                )
            conn.commit()

    def summarize_recent(self, limit: int = 200) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT skill_mode, request_type, selected_candidate_count, required_confirmation, final_action
                    FROM appointment_skill_logs
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (int(limit or 200),),
                )
                rows = cur.fetchall()
        total = len(rows)
        if not total:
            return {
                "sample_count": 0,
                "required_confirmation_rate": 0.0,
                "candidate_exposure_rate": 0.0,
                "final_action_distribution": {},
            }
        final_action_distribution = {}
        for row in rows:
            final_action_distribution[row[4] or "unspecified"] = final_action_distribution.get(row[4] or "unspecified", 0) + 1
        return {
            "sample_count": total,
            "required_confirmation_rate": round(sum(1 for row in rows if row[3]) / total, 4),
            "candidate_exposure_rate": round(sum(1 for row in rows if (row[2] or 0) > 0) / total, 4),
            "final_action_distribution": final_action_distribution,
        }
