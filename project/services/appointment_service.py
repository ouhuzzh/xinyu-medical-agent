from __future__ import annotations
import json
import uuid
from datetime import date
import psycopg
import config
from db.schema_manager import SchemaManager


class AppointmentService:
    def __init__(self):
        self._conninfo = (
            f"host={config.POSTGRES_HOST} "
            f"port={config.POSTGRES_PORT} "
            f"dbname={config.POSTGRES_DB} "
            f"user={config.POSTGRES_USER} "
            f"password={config.POSTGRES_PASSWORD}"
        )
        self._schema_manager = SchemaManager(self._conninfo)

    def _connect(self):
        self._schema_manager.apply_migrations()
        return psycopg.connect(self._conninfo)

    def ensure_patient_for_thread(self, thread_id: str, conn=None) -> int:
        patient_no = "thread-" + uuid.uuid5(uuid.NAMESPACE_URL, thread_id).hex
        patient_name = f"Session {thread_id[:8]}"
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
                cur.execute("SELECT patient_id FROM chat_sessions WHERE thread_id = %s", (thread_id,))
                row = cur.fetchone()
                if row and row[0]:
                    if owns_connection:
                        connection.commit()
                    return row[0]

                cur.execute(
                    """
                    INSERT INTO patients (patient_no, name)
                    VALUES (%s, %s)
                    ON CONFLICT (patient_no)
                    DO UPDATE SET name = EXCLUDED.name
                    RETURNING id
                    """,
                    (patient_no, patient_name),
                )
                patient_id = cur.fetchone()[0]
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET patient_id = %s
                    WHERE thread_id = %s
                    """,
                    (patient_id, thread_id),
                )
            if owns_connection:
                connection.commit()
        finally:
            if owns_connection:
                connection.close()
        return patient_id

    def find_department_by_name(self, name: str, conn=None):
        if not name:
            return None
        owns_connection = conn is None
        connection = conn or self._connect()
        try:
            with connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, code, name
                    FROM departments
                    WHERE lower(name) = lower(%s)
                       OR lower(code) = lower(%s)
                       OR lower(name) LIKE lower(%s)
                    ORDER BY CASE WHEN lower(name) = lower(%s) THEN 0 ELSE 1 END, id
                    LIMIT 1
                    """,
                    (name, name, f"%{name}%", name),
                )
                row = cur.fetchone()
        finally:
            if owns_connection:
                connection.close()
        if not row:
            return None
        return {"id": row[0], "code": row[1], "name": row[2]}

    def list_departments(self, query: str | None = None, limit: int = 12, conn=None):
        owns_connection = conn is None
        connection = conn or self._connect()
        try:
            with connection.cursor() as cur:
                if query:
                    cur.execute(
                        """
                        SELECT id, code, name
                        FROM departments
                        WHERE lower(name) LIKE lower(%s)
                           OR lower(code) LIKE lower(%s)
                        ORDER BY CASE WHEN lower(name) = lower(%s) THEN 0 ELSE 1 END, name, id
                        LIMIT %s
                        """,
                        (f"%{query}%", f"%{query}%", query, int(limit or 12)),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, code, name
                        FROM departments
                        ORDER BY name, id
                        LIMIT %s
                        """,
                        (int(limit or 12),),
                    )
                rows = cur.fetchall()
        finally:
            if owns_connection:
                connection.close()
        return [{"id": row[0], "code": row[1], "name": row[2]} for row in rows]

    def find_available_schedule(self, department: str, schedule_date: date, time_slot: str, doctor_name: str | None = None, conn=None):
        owns_connection = conn is None
        connection = conn or self._connect()
        department_row = self.find_department_by_name(department, conn=connection)
        if not department_row:
            if owns_connection:
                connection.close()
            return None

        try:
            with connection.cursor() as cur:
                if doctor_name:
                    cur.execute(
                        """
                        SELECT ds.id, ds.doctor_id, ds.department_id, ds.schedule_date, ds.time_slot,
                               ds.quota_available, d.name
                        FROM doctor_schedules ds
                        JOIN doctors d ON d.id = ds.doctor_id
                        WHERE ds.department_id = %s
                          AND ds.schedule_date = %s
                          AND ds.time_slot = %s
                          AND ds.quota_available > 0
                          AND lower(d.name) LIKE lower(%s)
                        ORDER BY ds.id
                        LIMIT 1
                        """,
                        (department_row["id"], schedule_date, time_slot, f"%{doctor_name}%"),
                    )
                else:
                    cur.execute(
                        """
                        SELECT ds.id, ds.doctor_id, ds.department_id, ds.schedule_date, ds.time_slot,
                               ds.quota_available, d.name
                        FROM doctor_schedules ds
                        JOIN doctors d ON d.id = ds.doctor_id
                        WHERE ds.department_id = %s
                          AND ds.schedule_date = %s
                          AND ds.time_slot = %s
                          AND ds.quota_available > 0
                        ORDER BY ds.id
                        LIMIT 1
                        """,
                        (department_row["id"], schedule_date, time_slot),
                    )
                row = cur.fetchone()
        finally:
            if owns_connection:
                connection.close()
        if not row:
            return None
        return {
            "schedule_id": row[0],
            "doctor_id": row[1],
            "department_id": row[2],
            "schedule_date": row[3],
            "time_slot": row[4],
            "quota_available": row[5],
            "doctor_name": row[6],
            "department_name": department_row["name"],
        }

    def list_available_doctors(self, department: str, schedule_date: date, time_slot: str, conn=None):
        owns_connection = conn is None
        connection = conn or self._connect()
        department_row = self.find_department_by_name(department, conn=connection)
        if not department_row:
            if owns_connection:
                connection.close()
            return []

        try:
            with connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT ds.id, ds.doctor_id, ds.department_id, ds.schedule_date, ds.time_slot,
                           ds.quota_available, d.name
                    FROM doctor_schedules ds
                    JOIN doctors d ON d.id = ds.doctor_id
                    WHERE ds.department_id = %s
                      AND ds.schedule_date = %s
                      AND ds.time_slot = %s
                      AND ds.quota_available > 0
                    ORDER BY d.name, ds.id
                    """,
                    (department_row["id"], schedule_date, time_slot),
                )
                rows = cur.fetchall()
        finally:
            if owns_connection:
                connection.close()

        return [
            {
                "schedule_id": row[0],
                "doctor_id": row[1],
                "department_id": row[2],
                "schedule_date": row[3],
                "time_slot": row[4],
                "quota_available": row[5],
                "doctor_name": row[6],
                "department_name": department_row["name"],
            }
            for row in rows
        ]

    def get_doctor_availability(
        self,
        doctor_name: str,
        *,
        department: str | None = None,
        schedule_date: date | None = None,
        time_slot: str | None = None,
        limit: int = 6,
        conn=None,
    ):
        if not doctor_name:
            return []
        owns_connection = conn is None
        connection = conn or self._connect()
        try:
            params = [f"%{doctor_name}%"]
            conditions = ["lower(d.name) LIKE lower(%s)", "ds.quota_available > 0"]
            if department:
                department_row = self.find_department_by_name(department, conn=connection)
                if not department_row:
                    return []
                conditions.append("ds.department_id = %s")
                params.append(department_row["id"])
            if schedule_date:
                conditions.append("ds.schedule_date = %s")
                params.append(schedule_date)
            if time_slot:
                conditions.append("ds.time_slot = %s")
                params.append(time_slot)
            else:
                conditions.append("ds.schedule_date >= %s")
                params.append(date.today())
            params.append(int(limit or 6))
            with connection.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        ds.id, ds.doctor_id, ds.department_id, ds.schedule_date, ds.time_slot,
                        ds.quota_available, d.name, dep.name
                    FROM doctor_schedules ds
                    JOIN doctors d ON d.id = ds.doctor_id
                    JOIN departments dep ON dep.id = ds.department_id
                    WHERE {' AND '.join(conditions)}
                    ORDER BY ds.schedule_date, ds.time_slot, d.name, ds.id
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        finally:
            if owns_connection:
                connection.close()

        return [
            {
                "schedule_id": row[0],
                "doctor_id": row[1],
                "department_id": row[2],
                "schedule_date": row[3],
                "time_slot": row[4],
                "quota_available": row[5],
                "doctor_name": row[6],
                "department_name": row[7],
            }
            for row in rows
        ]

    def list_upcoming_availability(
        self,
        department: str,
        *,
        doctor_name: str | None = None,
        start_date: date | None = None,
        limit: int = 6,
        conn=None,
    ):
        owns_connection = conn is None
        connection = conn or self._connect()
        department_row = self.find_department_by_name(department, conn=connection)
        if not department_row:
            if owns_connection:
                connection.close()
            return []
        try:
            params = [department_row["id"], start_date or date.today()]
            conditions = [
                "ds.department_id = %s",
                "ds.schedule_date >= %s",
                "ds.quota_available > 0",
            ]
            if doctor_name:
                conditions.append("lower(d.name) LIKE lower(%s)")
                params.append(f"%{doctor_name}%")
            params.append(int(limit or 6))
            with connection.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        ds.id, ds.doctor_id, ds.department_id, ds.schedule_date, ds.time_slot,
                        ds.quota_available, d.name
                    FROM doctor_schedules ds
                    JOIN doctors d ON d.id = ds.doctor_id
                    WHERE {' AND '.join(conditions)}
                    ORDER BY ds.schedule_date, ds.time_slot, d.name, ds.id
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        finally:
            if owns_connection:
                connection.close()

        return [
            {
                "schedule_id": row[0],
                "doctor_id": row[1],
                "department_id": row[2],
                "schedule_date": row[3],
                "time_slot": row[4],
                "quota_available": row[5],
                "doctor_name": row[6],
                "department_name": department_row["name"],
            }
            for row in rows
        ]

    def create_appointment(self, thread_id: str, department: str, schedule_date: date, time_slot: str, doctor_name: str | None = None):
        appointment_no = "APT" + uuid.uuid4().hex[:10].upper()
        request_payload = {
            "department": department,
            "date": schedule_date.isoformat(),
            "time_slot": time_slot,
            "doctor_name": doctor_name or "",
        }
        with self._connect() as conn:
            patient_id = self.ensure_patient_for_thread(thread_id, conn=conn)
            schedule = self.find_available_schedule(
                department,
                schedule_date,
                time_slot,
                doctor_name=doctor_name,
                conn=conn,
            )
            if not schedule:
                return None

            with conn.cursor() as cur:
                # PostgreSQL will take a row-level lock for this UPDATE, so only one
                # concurrent transaction can decrement the same schedule record when
                # quota_available is down to the last remaining slot.
                cur.execute(
                    """
                    UPDATE doctor_schedules
                    SET quota_available = quota_available - 1
                    WHERE id = %s AND quota_available > 0
                    RETURNING id
                    """,
                    (schedule["schedule_id"],),
                )
                locked = cur.fetchone()
                if not locked:
                    conn.rollback()
                    return None

                cur.execute(
                    """
                    INSERT INTO appointments (
                        appointment_no, patient_id, doctor_id, department_id, schedule_id,
                        appointment_date, time_slot, status, created_by
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'booked', 'ai_agent')
                    RETURNING id
                    """,
                    (
                        appointment_no,
                        patient_id,
                        schedule["doctor_id"],
                        schedule["department_id"],
                        schedule["schedule_id"],
                        schedule["schedule_date"],
                        schedule["time_slot"],
                    ),
                )
                appointment_id = cur.fetchone()[0]
                response_payload = {
                    "appointment_no": appointment_no,
                    "department": schedule["department_name"],
                    "date": schedule["schedule_date"].isoformat(),
                    "time_slot": schedule["time_slot"],
                    "doctor_name": schedule["doctor_name"],
                    "status": "booked",
                }
                cur.execute(
                    """
                    INSERT INTO appointment_logs (appointment_id, thread_id, action, request_payload, response_payload)
                    VALUES (%s, %s, 'book', %s::jsonb, %s::jsonb)
                    """,
                    (
                        appointment_id,
                        thread_id,
                        json.dumps(request_payload, ensure_ascii=False),
                        json.dumps(response_payload, ensure_ascii=False),
                    ),
                )
            conn.commit()
        return response_payload

    def find_candidate_appointments(self, thread_id: str, appointment_no: str | None = None, department: str | None = None, schedule_date: date | None = None, conn=None):
        owns_connection = conn is None
        connection = conn or self._connect()
        patient_id = self.ensure_patient_for_thread(thread_id, conn=connection)
        conditions = ["a.patient_id = %s", "a.status = 'booked'"]
        params = [patient_id]
        if appointment_no:
            conditions.append("a.appointment_no = %s")
            params.append(appointment_no.upper())
        else:
            if department:
                conditions.append("lower(dep.name) LIKE lower(%s)")
                params.append(f"%{department}%")
            if schedule_date:
                conditions.append("a.appointment_date = %s")
                params.append(schedule_date)

        query = f"""
            SELECT a.id, a.appointment_no, a.appointment_date, a.time_slot, a.schedule_id, dep.name, d.name
            FROM appointments a
            JOIN departments dep ON dep.id = a.department_id
            LEFT JOIN doctors d ON d.id = a.doctor_id
            WHERE {' AND '.join(conditions)}
            ORDER BY a.appointment_date, a.time_slot, a.id
        """
        try:
            with connection.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        finally:
            if owns_connection:
                connection.close()
        return [
            {
                "appointment_id": row[0],
                "appointment_no": row[1],
                "appointment_date": row[2],
                "time_slot": row[3],
                "schedule_id": row[4],
                "department": row[5],
                "doctor_name": row[6],
            }
            for row in rows
        ]

    def list_user_appointments(self, thread_id: str, limit: int = 8, conn=None):
        owns_connection = conn is None
        connection = conn or self._connect()
        patient_id = self.ensure_patient_for_thread(thread_id, conn=connection)
        try:
            with connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        a.id, a.appointment_no, a.appointment_date, a.time_slot, a.schedule_id,
                        dep.name, d.name
                    FROM appointments a
                    JOIN departments dep ON dep.id = a.department_id
                    LEFT JOIN doctors d ON d.id = a.doctor_id
                    WHERE a.patient_id = %s AND a.status = 'booked'
                    ORDER BY a.appointment_date, a.time_slot, a.id
                    LIMIT %s
                    """,
                    (patient_id, int(limit or 8)),
                )
                rows = cur.fetchall()
        finally:
            if owns_connection:
                connection.close()
        return [
            {
                "appointment_id": row[0],
                "appointment_no": row[1],
                "appointment_date": row[2],
                "time_slot": row[3],
                "schedule_id": row[4],
                "department": row[5],
                "doctor_name": row[6],
            }
            for row in rows
        ]

    def cancel_appointment(self, thread_id: str, appointment_id: int):
        with self._connect() as conn:
            patient_id = self.ensure_patient_for_thread(thread_id, conn=conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT a.id, a.appointment_no, a.appointment_date, a.time_slot, a.schedule_id, dep.name
                    FROM appointments
                    a
                    JOIN departments dep ON dep.id = a.department_id
                    WHERE a.id = %s AND a.patient_id = %s AND a.status = 'booked'
                    """,
                    (appointment_id, patient_id),
                )
                row = cur.fetchone()
                if not row:
                    conn.rollback()
                    return None

                cur.execute(
                    """
                    UPDATE appointments
                    SET status = 'cancelled', updated_at = NOW()
                    WHERE id = %s
                    """,
                    (appointment_id,),
                )
                cur.execute(
                    """
                    UPDATE doctor_schedules
                    SET quota_available = quota_available + 1
                    WHERE id = %s
                    """,
                    (row[4],),
                )
                response_payload = {
                    "appointment_no": row[1],
                    "date": row[2].isoformat(),
                    "time_slot": row[3],
                    "department": row[5],
                    "status": "cancelled",
                }
                cur.execute(
                    """
                    INSERT INTO appointment_logs (appointment_id, thread_id, action, request_payload, response_payload)
                    VALUES (%s, %s, 'cancel', %s::jsonb, %s::jsonb)
                    """,
                    (
                        appointment_id,
                        thread_id,
                        json.dumps({"appointment_id": appointment_id}, ensure_ascii=False),
                        json.dumps(response_payload, ensure_ascii=False),
                    ),
                )
            conn.commit()
        return response_payload

    def reschedule_appointment(
        self,
        thread_id: str,
        appointment_id: int,
        department: str,
        schedule_date: date,
        time_slot: str,
        doctor_name: str | None = None,
    ):
        request_payload = {
            "appointment_id": appointment_id,
            "department": department,
            "date": schedule_date.isoformat(),
            "time_slot": time_slot,
            "doctor_name": doctor_name or "",
        }
        with self._connect() as conn:
            patient_id = self.ensure_patient_for_thread(thread_id, conn=conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        a.id, a.appointment_no, a.schedule_id, a.department_id, a.doctor_id,
                        a.appointment_date, a.time_slot, dep.name, COALESCE(d.name, '')
                    FROM appointments a
                    JOIN departments dep ON dep.id = a.department_id
                    LEFT JOIN doctors d ON d.id = a.doctor_id
                    WHERE a.id = %s AND a.patient_id = %s AND a.status = 'booked'
                    FOR UPDATE
                    """,
                    (appointment_id, patient_id),
                )
                current = cur.fetchone()
                if not current:
                    conn.rollback()
                    return None

            schedule = self.find_available_schedule(
                department,
                schedule_date,
                time_slot,
                doctor_name=doctor_name,
                conn=conn,
            )
            if not schedule:
                conn.rollback()
                return None

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE doctor_schedules
                    SET quota_available = quota_available - 1
                    WHERE id = %s AND quota_available > 0
                    RETURNING id
                    """,
                    (schedule["schedule_id"],),
                )
                locked = cur.fetchone()
                if not locked:
                    conn.rollback()
                    return None

                if schedule["schedule_id"] != current[2]:
                    cur.execute(
                        """
                        UPDATE doctor_schedules
                        SET quota_available = quota_available + 1
                        WHERE id = %s
                        """,
                        (current[2],),
                    )

                cur.execute(
                    """
                    UPDATE appointments
                    SET doctor_id = %s,
                        department_id = %s,
                        schedule_id = %s,
                        appointment_date = %s,
                        time_slot = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        schedule["doctor_id"],
                        schedule["department_id"],
                        schedule["schedule_id"],
                        schedule["schedule_date"],
                        schedule["time_slot"],
                        appointment_id,
                    ),
                )

                response_payload = {
                    "appointment_no": current[1],
                    "department": schedule["department_name"],
                    "date": schedule["schedule_date"].isoformat(),
                    "time_slot": schedule["time_slot"],
                    "doctor_name": schedule["doctor_name"],
                    "previous_department": current[7],
                    "previous_date": current[5].isoformat(),
                    "previous_time_slot": current[6],
                    "previous_doctor_name": current[8],
                    "status": "booked",
                }
                cur.execute(
                    """
                    INSERT INTO appointment_logs (appointment_id, thread_id, action, request_payload, response_payload)
                    VALUES (%s, %s, 'reschedule', %s::jsonb, %s::jsonb)
                    """,
                    (
                        appointment_id,
                        thread_id,
                        json.dumps(request_payload, ensure_ascii=False),
                        json.dumps(response_payload, ensure_ascii=False),
                    ),
                )
            conn.commit()
        return response_payload
