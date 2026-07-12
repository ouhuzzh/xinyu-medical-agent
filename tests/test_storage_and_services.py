import sys
import unittest
from datetime import date
from unittest.mock import Mock, patch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from memory.summary_store import SummaryStore  # noqa: E402
from services.appointment_service import AppointmentService  # noqa: E402


class FakeSummaryCursor:
    def __init__(self, state):
        self.state = state
        self.fetchone_result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        sql = " ".join(query.split())
        params = params or ()
        if "INSERT INTO chat_sessions" in sql:
            self.state["sessions"].add(params[0])
            self.fetchone_result = None
        elif "SELECT summary_content" in sql:
            key = (params[0], "long_term")
            self.fetchone_result = (self.state["summaries"][key]["summary_content"],) if key in self.state["summaries"] else None
        elif sql.startswith("INSERT INTO chat_session_summaries"):
            key = (params[0], "long_term")
            self.state["summaries"][key] = {
                "summary_content": params[1],
                "last_message_index": params[2],
            }
            self.fetchone_result = None
        elif "DELETE FROM chat_session_summaries" in sql:
            self.state["summaries"].pop((params[0], "long_term"), None)
            self.fetchone_result = None
        elif "DELETE FROM chat_sessions" in sql:
            self.state["sessions"].discard(params[0])
            self.fetchone_result = None
        else:
            raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchone(self):
        return self.fetchone_result


class FakeSummaryConnection:
    def __init__(self, state):
        self.state = state
        self.commits = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def cursor(self):
        return FakeSummaryCursor(self.state)

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


class SummaryStoreSpy(SummaryStore):
    def __init__(self):
        super().__init__()
        self.connect_calls = 0
        self.state = {"sessions": set(), "summaries": {}}

    def _connect(self):
        self.connect_calls += 1
        return FakeSummaryConnection(self.state)


class FakeAppointmentCursor:
    def __init__(self):
        self.fetchone_results = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        sql = " ".join(query.split())
        if sql.startswith("UPDATE doctor_schedules"):
            self.fetchone_results.append((11,))
        elif sql.startswith("INSERT INTO appointments"):
            self.fetchone_results.append((22,))
        elif sql.startswith("INSERT INTO appointment_logs"):
            self.fetchone_results.append(None)
        else:
            raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchone(self):
        return self.fetchone_results.pop(0)


class FakeAppointmentConnection:
    def __init__(self):
        self.cursor_obj = FakeAppointmentCursor()
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def close(self):
        pass


class AppointmentServiceSpy(AppointmentService):
    def __init__(self):
        super().__init__()
        self.connect_calls = 0
        self.ensure_received_conn = False
        self.schedule_received_conn = False
        self.connection = FakeAppointmentConnection()

    def _connect(self):
        self.connect_calls += 1
        return self.connection

    def ensure_patient_for_thread(self, thread_id: str, conn=None) -> int:
        self.ensure_received_conn = conn is not None
        return 7

    def find_available_schedule(self, department, schedule_date, time_slot, doctor_name=None, conn=None):
        self.schedule_received_conn = conn is not None
        return {
            "schedule_id": 5,
            "doctor_id": 6,
            "department_id": 8,
            "schedule_date": schedule_date,
            "time_slot": time_slot,
            "quota_available": 4,
            "doctor_name": doctor_name or "张医生",
            "department_name": department,
        }


class StorageAndServiceTests(unittest.TestCase):
    def test_summary_store_save_summary_updates_single_row(self):
        store = SummaryStoreSpy()

        store.save_summary("thread-a", "summary-1", 2)
        store.save_summary("thread-a", "summary-2", 4)

        self.assertEqual(store.connect_calls, 2)
        self.assertEqual(len(store.state["summaries"]), 1)
        self.assertEqual(store.state["summaries"][("thread-a", "long_term")]["summary_content"], "summary-2")
        self.assertEqual(store.get_summary("thread-a"), "summary-2")

    def test_create_appointment_reuses_single_connection(self):
        service = AppointmentServiceSpy()

        result = service.create_appointment(
            thread_id="thread-b",
            department="呼吸内科",
            schedule_date=date(2026, 4, 20),
            time_slot="morning",
            doctor_name="张医生",
        )

        self.assertEqual(service.connect_calls, 1)
        self.assertTrue(service.ensure_received_conn)
        self.assertTrue(service.schedule_received_conn)
        self.assertEqual(result["appointment_no"][:3], "APT")
        self.assertEqual(result["department"], "呼吸内科")

    def test_appointment_connect_does_not_apply_migrations_each_time(self):
        schema_manager = Mock()
        connection = Mock()
        with patch("services.appointment_service.SchemaManager", return_value=schema_manager), patch(
            "services.appointment_service.connect",
            return_value=connection,
        ):
            service = AppointmentService()
            first = service._connect()
            second = service._connect()

        self.assertIs(first, connection)
        self.assertIs(second, connection)
        schema_manager.apply_migrations.assert_not_called()


if __name__ == "__main__":
    unittest.main()
