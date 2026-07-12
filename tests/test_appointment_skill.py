import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from services.appointment_skill import AppointmentSkill  # noqa: E402


class FakeAppointmentService:
    def __init__(self):
        self.next_departments = []
        self.all_departments = []
        self.next_doctors = []
        self.next_availability = []
        self.next_appointments = []
        self.next_schedule = None
        self.next_candidates = []
        self.next_cancel_result = None
        self.next_booking = None

    def list_departments(self, query=None, limit=10):
        if query:
            return list(self.next_departments)[:limit]
        return list(self.all_departments or self.next_departments)[:limit]

    def list_available_doctors(self, department, schedule_date, time_slot):
        return list(self.next_doctors)

    def get_doctor_availability(self, doctor_name, department=None, schedule_date=None, time_slot=None, limit=6):
        return list(self.next_availability)[:limit]

    def list_upcoming_availability(self, department, doctor_name=None, start_date=None, limit=6):
        return list(self.next_availability)[:limit]

    def list_user_appointments(self, thread_id, limit=8):
        return list(self.next_appointments)[:limit]

    def find_available_schedule(self, department, schedule_date, time_slot, doctor_name=None):
        return self.next_schedule

    def create_appointment(self, thread_id, department, schedule_date, time_slot, doctor_name=None):
        return self.next_booking

    def find_candidate_appointments(self, thread_id, appointment_no=None, department=None, schedule_date=None):
        return list(self.next_candidates)

    def cancel_appointment(self, thread_id, appointment_id):
        return self.next_cancel_result


class AppointmentSkillTests(unittest.TestCase):
    def test_discover_departments_falls_back_to_default_list_when_query_has_no_match(self):
        service = FakeAppointmentService()
        service.next_departments = []
        service.all_departments = [
            {"id": 1, "code": "resp", "name": "呼吸内科"},
            {"id": 2, "code": "card", "name": "心内科"},
        ]
        skill = AppointmentSkill(service)

        message = skill.discover_departments("我要挂号")

        self.assertIn("呼吸内科", message)
        self.assertIn("心内科", message)

    def test_discover_doctors_returns_read_only_listing(self):
        service = FakeAppointmentService()
        tomorrow = date.today() + timedelta(days=1)
        service.next_doctors = [
            {
                "doctor_name": "张医生",
                "department_name": "呼吸内科",
                "schedule_date": tomorrow,
                "time_slot": "morning",
                "quota_available": 3,
            },
            {
                "doctor_name": "李医生",
                "department_name": "呼吸内科",
                "schedule_date": tomorrow,
                "time_slot": "morning",
                "quota_available": 1,
            },
        ]
        skill = AppointmentSkill(service)

        message, doctors = skill.discover_doctors("呼吸内科", schedule_date=tomorrow, time_slot="morning")

        self.assertEqual(len(doctors), 2)
        self.assertIn("可预约的医生有", message)
        self.assertIn("张医生", message)
        self.assertIn("李医生", message)

    def test_prepare_appointment_uses_any_available_doctor(self):
        service = FakeAppointmentService()
        tomorrow = date.today() + timedelta(days=1)
        service.next_doctors = [
            {
                "schedule_id": 1,
                "doctor_name": "张医生",
                "department_name": "呼吸内科",
                "schedule_date": tomorrow,
                "time_slot": "morning",
                "quota_available": 3,
            },
            {
                "schedule_id": 2,
                "doctor_name": "李医生",
                "department_name": "呼吸内科",
                "schedule_date": tomorrow,
                "time_slot": "morning",
                "quota_available": 1,
            },
        ]
        service.next_schedule = service.next_doctors[0]
        skill = AppointmentSkill(service)

        preview, doctor_options, alternatives = skill.prepare_appointment(
            department="呼吸内科",
            schedule_date=tomorrow,
            time_slot="morning",
            doctor_name="",
            allow_any_doctor=True,
        )

        self.assertIsNotNone(preview)
        self.assertEqual(preview.doctor_name, "张医生")
        self.assertEqual(len(doctor_options), 2)
        self.assertEqual(alternatives, [])

    def test_prepare_cancellation_without_filters_lists_existing_appointments(self):
        service = FakeAppointmentService()
        tomorrow = date.today() + timedelta(days=1)
        service.next_appointments = [
            {
                "appointment_id": 1,
                "appointment_no": "APT001",
                "department": "心内科",
                "appointment_date": tomorrow,
                "time_slot": "morning",
                "doctor_name": "李医生",
            }
        ]
        skill = AppointmentSkill(service)

        preview, candidates = skill.prepare_cancellation("thread-1")

        self.assertIsNone(preview)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["appointment_no"], "APT001")


if __name__ == "__main__":
    unittest.main()
