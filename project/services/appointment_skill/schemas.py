from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AppointmentCandidate:
    candidate_type: str
    label: str
    payload: dict = field(default_factory=dict)


@dataclass
class AppointmentDiscoveryResult:
    skill_mode: str
    message: str
    candidates: list[AppointmentCandidate] = field(default_factory=list)
    appointment_context: dict = field(default_factory=dict)


@dataclass
class AppointmentPreview:
    department: str
    date: str
    time_slot: str
    doctor_name: str = ""
    action: str = "book"


@dataclass
class CancellationPreview:
    appointment_id: str
    appointment_no: str
    department: str
    date: str
    time_slot: str
    doctor_name: str = ""
    action: str = "cancel"


@dataclass
class ReschedulePreview:
    appointment_id: str
    appointment_no: str
    department: str
    date: str
    time_slot: str
    doctor_name: str = ""
    previous_department: str = ""
    previous_date: str = ""
    previous_time_slot: str = ""
    previous_doctor_name: str = ""
    action: str = "reschedule"


@dataclass
class AppointmentSkillAction:
    """Internal service-layer action descriptor for appointment skill operations.

    Distinct from the Pydantic AppointmentSkillRequest in rag_agent/schemas.py
    which is used for LLM structured output (tool calling).
    """
    action: str = ""
    department: str = ""
    date: str = ""
    time_slot: str = ""
    doctor_name: str = ""
    appointment_no: str = ""
    clarification: str = ""
