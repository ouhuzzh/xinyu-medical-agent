from typing import List, Literal
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )
    intent: str = Field(
        default="medical_rag",
        description="Intent classification: medical_rag, triage, appointment, cancel_appointment, or clarification."
    )


class IntentAnalysis(BaseModel):
    intent: Literal["medical_rag", "triage", "appointment", "cancel_appointment", "clarification"] = Field(
        description="Intent classification. Must be one of: medical_rag, triage, appointment, cancel_appointment, clarification."
    )
    is_clear: bool = Field(
        description="Whether the user's request is clear enough to continue."
    )
    clarification_needed: str = Field(
        description="Clarification question if the request is not clear enough."
    )


class DepartmentRecommendation(BaseModel):
    department: str = Field(
        description="Single primary department recommendation."
    )
    reason: str = Field(
        description="Short reason for the department recommendation."
    )
    needs_clarification: bool = Field(
        description="Whether more information is required before making a recommendation."
    )
    clarification_needed: str = Field(
        description="Clarification question when more information is needed."
    )


class AppointmentRequest(BaseModel):
    department: str = Field(description="Department name if available, otherwise empty string.")
    date: str = Field(description="Appointment date phrase or ISO date string if available, otherwise empty string.")
    time_slot: str = Field(description="Preferred time slot such as 上午/下午/晚上 or morning/afternoon/evening, otherwise empty string.")
    doctor_name: str = Field(description="Doctor name if explicitly requested, otherwise empty string.")
    needs_clarification: bool = Field(description="Whether more information is required before booking.")
    clarification_needed: str = Field(description="Clarification question when required fields are missing.")


class CancelAppointmentRequest(BaseModel):
    appointment_no: str = Field(description="Appointment number if explicitly provided, otherwise empty string.")
    department: str = Field(description="Department name if available, otherwise empty string.")
    date: str = Field(description="Appointment date phrase or ISO date string if available, otherwise empty string.")
    needs_clarification: bool = Field(description="Whether more information is required before cancellation.")
    clarification_needed: str = Field(description="Clarification question when the appointment cannot be identified yet.")


class AppointmentActionCall(BaseModel):
    action: Literal["clarify", "prepare_booking"] = Field(
        description="Either ask for missing booking information or prepare a booking preview for confirmation."
    )
    department: str = Field(description="Department name if available, otherwise empty string.")
    date: str = Field(
        description="Preferred appointment date in YYYY-MM-DD format when known, otherwise empty string.",
        pattern=r"^$|^\d{4}-\d{2}-\d{2}$",
    )
    time_slot: Literal["", "morning", "afternoon", "evening"] = Field(
        description="Preferred standardized time slot when known, otherwise empty string."
    )
    doctor_name: str = Field(description="Doctor name if explicitly requested, otherwise empty string.")
    clarification: str = Field(description="Short clarification question when action is clarify, otherwise empty string.")


class CancelActionCall(BaseModel):
    action: Literal["clarify", "prepare_cancellation"] = Field(
        description="Either ask for missing cancellation information or prepare a cancellation preview for confirmation."
    )
    appointment_no: str = Field(description="Appointment number if available, otherwise empty string.")
    department: str = Field(description="Department name if available, otherwise empty string.")
    date: str = Field(
        description="Appointment date in YYYY-MM-DD format when known, otherwise empty string.",
        pattern=r"^$|^\d{4}-\d{2}-\d{2}$",
    )
    clarification: str = Field(description="Short clarification question when action is clarify, otherwise empty string.")


class AppointmentSkillRequest(BaseModel):
    action: Literal[
        "clarify",
        "discover_department",
        "discover_doctor",
        "discover_availability",
        "list_my_appointments",
        "prepare_appointment",
        "confirm_appointment",
        "prepare_cancellation",
        "confirm_cancellation",
        "prepare_reschedule",
    ] = Field(description="Appointment-skill action for discovery, planning, or controlled execution.")
    department: str = Field(description="Department name if available, otherwise empty string.")
    date: str = Field(
        description="Date in YYYY-MM-DD format when known, otherwise empty string.",
        pattern=r"^$|^\d{4}-\d{2}-\d{2}$",
    )
    time_slot: Literal["", "morning", "afternoon", "evening"] = Field(
        description="Preferred standardized time slot when known, otherwise empty string."
    )
    doctor_name: str = Field(description="Doctor name if explicitly requested, otherwise empty string.")
    appointment_no: str = Field(description="Appointment number if explicitly referenced, otherwise empty string.")
    clarification: str = Field(description="Short clarification prompt when action is clarify, otherwise empty string.")


class RetrievalQueryPlan(BaseModel):
    queries: List[str] = Field(description="Ordered list of 2-4 retrieval-friendly queries.")


class RetrievalDocumentGrade(BaseModel):
    keep: bool = Field(description="Whether this retrieved document should be retained.")
    relevance: Literal["high", "medium", "low"] = Field(description="Estimated relevance grade.")
    reason: str = Field(description="Short reason for the relevance grade.")


class EvidenceSufficiency(BaseModel):
    is_sufficient: bool = Field(description="Whether current evidence is sufficient to answer the question.")
    reason: str = Field(description="Short explanation for the sufficiency judgment.")
    retry_query: str = Field(description="One improved retry query when evidence is insufficient, otherwise empty string.")


class GroundedAnswerCheck(BaseModel):
    grounded: bool = Field(description="Whether the answer stays within the provided evidence.")
    revised_answer: str = Field(description="A conservative revised answer if the original answer was not grounded.")
    note: str = Field(description="Short note describing the grounding decision.")
