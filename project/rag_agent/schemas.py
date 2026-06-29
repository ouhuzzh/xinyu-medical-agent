from typing import List, Literal, Optional, Sequence
from pydantic import BaseModel, Field, create_model

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
    """Base intent analysis schema with flexible str intent.

    For structured output with Literal constraints, use
    ``build_intent_analysis_schema()`` which dynamically builds a
    Pydantic model with a Literal[intent_labels] field.
    """
    intent: str = Field(
        description="Intent classification. Must be one of the valid intent labels."
    )
    is_clear: bool = Field(
        description="Whether the user's request is clear enough to continue."
    )
    clarification_needed: str = Field(
        description="Clarification question if the request is not clear enough."
    )


def build_intent_analysis_schema(
    intent_labels: Optional[Sequence[str]] = None,
) -> type[BaseModel]:
    """Build a Pydantic model for intent analysis with a Literal[intent] field.

    Dynamically creates a model whose ``intent`` field uses Literal to
    constrain output to the *current* set of intent labels (core + skills),
    rather than a hardcoded set that would need manual updates when skills
    are added.

    Args:
        intent_labels: Ordered sequence of valid intent strings.
            Defaults to the core 6 intents if not provided.

    Returns:
        A dynamically-created Pydantic model class with:
          - intent: Literal[...]  (validated against intent_labels)
          - is_clear: bool
          - clarification_needed: str
    """
    if not intent_labels:
        intent_labels = ("medical_rag", "triage", "appointment",
                         "cancel_appointment", "greeting", "clarification")

    literal_type = Literal[tuple(intent_labels)]  # type: ignore[valid-type]

    return create_model(
        "DynamicIntentAnalysis",
        intent=(literal_type, Field(
            description=f"Intent classification. Must be one of: {', '.join(intent_labels)}"
        )),
        is_clear=(bool, Field(
            description="Whether the user's request is clear enough to continue."
        )),
        clarification_needed=(str, Field(
            description="Clarification question if the request is not clear enough."
        )),
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


class GroundingCritique(BaseModel):
    critique: str = Field(description="哪些回答内容超出检索证据、缺证据或与证据矛盾。")
    revised_answer: str = Field(description="基于现有证据重写后的回答，收窄到证据范围内，不加免责声明。")


class TaskDecomposition(BaseModel):
    needs_decomposition: bool = Field(description="用户问题是否包含多个可独立检索的子问题/facet。")
    sub_questions: List[str] = Field(description="分解后的独立子问题；不复合时为仅含原问题的单元素列表。")
    reason: str = Field(description="简短说明是否复合的判断依据。")
