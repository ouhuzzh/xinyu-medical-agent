from __future__ import annotations
from typing import Any, List, Annotated, Set, Dict
from langgraph.graph import MessagesState
import operator

def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b


def keep_latest_non_empty(existing: str, new: str) -> str:
    new_value = str(new or "").strip()
    if new_value:
        return new_value
    return str(existing or "")

class State(MessagesState):
    """State for main agent graph"""
    questionIsClear: bool = False
    conversation_summary: str = ""
    recent_context: Annotated[str, keep_latest_non_empty] = ""
    request_id: Annotated[str, keep_latest_non_empty] = ""
    topic_focus: Annotated[str, keep_latest_non_empty] = ""
    originalQuery: str = ""
    thread_id: str = ""
    intent: str = ""
    primary_intent: str = ""
    secondary_intent: str = ""
    primary_user_query: str = ""
    secondary_user_query: str = ""
    planned_queries: List[str] = []
    decision_source: str = ""
    route_reason: str = ""
    last_route_reason: str = ""
    risk_level: str = "normal"
    pending_clarification: str = ""
    clarification_attempts: int = 0
    clarification_target: str = ""
    deferred_user_question: str = ""
    recommended_department: str = ""
    appointment_context: Dict[str, str] = {}
    appointment_skill_mode: str = ""
    appointment_candidates: List[dict] = []
    selected_doctor: str = ""
    selected_schedule_id: str = ""
    deferred_confirmation_action: str = ""
    skill_last_prompt: str = ""
    last_appointment_no: str = ""
    pending_action_type: str = ""
    pending_action_payload: Dict[str, str] = {}
    pending_confirmation_id: str = ""
    pending_candidates: List[dict] = []
    rewrittenQuestions: List[str] = []
    grounding_evidence_score: float | None = None
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []
    skill_data: Dict[str, Any] = {}
    user_memories: str = ""

class AgentState(MessagesState):
    """State for individual agent subgraph"""
    question: str = ""
    question_index: int = 0
    query_plan: List[str] = []
    context_summary: str = ""
    recent_context: str = ""
    topic_focus: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    final_answer: str = ""
    agent_answers: List[dict] = []
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
