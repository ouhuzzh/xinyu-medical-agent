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
    intent_confidence: float = 0.0  # 0.0-1.0, source layer determines fast-path vs LLM
    intent_source: str = ""  # "l1_rule" | "l2_embedding" | "l3_llm" | "skill:*"
    secondary_intent: str = ""
    primary_user_query: str = ""
    secondary_user_query: str = ""
    planned_queries: List[str] = []
    sub_questions: List[str] = []
    # P4: multi-agent supervisor — loop flags at medical_rag exit
    supervisor_active: bool = False
    supervisor_rounds: int = 0
    supervisor_next: str = "FINISH"
    # P5: online self-eval — LLM-as-judge score + details at turn end
    self_eval_score: float | None = None
    self_eval_details: dict = {}
    decision_source: str = ""
    route_reason: str = ""
    last_route_reason: str = ""
    risk_level: str = "normal"
    pending_clarification: str = ""
    clarification_attempts: int = 0
    clarification_target: str = ""
    deferred_user_question: str = ""
    # Compound-request drain queue: segments beyond the 2nd, each {"intent","query"}.
    # prepare_secondary_turn pops one per turn so 3+ segment compounds are answered
    # across turns instead of silently dropped.
    deferred_extra_tasks: List[dict] = []
    # Phase 2 turn planner: LLM-produced cross-intent task list + drained results.
    # dispatch_next_task pops the next undone task; completeness_gate checks coverage.
    planned_tasks: List[dict] = []
    task_results: Annotated[List[dict], accumulate_or_reset] = []
    planner_replan_count: int = 0
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
    pending_stale_count: int = 0  # consecutive irrelevant turns while pending — auto-clear at 2
    rewrittenQuestions: List[str] = []
    grounding_evidence_score: float | None = None
    grounding_passed: bool = False
    grounding_critique: str = ""
    grounding_rounds: int = 0
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []
    skill_data: Dict[str, Any] = {}
    user_memories: str = ""
    user_id: str = ""

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
    # P1: agentic retrieval — evidence-sufficiency reflection loop
    evidence_rounds: int = 0
    evidence_critique: str = ""
    last_refined_query: str = ""
    refined_queries: Annotated[List[str], operator.add] = []
    evidence_sufficient: bool = False
