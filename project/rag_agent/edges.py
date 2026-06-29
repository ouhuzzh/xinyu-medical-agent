from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Send
from .graph_state import State, AgentState
import config
from config import MAX_ITERATIONS, MAX_TOOL_CALLS, MAX_EVIDENCE_ROUNDS, MAX_GROUNDING_ROUNDS, MAX_SUB_QUESTIONS


def route_after_analyze_turn(state: State) -> Literal["rewrite_query", "intent_router"]:
    """Skip intent_router LLM call when rules were inconclusive.

    When analyze_turn rules already determined a concrete intent
    (greeting, cancel, appointment, triage, skill-registered), go to
    intent_router which will short-circuit (no LLM call).

    When rules are inconclusive (empty primary_intent), go directly to
    rewrite_query — which now also classifies intent in the same LLM
    call.  Saves one 14B LLM call (~4.5s) per medical query.
    """
    if state.get("primary_intent", ""):
        return "intent_router"
    return "rewrite_query"


def route_after_intent(state: State) -> str:
    """Route based on intent.  Prefers skill-registered routes, falls back
    to legacy static routes when the skill registry is not available."""
    intent = state.get("intent", "")

    # Check skill-registered routes first (single source of truth)
    try:
        from skills.registry import get_skill_registry
        registry = get_skill_registry()
        if registry.skills:
            skill_routes = registry.get_route_mapping()
            if intent in skill_routes:
                return skill_routes[intent]
    except Exception:
        pass

    # Legacy fallback — used when skill registry is not available or disabled
    _LEGACY_ROUTES = {
        "greeting": "__end__",
        "triage": "recommend_department",
        "appointment": "handle_appointment_skill",
        "cancel_appointment": "handle_appointment_skill",
        "clarification": "request_clarification",
    }
    if intent in _LEGACY_ROUTES:
        return _LEGACY_ROUTES[intent]

    # Default: medical RAG pipeline
    return "rewrite_query"


def route_after_rewrite(state: State) -> str:
    """Route after rewrite_query classifies intent + rewrites query.

    When the LLM (L3) classifies a non-medical_rag intent, route directly
    to the appropriate handler instead of always going to the RAG pipeline.
    This keeps classification cost at 1 LLM call (merged with rewrite).
    """
    if not state.get("questionIsClear", False):
        return "request_clarification"

    intent = state.get("intent", "medical_rag")

    # Check skill-registered routes first (single source of truth)
    try:
        from skills.registry import get_skill_registry
        registry = get_skill_registry()
        if registry.skills:
            skill_routes = registry.get_route_mapping()
            if intent in skill_routes:
                target = skill_routes[intent]
                if target != "rewrite_query":
                    return target
    except Exception:
        pass

    # Legacy fallback for when skill registry is unavailable
    _LEGACY_ROUTES = {
        "appointment": "handle_appointment_skill",
        "cancel_appointment": "handle_appointment_skill",
        "triage": "recommend_department",
        "greeting": "__end__",
    }
    if intent in _LEGACY_ROUTES:
        return _LEGACY_ROUTES[intent]

    # Default: medical RAG pipeline (P3: decompose into sub-questions first)
    return "decompose_tasks"


def route_after_query_plan(state: State):
    """P3: fan out one Send per sub-question; LangGraph runs them in parallel.

    Each sub-question enters the agent subgraph independently and runs the P1
    retrieval loop. collect_answer writes agent_answers with question_index=i;
    the accumulate_or_reset reducer merges them; grounded_answer_generation
    sorts by index and synthesizes. With a single sub-question this is exactly
    today's single-Send path.
    """
    summary = state.get("conversation_summary", "")
    recent_context = state.get("recent_context", "")
    topic_focus = state.get("topic_focus", "")
    user_memories = state.get("user_memories", "")

    rewritten = [str(q).strip() for q in (state.get("rewrittenQuestions") or []) if str(q).strip()]
    primary = (rewritten[0] if rewritten else "") or state.get("originalQuery", "") or state.get("primary_user_query", "")

    subs = [str(s).strip() for s in (state.get("sub_questions") or []) if str(s).strip()]
    if not subs:
        # Always emit at least one Send (even with an empty primary) so the graph
        # still flows through collect_answer → grounded_answer_generation and
        # produces an answer, rather than silently terminating with no output.
        subs = [primary]
    subs = subs[:MAX_SUB_QUESTIONS]

    payload_base = {
        "messages": [],
        "context_summary": summary,
        "recent_context": recent_context,
        "topic_focus": topic_focus,
        "user_memories": user_memories,
    }
    return [
        Send(
            "agent",
            {
                **payload_base,
                "question": q,
                "question_index": i,
                "query_plan": [q],
            },
        )
        for i, q in enumerate(subs)
    ]


def route_after_clarification(state: State) -> Literal["intent_router", "rewrite_query", "recommend_department", "handle_appointment_skill", "handle_appointment", "handle_cancel_appointment"]:
    target = state.get("clarification_target", "") or "intent_router"
    if target == "rewrite_query":
        return "rewrite_query"
    if target == "recommend_department":
        return "recommend_department"
    if target == "handle_appointment_skill":
        return "handle_appointment_skill"
    if target == "handle_appointment":
        return "handle_appointment"
    if target == "handle_cancel_appointment":
        return "handle_cancel_appointment"
    return "intent_router"


def route_after_orchestrator_call(state: AgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"

    if _has_repeated_no_evidence(state) or _has_repeated_search_query(state):
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"
    
    return "tools"


def route_after_action(state: State) -> str:
    """Route after an action specialist (appointment/triage) finishes.

    Priority: pending clarification > secondary turn > supervisor loop > END.
    The supervisor_active branch (P4) is lowest priority so that explicit
    pending/secondary signals (stronger closure intents) win.
    """
    if state.get("pending_clarification") and state.get("clarification_target"):
        return "request_clarification"
    if (
        state.get("secondary_intent")
        and state.get("deferred_user_question")
        and not state.get("pending_clarification")
        and not state.get("pending_action_type")
        and not state.get("pending_candidates")
        and not state.get("deferred_confirmation_action")
    ):
        return "prepare_secondary_turn"
    if bool(state.get("supervisor_active", False)):
        return "supervise"
    return "__end__"


def route_after_prepare_secondary_turn(state: State) -> Literal["rewrite_query", "handle_appointment", "handle_cancel_appointment", "recommend_department"]:
    intent = state.get("primary_intent") or state.get("intent") or ""

    # Check skill-registered routes first
    try:
        from skills.registry import get_skill_registry
        registry = get_skill_registry()
        if registry.skills:
            skill_routes = registry.get_route_mapping()
            if intent in skill_routes:
                target = skill_routes[intent]
                # Only return targets that are valid for this edge's mapping
                if target in ("handle_appointment", "handle_cancel_appointment",
                              "recommend_department", "rewrite_query"):
                    return target
    except Exception:
        pass

    # Legacy fallback
    _LEGACY_ROUTES = {
        "appointment": "handle_appointment",
        "cancel_appointment": "handle_cancel_appointment",
        "triage": "recommend_department",
    }
    if intent in _LEGACY_ROUTES:
        return _LEGACY_ROUTES[intent]
    return "rewrite_query"


def _has_repeated_no_evidence(state: AgentState) -> bool:
    tool_messages = [msg for msg in state.get("messages", []) if isinstance(msg, ToolMessage)]
    if len(tool_messages) < 2:
        return False
    last_two_contents = [str(msg.content or "") for msg in tool_messages[-2:]]
    return all("NO_EVIDENCE" in content for content in last_two_contents)


def _has_repeated_search_query(state: AgentState) -> bool:
    queries = []
    for msg in reversed(state.get("messages", [])):
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tool_call in tool_calls:
            if tool_call.get("name") != "search_child_chunks":
                continue
            query = str((tool_call.get("args") or {}).get("query") or "").strip().lower()
            if query:
                queries.append(query)
            if len(queries) >= 2:
                return queries[0] == queries[1]
    return False


# Diverges from siblings: compares latest entry against ALL earlier entries (catches A,B,A), whereas _has_repeated_no_evidence/_has_repeated_search_query compare only the last two.
def _has_repeated_refined_query(state: AgentState) -> bool:
    """No-progress guard: the latest refined query already appeared earlier."""
    refined = [str(q or "").strip().lower() for q in (state.get("refined_queries") or []) if str(q or "").strip()]
    if len(refined) < 2:
        return False
    return refined[-1] in refined[:-1]


def route_after_evidence(state: AgentState) -> Literal["should_compress_context", "fallback_response"]:
    """P1: route after evidence reflection.

    - sufficient → should_compress_context (then orchestrator → collect_answer)
    - insufficient + budget + progress → should_compress_context (re-search)
    - insufficient + exhausted/no-progress → fallback_response
    """
    is_sufficient = bool(state.get("evidence_sufficient", False))
    if is_sufficient:
        return "should_compress_context"

    rounds = int(state.get("evidence_rounds", 0) or 0)
    last_refined = str(state.get("last_refined_query", "") or "").strip()

    # No-progress guards fire before the budget guard so a stalled loop
    # terminates as early as possible (a repeated refined query / repeated
    # NO_EVIDENCE means more rounds cannot help).
    if _has_repeated_refined_query(state):
        return "fallback_response"
    if _has_repeated_no_evidence(state):
        return "fallback_response"
    if not last_refined:
        return "fallback_response"
    if rounds >= MAX_EVIDENCE_ROUNDS:
        return "fallback_response"

    return "should_compress_context"


def route_after_grounding(state: State) -> Literal["__end__", "revise_answer", "supervise"]:
    """P2/P4: route after the answer grounding check.

    - grounded (grounding_passed=True) → supervise (P4) when supervisor enabled, else END
    - not grounded + budget remaining (grounding_rounds < MAX_GROUNDING_ROUNDS) → revise_answer
    - not grounded + budget exhausted → supervise (P4) when supervisor enabled, else END
    """
    _to_supervisor = "supervise" if config.ENABLE_MULTI_AGENT_SUPERVISOR else "__end__"
    if bool(state.get("grounding_passed", False)):
        return _to_supervisor
    rounds = int(state.get("grounding_rounds", 0) or 0)
    if rounds < MAX_GROUNDING_ROUNDS:
        return "revise_answer"
    return _to_supervisor


def route_after_supervisor(state: State) -> str:
    """P4: dispatch the supervisor's chosen agent, or finish."""
    nxt = str(state.get("supervisor_next", "FINISH") or "FINISH").strip()
    if nxt == "appointment":
        return "handle_appointment_skill"
    if nxt == "triage":
        return "recommend_department"
    return "__end__"
