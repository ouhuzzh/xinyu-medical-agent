"""Turn planner nodes: decompose a compound turn into planned_tasks and drain them.

plan_tasks (LLM decomposer) -> dispatch_next_task -> handler -> advance_task ->
route_to_next_or_gate -> dispatch|completeness_gate. Extracted from rag_nodes.
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .graph_state import State
from .schemas import build_turn_plan_schema
from .prompts import get_turn_planner_prompt
import config
from .node_helpers import (
    _structured_output_llm,
    collect_skill_hints,
    _next_undone_task,
    _undone_tasks,
    _clear_per_task_rag_state,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turn planner (plan_tasks / dispatch_next_task / advance_task / completeness_gate).
# ---------------------------------------------------------------------------

def _planner_user_query(state: State) -> str:
    q = str(state.get("primary_user_query") or state.get("originalQuery") or "").strip()
    if not q and state.get("messages"):
        q = str(getattr(state["messages"][-1], "content", "") or "").strip()
    return q


def plan_tasks(state: State, llm):
    """Turn planner: decompose the user's message into an ordered list
    of independent cross-intent tasks. Replaces the legacy rule-based compound split.

    Each task is {id, intent, query}. Single-intent -> one task; cross-intent
    compound -> N. Within-medical multi-facet decomposition stays with
    decompose_tasks (parallel fan-out), so the planner merges same-intent
    medical facets into one task. LLM failure falls back to a single task so the
    node never breaks the graph.
    """
    user_query = _planner_user_query(state)

    skill_hints, intent_labels = collect_skill_hints()
    l1_intent = ""
    try:
        from skills.registry import get_skill_registry
        l1_match = get_skill_registry().classify_by_keywords(user_query)
        if l1_match:
            l1_intent = l1_match[0]
    except Exception:
        pass

    def _single(intent: str) -> dict:
        return {
            "planned_tasks": [{"id": 0, "intent": intent or "medical_rag", "query": user_query}],
        }

    if not user_query:
        return _single("medical_rag")

    try:
        schema_cls = build_turn_plan_schema(intent_labels)
        parser = _structured_output_llm(llm, schema_cls, max_tokens=config.LLM_STRUCTURED_MAX_TOKENS)
        sys_msg = SystemMessage(content=get_turn_planner_prompt(skill_hints, max_tasks=config.MAX_PLANNED_TASKS))
        user_payload = (
            f"用户消息：{user_query}\n"
            f"对话摘要：{state.get('conversation_summary', '')}\n"
            f"近期对话：{state.get('recent_context', '')}\n"
            f"话题焦点：{state.get('topic_focus', '')}"
        )
        verdict = parser.invoke([sys_msg, HumanMessage(content=user_payload)])
    except Exception:
        logger.exception("plan_tasks structured output failed; falling back to single task.")
        return _single(l1_intent)

    raw_tasks = list(getattr(verdict, "tasks", []) or [])
    tasks: list = []
    for i, t in enumerate(raw_tasks):
        intent = str(getattr(t, "intent", "") or "").strip()
        query = str(getattr(t, "query", "") or "").strip()
        if not query:
            continue
        if not intent:
            intent = "medical_rag"
        tasks.append({"id": i, "intent": intent, "query": query})

    if not tasks:
        return _single(l1_intent)

    return {"planned_tasks": tasks[: config.MAX_PLANNED_TASKS]}


def dispatch_next_task(state: State):
    """Stage the next undone planned task for execution: set intent + query so
    the downstream handler (rewrite_query / handle_appointment_skill / ...) acts
    on just this task. _get_user_query reads primary_user_query first, so
    setting it focuses each handler on the staged sub-query.

    Resets per-task medical fields so task N+1's RAG loop doesn't see task N's
    leftovers. Injects a HumanMessage for non-first tasks (the first task's
    original message is already in history).
    """
    task = _next_undone_task(state)
    if not task:
        return {}
    intent = str(task.get("intent", "") or "").strip() or "medical_rag"
    query = str(task.get("query", "") or "").strip()
    if not query:
        return {}

    is_first = not bool(state.get("task_results"))
    update: dict = {
        "intent": intent,
        "primary_intent": intent,
        "primary_user_query": query,
        "originalQuery": query,
        # Per-task RAG state reset (avoid bleeding task N into task N+1).
        **_clear_per_task_rag_state(),
        # Clear Phase 1 compound fields so they don't interfere.
        "secondary_intent": "",
        "deferred_user_question": "",
    }
    if not is_first:
        update["messages"] = [HumanMessage(content=query)]
    return update


def advance_task(state: State):
    """Record the just-executed task (lowest undone id) as done in task_results,
    so the next dispatch picks the following task. Runs after every handler
    before the drain edge decides whether to dispatch the next task or gate.
    """
    task = _next_undone_task(state)
    if not task:
        return {}
    try:
        tid = int(task.get("id", -1))
    except (TypeError, ValueError):
        return {}
    return {"task_results": [{"id": tid, "intent": str(task.get("intent", "") or ""), "status": "done"}]}


def completeness_gate(state: State):
    """Terminal gate: compare planned_tasks vs task_results and append a caveat
    naming any task that was never executed (e.g. its handler errored or the
    plan was interrupted by a pending action). Detection-only (no replan -
    replan is Phase 3). When all tasks completed, this is a no-op pass-through.
    """
    missing = [str(t.get("query", "") or "").strip() for t in _undone_tasks(state)]
    missing = [q for q in missing if q]
    if not missing:
        return {}
    items = "、".join(f"「{q}」" for q in missing)
    caveat = (
        f"\n\n⚠️ 您的问题包含多个部分，其中关于{items}暂未能给出回答，可否再单独描述一下？"
    )
    return {"messages": [AIMessage(content=caveat, name="completeness_gate")]}
