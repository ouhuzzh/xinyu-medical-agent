"""RAG retrieval, generation, and grounding nodes.

Query rewriting, retrieval query planning, the orchestrator loop, context
compression, answer collection/grounding, and fallback generation live here.
Private helpers that are only used by these public nodes are kept local.
"""

import re
import logging
from typing import Literal, Set
from datetime import date

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langgraph.types import Command

from .graph_state import State, AgentState
from .schemas import (
    QueryAnalysis,
    RetrievalQueryPlan,
    GroundedAnswerCheck,
    EvidenceSufficiency,
    GroundingCritique,
    SupervisorDecision,
    TaskDecomposition,
)
from .prompts import (
    get_rewrite_query_prompt,
    get_retrieval_query_plan_prompt,
    get_orchestrator_prompt,
    get_fallback_response_prompt,
    get_context_compression_prompt,
    get_aggregation_prompt,
    get_evidence_sufficiency_prompt,
    get_grounding_critique_prompt,
    get_task_decomposition_prompt,
)
from utils import estimate_context_tokens
import config
from config import BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR
from rag_agent.tools import plan_queries, ground_answer, check_sufficiency

from .node_helpers import (
    _build_history_reset_messages,
    _build_medical_fallback_notice,
    _build_recent_context,
    _confidence_bucket_explanation,
    _confidence_bucket_label,
    _extract_topic_focus,
    _format_reference_lines,
    _get_user_query,
    _infer_risk_level,
    _looks_like_general_non_medical_query,
    _looks_like_medical_follow_up,
    _looks_like_medical_knowledge_question,
    _looks_like_medical_request,
    _needs_strict_medical_safety,
    _next_clarification_attempt,
    _sanitize_final_answer_text,
    _strip_leading_query_plan_blob,
    _structured_output_llm,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers (only used by RAG nodes)
# ---------------------------------------------------------------------------

def _extract_source_citations(messages) -> list[dict]:
    citations = []
    seen = set()
    current_confidence = ""
    current_evidence_score = None
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        text = str(message.content or "")
        confidence_match = re.search(r"Confidence Bucket:\s*(\w+)", text, re.IGNORECASE)
        if confidence_match and not current_confidence:
            current_confidence = confidence_match.group(1).strip().lower()
        score_match = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if score_match:
            try:
                score_value = float(score_match.group(1))
            except ValueError:
                score_value = None
            if score_value is not None and current_evidence_score is None:
                current_evidence_score = score_value
        for block in text.split("\n\n"):
            title_match = re.search(r"Source Title:\s*(.+)", block)
            source_type_match = re.search(r"Source Type:\s*(.+)", block)
            url_match = re.search(r"Original URL:\s*(.+)", block)
            source_match = re.search(r"File Name:\s*(.+)", block)
            freshness_match = re.search(r"Freshness Bucket:\s*(.+)", block)
            score_match = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", block, re.IGNORECASE)
            if not any((title_match, source_match)):
                continue
            title = (title_match.group(1).strip() if title_match else source_match.group(1).strip())
            source_type = source_type_match.group(1).strip() if source_type_match else "unknown"
            original_url = url_match.group(1).strip() if url_match else ""
            freshness_bucket = freshness_match.group(1).strip().lower() if freshness_match else ""
            evidence_score = current_evidence_score
            if score_match:
                try:
                    evidence_score = float(score_match.group(1))
                except ValueError:
                    pass
            key = (title, source_type, original_url, freshness_bucket)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "title": title,
                    "source_type": source_type,
                    "original_url": original_url,
                    "confidence_bucket": current_confidence or "",
                    "freshness_bucket": freshness_bucket,
                    "evidence_score": evidence_score,
                }
            )
    return citations


# ---------------------------------------------------------------------------
# Public RAG nodes
# ---------------------------------------------------------------------------

def rewrite_query(state: State, llm):
    conversation_summary = state.get("conversation_summary", "")
    recent_context = state.get("recent_context") or _build_recent_context(state.get("messages", []))
    user_query = _get_user_query(state)
    topic_focus = state.get("topic_focus", "")

    if state.get("intent") == "medical_rag" and _looks_like_general_non_medical_query(user_query):
        delete_all = _build_history_reset_messages(state["messages"])
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": user_query,
            "rewrittenQuestions": [user_query],
            "recent_context": recent_context,
            "topic_focus": topic_focus,
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
        }

    context_parts = []
    if conversation_summary.strip():
        context_parts.append(f"Conversation Context:\n{conversation_summary}\n")
    if recent_context.strip():
        context_parts.append(f"Recent Dialogue Context:\n{recent_context}\n")
    if state.get("user_memories"):
        context_parts.append(f"Known user context:\n{state['user_memories']}\n")
    if topic_focus.strip():
        context_parts.append(f"Topic focus:\n{topic_focus}\n")
    context_parts.append(f"User Query:\n{user_query}\n")
    context_section = "".join(context_parts)

    try:
        # Collect skill L3 hints for dynamic prompt injection
        skill_hints = []
        try:
            from skills.registry import get_skill_registry
            _reg = get_skill_registry()
            skill_hints = _reg.collect_llm_hints()
        except Exception:
            pass

        llm_with_structure = _structured_output_llm(llm, QueryAnalysis, temperature=0.1)
        response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt(skill_hints)), HumanMessage(content=context_section)])
    except Exception:
        logger.exception("Rewrite query structured output failed; using original query.")
        return {
            "questionIsClear": True,
            "messages": _build_history_reset_messages(state["messages"]),
            "originalQuery": user_query,
            "rewrittenQuestions": [user_query],
            "recent_context": recent_context,
            "topic_focus": topic_focus or _extract_topic_focus(user_query, topic_focus),
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
        }

    if response.questions and response.is_clear:
        llm_intent = getattr(response, "intent", "medical_rag") or "medical_rag"
        delete_all = _build_history_reset_messages(state["messages"])
        extra_msgs: list = list(delete_all)
        if llm_intent == "greeting":
            greeting_msg = AIMessage(content="你好！我是你的医疗助手，可以帮你：\n- 🏥 推荐就诊科室\n- 📅 预约挂号\n- ❌ 取消预约\n- 💊 解答医疗健康问题\n\n请问有什么可以帮你的？")
            extra_msgs.insert(0, greeting_msg)
        return {
            "intent": llm_intent,
            "questionIsClear": True,
            "messages": extra_msgs,
            "originalQuery": user_query,
            "rewrittenQuestions": response.questions,
            "recent_context": recent_context,
            "topic_focus": topic_focus,
            "primary_intent": llm_intent,
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
        }

    if _looks_like_medical_knowledge_question(user_query):
        delete_all = _build_history_reset_messages(state["messages"])
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": user_query,
            "rewrittenQuestions": [user_query],
            "recent_context": recent_context,
            "topic_focus": topic_focus,
            "primary_intent": "medical_rag",
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
        }

    if state.get("intent") == "medical_rag" and _looks_like_medical_follow_up(user_query, "\n".join(part for part in (conversation_summary, topic_focus) if part), recent_context):
        delete_all = _build_history_reset_messages(state["messages"])
        fallback_query = response.questions[0] if response.questions else f"{topic_focus or recent_context or conversation_summary}\nFollow-up: {user_query}"
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": user_query,
            "rewrittenQuestions": [fallback_query],
            "recent_context": recent_context,
            "topic_focus": topic_focus,
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
        }

    if state.get("intent") == "medical_rag" and state.get("pending_clarification"):
        delete_all = _build_history_reset_messages(state["messages"])
        fallback_query = response.questions[0] if response.questions else (f"{topic_focus or recent_context or conversation_summary}\nQuestion: {user_query}" if (topic_focus or recent_context or conversation_summary) else user_query)
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": user_query,
            "rewrittenQuestions": [fallback_query],
            "recent_context": recent_context,
            "topic_focus": topic_focus,
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
        }

    clarification_attempts = _next_clarification_attempt(state)
    if clarification_attempts > 1:
        fallback_query = response.questions[0] if response.questions else (topic_focus or user_query)
        return {
            "questionIsClear": True,
            "messages": _build_history_reset_messages(state["messages"]),
            "originalQuery": user_query,
            "rewrittenQuestions": [fallback_query],
            "recent_context": recent_context,
            "topic_focus": topic_focus or _extract_topic_focus(user_query, topic_focus),
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": clarification_attempts,
        }

    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "我可以继续帮你，但还差一点关键信息。你能再具体一点吗？"
    return {
        "questionIsClear": False,
        "recent_context": recent_context,
        "topic_focus": topic_focus or _extract_topic_focus(user_query, topic_focus),
        "pending_clarification": clarification,
        "clarification_target": "rewrite_query",
        "clarification_attempts": clarification_attempts,
        "messages": [AIMessage(content=clarification)],
    }


def plan_retrieval_queries(state: State, llm):
    """Build retrieval queries from rewritten questions using rule-based expansion.

    Previously used an LLM call here, but the rule-based plan_queries fallback
    produces equivalent quality for ~0 cost (no LLM latency).  The rewrite_query
    node already does the heavy semantic lifting.
    """
    rewritten = [item for item in (state.get("rewrittenQuestions") or []) if str(item).strip()]
    original_query = state.get("originalQuery") or state.get("primary_user_query") or ""
    base_query = rewritten[0] if rewritten else original_query
    planned = plan_queries(base_query, topic_focus=state.get("topic_focus", ""), recent_context=state.get("recent_context", ""))
    return {"planned_queries": planned}


def reset_supervisor_state(state: State):
    """P4: clear supervisor loop flags at turn start.

    LangGraph's checkpointer persists State across turns. If a supervisor-
    dispatched specialist interrupted (e.g. appointment needs clarification),
    the leftover supervisor_active=True would mis-route the resumed specialist
    back to supervise. This node resets those flags every turn, before
    analyze_turn, with zero invasion of analyze_turn's return paths.
    """
    return {"supervisor_active": False, "supervisor_rounds": 0}


def decompose_tasks(state: State, llm):
    """P3: judge whether a medical question is compound and split into sub-questions.

    A light LLM (via TaskDecomposition schema) decides whether the question
    contains multiple independent facets. If compound, it produces 1-N
    self-contained sub-questions (N <= MAX_SUB_QUESTIONS); if not, the result
    is a single-element list holding the primary query — which makes the
    downstream route_after_query_plan fall back to today's single-Send path.
    LLM failure (empty sub_questions) falls back to [primary] so the node never
    breaks. When ENABLE_TASK_DECOMPOSITION is False the LLM is skipped and
    [primary] is returned directly (rollback).
    """
    rewritten = [str(q).strip() for q in (state.get("rewrittenQuestions") or []) if str(q).strip()]
    original_query = str(state.get("originalQuery") or state.get("primary_user_query") or "").strip()
    primary = rewritten[0] if rewritten else original_query

    if not config.ENABLE_TASK_DECOMPOSITION or not primary:
        return {"sub_questions": [primary] if primary else []}

    sys_msg = SystemMessage(content=get_task_decomposition_prompt())
    user_payload = (
        f"用户原始问题：{original_query}\n"
        f"重写后问题：{rewritten}\n"
        f"上下文摘要：{state.get('conversation_summary', '')}\n"
        f"近期对话：{state.get('recent_context', '')}\n"
        f"话题焦点：{state.get('topic_focus', '')}"
    )
    parser = _structured_output_llm(llm, TaskDecomposition, max_tokens=config.LLM_STRUCTURED_MAX_TOKENS)
    verdict = parser.invoke([sys_msg, HumanMessage(content=user_payload)])

    subs = [str(s).strip() for s in (getattr(verdict, "sub_questions", []) or []) if str(s).strip()]

    # Fallback: LLM gave no usable sub-questions → single primary path.
    if not subs:
        return {"sub_questions": [primary] if primary else []}

    return {"sub_questions": subs[: config.MAX_SUB_QUESTIONS]}


def _latest_tool_message(state: AgentState):
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, ToolMessage):
            return msg
    return None


def _evidence_docs_from_tool_message(text: str) -> list[Document]:
    """Parse graded retrieval-result blocks from a ToolMessage content string.

    The retrieval tool (tools.py:654-672) joins per-result blocks with
    "\\n\\n"; each block carries `Score: 0.8732` and `Relevance Grade: high`
    lines. We rebuild Document objects carrying the same metadata shape that
    `check_sufficiency` (tools.py:190) and `_doc_score` (tools.py:84) read, so
    the rule-based sufficiency check sees genuine scores/grades instead of a
    hardcoded weak doc. Returns an empty list when no parseable blocks exist.
    """
    docs: list[Document] = []
    if not text:
        return docs
    for block in text.split("\n\n"):
        score_match = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", block, re.IGNORECASE)
        if not score_match:
            continue
        try:
            score = float(score_match.group(1))
        except ValueError:
            continue
        grade_match = re.search(r"Relevance Grade:\s*(\w+)", block, re.IGNORECASE)
        grade = grade_match.group(1).strip().lower() if grade_match else ""
        docs.append(Document(page_content=block, metadata={"score": score, "relevance_grade": grade}))
    return docs


def evaluate_evidence(state: AgentState, llm):
    """P1: reflect on retrieved evidence; decide sufficiency and refine query.

    Fast path: rule-based check_sufficiency says sufficient → skip LLM.
    Reflection path: rule says insufficient → light LLM judges via
    EvidenceSufficiency schema and produces a refined retry_query.
    Failure path: LLM parse/empty → fall back to rule's retry_query.
    """
    tool_msg = _latest_tool_message(state)
    evidence_text = str(getattr(tool_msg, "content", "") or "")
    question = str(state.get("question") or "").strip()
    query_plan = [str(q).strip() for q in (state.get("query_plan") or []) if str(q).strip()]

    # Parse genuine graded docs from the latest tool result so check_sufficiency
    # sees real scores/grades — this is what makes the fast path reachable.
    evidence_docs = _evidence_docs_from_tool_message(evidence_text)

    rule = check_sufficiency(question or (query_plan[0] if query_plan else ""), evidence_docs)

    rounds = int(state.get("evidence_rounds", 0) or 0)

    # Fast path: rule says sufficient — skip LLM entirely.
    if rule.get("is_sufficient"):
        return {
            "evidence_critique": rule.get("reason", ""),
            "last_refined_query": "",
            "evidence_rounds": rounds,
            "evidence_sufficient": True,
        }

    # Reflection path.
    sys_msg = SystemMessage(content=get_evidence_sufficiency_prompt())
    user_payload = (
        f"用户问题：{question}\n"
        f"已有检索式：{query_plan}\n"
        f"最新检索证据：\n{evidence_text[:2000]}"
    )
    parser = _structured_output_llm(llm, EvidenceSufficiency, max_tokens=config.LLM_STRUCTURED_MAX_TOKENS)
    verdict = parser.invoke([sys_msg, HumanMessage(content=user_payload)])

    refined = (verdict.retry_query or "").strip()
    critique = (verdict.reason or rule.get("reason", "")).strip()

    # Failure fallback: LLM gave insufficient but no usable refined query.
    if not verdict.is_sufficient and not refined:
        refined = (rule.get("retry_query") or "").strip()

    is_sufficient = bool(verdict.is_sufficient)

    delta = {
        "evidence_critique": critique,
        "evidence_rounds": rounds + 1 if not is_sufficient else rounds,
        "evidence_sufficient": is_sufficient,
    }
    if refined and not is_sufficient:
        delta["last_refined_query"] = refined
        delta["refined_queries"] = [refined]
    else:
        delta["last_refined_query"] = ""
    return delta


# --- Agent Nodes ---
def orchestrator(state: AgentState, llm_with_tools):
    context_summary = state.get("context_summary", "").strip()
    recent_context = state.get("recent_context", "").strip()
    topic_focus = state.get("topic_focus", "").strip()
    question = str(state.get("question") or "").strip()
    query_plan = [str(item).strip() for item in (state.get("query_plan") or []) if str(item).strip()]
    is_medical_request = _looks_like_medical_request(
        question,
        conversation_summary=context_summary,
        recent_context=recent_context,
        topic_focus=topic_focus,
    )
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    recent_context_injection = (
        [HumanMessage(content=f"[RECENT DIALOGUE CONTEXT]\n\n{recent_context}")]
        if recent_context else []
    )
    topic_focus_injection = (
        [HumanMessage(content=f"[TOPIC FOCUS]\n\n{topic_focus}")]
        if topic_focus else []
    )
    user_memories = state.get("user_memories", "").strip()
    user_memories_injection = (
        [HumanMessage(content=f"[已知用户信息: 高血压/过敏/偏好等]\n\n{user_memories}")]
        if user_memories else []
    )
    query_plan_injection = (
        [HumanMessage(content="[RETRIEVAL QUERY PLAN]\n\n" + "\n".join(f"- {item}" for item in query_plan))]
        if query_plan else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=question)
        base_messages = [sys_msg] + summary_injection + recent_context_injection + topic_focus_injection + user_memories_injection + query_plan_injection + [human_msg]
        if is_medical_request:
            retrieval_hint = (
                "For this medical question, call 'search_child_chunks' first unless the injected context already provides enough evidence. "
                "Pass the retrieval query plan into the tool when available. Prefer the current question first; if the first retrieval is weak, you may try one alternate query from the retrieval query plan, but avoid repeating the same search."
            )
            base_messages.append(HumanMessage(content=retrieval_hint))
        # P1: inject refined-query hint after evidence reflection found evidence insufficient.
        # When both retrieval_hint and refined_hint are present, refined_hint takes precedence (it is more specific and recent).
        refined_query = str(state.get("last_refined_query", "") or "").strip()
        if refined_query:
            critique = str(state.get("evidence_critique", "") or "").strip()
            refined_hint = (
                f"上一次检索证据不足，原因：{critique}。"
                f"请用以下检索式重新调用 search_child_chunks，不要重复之前的查询：{refined_query}"
            )
            base_messages.append(HumanMessage(content=refined_hint))
        response = llm_with_tools.invoke(base_messages)
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1, "last_refined_query": ""}

    reuse_messages = [sys_msg] + summary_injection + recent_context_injection + topic_focus_injection + user_memories_injection + query_plan_injection + state["messages"]
    refined_query = str(state.get("last_refined_query", "") or "").strip()
    if refined_query:
        critique = str(state.get("evidence_critique", "") or "").strip()
        reuse_messages.append(HumanMessage(
            content=f"上一次检索证据不足，原因：{critique}。请用以下检索式重新调用 search_child_chunks，不要重复之前的查询：{refined_query}"
        ))
    response = llm_with_tools.invoke(reuse_messages)
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1, "last_refined_query": ""}

def fallback_response(state: AgentState, llm):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()
    recent_context = state.get("recent_context", "").strip()
    topic_focus = state.get("topic_focus", "").strip()
    user_query = str(state.get("question") or "").strip()
    is_medical_request = _looks_like_medical_request(
        user_query,
        conversation_summary=context_summary,
        recent_context=recent_context,
        topic_focus=topic_focus,
    )
    inferred_risk = _infer_risk_level(user_query, "normal")
    risk_level = "high" if _needs_strict_medical_safety(user_query, inferred_risk) else "normal"
    has_no_evidence = any("NO_EVIDENCE" in content for content in unique_contents)

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if recent_context:
        context_parts.append(f"## Recent Dialogue Context\n\n{recent_context}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {user_query}\n\n"
        f"REQUEST TYPE: {'medical' if is_medical_request else 'general_or_non_medical'}\n"
        f"RISK LEVEL: {risk_level}\n"
        f"KNOWLEDGE STATUS: {'no_evidence' if has_no_evidence else 'limited_or_partial'}\n\n"
        f"{context_text}\n\n"
        "INSTRUCTION:\n"
        "- If this is a medical request with weak or missing evidence, still provide a concise general medical-information answer when reasonably safe.\n"
        "- For that medical fallback mode, clearly say the answer was not sufficiently based on knowledge-base retrieval and cannot replace in-person medical diagnosis.\n"
        "- For severe symptoms, worsening symptoms, or medication/dosing questions, add a stronger safety reminder.\n"
        "- For non-medical or casual questions, answer naturally and do not force a medical refusal."
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}

def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)

def compress_context(state: AgentState, llm):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}


def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = _sanitize_final_answer_text(last_message.content) if is_valid else "Unable to generate an answer."
    citations = _extract_source_citations(state.get("messages", []))
    confidence_bucket = next((item.get("confidence_bucket") for item in citations if item.get("confidence_bucket")), "")
    if not confidence_bucket:
        confidence_bucket = "no_evidence" if any("NO_EVIDENCE" in str(msg.content or "") for msg in state.get("messages", []) if isinstance(msg, ToolMessage)) else "low"
    evidence_scores = [
        float(item.get("evidence_score"))
        for item in citations
        if item.get("evidence_score") is not None
    ]
    return {
        "final_answer": answer,
        "agent_answers": [{
            "index": state["question_index"],
            "question": state["question"],
            "answer": answer,
            "query_plan": state.get("query_plan", []),
            "confidence_bucket": confidence_bucket,
            "evidence_score": max(evidence_scores) if evidence_scores else None,
            "sources": citations[:3],
        }]
    }
# --- End of Agent Nodes---

def grounded_answer_generation(state: State, llm):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    all_sources = []
    seen_sources = set()
    confidence_levels = []
    evidence_scores = []
    for answer in sorted_answers:
        bucket = str(answer.get("confidence_bucket") or "").strip().lower()
        if bucket:
            confidence_levels.append(bucket)
        score = answer.get("evidence_score")
        if score is not None:
            try:
                evidence_scores.append(float(score))
            except (TypeError, ValueError):
                pass
        for item in answer.get("sources") or []:
            key = (item.get("title", ""), item.get("source_type", ""), item.get("original_url", ""))
            if key in seen_sources:
                continue
            seen_sources.add(key)
            all_sources.append(item)

    confidence_bucket = "high"
    if "no_evidence" in confidence_levels:
        confidence_bucket = "no_evidence"
    elif "low" in confidence_levels:
        confidence_bucket = "low"
    elif "medium" in confidence_levels:
        confidence_bucket = "medium"
    aggregate_evidence_score = max(evidence_scores) if evidence_scores else None

    citation_lines = _format_reference_lines(all_sources)

    original_query = state.get("originalQuery", "")
    is_medical_request = _looks_like_medical_request(
        original_query,
        conversation_summary=state.get("conversation_summary", ""),
        recent_context=state.get("recent_context", ""),
        topic_focus=state.get("topic_focus", ""),
    )
    risk_level = _infer_risk_level(original_query, state.get("risk_level", "normal"))

    confidence_note = ""
    confidence_label = _confidence_bucket_label(confidence_bucket)
    confidence_explanation = _confidence_bucket_explanation(
        confidence_bucket,
        is_medical_request=is_medical_request,
    )
    if is_medical_request and confidence_bucket in {"no_evidence", "low"}:
        confidence_note = f"\n\n证据强度：`{confidence_label}`。{confidence_explanation}\n\n" + _build_medical_fallback_notice(
            risk_level="high" if _needs_strict_medical_safety(original_query, risk_level) else "normal",
            confidence_bucket=confidence_bucket,
        )
    elif confidence_bucket == "medium":
        confidence_note = f"\n\n证据强度：`{confidence_label}`。{confidence_explanation}"
    elif confidence_bucket == "high":
        confidence_note = f"\n\n证据强度：`{confidence_label}`。{confidence_explanation}"
    if any(item.get("freshness_bucket") == "outdated" for item in all_sources):
        confidence_note += "\n\n版本提醒：当前命中了较旧资料，请结合最新指南或线下医生意见一起判断。"

    citation_block = ""
    if citation_lines:
        citation_block = "\n\n参考来源：\n" + "\n".join(citation_lines)

    final_content = _sanitize_final_answer_text(synthesis_response.content)
    return {
        "messages": [AIMessage(content=f"{final_content}{confidence_note}{citation_block}")],
        "clarification_attempts": 0,
        "grounding_evidence_score": aggregate_evidence_score,
    }


def answer_grounding_check(state: State, llm):
    latest_message = state["messages"][-1] if state.get("messages") else None
    current_answer = str(getattr(latest_message, "content", "") or "").strip()
    confidence_levels = [
        str(item.get("confidence_bucket") or "").strip().lower()
        for item in state.get("agent_answers") or []
        if str(item.get("confidence_bucket") or "").strip()
    ]
    # Fast-path: skip grounding check when evidence is clearly strong
    has_low = any(c in ("low", "no_evidence") for c in confidence_levels)
    evidence_score = state.get("grounding_evidence_score")
    if not has_low and evidence_score is not None and evidence_score >= config.RAG_HIGH_CONFIDENCE_SCORE:
        return {"grounding_passed": True}
    # Build evidence docs from agent_answers (each answer carries its retrieval
    # evidence metadata — score, source citation).  If agent_answers is empty,
    # fall back to the legacy numerical score.
    evidence_docs = []
    for item in (state.get("agent_answers") or []):
        if isinstance(item, dict):
            content = item.get("answer", "") or item.get("content", "") or ""
            score = item.get("score", item.get("evidence_score", None))
            source = item.get("source", "") or item.get("citation", "") or ""
            if not source and score is not None:
                source = f"evidence_score={score}"
            if source:
                evidence_docs.append(Document(page_content=str(content), metadata={"score": float(score) if score else 0.0, "source": str(source)}))
    if not evidence_docs and evidence_score is not None:
        evidence_docs = [Document(page_content="", metadata={"score": float(evidence_score)})]
    if not evidence_docs:
        if "no_evidence" in confidence_levels:
            evidence_docs = []
        else:
            evidence_docs = [Document(page_content="", metadata={"score": 0.88 if "high" in confidence_levels else 0.68})]
    original_query = state.get("originalQuery", "")
    risk_level = _infer_risk_level(original_query, state.get("risk_level", "normal"))
    grounded = ground_answer(
        current_answer,
        evidence_docs,
        question=original_query,
        medical_mode=_looks_like_medical_request(
            original_query,
            conversation_summary=state.get("conversation_summary", ""),
            recent_context=state.get("recent_context", ""),
            topic_focus=state.get("topic_focus", ""),
        ),
        high_risk=_needs_strict_medical_safety(original_query, risk_level),
    )
    final_answer = _strip_leading_query_plan_blob(grounded.get("revised_answer", current_answer))
    is_grounded = bool(grounded.get("grounded"))
    delta: dict = {"grounding_passed": is_grounded}
    # Append the (passive disclaimer) revised answer only when it differs —
    # this is the termination-branch safe degrade; if revise_answer runs next
    # it appends an evidence-bounded rewrite that becomes the latest message.
    if final_answer != current_answer:
        delta["messages"] = [AIMessage(content=final_answer)]
    return delta


def revise_answer(state: State, llm):
    """P2: critique an un-grounded answer and rewrite it within evidence bounds.

    Reads the current (un-grounded) answer from the last message, the evidence
    from agent_answers, and asks the light LLM (via GroundingCritique schema)
    for a structured critique + an evidence-bounded rewrite. The rewrite is
    appended as the latest message; control returns to answer_grounding_check
    for a re-check. LLM failure (empty revised_answer) falls back to
    ground_answer's passive-disclaimer revised_answer so the node never breaks.
    """
    latest_message = state["messages"][-1] if state.get("messages") else None
    current_answer = str(getattr(latest_message, "content", "") or "").strip()

    # Evidence block from agent_answers (same shape answer_grounding_check builds).
    evidence_docs = []
    for item in (state.get("agent_answers") or []):
        if isinstance(item, dict):
            content = item.get("answer", "") or item.get("content", "") or ""
            score = item.get("score", item.get("evidence_score", None))
            source = item.get("source", "") or item.get("citation", "") or ""
            if not source and score is not None:
                source = f"evidence_score={score}"
            if source:
                evidence_docs.append(Document(page_content=str(content), metadata={"score": float(score) if score else 0.0, "source": str(source)}))
    evidence_text = "\n\n".join(
        f"[证据{i}] {getattr(d, 'page_content', '')}".strip()
        for i, d in enumerate(evidence_docs, start=1)
    ) or "（无结构化证据）"

    original_query = str(state.get("originalQuery", "") or "").strip()
    rounds = int(state.get("grounding_rounds", 0) or 0)
    risk_level = _infer_risk_level(original_query, state.get("risk_level", "normal"))

    # Passive-disclaimer fallback (used if the LLM critique yields no rewrite).
    fallback = ground_answer(
        current_answer,
        evidence_docs,
        question=original_query,
        medical_mode=_looks_like_medical_request(
            original_query,
            conversation_summary=state.get("conversation_summary", ""),
            recent_context=state.get("recent_context", ""),
            topic_focus=state.get("topic_focus", ""),
        ),
        high_risk=_needs_strict_medical_safety(original_query, risk_level),
    )
    fallback_revised = _strip_leading_query_plan_blob(fallback.get("revised_answer", current_answer))

    sys_msg = SystemMessage(content=get_grounding_critique_prompt())
    user_payload = (
        f"用户问题：{original_query}\n"
        f"检索证据：\n{evidence_text[:2000]}\n"
        f"待评审回答：\n{current_answer}"
    )
    parser = _structured_output_llm(llm, GroundingCritique, max_tokens=config.LLM_STRUCTURED_MAX_TOKENS)
    verdict = parser.invoke([sys_msg, HumanMessage(content=user_payload)])

    critique = (getattr(verdict, "critique", "") or "").strip()
    revised = (getattr(verdict, "revised_answer", "") or "").strip()

    if not revised:
        revised = fallback_revised
        if not critique:
            critique = (fallback.get("note", "") or "").strip()

    return {
        "messages": [AIMessage(content=revised)],
        "grounding_critique": critique,
        "grounding_rounds": rounds + 1,
    }


def aggregate_answers(state: State, llm):
    return grounded_answer_generation(state, llm)


__all__ = [
    "aggregate_answers",
    "answer_grounding_check",
    "collect_answer",
    "compress_context",
    "decompose_tasks",
    "evaluate_evidence",
    "fallback_response",
    "grounded_answer_generation",
    "orchestrator",
    "plan_retrieval_queries",
    "revise_answer",
    "reset_supervisor_state",
    "rewrite_query",
    "should_compress_context",
]
