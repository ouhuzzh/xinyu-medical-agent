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
)
from .prompts import (
    get_rewrite_query_prompt,
    get_retrieval_query_plan_prompt,
    get_orchestrator_prompt,
    get_fallback_response_prompt,
    get_context_compression_prompt,
    get_aggregation_prompt,
)
from utils import estimate_context_tokens
import config
from config import BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR
from rag_agent.tools import plan_queries, ground_answer

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
    if state.get("core_memory"):
        context_parts.append(f"[Core Memory - Always Visible]\n{state['core_memory']}\n")
    if conversation_summary.strip():
        context_parts.append(f"Conversation Context:\n{conversation_summary}\n")
    if recent_context.strip():
        context_parts.append(f"Recent Dialogue Context:\n{recent_context}\n")
    if state.get("user_memories"):
        context_parts.append(f"Known user context:\n{state['user_memories']}\n")
    if state.get("reflection_memories"):
        context_parts.append(f"Insights about user:\n{state['reflection_memories']}\n")
    if state.get("episodic_memories"):
        context_parts.append(f"Relevant past conversations:\n{state['episodic_memories']}\n")
    if topic_focus.strip():
        context_parts.append(f"Topic focus:\n{topic_focus}\n")
    context_parts.append(f"User Query:\n{user_query}\n")
    context_section = "".join(context_parts)

    try:
        llm_with_structure = _structured_output_llm(llm, QueryAnalysis, temperature=0.1)
        response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])
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
        delete_all = _build_history_reset_messages(state["messages"])
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": user_query,
            "rewrittenQuestions": response.questions,
            "recent_context": recent_context,
            "topic_focus": topic_focus,
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
    query_plan_injection = (
        [HumanMessage(content="[RETRIEVAL QUERY PLAN]\n\n" + "\n".join(f"- {item}" for item in query_plan))]
        if query_plan else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=question)
        base_messages = [sys_msg] + summary_injection + recent_context_injection + topic_focus_injection + query_plan_injection + [human_msg]
        if is_medical_request:
            retrieval_hint = (
                "For this medical question, call 'search_child_chunks' first unless the injected context already provides enough evidence. "
                "Pass the retrieval query plan into the tool when available. Prefer the current question first; if the first retrieval is weak, you may try one alternate query from the retrieval query plan, but avoid repeating the same search."
            )
            base_messages.append(HumanMessage(content=retrieval_hint))
        response = llm_with_tools.invoke(base_messages)
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    response = llm_with_tools.invoke([sys_msg] + summary_injection + recent_context_injection + topic_focus_injection + query_plan_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}

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
        return {}
    evidence_score = state.get("grounding_evidence_score")
    pseudo_docs = []
    if evidence_score is not None:
        try:
            pseudo_docs = [Document(page_content="", metadata={"score": float(evidence_score)})]
        except (TypeError, ValueError):
            pseudo_docs = []
    if not pseudo_docs:
        if "no_evidence" in confidence_levels:
            pseudo_docs = []
        elif "low" in confidence_levels:
            pseudo_docs = [Document(page_content="", metadata={"score": 0.68})]
        else:
            pseudo_docs = [Document(page_content="", metadata={"score": 0.88})]
    original_query = state.get("originalQuery", "")
    risk_level = _infer_risk_level(original_query, state.get("risk_level", "normal"))
    grounded = ground_answer(
        current_answer,
        pseudo_docs,
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
    if final_answer == current_answer:
        return {}
    return {"messages": [AIMessage(content=final_answer)]}


def aggregate_answers(state: State, llm):
    return grounded_answer_generation(state, llm)


__all__ = [
    "aggregate_answers",
    "answer_grounding_check",
    "collect_answer",
    "compress_context",
    "fallback_response",
    "grounded_answer_generation",
    "orchestrator",
    "plan_retrieval_queries",
    "rewrite_query",
    "should_compress_context",
]
