"""Answer grounding + aggregation nodes.

grounded_answer_generation (merge by index), answer_grounding_check (critique),
revise_answer (evidence-bounded rewrite), aggregate_answers. Extracted from rag_nodes.
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from .graph_state import State, AgentState
from .schemas import GroundingCritique
from .prompts import get_aggregation_prompt, get_grounding_critique_prompt
import config
from .tools import ground_answer
from .node_helpers import (
    _structured_output_llm,
    _format_reference_lines,
    _confidence_bucket_explanation,
    _confidence_bucket_label,
    _infer_risk_level,
    _looks_like_medical_request,
    _needs_strict_medical_safety,
    _sanitize_final_answer_text,
    _strip_leading_query_plan_blob,
)

logger = logging.getLogger(__name__)


def _build_missing_subquestion_caveat(state, sorted_answers) -> str:
    """Detect sub-questions that decompose planned but weren't reliably answered,
    and return an explicit caveat naming them.

    A sub-question counts as unaddressed if its index never produced an
    agent_answer, or the answer is empty / the "Unable to generate" fallback.
    Returns "" when no decomposition happened (single question) or every planned
    sub-question was answered - so the caveat only fires for genuine compound
    medical questions that came back partial.
    """
    subs = [str(s).strip() for s in (state.get("sub_questions") or []) if str(s).strip()]
    if len(subs) <= 1:
        return ""
    answered: dict = {}
    for a in sorted_answers:
        if not isinstance(a, dict):
            continue
        try:
            idx = int(a.get("index", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0:
            continue
        answered[idx] = str(a.get("answer", "") or "").strip()
    missing: list = []
    for i, q in enumerate(subs):
        ans = answered.get(i)
        if ans is None or not ans or "Unable to generate" in ans:
            missing.append(q)
    if not missing:
        return ""
    items = "、".join(f"「{q}」" for q in missing)
    return (
        f"\n\n⚠️ 您的问题包含多个部分，其中关于{items}暂未能给出可靠回答，"
        f"可否再单独描述一下这部分？"
    )


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
    missing_caveat = _build_missing_subquestion_caveat(state, sorted_answers)
    return {
        "messages": [AIMessage(content=f"{final_content}{confidence_note}{missing_caveat}{citation_block}")],
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
    "reset_turn_state",
    "self_eval",
    "rewrite_query",
    "should_compress_context",
]
