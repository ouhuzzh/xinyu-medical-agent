"""Retrieval quality evaluator — offline benchmarking for the RAG pipeline.

Evaluates:
    - Retrieval quality: source type match, keyword coverage, confidence buckets
    - Answer quality: keyword coverage, safety keywords, tone (patient-friendly vs clinical)
    - Route quality: intent classification hit rate, compound request handling
    - Supports pipeline_config for ablation studies
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from rag_agent.tools import ToolFactory
from rag_agent.routing_nodes import classify_query_offline


logger = logging.getLogger(__name__)


CLARIFICATION_PATTERNS = (
    "请问",
    "能否补充",
    "可以补充",
    "方便补充",
    "具体是",
    "想了解",
    "还需要知道",
    "请再描述",
    "需要更多信息",
)
PATIENT_FRIENDLY_MARKERS = (
    "建议",
    "可以",
    "通常",
    "一般",
    "平时",
    "注意",
    "尽快就医",
    "按医嘱",
    "先",
)
CLINICAL_JARGON_MARKERS = (
    "危险分层",
    "合并症",
    "一线药物",
    "首选药物",
    "剂量调整逻辑",
    "临床评估",
    "protocol",
    "guideline",
    "dosing",
    "risk stratification",
)
NO_EVIDENCE_PATTERNS = (
    "暂无相关信息",
    "暂无相关资料",
    "没有相关证据",
    "未找到相关信息",
    "知识库暂无",
    "没有在知识库里找到",
)


@dataclass
class QAEvalSample:
    sample_id: str
    question: str
    search_query: str = ""
    conversation_summary: str = ""
    transcript_turns: List[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)
    expected_source_type: str = ""
    expected_source_contains: List[str] = field(default_factory=list)
    expected_retrieval_keywords: List[str] = field(default_factory=list)
    expected_answer_keywords: List[str] = field(default_factory=list)
    expected_safety_keywords: List[str] = field(default_factory=list)
    forbidden_answer_keywords: List[str] = field(default_factory=list)
    expected_no_evidence: bool = False
    must_not_clarify: bool = False
    preferred_answer_style: str = ""
    note: str = ""
    expected_primary_intent: str = ""
    expected_secondary_intent: str = ""

    @classmethod
    def from_dict(cls, payload: dict):
        return cls(
            sample_id=str(payload.get("id") or payload.get("sample_id") or "").strip(),
            question=str(payload.get("question") or "").strip(),
            search_query=str(payload.get("search_query") or "").strip(),
            conversation_summary=str(payload.get("conversation_summary") or "").strip(),
            transcript_turns=[str(item).strip() for item in payload.get("transcript_turns", []) if str(item).strip()],
            category=str(payload.get("category") or "general").strip().lower(),
            difficulty=str(payload.get("difficulty") or "medium").strip().lower(),
            tags=[str(item).strip().lower() for item in payload.get("tags", []) if str(item).strip()],
            expected_source_type=str(payload.get("expected_source_type") or "").strip().lower(),
            expected_source_contains=[str(item).strip() for item in payload.get("expected_source_contains", []) if str(item).strip()],
            expected_retrieval_keywords=[str(item).strip() for item in payload.get("expected_retrieval_keywords", []) if str(item).strip()],
            expected_answer_keywords=[str(item).strip() for item in payload.get("expected_answer_keywords", []) if str(item).strip()],
            expected_safety_keywords=[str(item).strip() for item in payload.get("expected_safety_keywords", []) if str(item).strip()],
            forbidden_answer_keywords=[str(item).strip() for item in payload.get("forbidden_answer_keywords", []) if str(item).strip()],
            expected_no_evidence=bool(payload.get("expected_no_evidence", False)),
            must_not_clarify=bool(payload.get("must_not_clarify", False)),
            preferred_answer_style=str(payload.get("preferred_answer_style") or "").strip().lower(),
            note=str(payload.get("note") or "").strip(),
            expected_primary_intent=str(payload.get("expected_primary_intent") or "").strip(),
            expected_secondary_intent=str(payload.get("expected_secondary_intent") or "").strip(),
        )

    def validate(self):
        if not self.sample_id:
            raise ValueError("QA sample is missing `id`.")
        if not self.question:
            raise ValueError(f"QA sample `{self.sample_id}` is missing `question`.")


@dataclass
class RetrievalEvalResult:
    sample_id: str
    question: str
    category: str
    difficulty: str
    transcript_turns: List[str]
    route_primary_intent: str
    route_secondary_intent: str
    route_decision_source: str
    route_reason: str
    route_hit: bool
    secondary_route_hit: bool
    preferred_source_layers: List[str]
    confidence_bucket: str
    retrieval_score: float
    answer_score: Optional[float]
    overall_score: float
    retrieval_relevance_hit: bool
    evidence_sufficient: bool
    grounding_violation_detected: bool
    top_source_type: str
    top_source: str
    matched_retrieval_keywords: List[str]
    missing_retrieval_keywords: List[str]
    matched_answer_keywords: List[str]
    missing_answer_keywords: List[str]
    matched_safety_keywords: List[str]
    missing_safety_keywords: List[str]
    clarification_detected: bool
    patient_friendly_detected: bool
    no_evidence_answer_detected: bool
    safety_score: Optional[float]
    tone_score: Optional[float]
    source_type_hit: bool
    source_contains_hit: bool
    no_evidence_detected: bool
    retrieved_sources: List[str]
    retrieved_source_types: List[str]
    answer_text: str
    snippets: List[str]
    retrieval_latency_ms: float = 0.0
    vector_search_latency_ms: float = 0.0
    keyword_search_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


def load_qa_samples(path: str | Path) -> List[QAEvalSample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = [QAEvalSample.from_dict(item) for item in payload]
    for sample in samples:
        sample.validate()
    return samples


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _keyword_matches(text: str, keywords: Iterable[str]) -> tuple[list[str], list[str]]:
    normalized = _normalize_text(text)
    matched, missing = [], []
    for keyword in keywords:
        if _normalize_text(keyword) in normalized:
            matched.append(keyword)
        else:
            missing.append(keyword)
    return matched, missing


def _weighted_score(components: list[tuple[float, float]]) -> float:
    if not components:
        return 1.0
    total_weight = sum(weight for weight, _ in components)
    if total_weight <= 0:
        return 1.0
    return round(sum(weight * value for weight, value in components) / total_weight, 4)


def _group_summary(results: list[RetrievalEvalResult], attr_name: str) -> dict:
    groups = {}
    for item in results:
        group_name = getattr(item, attr_name, "") or "unspecified"
        groups.setdefault(group_name, []).append(item)

    summary = {}
    for group_name, items in groups.items():
        answer_scored = [entry.answer_score for entry in items if entry.answer_score is not None]
        safety_scored = [entry.safety_score for entry in items if entry.safety_score is not None]
        tone_scored = [entry.tone_score for entry in items if entry.tone_score is not None]
        summary[group_name] = {
            "sample_count": len(items),
            "avg_retrieval_score": round(sum(entry.retrieval_score for entry in items) / len(items), 4),
            "avg_overall_score": round(sum(entry.overall_score for entry in items) / len(items), 4),
            "avg_answer_score": round(sum(answer_scored) / len(answer_scored), 4) if answer_scored else None,
            "avg_safety_score": round(sum(safety_scored) / len(safety_scored), 4) if safety_scored else None,
            "avg_tone_score": round(sum(tone_scored) / len(tone_scored), 4) if tone_scored else None,
            "pass_rate_085": round(sum(1 for entry in items if entry.overall_score >= 0.85) / len(items), 4),
            "route_hit_rate": round(sum(1 for entry in items if entry.route_hit) / len(items), 4),
            "retrieval_relevance_hit_rate": round(sum(1 for entry in items if entry.retrieval_relevance_hit) / len(items), 4),
            "evidence_sufficiency_pass_rate": round(sum(1 for entry in items if entry.evidence_sufficient) / len(items), 4),
            "grounding_violation_rate": round(sum(1 for entry in items if entry.grounding_violation_detected) / len(items), 4),
        }
    return summary


def _messages_from_transcript(sample: QAEvalSample) -> list:
    messages = []
    for turn in sample.transcript_turns:
        text = str(turn or "").strip()
        if not text:
            continue
        if text.lower().startswith("user:"):
            messages.append(HumanMessage(content=text.split(":", 1)[1].strip()))
        elif text.lower().startswith("assistant:"):
            messages.append(AIMessage(content=text.split(":", 1)[1].strip()))
    if not messages or not isinstance(messages[-1], HumanMessage):
        messages.append(HumanMessage(content=sample.question))
    return messages


def _confidence_bucket_from_docs(docs) -> str:
    if not docs:
        return "no_evidence"
    metadata = docs[0].metadata or {}
    score = float(metadata.get("rerank_score") or metadata.get("fusion_score") or metadata.get("score") or 0.0)
    if score >= 0.85 and len(docs) >= 2:
        return "high"
    if score >= 0.72:
        return "medium"
    return "low"


class RetrievalQualityEvaluator:
    def __init__(self, collection, *, limit: int = 3, score_threshold: float = 0.7,
                 pipeline_config=None):
        # ``analyze_turn`` delegates rule routing to the skill registry.  The
        # live application bootstraps that registry during graph compilation,
        # but offline evaluators run independently of the application startup.
        # Without this registration every expected route is reported as an LLM
        # fallback, producing a misleading 0% route hit rate.
        try:
            from core.skill_bootstrapper import SkillBootstrapper

            SkillBootstrapper().bootstrap()
        except Exception:
            logger.warning("QA evaluator could not bootstrap skills; route metrics may be incomplete.", exc_info=True)
        self.tool_factory = ToolFactory(collection)
        self.limit = limit
        self.score_threshold = score_threshold
        self.pipeline_config = pipeline_config

    @staticmethod
    def _score_answer(sample: QAEvalSample, answer_text: str) -> dict:
        if answer_text is None:
            return {
                "answer_score": None,
                "matched_answer_keywords": [],
                "missing_answer_keywords": [],
                "matched_safety_keywords": [],
                "missing_safety_keywords": [],
                "clarification_detected": False,
                "patient_friendly_detected": False,
                "no_evidence_answer_detected": False,
                "safety_score": None,
                "tone_score": None,
                "grounding_violation_detected": False,
            }

        answer_text = answer_text.strip()
        normalized_answer = _normalize_text(answer_text)
        matched_keywords, missing_keywords = _keyword_matches(answer_text, sample.expected_answer_keywords)
        matched_safety_keywords, missing_safety_keywords = _keyword_matches(answer_text, sample.expected_safety_keywords)
        clarification_detected = any(token in normalized_answer for token in CLARIFICATION_PATTERNS)
        patient_friendly_hits = sum(1 for token in PATIENT_FRIENDLY_MARKERS if token in answer_text)
        jargon_hits = sum(1 for token in CLINICAL_JARGON_MARKERS if token in normalized_answer)
        patient_friendly_detected = patient_friendly_hits > 0 and jargon_hits < 2
        no_evidence_answer_detected = any(token in answer_text for token in NO_EVIDENCE_PATTERNS)

        forbidden_hit = False
        components = []
        if sample.expected_answer_keywords:
            coverage = len(matched_keywords) / len(sample.expected_answer_keywords)
            components.append((0.45, coverage))
        safety_score = None
        if sample.expected_safety_keywords:
            safety_score = len(matched_safety_keywords) / len(sample.expected_safety_keywords)
            components.append((0.25, safety_score))
        tone_score = None
        if sample.preferred_answer_style:
            if sample.preferred_answer_style == "patient_friendly":
                tone_score = _weighted_score([
                    (0.6, 1.0 if patient_friendly_hits else 0.0),
                    (0.4, 0.0 if jargon_hits >= 2 else 1.0),
                ])
            elif sample.preferred_answer_style == "no_evidence":
                tone_score = _weighted_score([
                    (0.8, 1.0 if no_evidence_answer_detected else 0.0),
                    (0.2, 0.0 if clarification_detected else 1.0),
                ])
            else:
                tone_score = 1.0
            components.append((0.15, tone_score))
        if sample.forbidden_answer_keywords:
            forbidden_hit = any(_normalize_text(token) in normalized_answer for token in sample.forbidden_answer_keywords)
            components.append((0.1, 0.0 if forbidden_hit else 1.0))
        if sample.must_not_clarify:
            components.append((0.05, 0.0 if clarification_detected else 1.0))

        grounding_violation_detected = forbidden_hit or (sample.expected_no_evidence and not no_evidence_answer_detected)

        return {
            "answer_score": _weighted_score(components),
            "matched_answer_keywords": matched_keywords,
            "missing_answer_keywords": missing_keywords,
            "matched_safety_keywords": matched_safety_keywords,
            "missing_safety_keywords": missing_safety_keywords,
            "clarification_detected": clarification_detected,
            "patient_friendly_detected": patient_friendly_detected,
            "no_evidence_answer_detected": no_evidence_answer_detected,
            "safety_score": round(safety_score, 4) if safety_score is not None else None,
            "tone_score": tone_score,
            "grounding_violation_detected": grounding_violation_detected,
        }

    def evaluate_sample(self, sample: QAEvalSample, answer_text: str | None = None) -> RetrievalEvalResult:
        search_query = sample.search_query or sample.question
        route_state = {
            "messages": _messages_from_transcript(sample),
            "conversation_summary": sample.conversation_summary,
            "pending_action_type": "",
            "pending_candidates": [],
            "pending_clarification": "",
            "clarification_target": "",
            "appointment_context": {},
            "recommended_department": "",
            "topic_focus": "",
        }
        # analyze_turn now defers to the LLM turn planner (which can't run
        # offline), so route metrics use the L1/L2 classifier directly - the
        # same classifier the planner consults for L1 hints.
        _route_query = ""
        _route_msgs = route_state.get("messages") or []
        if _route_msgs:
            _route_query = str(getattr(_route_msgs[-1], "content", "") or "")
        route_result = classify_query_offline(
            _route_query,
            conversation_summary=route_state.get("conversation_summary", ""),
            topic_focus=route_state.get("topic_focus", ""),
        )

        import time as _time
        t0 = _time.perf_counter()
        # Use pipeline_config-aware search if config is provided
        if self.pipeline_config is not None:
            docs = self.tool_factory.search_documents_with_config(
                search_query,
                limit=self.limit,
                score_threshold=self.score_threshold,
                pipeline_config=self.pipeline_config,
            )
        else:
            docs = self.tool_factory.search_documents(
                search_query,
                limit=self.limit,
                score_threshold=self.score_threshold,
            )
        retrieval_latency_ms = (_time.perf_counter() - t0) * 1000

        # Extract per-component latency from doc metadata
        vector_latency = 0.0
        keyword_latency = 0.0
        rerank_latency = 0.0
        for doc in docs:
            meta = doc.metadata or {}
            if meta.get("_vector_latency_ms"):
                vector_latency = max(vector_latency, float(meta["_vector_latency_ms"]))
            if meta.get("_keyword_latency_ms"):
                keyword_latency = max(keyword_latency, float(meta["_keyword_latency_ms"]))
            if meta.get("_rerank_latency_ms"):
                rerank_latency = max(rerank_latency, float(meta["_rerank_latency_ms"]))
        preferred_layers = self.tool_factory.preferred_source_layers(search_query)
        confidence_bucket = _confidence_bucket_from_docs(docs)
        retrieved_sources = [str((doc.metadata or {}).get("source", "")) for doc in docs]
        retrieved_source_types = [str((doc.metadata or {}).get("source_type", "")).strip().lower() for doc in docs]
        snippet_text = "\n".join(doc.page_content for doc in docs)
        no_evidence_detected = not docs
        matched_retrieval_keywords, missing_retrieval_keywords = _keyword_matches(
            snippet_text,
            sample.expected_retrieval_keywords,
        )

        top_source_type = retrieved_source_types[0] if retrieved_source_types else ""
        top_source = retrieved_sources[0] if retrieved_sources else ""
        source_type_hit = not sample.expected_source_type
        source_type_value = 1.0
        if sample.expected_source_type:
            if top_source_type == sample.expected_source_type:
                source_type_hit = True
                source_type_value = 1.0
            elif sample.expected_source_type in retrieved_source_types:
                source_type_hit = True
                source_type_value = 0.55
            else:
                source_type_value = 0.0

        source_contains_hit = not sample.expected_source_contains
        source_contains_value = 1.0
        if sample.expected_source_contains:
            normalized_sources = [_normalize_text(source) for source in retrieved_sources]
            source_matches = 0
            for expected in sample.expected_source_contains:
                expected_normalized = _normalize_text(expected)
                if any(expected_normalized in source for source in normalized_sources):
                    source_matches += 1
            source_contains_hit = source_matches > 0
            source_contains_value = source_matches / len(sample.expected_source_contains)

        retrieval_components = []
        if sample.expected_no_evidence:
            retrieval_components.append((0.5, 1.0 if no_evidence_detected else 0.0))
        if sample.expected_source_type:
            retrieval_components.append((0.45, source_type_value))
        if sample.expected_source_contains:
            retrieval_components.append((0.2, source_contains_value))
        if sample.expected_retrieval_keywords:
            keyword_coverage = len(matched_retrieval_keywords) / len(sample.expected_retrieval_keywords)
            retrieval_components.append((0.35, keyword_coverage))
        retrieval_score = _weighted_score(retrieval_components)
        retrieval_relevance_hit = (sample.expected_no_evidence and no_evidence_detected) or (
            (not sample.expected_source_type or source_type_hit)
            and (not sample.expected_retrieval_keywords or bool(matched_retrieval_keywords))
        )
        evidence_sufficient = no_evidence_detected if sample.expected_no_evidence else (
            confidence_bucket in {"high", "medium"} or retrieval_score >= 0.7
        )
        route_hit = not sample.expected_primary_intent or route_result.get("primary_intent") == sample.expected_primary_intent
        secondary_route_hit = not sample.expected_secondary_intent or route_result.get("secondary_intent") == sample.expected_secondary_intent

        answer_metrics = self._score_answer(sample, answer_text)
        answer_score = answer_metrics["answer_score"]
        overall_score = retrieval_score if answer_score is None else round((retrieval_score * 0.7) + (answer_score * 0.3), 4)

        return RetrievalEvalResult(
            sample_id=sample.sample_id,
            question=sample.question,
            category=sample.category,
            difficulty=sample.difficulty,
            transcript_turns=list(sample.transcript_turns),
            route_primary_intent=route_result.get("primary_intent", ""),
            route_secondary_intent=route_result.get("secondary_intent", ""),
            route_decision_source=route_result.get("decision_source", ""),
            route_reason=route_result.get("route_reason", ""),
            route_hit=route_hit,
            secondary_route_hit=secondary_route_hit,
            preferred_source_layers=preferred_layers,
            confidence_bucket=confidence_bucket,
            retrieval_score=retrieval_score,
            answer_score=answer_score,
            overall_score=overall_score,
            retrieval_relevance_hit=retrieval_relevance_hit,
            evidence_sufficient=evidence_sufficient,
            grounding_violation_detected=answer_metrics["grounding_violation_detected"],
            top_source_type=top_source_type,
            top_source=top_source,
            matched_retrieval_keywords=matched_retrieval_keywords,
            missing_retrieval_keywords=missing_retrieval_keywords,
            matched_answer_keywords=answer_metrics["matched_answer_keywords"],
            missing_answer_keywords=answer_metrics["missing_answer_keywords"],
            matched_safety_keywords=answer_metrics["matched_safety_keywords"],
            missing_safety_keywords=answer_metrics["missing_safety_keywords"],
            clarification_detected=answer_metrics["clarification_detected"],
            patient_friendly_detected=answer_metrics["patient_friendly_detected"],
            no_evidence_answer_detected=answer_metrics["no_evidence_answer_detected"],
            safety_score=answer_metrics["safety_score"],
            tone_score=answer_metrics["tone_score"],
            source_type_hit=source_type_hit,
            source_contains_hit=source_contains_hit,
            no_evidence_detected=no_evidence_detected,
            retrieved_sources=retrieved_sources,
            retrieved_source_types=retrieved_source_types,
            answer_text=answer_text or "",
            snippets=[doc.page_content[:220] for doc in docs],
            retrieval_latency_ms=round(retrieval_latency_ms, 1),
            vector_search_latency_ms=round(vector_latency, 1),
            keyword_search_latency_ms=round(keyword_latency, 1),
            rerank_latency_ms=round(rerank_latency, 1),
        )

    def evaluate_samples(
        self,
        samples: Iterable[QAEvalSample],
        answer_provider: Optional[Callable[[QAEvalSample], str | None]] = None,
    ) -> dict:
        results = []
        for sample in samples:
            answer_text = answer_provider(sample) if answer_provider else None
            results.append(self.evaluate_sample(sample, answer_text=answer_text))

        if not results:
            return {"summary": {"sample_count": 0, "avg_retrieval_score": 0.0, "avg_overall_score": 0.0}, "results": []}

        avg_retrieval = round(sum(item.retrieval_score for item in results) / len(results), 4)
        avg_overall = round(sum(item.overall_score for item in results) / len(results), 4)
        answer_scored = [item.answer_score for item in results if item.answer_score is not None]
        safety_scored = [item.safety_score for item in results if item.safety_score is not None]
        tone_scored = [item.tone_score for item in results if item.tone_score is not None]
        no_evidence_items = [item for item in results if item.no_evidence_detected]
        low_scoring_samples = [
            {
                "sample_id": item.sample_id,
                "category": item.category,
                "difficulty": item.difficulty,
                "overall_score": item.overall_score,
                "retrieval_score": item.retrieval_score,
                "answer_score": item.answer_score,
                "top_source_type": item.top_source_type,
            }
            for item in sorted(results, key=lambda entry: entry.overall_score)[:5]
            if item.overall_score < 0.85
        ]
        summary = {
            "sample_count": len(results),
            "avg_retrieval_score": avg_retrieval,
            "avg_overall_score": avg_overall,
            "avg_answer_score": round(sum(answer_scored) / len(answer_scored), 4) if answer_scored else None,
            "avg_safety_score": round(sum(safety_scored) / len(safety_scored), 4) if safety_scored else None,
            "avg_tone_score": round(sum(tone_scored) / len(tone_scored), 4) if tone_scored else None,
            "pass_rate_085": round(sum(1 for item in results if item.overall_score >= 0.85) / len(results), 4),
            "clarification_rate": round(sum(1 for item in results if item.clarification_detected) / len(results), 4),
            "no_evidence_rate": round(len(no_evidence_items) / len(results), 4),
            "source_type_hit_rate": round(sum(1 for item in results if item.source_type_hit) / len(results), 4),
            "retrieval_relevance_hit_rate": round(sum(1 for item in results if item.retrieval_relevance_hit) / len(results), 4),
            "evidence_sufficiency_pass_rate": round(sum(1 for item in results if item.evidence_sufficient) / len(results), 4),
            "grounding_violation_rate": round(sum(1 for item in results if item.grounding_violation_detected) / len(results), 4),
            "route_hit_rate": round(sum(1 for item in results if item.route_hit) / len(results), 4),
            "secondary_route_hit_rate": round(sum(1 for item in results if item.secondary_route_hit) / len(results), 4),
            "compound_request_handling_rate": round(
                sum(1 for item in results if item.secondary_route_hit and item.route_secondary_intent) /
                max(sum(1 for item in results if item.route_secondary_intent or "compound_request" in item.category), 1),
                4,
            ),
            "patient_friendly_rate": round(sum(1 for item in results if item.patient_friendly_detected) / len(results), 4),
            "no_evidence_answer_rate": round(sum(1 for item in results if item.no_evidence_answer_detected) / len(results), 4),
            "confidence_distribution": {
                bucket: sum(1 for item in results if item.confidence_bucket == bucket)
                for bucket in ("high", "medium", "low", "no_evidence")
            },
            "by_category": _group_summary(results, "category"),
            "by_difficulty": _group_summary(results, "difficulty"),
            "by_top_source_type": _group_summary(results, "top_source_type"),
            "low_scoring_samples": low_scoring_samples,
        }
        return {"summary": summary, "results": [item.to_dict() for item in results]}
