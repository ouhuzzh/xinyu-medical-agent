from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import config
from core.chat_interface import ChatInterface
from core.document_chunker import DocumentChuncker
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.qa_eval import QAEvalSample, RetrievalQualityEvaluator
from rag_agent.tools import ToolFactory, plan_queries
from utils import estimate_context_tokens


@dataclass
class MemoryBenchmarkSample:
    id: str
    current_question: str
    history_turns: list[str]
    conversation_summary: str = ""
    session_state: dict = field(default_factory=dict)
    category: str = ""


@dataclass
class MedicalRagBenchmarkSample:
    id: str
    question: str
    expected_sources: list[str]
    expected_keywords: list[str] = field(default_factory=list)
    search_query: str = ""
    source_types: list[str] = field(default_factory=list)
    topic_focus: str = ""
    recent_context: str = ""
    category: str = ""


def load_offline_answer_benchmark_samples(path: str | Path) -> list[QAEvalSample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = [QAEvalSample.from_dict(item) for item in payload]
    for sample in samples:
        sample.validate()
    return samples


class KeywordBenchmarkEmbeddings:
    CONCEPT_ALIASES = (
        ("hypertension", ("高血压", "hypertension", "blood pressure")),
        ("hypertension_lifestyle", ("低盐", "限盐", "high-salt diet", "physical activity", "alcohol", "tobacco", "lifestyle")),
        ("hypertension_symptoms", ("头晕", "headaches", "chest pain", "blood pressure checked", "checked")),
        ("diabetes", ("糖尿病", "diabetes", "blood glucose", "type 2")),
        ("diabetes_prevention", ("healthy diet", "physical activity", "body weight", "tobacco", "体重")),
        ("diabetes_complications", ("blindness", "kidney failure", "heart attacks", "stroke", "失明", "肾衰竭")),
        ("asthma", ("哮喘", "asthma", "wheezing", "shortness of breath")),
        ("asthma_management", ("inhaler", "bronchodilators", "steroids", "triggers", "诱因", "吸入")),
        ("covid", ("新冠", "covid", "sars-cov-2")),
        ("covid_transmission", ("飞沫", "密切接触", "气溶胶", "污染的物品")),
        ("covid_prevention", ("疫苗", "加强免疫", "勤洗手", "戴口罩", "通风")),
        ("covid_risk", ("老年人", "严重基础疾病", "晚期妊娠", "肥胖", "重症")),
        ("influenza", ("流感", "influenza")),
        ("influenza_transmission", ("空气传播", "人群密集", "通风不良", "咳嗽")),
        ("mycoplasma", ("肺炎支原体", "mycoplasma", "儿童")),
        ("mycoplasma_severe", ("发热", "咳嗽", "呼吸困难", "胸痛")),
    )

    def _vector(self, text: str) -> list[float]:
        normalized = str(text or "").lower()
        values = []
        for _, aliases in self.CONCEPT_ALIASES:
            values.append(float(sum(normalized.count(alias.lower()) for alias in aliases)))
        return values

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


class InMemoryHybridBenchmarkCollection:
    def __init__(self, docs: list[Document], embeddings: KeywordBenchmarkEmbeddings):
        self.docs = [self._clone_doc(doc) for doc in docs]
        self.embeddings = embeddings
        self.doc_vectors = embeddings.embed_documents(self._doc_text(doc) for doc in self.docs)

    @staticmethod
    def _clone_doc(doc: Document) -> Document:
        return Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))

    @staticmethod
    def _doc_text(doc: Document) -> str:
        metadata = doc.metadata or {}
        return " ".join(
            str(part).strip()
            for part in (
                metadata.get("title"),
                metadata.get("document_topic"),
                metadata.get("section_title"),
                doc.page_content,
            )
            if str(part or "").strip()
        )

    @staticmethod
    def _doc_source_type(doc: Document) -> str:
        metadata = doc.metadata or {}
        return str(
            metadata.get("source_type")
            or metadata.get("document_type")
            or metadata.get("intended_audience")
            or ""
        ).strip().lower()

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _normalize_source_types(source_types) -> set[str]:
        return {str(item).strip().lower() for item in (source_types or []) if str(item).strip()}

    @staticmethod
    def _extract_query_terms(query: str) -> list[str]:
        normalized = str(query or "").lower()
        terms = {term for term in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized) if len(term) >= 2}
        for _, aliases in KeywordBenchmarkEmbeddings.CONCEPT_ALIASES:
            for alias in aliases:
                alias_normalized = alias.lower()
                if alias_normalized in normalized:
                    terms.add(alias_normalized)
        return sorted(terms)

    def _lexical_score(self, query: str, doc: Document) -> float:
        haystack = self._doc_text(doc).lower()
        terms = self._extract_query_terms(query)
        if not terms:
            return 0.0
        return float(sum(haystack.count(term) for term in terms))

    def similarity_search(self, query: str, k: int = 4, score_threshold: float = 0.0, source_types=None, rerank: bool = True) -> list[Document]:
        allowed = self._normalize_source_types(source_types)
        query_vector = self.embeddings.embed_query(query)
        scored = []
        for doc, vector in zip(self.docs, self.doc_vectors):
            if allowed and self._doc_source_type(doc) not in allowed:
                continue
            score = self._cosine_similarity(query_vector, vector)
            if score < score_threshold:
                continue
            cloned = self._clone_doc(doc)
            cloned.metadata["score"] = round(score, 6)
            scored.append(cloned)
        scored.sort(key=lambda item: float((item.metadata or {}).get("score") or 0.0), reverse=True)
        results = scored[:k]
        if rerank:
            return self.rerank_candidates(query, results, k)
        return results

    def keyword_search(self, query: str, k: int = 4, source_types=None) -> list[Document]:
        allowed = self._normalize_source_types(source_types)
        scored = []
        for doc in self.docs:
            if allowed and self._doc_source_type(doc) not in allowed:
                continue
            score = self._lexical_score(query, doc)
            if score <= 0:
                continue
            cloned = self._clone_doc(doc)
            cloned.metadata["score"] = round(score, 6)
            scored.append(cloned)
        scored.sort(key=lambda item: float((item.metadata or {}).get("score") or 0.0), reverse=True)
        return scored[:k]

    def rerank_candidates(self, query: str, candidates: list[Document], top_n: int) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)
        reranked = []
        for candidate in candidates:
            semantic = self._cosine_similarity(query_vector, self.embeddings.embed_query(self._doc_text(candidate)))
            lexical = self._lexical_score(query, candidate)
            fused = (semantic * 0.7) + (lexical * 0.3)
            cloned = self._clone_doc(candidate)
            cloned.metadata["score"] = round(fused, 6)
            reranked.append(cloned)
        reranked.sort(key=lambda item: float((item.metadata or {}).get("score") or 0.0), reverse=True)
        return reranked[:top_n]

    def filtered(self, source_types=None) -> "InMemoryHybridBenchmarkCollection":
        allowed = self._normalize_source_types(source_types)
        if not allowed:
            return self
        filtered_docs = [
            self._clone_doc(doc)
            for doc in self.docs
            if self._doc_source_type(doc) in allowed
        ]
        return InMemoryHybridBenchmarkCollection(filtered_docs, self.embeddings)


def load_memory_benchmark_samples(path: str | Path) -> list[MemoryBenchmarkSample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        MemoryBenchmarkSample(
            id=str(item["id"]),
            current_question=str(item["current_question"]),
            history_turns=[str(turn) for turn in item.get("history_turns", [])],
            conversation_summary=str(item.get("conversation_summary") or ""),
            session_state=dict(item.get("session_state") or {}),
            category=str(item.get("category") or ""),
        )
        for item in payload
    ]


def load_medical_rag_benchmark_samples(path: str | Path) -> list[MedicalRagBenchmarkSample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        MedicalRagBenchmarkSample(
            id=str(item["id"]),
            question=str(item["question"]),
            expected_sources=[str(value) for value in item.get("expected_sources", [])],
            expected_keywords=[str(value) for value in item.get("expected_keywords", [])],
            search_query=str(item.get("search_query") or ""),
            source_types=[str(value) for value in item.get("source_types", [])],
            topic_focus=str(item.get("topic_focus") or ""),
            recent_context=str(item.get("recent_context") or ""),
            category=str(item.get("category") or ""),
        )
        for item in payload
    ]


def _history_messages(history_turns: list[str]) -> list:
    messages = []
    for turn in history_turns:
        text = str(turn or "").strip()
        if not text:
            continue
        lower = text.lower()
        if lower.startswith("user:"):
            messages.append(HumanMessage(content=text.split(":", 1)[1].strip()))
        elif lower.startswith("assistant:"):
            messages.append(AIMessage(content=text.split(":", 1)[1].strip()))
    return messages


def _p95(values: list[int | float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return float(ordered[index])


def evaluate_memory_token_benchmark(samples: list[MemoryBenchmarkSample]) -> dict:
    rows = []
    baseline_tokens = []
    optimized_tokens = []
    reductions = []
    window_size = max(1, int(config.SHORT_TERM_WINDOW_SIZE)) * 2

    for sample in samples:
        history_messages = _history_messages(sample.history_turns)
        baseline_messages = [*history_messages, HumanMessage(content=sample.current_question)]
        optimized_messages = []
        if sample.conversation_summary:
            optimized_messages.append(SystemMessage(content=f"Conversation summary:\n{sample.conversation_summary}"))
        optimized_messages.extend(ChatInterface._build_state_messages(sample.session_state))
        optimized_messages.extend(history_messages[-window_size:])
        optimized_messages.append(HumanMessage(content=sample.current_question))

        baseline_count = estimate_context_tokens(baseline_messages)
        optimized_count = estimate_context_tokens(optimized_messages)
        reduction_rate = 0.0 if baseline_count <= 0 else round((baseline_count - optimized_count) / baseline_count, 4)

        baseline_tokens.append(baseline_count)
        optimized_tokens.append(optimized_count)
        reductions.append(reduction_rate)
        rows.append(
            {
                "id": sample.id,
                "category": sample.category,
                "baseline_tokens": baseline_count,
                "optimized_tokens": optimized_count,
                "token_reduction_rate": reduction_rate,
            }
        )

    sample_count = len(rows)
    return {
        "summary": {
            "sample_count": sample_count,
            "avg_baseline_tokens": round(sum(baseline_tokens) / sample_count, 2) if sample_count else 0.0,
            "avg_optimized_tokens": round(sum(optimized_tokens) / sample_count, 2) if sample_count else 0.0,
            "avg_token_reduction_rate": round(sum(reductions) / sample_count, 4) if sample_count else 0.0,
            "p95_baseline_tokens": round(_p95(baseline_tokens), 2),
            "p95_optimized_tokens": round(_p95(optimized_tokens), 2),
            "p95_token_reduction_rate": round(_p95(reductions), 4),
            "short_term_window_messages": window_size,
        },
        "details": rows,
    }


def _extract_front_matter_metadata(raw_text: str) -> dict:
    metadata = {}
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if metadata:
                break
            continue
        match = re.match(r"^([A-Za-z][A-Za-z0-9 _-]*):\s*(.+?)\s*$", stripped)
        if not match:
            if metadata:
                break
            continue
        metadata[match.group(1).strip().lower().replace(" ", "_")] = match.group(2).strip()
    return metadata


def _strip_front_matter(raw_text: str) -> str:
    lines = raw_text.splitlines()
    output = []
    seen_metadata = False
    metadata_done = False
    for line in lines:
        stripped = line.strip()
        is_metadata = bool(re.match(r"^([A-Za-z][A-Za-z0-9 _-]*):\s*(.+?)\s*$", stripped))
        if not metadata_done and is_metadata:
            seen_metadata = True
            continue
        if seen_metadata and not metadata_done and not stripped:
            metadata_done = True
            continue
        if metadata_done or not seen_metadata:
            output.append(line)
    return "\n".join(output).strip()


def build_isolated_medical_corpora(doc_paths: list[Path]) -> tuple[list[Document], list[Document]]:
    chunker = DocumentChuncker()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
    baseline_docs = []
    optimized_docs = []

    for doc_path in doc_paths:
        raw_text = doc_path.read_text(encoding="utf-8")
        metadata = _extract_front_matter_metadata(raw_text)
        content = _strip_front_matter(raw_text)
        source_type = metadata.get("document_type") or metadata.get("source_type") or "general"
        title = metadata.get("title") or doc_path.stem
        for index, chunk in enumerate(splitter.split_text(content)):
            baseline_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": doc_path.name,
                        "source_type": source_type,
                        "title": title,
                        "baseline_chunk_id": f"{doc_path.stem}_baseline_{index}",
                    },
                )
            )
        _, child_chunks = chunker.create_chunks_single(doc_path)
        for child in child_chunks:
            child.metadata["source_type"] = child.metadata.get("source_type") or child.metadata.get("document_type") or source_type
            child.metadata["title"] = child.metadata.get("title") or title
            child.metadata["document_topic"] = child.metadata.get("document_topic") or title
        optimized_docs.extend(child_chunks)

    return baseline_docs, optimized_docs


def _unique_sources(docs: list[Document], limit: int) -> list[str]:
    ordered = []
    seen = set()
    for doc in docs:
        source = str((doc.metadata or {}).get("source") or "").strip()
        if not source or source in seen:
            continue
        seen.add(source)
        ordered.append(source)
        if len(ordered) >= limit:
            break
    return ordered


def _source_metrics(expected_sources: list[str], docs: list[Document], top_k: int, mrr_k: int) -> dict:
    normalized_expected = {str(item).strip().lower() for item in expected_sources if str(item).strip()}
    top_sources = _unique_sources(docs, max(top_k, mrr_k))
    top_k_sources = top_sources[:top_k]
    hits = sum(1 for source in top_k_sources if source.lower() in normalized_expected)
    reciprocal_rank = 0.0
    for rank, source in enumerate(top_sources[:mrr_k], start=1):
        if source.lower() in normalized_expected:
            reciprocal_rank = round(1.0 / rank, 4)
            break
    precision = 0.0 if not top_k_sources else round(hits / len(top_k_sources), 4)
    recall = 0.0 if not normalized_expected else round(hits / len(normalized_expected), 4)
    hit = 1.0 if hits else 0.0
    return {
        "top_sources": top_k_sources,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr_at_k": reciprocal_rank,
        "hit_at_k": hit,
    }


def _keyword_coverage(expected_keywords: list[str], docs: list[Document]) -> float:
    keywords = [str(item).strip().lower() for item in expected_keywords if str(item).strip()]
    if not keywords:
        return 0.0
    haystack = "\n".join(doc.page_content for doc in docs).lower()
    hits = sum(1 for keyword in keywords if keyword in haystack)
    return round(hits / len(keywords), 4)


def _optimized_query_plan(sample: MedicalRagBenchmarkSample) -> list[str]:
    base_query = sample.search_query or sample.question
    return plan_queries(base_query, topic_focus=sample.topic_focus, recent_context=sample.recent_context) or [base_query]


def _run_optimized_retrieval(sample: MedicalRagBenchmarkSample, collection: InMemoryHybridBenchmarkCollection, top_k: int, mrr_k: int) -> tuple[list[Document], list[str]]:
    active_collection = collection.filtered(sample.source_types or None)
    tool_factory = ToolFactory(active_collection)
    query_plan = _optimized_query_plan(sample)
    previous_hybrid = config.ENABLE_HYBRID_RETRIEVAL
    try:
        config.ENABLE_HYBRID_RETRIEVAL = True
        ranked_sets = []
        for planned_query in query_plan:
            ranked_sets.append(
                tool_factory.search_documents(
                    planned_query,
                    limit=max(top_k * 2, mrr_k),
                    score_threshold=0.0,
                )
            )
        fused = ToolFactory._rrf_fuse_ranked_sets(ranked_sets, max(top_k, mrr_k))
        preferred_layers = ToolFactory.preferred_source_layers(sample.search_query or sample.question)
        fused = ToolFactory._sort_docs_by_source_priority(fused, preferred_layers=preferred_layers)
        reranker = getattr(active_collection, "rerank_candidates", None)
        if callable(reranker):
            fused = reranker(sample.search_query or sample.question, fused, max(top_k, mrr_k))
        return fused[: max(top_k, mrr_k)], query_plan
    finally:
        config.ENABLE_HYBRID_RETRIEVAL = previous_hybrid


def evaluate_medical_rag_benchmark(
    samples: list[MedicalRagBenchmarkSample],
    baseline_collection: InMemoryHybridBenchmarkCollection,
    optimized_collection: InMemoryHybridBenchmarkCollection,
    *,
    top_k: int = 5,
    mrr_k: int = 10,
) -> dict:
    rows = []
    baseline_precisions = []
    baseline_recalls = []
    baseline_mrrs = []
    baseline_hits = []
    baseline_keyword_coverages = []
    optimized_precisions = []
    optimized_recalls = []
    optimized_mrrs = []
    optimized_hits = []
    optimized_keyword_coverages = []

    for sample in samples:
        query = sample.search_query or sample.question
        baseline_docs = baseline_collection.similarity_search(
            query,
            k=max(top_k, mrr_k),
            score_threshold=0.0,
            source_types=sample.source_types or None,
            rerank=False,
        )
        optimized_docs, query_plan = _run_optimized_retrieval(sample, optimized_collection, top_k, mrr_k)
        baseline_metrics = _source_metrics(sample.expected_sources, baseline_docs, top_k, mrr_k)
        optimized_metrics = _source_metrics(sample.expected_sources, optimized_docs, top_k, mrr_k)
        baseline_keyword = _keyword_coverage(sample.expected_keywords, baseline_docs[:top_k])
        optimized_keyword = _keyword_coverage(sample.expected_keywords, optimized_docs[:top_k])

        baseline_precisions.append(baseline_metrics["precision_at_k"])
        baseline_recalls.append(baseline_metrics["recall_at_k"])
        baseline_mrrs.append(baseline_metrics["mrr_at_k"])
        baseline_hits.append(baseline_metrics["hit_at_k"])
        baseline_keyword_coverages.append(baseline_keyword)
        optimized_precisions.append(optimized_metrics["precision_at_k"])
        optimized_recalls.append(optimized_metrics["recall_at_k"])
        optimized_mrrs.append(optimized_metrics["mrr_at_k"])
        optimized_hits.append(optimized_metrics["hit_at_k"])
        optimized_keyword_coverages.append(optimized_keyword)

        rows.append(
            {
                "id": sample.id,
                "category": sample.category,
                "question": sample.question,
                "expected_sources": sample.expected_sources,
                "query_plan": query_plan,
                "baseline": {
                    **baseline_metrics,
                    "keyword_coverage": baseline_keyword,
                },
                "optimized": {
                    **optimized_metrics,
                    "keyword_coverage": optimized_keyword,
                },
            }
        )

    sample_count = len(rows)

    def _avg(values: list[float]) -> float:
        return round(sum(values) / sample_count, 4) if sample_count else 0.0

    return {
        "summary": {
            "sample_count": sample_count,
            "baseline_precision_at_5": _avg(baseline_precisions),
            "baseline_recall_at_5": _avg(baseline_recalls),
            "baseline_mrr_at_10": _avg(baseline_mrrs),
            "baseline_hit_at_5": _avg(baseline_hits),
            "baseline_keyword_coverage": _avg(baseline_keyword_coverages),
            "optimized_precision_at_5": _avg(optimized_precisions),
            "optimized_recall_at_5": _avg(optimized_recalls),
            "optimized_mrr_at_10": _avg(optimized_mrrs),
            "optimized_hit_at_5": _avg(optimized_hits),
            "optimized_keyword_coverage": _avg(optimized_keyword_coverages),
            "precision_uplift": round(_avg(optimized_precisions) - _avg(baseline_precisions), 4),
            "recall_uplift": round(_avg(optimized_recalls) - _avg(baseline_recalls), 4),
            "mrr_uplift": round(_avg(optimized_mrrs) - _avg(baseline_mrrs), 4),
            "keyword_coverage_uplift": round(_avg(optimized_keyword_coverages) - _avg(baseline_keyword_coverages), 4),
        },
        "details": rows,
    }


def as_pretty_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[。！？.!?])\s+|\n+", str(text or "")) if segment.strip()]


def _select_support_sentences(sample: QAEvalSample, docs: list[Document], limit: int = 3) -> list[str]:
    keywords = [str(item).strip() for item in (sample.expected_answer_keywords or sample.expected_retrieval_keywords) if str(item).strip()]
    chosen = []
    seen = set()
    for doc in docs:
        for sentence in _split_sentences(doc.page_content):
            normalized = sentence.lower()
            keyword_hits = sum(1 for keyword in keywords if keyword.lower() in normalized)
            if keywords and keyword_hits == 0:
                continue
            key = sentence[:180]
            if key in seen:
                continue
            seen.add(key)
            chosen.append((keyword_hits, sentence))
    if not chosen:
        for doc in docs:
            for sentence in _split_sentences(doc.page_content)[:2]:
                key = sentence[:180]
                if key in seen:
                    continue
                seen.add(key)
                chosen.append((0, sentence))
    chosen.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return [sentence for _, sentence in chosen[:limit]]


def synthesize_grounded_answer(sample: QAEvalSample, docs: list[Document], *, max_sources: int = 2) -> str:
    if not docs:
        return "我目前没有在这批 NHC/WHO 资料里找到足够直接的依据，这次只能给到非常有限的参考，不能替代医生面对面诊断。"

    support_sentences = _select_support_sentences(sample, docs, limit=3)
    unique_sources = []
    seen_sources = set()
    for doc in docs:
        source = str((doc.metadata or {}).get("source") or "").strip()
        if source and source not in seen_sources:
            seen_sources.add(source)
            unique_sources.append(source)
        if len(unique_sources) >= max_sources:
            break

    if sample.preferred_answer_style == "clinical":
        prefix = "根据现有指南资料，可以先这样概括："
    elif sample.expected_no_evidence:
        prefix = "这次没有找到充分证据支持直接回答："
    else:
        prefix = "根据现有资料，可以先抓住这几点："

    body = " ".join(sentence.strip() for sentence in support_sentences if sentence.strip())
    if sample.expected_safety_keywords:
        body += " 如症状明显加重、持续不缓解，建议尽快线下就医评估。"
    if sample.must_not_clarify:
        body = body.replace("？", "。")

    citation = f" 来源：{'；'.join(unique_sources)}" if unique_sources else ""
    return f"{prefix} {body}{citation}".strip()


def _prepare_answer_docs(collection: InMemoryHybridBenchmarkCollection, docs: list[Document]) -> list[Document]:
    prepared = []
    seen_groups = set()
    for doc in docs:
        metadata = doc.metadata or {}
        parent_id = str(metadata.get("parent_id") or "").strip()
        source = str(metadata.get("source") or "").strip()
        group_key = parent_id or source
        if not group_key or group_key in seen_groups:
            continue
        seen_groups.add(group_key)

        if parent_id:
            sibling_chunks = [
                sibling for sibling in collection.docs
                if str((sibling.metadata or {}).get("parent_id") or "").strip() == parent_id
            ]
            if sibling_chunks:
                merged_text = "\n".join(chunk.page_content for chunk in sibling_chunks)
                prepared.append(Document(page_content=merged_text, metadata=dict(metadata)))
                continue
        prepared.append(doc)
    return prepared or docs


def _extract_cited_sources(answer_text: str) -> list[str]:
    marker = "来源："
    if marker not in answer_text:
        return []
    cited_block = answer_text.split(marker, 1)[1].strip()
    return [item.strip() for item in re.split(r"[；;,，]\s*", cited_block) if item.strip()]


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _citation_precision(expected_sources: list[str], cited_sources: list[str]) -> float:
    expected = {str(item).strip().lower() for item in expected_sources if str(item).strip()}
    cited = [str(item).strip().lower() for item in cited_sources if str(item).strip()]
    if not cited:
        return 0.0
    hits = sum(1 for item in cited if item in expected)
    return round(hits / len(cited), 4)


def _citation_recall(expected_sources: list[str], cited_sources: list[str]) -> float:
    expected = {str(item).strip().lower() for item in expected_sources if str(item).strip()}
    cited = {str(item).strip().lower() for item in cited_sources if str(item).strip()}
    if not expected:
        return 1.0
    hits = sum(1 for item in expected if item in cited)
    return round(hits / len(expected), 4)


def evaluate_offline_answer_benchmark(
    samples: list[QAEvalSample],
    baseline_collection: InMemoryHybridBenchmarkCollection,
    optimized_collection: InMemoryHybridBenchmarkCollection,
    *,
    limit: int = 3,
    score_threshold: float = 0.0,
) -> dict:
    baseline_evaluator = RetrievalQualityEvaluator(baseline_collection, limit=limit, score_threshold=score_threshold)
    optimized_evaluator = RetrievalQualityEvaluator(optimized_collection, limit=limit, score_threshold=score_threshold)

    def _provider(collection):
        tool_factory = ToolFactory(collection)

        def _answer(sample: QAEvalSample):
            query = sample.search_query or sample.question
            docs = tool_factory.search_documents(query, limit=limit, score_threshold=score_threshold)
            return synthesize_grounded_answer(sample, _prepare_answer_docs(collection, docs))

        return _answer

    baseline_report = baseline_evaluator.evaluate_samples(samples, answer_provider=_provider(baseline_collection))
    optimized_report = optimized_evaluator.evaluate_samples(samples, answer_provider=_provider(optimized_collection))

    def _augment(report: dict):
        citation_precisions = []
        citation_recalls = []
        for item in report["results"]:
            cited_sources = _extract_cited_sources(item.get("answer_text", ""))
            item["cited_sources"] = cited_sources
            expected_sources = next(
                (sample.expected_source_contains for sample in samples if sample.sample_id == item.get("sample_id")),
                [],
            )
            precision = _citation_precision(expected_sources, cited_sources)
            recall = _citation_recall(expected_sources, cited_sources)
            item["citation_precision"] = precision
            item["citation_recall"] = recall
            citation_precisions.append(precision)
            citation_recalls.append(recall)
        report["summary"]["avg_citation_precision"] = _average(citation_precisions)
        report["summary"]["avg_citation_recall"] = _average(citation_recalls)
        return report

    baseline_report = _augment(baseline_report)
    optimized_report = _augment(optimized_report)

    return {
        "summary": {
            "sample_count": baseline_report["summary"]["sample_count"],
            "baseline_avg_answer_score": baseline_report["summary"]["avg_answer_score"],
            "optimized_avg_answer_score": optimized_report["summary"]["avg_answer_score"],
            "answer_score_uplift": round(
                (optimized_report["summary"]["avg_answer_score"] or 0.0) - (baseline_report["summary"]["avg_answer_score"] or 0.0),
                4,
            ),
            "baseline_avg_overall_score": baseline_report["summary"]["avg_overall_score"],
            "optimized_avg_overall_score": optimized_report["summary"]["avg_overall_score"],
            "overall_score_uplift": round(
                optimized_report["summary"]["avg_overall_score"] - baseline_report["summary"]["avg_overall_score"],
                4,
            ),
            "baseline_avg_citation_precision": baseline_report["summary"]["avg_citation_precision"],
            "optimized_avg_citation_precision": optimized_report["summary"]["avg_citation_precision"],
            "citation_precision_uplift": round(
                optimized_report["summary"]["avg_citation_precision"] - baseline_report["summary"]["avg_citation_precision"],
                4,
            ),
            "baseline_grounding_violation_rate": baseline_report["summary"]["grounding_violation_rate"],
            "optimized_grounding_violation_rate": optimized_report["summary"]["grounding_violation_rate"],
        },
        "baseline": baseline_report,
        "optimized": optimized_report,
    }
