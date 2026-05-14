from __future__ import annotations
from contextvars import ContextVar
from typing import List
import logging
import re

import config
from langchain_core.tools import tool
from langchain_core.documents import Document
from db.parent_store_manager import ParentStoreManager


_SOURCE_TYPE_PRIORITY = {
    "patient_education": 0,
    "public_health": 1,
    "clinical_guideline": 2,
    "research_article": 3,
}
_DEFAULT_LAYERED_SOURCE_TYPES = ["patient_education", "public_health", "clinical_guideline"]
_NO_EVIDENCE_RESPONSE = "NO_EVIDENCE: 知识库中暂无相关信息或足够相关证据。若问题属于医学问题，可给出通用医学信息回答，但必须说明这次回答未充分基于知识库检索结果。"
_RRF_K = 60
_RETRIEVAL_CONTEXT = ContextVar("retrieval_context", default={})
logger = logging.getLogger(__name__)
_QUERY_STOPWORDS = {
    "什么", "怎么", "如何", "一下", "一下子", "请问", "这个", "那个", "情况", "问题", "还要", "需要",
    "应该", "一般", "现在", "最近", "一下吧", "一个", "哪些", "可以", "是不是",
}
_QUERY_TYPE_KEYWORDS = {
    "public_health": (
        "预防", "风险", "流行", "发病率", "传播", "疫苗", "筛查", "risk", "prevention",
        "incidence", "prevalence", "outbreak", "vaccine", "screening", "public health",
    ),
    "clinical_guideline": (
        "指南", "诊疗方案", "规范", "标准", "共识", "第十版", "剂量", "用法", "首选药",
        "protocol", "guideline", "criteria", "dose", "dosing", "first-line", "recommendation",
    ),
    "patient_education": (
        "是什么", "怎么办", "会不会", "症状", "表现", "原因", "怎么治疗", "怎么缓解", "严重吗",
        "what is", "symptom", "symptoms", "what should", "how to", "can it", "is it serious",
    ),
}


def set_retrieval_context(*, thread_id: str = "", original_query: str = "", query_plan=None, request_id: str = ""):
    return _RETRIEVAL_CONTEXT.set(
        {
            "thread_id": str(thread_id or "").strip(),
            "original_query": str(original_query or "").strip(),
            "query_plan": list(query_plan or []),
            "request_id": str(request_id or "").strip(),
        }
    )


def reset_retrieval_context(token):
    if token is not None:
        try:
            _RETRIEVAL_CONTEXT.reset(token)
        except ValueError:
            # Starlette/anyio may advance a synchronous SSE iterator in a
            # different execution context from the one that created the token.
            # The request is ending either way, so clear the current context
            # instead of surfacing a false chat failure to the user.
            logger.warning("Retrieval context token was reset from a different context; clearing current context.")
            _RETRIEVAL_CONTEXT.set({})


def get_retrieval_context() -> dict:
    value = _RETRIEVAL_CONTEXT.get()
    return dict(value) if isinstance(value, dict) else {}


def _doc_score(metadata: dict | None) -> float:
    metadata = metadata or {}
    raw = metadata.get("rerank_score")
    if raw in (None, ""):
        raw = metadata.get("score")
    if raw in (None, ""):
        raw = metadata.get("fusion_score")
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _confidence_bucket(results: List[Document]) -> str:
    if not results:
        return "no_evidence"
    top_doc = results[0]
    score = _doc_score(top_doc.metadata)
    if score >= config.RAG_HIGH_CONFIDENCE_SCORE and len(results) >= 2:
        return "high"
    if score >= config.RAG_MEDIUM_CONFIDENCE_SCORE:
        return "medium"
    return "low"


def _append_once(text: str, addition: str) -> str:
    addition = str(addition or "").strip()
    if not addition:
        return text
    if addition in text:
        return text
    return f"{text.rstrip()}\n\n{addition}".strip()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _query_keywords(query: str) -> set[str]:
    normalized = _normalize_text(query)
    tokens = set(re.findall(r"[\u4e00-\u9fff]{1,8}|[a-zA-Z][a-zA-Z0-9_-]+", normalized))
    return {token for token in tokens if token not in _QUERY_STOPWORDS and len(token) > 1}


def _lexical_overlap_score(query: str, doc: Document) -> float:
    query_terms = _query_keywords(query)
    if not query_terms:
        return 0.0
    haystack = _normalize_text(doc.page_content + " " + str(doc.metadata or {}))
    hits = sum(1 for term in query_terms if term in haystack)
    return hits / max(len(query_terms), 1)


def plan_queries(query: str, topic_focus: str = "", recent_context: str = "") -> list[str]:
    base_query = str(query or "").strip()
    if not base_query:
        return []
    planned = [base_query]
    normalized = _normalize_text(base_query)
    if topic_focus and topic_focus.strip() and topic_focus.strip() not in base_query:
        planned.append(f"{topic_focus.strip()} {base_query}".strip())
    if recent_context.strip() and any(token in normalized for token in ("那", "这个", "这种情况", "这会", "还要", "要紧吗")):
        planned.append(f"{recent_context.strip()} {base_query}".strip())
    if any(keyword in normalized for keyword in _QUERY_TYPE_KEYWORDS["clinical_guideline"]):
        planned.append(f"{base_query} 指南 诊疗方案")
    elif any(keyword in normalized for keyword in _QUERY_TYPE_KEYWORDS["public_health"]):
        planned.append(f"{base_query} 预防 风险 传播")
    else:
        planned.append(f"{base_query} 症状 治疗 注意事项")

    deduped = []
    seen = set()
    for item in planned:
        text = re.sub(r"\s+", " ", item).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped[:4]


def grade_documents(query: str, docs: List[Document]) -> List[Document]:
    graded = []
    for doc in docs or []:
        metadata = doc.metadata or {}
        score = _doc_score(metadata)
        overlap = _lexical_overlap_score(query, doc)
        if score >= config.RAG_HIGH_RELEVANCE_SCORE or overlap >= config.RAG_HIGH_LEXICAL_OVERLAP:
            grade = "high"
            keep = True
        elif score >= config.RAG_MEDIUM_RELEVANCE_SCORE or overlap >= config.RAG_MEDIUM_LEXICAL_OVERLAP:
            grade = "medium"
            keep = True
        else:
            grade = "low"
            keep = score >= config.RAG_LOW_RELEVANCE_SCORE and overlap >= config.RAG_LOW_LEXICAL_OVERLAP
        metadata["relevance_grade"] = grade
        metadata["lexical_overlap"] = round(overlap, 4)
        metadata["keep"] = bool(keep)
        doc.metadata = metadata
        if keep:
            graded.append(doc)
    return graded


def check_sufficiency(query: str, docs: List[Document]) -> dict:
    if not docs:
        return {"is_sufficient": False, "reason": "no_relevant_documents", "retry_query": f"{query} 医学资料"}
    top_score = _doc_score(docs[0].metadata)
    high_grade_count = sum(1 for doc in docs if (doc.metadata or {}).get("relevance_grade") == "high")
    if high_grade_count >= 1 and top_score >= config.RAG_DIRECT_EVIDENCE_SCORE:
        return {"is_sufficient": True, "reason": "direct_evidence", "retry_query": ""}
    if len(docs) >= 2 and top_score >= config.RAG_LIMITED_EVIDENCE_SCORE:
        return {"is_sufficient": True, "reason": "limited_but_usable", "retry_query": ""}
    query_terms = list(_query_keywords(query))
    retry_terms = [term for term in query_terms[:4] if len(term) > 1]
    retry_query = " ".join(retry_terms) if retry_terms else f"{query} 医疗 指南"
    return {"is_sufficient": False, "reason": "weak_or_sparse_evidence", "retry_query": retry_query}


def ground_answer(answer: str, docs: List[Document], *, question: str = "", medical_mode: bool = False, high_risk: bool = False) -> dict:
    text = str(answer or "").strip()
    if not docs:
        if not medical_mode:
            return {
                "grounded": True,
                "revised_answer": text,
                "note": "no_evidence_non_medical",
            }
        revised = text
        if "回答模式：" not in revised and "通用医学信息回答" not in revised:
            revised = f"回答模式：通用医学信息回答（本次未充分基于知识库检索结果）\n\n{revised}".strip()
        revised = _append_once(
            revised,
            "提醒：以上内容仅供一般医学信息参考，当前回答未充分基于知识库检索结果，不能替代专业医生面对面诊断。",
        )
        if high_risk:
            revised = _append_once(
                revised,
                "如果症状严重、持续加重，或涉及用药、急症、呼吸困难、胸痛等情况，请尽快线下就医或急诊评估。",
            )
        else:
            revised = _append_once(
                revised,
                "如果症状持续加重，或涉及用药调整、急症判断，请及时就医。",
            )
        return {
            "grounded": False,
            "revised_answer": revised,
            "note": "medical_generic_fallback",
        }
    confidence = _confidence_bucket(docs)
    if confidence in {"no_evidence", "low"}:
        if not medical_mode:
            return {"grounded": True, "revised_answer": text, "note": "low_confidence_non_medical"}
        revised = text
        if "回答模式：" not in revised and "通用医学信息回答" not in revised and "非知识库增强" not in revised:
            revised = f"回答模式：通用医学信息回答（知识库证据有限）\n\n{revised}".strip()
        revised = _append_once(
            revised,
            "提醒：以上内容仅供一般医学信息参考，当前知识库证据有限，不能替代专业医生面对面诊断。",
        )
        if high_risk:
            revised = _append_once(
                revised,
                "如果症状严重、持续加重，或涉及用药、剂量、急症判断，请尽快线下就医或急诊评估。",
            )
        return {"grounded": False, "revised_answer": revised, "note": "low_confidence_guardrail"}
    return {"grounded": True, "revised_answer": text, "note": "grounded"}


class ToolFactory:
    
    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()

    @staticmethod
    def _sort_docs_by_source_priority(results: List[Document], preferred_layers: List[str] | None = None) -> List[Document]:
        layer_priority = {
            str(source_type).strip().lower(): index
            for index, source_type in enumerate(preferred_layers or _DEFAULT_LAYERED_SOURCE_TYPES)
        }

        def sort_key(doc: Document):
            metadata = doc.metadata or {}
            source_type = str(metadata.get("source_type", "")).strip().lower()
            priority = layer_priority.get(source_type)
            if priority is None:
                priority = len(layer_priority) + _SOURCE_TYPE_PRIORITY.get(source_type, 99)
            score = float(metadata.get("fusion_score")) if metadata.get("fusion_score") not in (None, "") else _doc_score(metadata)
            return (priority, -score)

        return sorted(results, key=sort_key)

    @staticmethod
    def _preferred_source_layers(query: str) -> List[str]:
        normalized = (query or "").strip().lower()
        matches = {source_type: 0 for source_type in _DEFAULT_LAYERED_SOURCE_TYPES}
        for source_type, keywords in _QUERY_TYPE_KEYWORDS.items():
            matches[source_type] = sum(1 for keyword in keywords if keyword in normalized)

        if matches["clinical_guideline"] > max(matches["patient_education"], matches["public_health"]):
            return ["clinical_guideline", "patient_education", "public_health"]
        if matches["public_health"] > max(matches["patient_education"], matches["clinical_guideline"]):
            return ["public_health", "patient_education", "clinical_guideline"]
        return list(_DEFAULT_LAYERED_SOURCE_TYPES)

    @classmethod
    def preferred_source_layers(cls, query: str) -> List[str]:
        return cls._preferred_source_layers(query)

    @staticmethod
    def _dedupe_docs(results: List[Document]) -> List[Document]:
        deduped = []
        seen = set()
        for doc in results:
            metadata = doc.metadata or {}
            key = (
                metadata.get("parent_id"),
                metadata.get("source"),
                doc.page_content.strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
        return deduped

    @staticmethod
    def _doc_key(doc: Document):
        metadata = doc.metadata or {}
        return (
            metadata.get("parent_id"),
            metadata.get("source"),
            doc.page_content.strip(),
        )

    @staticmethod
    def _rrf_fuse(vector_results: List[Document], keyword_results: List[Document], limit: int) -> List[Document]:
        fused_scores = {}
        chosen_docs = {}
        for result_set in (vector_results, keyword_results):
            for rank, doc in enumerate(result_set, start=1):
                key = ToolFactory._doc_key(doc)
                fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (_RRF_K + rank))
                chosen_docs.setdefault(key, doc)

        fused_docs = []
        for key, doc in chosen_docs.items():
            doc.metadata["fusion_score"] = round(fused_scores[key], 6)
            if not doc.metadata.get("score"):
                doc.metadata["score"] = doc.metadata["fusion_score"]
            fused_docs.append(doc)
        fused_docs.sort(key=lambda item: float((item.metadata or {}).get("fusion_score") or 0.0), reverse=True)
        return fused_docs[:limit]

    @classmethod
    def _rrf_fuse_ranked_sets(cls, ranked_sets: List[List[Document]], limit: int) -> List[Document]:
        fused_scores = {}
        chosen_docs = {}
        best_raw_scores = {}
        for result_set in ranked_sets:
            for rank, doc in enumerate(result_set, start=1):
                key = cls._doc_key(doc)
                fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (_RRF_K + rank))
                current_raw = _doc_score(doc.metadata)
                if key not in chosen_docs or current_raw >= best_raw_scores.get(key, 0.0):
                    chosen_docs[key] = doc
                    best_raw_scores[key] = current_raw

        fused_docs = []
        for key, doc in chosen_docs.items():
            doc.metadata["fusion_score"] = round(fused_scores[key], 6)
            if not doc.metadata.get("score"):
                doc.metadata["score"] = round(best_raw_scores.get(key, 0.0), 6)
            fused_docs.append(doc)
        fused_docs.sort(key=lambda item: float((item.metadata or {}).get("fusion_score") or 0.0), reverse=True)
        return fused_docs[:limit]

    @staticmethod
    def _normalize_query_plan(query: str, query_plan) -> List[str]:
        deduped = []
        seen = set()
        for item in [query, *(query_plan or [])]:
            text = re.sub(r"\s+", " ", str(item or "").strip())
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return deduped[:4]

    def _similarity_search_with_optional_filters(self, query: str, limit: int, score_threshold: float, source_types=None, rerank=True) -> List[Document]:
        try:
            return self.collection.similarity_search(
                query,
                k=limit,
                score_threshold=score_threshold,
                source_types=source_types,
                rerank=rerank,
            )
        except TypeError:
            results = self.collection.similarity_search(query, k=limit, score_threshold=score_threshold)
            if source_types:
                allowed = {str(item).strip().lower() for item in source_types}
                results = [
                    doc for doc in results
                    if str((doc.metadata or {}).get("source_type", "")).strip().lower() in allowed
                ]
            return results[:limit]

    def _keyword_search_with_optional_filters(self, query: str, limit: int, source_types=None) -> List[Document]:
        keyword_search = getattr(self.collection, "keyword_search", None)
        if not callable(keyword_search):
            return []
        try:
            return keyword_search(query, k=limit, source_types=source_types)
        except TypeError:
            results = keyword_search(query, k=limit)
            if source_types:
                allowed = {str(item).strip().lower() for item in source_types}
                results = [
                    doc for doc in results
                    if str((doc.metadata or {}).get("source_type", "")).strip().lower() in allowed
                ]
            return results[:limit]

    def _layered_similarity_search(self, query: str, limit: int, score_threshold: float) -> List[Document]:
        per_tier_limit = max(limit, 3)
        layered_results = []
        source_layers = self._preferred_source_layers(query)
        for source_type in source_layers:
            vector_results = self._similarity_search_with_optional_filters(
                query,
                limit=per_tier_limit,
                score_threshold=score_threshold,
                source_types=[source_type],
                rerank=False,
            )
            tier_results = vector_results
            if config.ENABLE_HYBRID_RETRIEVAL:
                keyword_results = self._keyword_search_with_optional_filters(
                    query,
                    limit=per_tier_limit,
                    source_types=[source_type],
                )
                tier_results = self._rrf_fuse(vector_results, keyword_results, per_tier_limit)
            layered_results.extend(tier_results)
            deduped = self._dedupe_docs(layered_results)
            if len(deduped) >= limit:
                layered_results = deduped
                break
            layered_results = deduped

        if len(layered_results) < limit:
            fallback_vector_results = self._similarity_search_with_optional_filters(
                query,
                limit=max(limit * 2, 6),
                score_threshold=score_threshold,
                source_types=None,
                rerank=False,
            )
            fallback_results = fallback_vector_results
            if config.ENABLE_HYBRID_RETRIEVAL:
                fallback_keyword_results = self._keyword_search_with_optional_filters(
                    query,
                    limit=max(limit * 2, 6),
                    source_types=None,
                )
                fallback_results = self._rrf_fuse(
                    fallback_vector_results,
                    fallback_keyword_results,
                    max(limit * 2, 6),
                )
            layered_results = self._dedupe_docs(layered_results + fallback_results)

        layered_results = self._sort_docs_by_source_priority(layered_results, preferred_layers=source_layers)
        rerank_candidates = getattr(self.collection, "rerank_candidates", None)
        if callable(rerank_candidates):
            layered_results = rerank_candidates(query, layered_results, limit)
            layered_results = self._sort_docs_by_source_priority(layered_results, preferred_layers=source_layers)
        return layered_results[:limit]

    def search_documents(self, query: str, limit: int = 4, score_threshold: float = 0.7) -> List[Document]:
        return self._layered_similarity_search(query, limit=limit, score_threshold=score_threshold)

    def _log_retrieval(
        self,
        query: str,
        limit: int,
        results: List[Document],
        *,
        query_plan: list[str] | None = None,
        graded_doc_count: int = 0,
        sufficiency_result: str = "",
        retry_count: int = 0,
        final_confidence_bucket: str = "",
    ):
        log_func = getattr(self.collection, "log_retrieval", None)
        if not callable(log_func):
            return
        context = get_retrieval_context()
        log_func(
            request_id=context.get("request_id") or None,
            thread_id=context.get("thread_id") or None,
            query_text=context.get("original_query") or query,
            rewritten_query=query,
            retrieval_mode="hybrid_layered" if config.ENABLE_HYBRID_RETRIEVAL else "vector_layered",
            top_k=limit,
            result_count=len(results),
            selected_parent_ids=[doc.metadata.get("parent_id") for doc in results if (doc.metadata or {}).get("parent_id")],
            query_plan=query_plan or [query],
            graded_doc_count=graded_doc_count,
            sufficiency_result=sufficiency_result,
            retry_count=retry_count,
            final_confidence_bucket=final_confidence_bucket,
        )
    
    def _search_child_chunks(self, query: str, limit: int, query_plan: List[str] | None = None) -> str:
        """Search for the top K most relevant child chunks.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            query_plan: Optional alternate retrieval queries to execute and fuse
        """
        try:
            context = get_retrieval_context()
            normalized_plan = self._normalize_query_plan(
                query,
                query_plan or context.get("query_plan") or [],
            )
            per_query_limit = max(limit * 2, 6) if len(normalized_plan) > 1 else limit
            ranked_sets = []
            executed_queries = list(normalized_plan)

            def _retrieve_and_grade(planned_query: str) -> list[Document] | None:
                graded = grade_documents(
                    planned_query,
                    self._layered_similarity_search(
                        planned_query,
                        limit=per_query_limit,
                        score_threshold=0.7 if planned_query == query else 0.65,
                    ),
                )
                for doc in graded:
                    metadata = doc.metadata or {}
                    metadata["matched_query"] = planned_query
                    doc.metadata = metadata
                if graded:
                    return self._sort_docs_by_source_priority(
                        graded,
                        preferred_layers=self._preferred_source_layers(query),
                    )
                return None

            if len(normalized_plan) > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                plan_results: dict[str, list[Document]] = {}
                with ThreadPoolExecutor(max_workers=min(len(normalized_plan), 4)) as executor:
                    futures = {
                        executor.submit(_retrieve_and_grade, pq): pq
                        for pq in normalized_plan
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            plan_results[futures[future]] = result
                for pq in normalized_plan:
                    if pq in plan_results:
                        ranked_sets.append(plan_results[pq])
            else:
                result = _retrieve_and_grade(normalized_plan[0])
                if result is not None:
                    ranked_sets.append(result)

            results = self._sort_docs_by_source_priority(
                self._dedupe_docs(
                    self._rrf_fuse_ranked_sets(ranked_sets, per_query_limit) if ranked_sets else []
                ),
                preferred_layers=self._preferred_source_layers(query),
            )[:per_query_limit]
            sufficiency = check_sufficiency(query, results)
            retry_count = 0
            if retry_count < config.RAG_RETRY_LIMIT and not sufficiency["is_sufficient"] and sufficiency.get("retry_query"):
                retry_query = sufficiency["retry_query"]
                if _normalize_text(retry_query) not in {_normalize_text(item) for item in executed_queries}:
                    executed_queries.append(retry_query)
                    retry_count = 1
                    retry_results = self._sort_docs_by_source_priority(
                        grade_documents(
                            retry_query,
                            self._layered_similarity_search(retry_query, limit=per_query_limit, score_threshold=0.65),
                        ),
                        preferred_layers=self._preferred_source_layers(query),
                    )
                    if retry_results:
                        ranked_sets.append(retry_results)
                    results = self._sort_docs_by_source_priority(
                        self._dedupe_docs(
                            self._rrf_fuse_ranked_sets(ranked_sets, per_query_limit) if ranked_sets else []
                        ),
                        preferred_layers=self._preferred_source_layers(query),
                    )[:per_query_limit]
                    sufficiency = check_sufficiency(query, results)

            confidence_bucket = _confidence_bucket(results[:limit])
            self._log_retrieval(
                query,
                limit,
                results[:limit],
                query_plan=executed_queries,
                graded_doc_count=len(results),
                sufficiency_result=sufficiency["reason"],
                retry_count=retry_count,
                final_confidence_bucket=confidence_bucket,
            )
            if not results:
                return _NO_EVIDENCE_RESPONSE

            formatted_results = []
            for doc in results[:limit]:
                metadata = doc.metadata or {}
                formatted_results.append(
                    f"Parent ID: {metadata.get('parent_id', '')}\n"
                    f"File Name: {metadata.get('source', '')}\n"
                    f"Source Title: {metadata.get('title', metadata.get('source', ''))}\n"
                    f"Source Type: {metadata.get('source_type', 'unknown')}\n"
                    f"Original URL: {metadata.get('original_url', '')}\n"
                    f"Published At: {metadata.get('published_at', '')}\n"
                    f"Freshness Bucket: {metadata.get('freshness_bucket', '')}\n"
                    f"Score: {_doc_score(metadata):.4f}\n"
                    f"Relevance Grade: {metadata.get('relevance_grade', '')}\n"
                    f"Confidence Bucket: {confidence_bucket}\n"
                    f"Matched Query: {metadata.get('matched_query', query)}\n"
                    f"Content: {doc.page_content.strip()}"
                )

            return "\n\n".join(formatted_results)

        except Exception as e:
            logger.exception("Child chunk retrieval failed for query=%r", query)
            return f"RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_ids: List of parent chunk IDs to retrieve
        """
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"

            return "\n\n".join([
                f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                f"Source Title: {doc.get('metadata', {}).get('title', doc.get('metadata', {}).get('source', 'unknown'))}\n"
                f"Source Type: {doc.get('metadata', {}).get('source_type', 'unknown')}\n"
                f"Original URL: {doc.get('metadata', {}).get('original_url', '')}\n"
                f"Published At: {doc.get('metadata', {}).get('published_at', '')}\n"
                f"Freshness Bucket: {doc.get('metadata', {}).get('freshness_bucket', '')}\n"
                f"Content: {doc.get('content', '').strip()}"
                for doc in raw_parents
            ])            

        except Exception as e:
            logger.exception("Parent chunk batch retrieval failed for parent_ids=%r", parent_ids)
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Source Title: {parent.get('metadata', {}).get('title', parent.get('metadata', {}).get('source', 'unknown'))}\n"
                f"Source Type: {parent.get('metadata', {}).get('source_type', 'unknown')}\n"
                f"Original URL: {parent.get('metadata', {}).get('original_url', '')}\n"
                f"Published At: {parent.get('metadata', {}).get('published_at', '')}\n"
                f"Freshness Bucket: {parent.get('metadata', {}).get('freshness_bucket', '')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )          

        except Exception as e:
            logger.exception("Parent chunk retrieval failed for parent_id=%r", parent_id)
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"

    def create_tools(self) -> List:
        """Create and return the list of tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)

        return [search_tool, retrieve_tool]
