"""Text formatting, reference lines, confidence labels, and answer sanitization.

Extracted from node_helpers for focused reusability.
"""

import re
import logging

logger = logging.getLogger(__name__)


def _strip_leading_query_plan_blob(text: str) -> str:
    """Remove the query-plan block that sometimes leaks into the answer."""
    if not text:
        return text
    # Pattern: starts with "Query Plan:" or "检索计划" followed by numbered items
    pattern = r"^(?:Query\s*Plan[:：]|检索计划[:：])\s*\n(?:\s*[\d]+\.\s+.*\n)*\n*"
    return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)


def _strip_trailing_sources_block(text: str) -> str:
    """Remove the trailing 'Sources:' / '参考来源' block from answers."""
    if not text:
        return text
    pattern = r"\n*(?:Sources[:：]|参考来源[:：])\s*\n?.*$"
    return re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)


def _sanitize_final_answer_text(text: str) -> str:
    """Clean up the final answer text for display."""
    if not text:
        return text
    text = _strip_leading_query_plan_blob(text)
    text = _strip_trailing_sources_block(text)
    return text.strip()


def _build_medical_fallback_notice(*, risk_level: str = "normal", confidence_bucket: str = "no_evidence") -> str:
    """Build a medical safety notice for low-confidence or fallback answers."""
    parts = []
    if risk_level == "high":
        parts.append("⚠️ **高危提示**：您描述的症状可能需要紧急医疗处理，建议尽快前往急诊或拨打急救电话。")
    if confidence_bucket in ("no_evidence", "low"):
        parts.append("ℹ️ 以上回答**未基于知识库检索结果**，仅供一般健康信息参考，不能替代医生面对面诊断。")
    return "\n".join(parts)


def _confidence_bucket_label(confidence_bucket: str) -> str:
    """Return a human-readable label for a confidence bucket."""
    labels = {
        "high": "高置信度",
        "medium": "中等置信度",
        "low": "低置信度",
        "no_evidence": "无检索证据",
    }
    return labels.get(confidence_bucket, confidence_bucket)


def _confidence_bucket_explanation(confidence_bucket: str, *, is_medical_request: bool = False) -> str:
    """Return an explanation for a confidence bucket."""
    explanations = {
        "high": "回答基于高相关性检索结果，信息较为可靠。",
        "medium": "回答基于中等相关性检索结果，建议结合专业意见。",
        "low": "检索结果相关性较低，回答仅供参考。",
        "no_evidence": "未找到相关检索结果，回答基于模型通用知识。",
    }
    base = explanations.get(confidence_bucket, "")
    if is_medical_request and confidence_bucket in ("low", "no_evidence"):
        base += " 如有不适，请及时就医。"
    return base


def _source_type_label(source_type: str) -> str:
    """Return a human-readable label for a source type."""
    labels = {
        "official": "官方指南",
        "textbook": "医学教材",
        "research": "研究论文",
        "general": "通用参考",
        "unknown": "未知来源",
    }
    return labels.get(source_type, source_type)


def _freshness_bucket_label(bucket: str) -> str:
    """Return a human-readable label for a freshness bucket."""
    labels = {
        "very_fresh": "1 年内",
        "fresh": "1–3 年",
        "stale": "3–5 年",
        "very_stale": "5 年以上",
    }
    return labels.get(bucket, bucket)


def _format_reference_lines(sources: list[dict]) -> list[str]:
    """Format source dicts into readable reference lines."""
    lines = []
    for src in sources:
        title = src.get("title", "未命名文档")
        source_type = _source_type_label(src.get("source_type", "unknown"))
        freshness = _freshness_bucket_label(src.get("freshness_bucket", ""))
        relevance = src.get("relevance_score", 0)
        line = f"- [{source_type}] {title}"
        if freshness:
            line += f" ({freshness})"
        if relevance:
            line += f" [相关度: {relevance:.0%}]"
        lines.append(line)
    return lines
