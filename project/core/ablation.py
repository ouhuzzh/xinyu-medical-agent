"""Retrieval ablation study framework.

Supports disabling individual pipeline components (query rewrite, hybrid
search, rerank) to quantify their independent contribution to retrieval
quality.  Generates a Markdown comparison report.

Usage::

    from core.ablation import AblationStudy, RetrievalPipelineConfig
    study = AblationStudy(collection)
    results = study.run_all(samples)
    print(AblationStudy.format_comparison_report(results))
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from core.qa_eval import QAEvalSample, RetrievalEvalResult, RetrievalQualityEvaluator


# ---------------------------------------------------------------------------
# Pipeline config (what to enable / disable)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalPipelineConfig:
    """Controls which retrieval pipeline components are enabled."""

    enable_rewrite: bool = True
    enable_hybrid_search: bool = True
    enable_rerank: bool = True

    @property
    def label(self) -> str:
        """Human-readable label for this config variant."""
        parts = []
        if not self.enable_rewrite:
            parts.append("no_rewrite")
        if not self.enable_hybrid_search:
            parts.append("no_hybrid")
        if not self.enable_rerank:
            parts.append("no_rerank")
        return "+".join(parts) if parts else "baseline"


# ---------------------------------------------------------------------------
# Ablation result
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Results from one ablation variant."""

    config: RetrievalPipelineConfig
    precision_at_5: float
    mrr: float
    avg_latency_ms: float
    per_component_latency_ms: dict  # {"vector": 45.2, "keyword": 12.3, "rerank": 78.1}
    sample_count: int
    sample_details: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config_label"] = self.config.label
        return d


# ---------------------------------------------------------------------------
# MRR computation
# ---------------------------------------------------------------------------

def compute_mrr(
    results: List[RetrievalEvalResult],
    samples: List[QAEvalSample],
) -> float:
    """Compute Mean Reciprocal Rank across results.

    MRR = average of 1/rank where *rank* is the position (1-indexed) of the
    first retrieved doc whose ``page_content`` contains at least one expected
    retrieval keyword.  If no doc matches, that sample contributes 0.
    """
    if not results:
        return 0.0

    sample_by_id = {s.sample_id: s for s in samples}
    reciprocal_ranks: list[float] = []

    for r in results:
        sample = sample_by_id.get(r.sample_id)
        if not sample or not sample.expected_retrieval_keywords:
            reciprocal_ranks.append(0.0)
            continue

        # Check both snippets and retrieved source names for keyword presence
        found_rank = 0
        for idx in range(max(len(r.snippets), len(r.retrieved_sources))):
            text = ""
            if idx < len(r.snippets):
                text += (r.snippets[idx] or "")
            if idx < len(r.retrieved_sources):
                text += " " + (r.retrieved_sources[idx] or "")
            text_lower = text.lower()
            if any(kw.lower() in text_lower for kw in sample.expected_retrieval_keywords):
                found_rank = idx + 1
                break
        reciprocal_ranks.append(1.0 / found_rank if found_rank > 0 else 0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


# ---------------------------------------------------------------------------
# Ablation study runner
# ---------------------------------------------------------------------------

class AblationStudy:
    """Run retrieval quality evaluation with different pipeline configurations."""

    # Standard ablation variants
    STANDARD_CONFIGS = [
        RetrievalPipelineConfig(),  # baseline
        RetrievalPipelineConfig(enable_rewrite=False),  # no_rewrite
        RetrievalPipelineConfig(enable_hybrid_search=False),  # no_hybrid
        RetrievalPipelineConfig(enable_rerank=False),  # no_rerank
        RetrievalPipelineConfig(enable_rewrite=False, enable_rerank=False),  # no_rewrite+no_rerank
    ]

    def __init__(self, collection, *, limit: int = 3, score_threshold: float = 0.7):
        self.collection = collection
        self.limit = limit
        self.score_threshold = score_threshold

    def run_variant(
        self,
        samples: List[QAEvalSample],
        config: RetrievalPipelineConfig,
        *,
        answer_provider: Optional[Callable] = None,
    ) -> AblationResult:
        """Run evaluation for one ablation variant.

        When ``config.enable_rewrite`` is False, the original ``sample.question``
        is used as the search query instead of ``sample.search_query``.
        """
        # Build evaluator with pipeline config
        evaluator = RetrievalQualityEvaluator(
            self.collection,
            limit=self.limit,
            score_threshold=self.score_threshold,
            pipeline_config=config,
        )

        t0 = time.perf_counter()
        results: list[RetrievalEvalResult] = []
        for sample in samples:
            # no_rewrite: use original question directly
            search_query = sample.search_query or sample.question
            if not config.enable_rewrite:
                search_query = sample.question

            answer_text = None
            if answer_provider:
                answer_text = answer_provider(sample)

            result = evaluator.evaluate_sample(sample, answer_text=answer_text)
            results.append(result)
        total_ms = (time.perf_counter() - t0) * 1000

        # Compute metrics
        precisions = []
        component_latencies: dict[str, list[float]] = {
            "vector": [], "keyword": [], "rerank": [], "total": [],
        }

        for r in results:
            # Precision@5: fraction of top-5 that are retrieval-relevant
            precisions.append(1.0 if r.retrieval_relevance_hit else 0.0)
            if r.retrieval_latency_ms:
                component_latencies["total"].append(r.retrieval_latency_ms)
            if r.vector_search_latency_ms:
                component_latencies["vector"].append(r.vector_search_latency_ms)
            if r.keyword_search_latency_ms:
                component_latencies["keyword"].append(r.keyword_search_latency_ms)
            if r.rerank_latency_ms:
                component_latencies["rerank"].append(r.rerank_latency_ms)

        precision_at_5 = sum(precisions) / len(precisions) if precisions else 0.0
        mrr = compute_mrr(results, samples)
        avg_latency = total_ms / len(samples) if samples else 0.0
        avg_component = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in component_latencies.items()
        }

        return AblationResult(
            config=config,
            precision_at_5=round(precision_at_5, 4),
            mrr=round(mrr, 4),
            avg_latency_ms=round(avg_latency, 1),
            per_component_latency_ms=avg_component,
            sample_count=len(samples),
            sample_details=[r.to_dict() for r in results],
        )

    def run_all(
        self,
        samples: List[QAEvalSample],
        *,
        custom_configs: Optional[List[RetrievalPipelineConfig]] = None,
        answer_provider: Optional[Callable] = None,
    ) -> List[AblationResult]:
        """Run all standard ablation variants plus any custom ones."""
        configs = list(self.STANDARD_CONFIGS)
        if custom_configs:
            configs.extend(custom_configs)

        results: list[AblationResult] = []
        for cfg in configs:
            result = self.run_variant(samples, cfg, answer_provider=answer_provider)
            results.append(result)
        return results

    @staticmethod
    def format_comparison_report(results: List[AblationResult]) -> str:
        """Format results as a Markdown comparison table."""
        lines = [
            "# Retrieval Ablation Study Report",
            "",
            "| Variant | Precision@5 | MRR | Avg Latency (ms) | Vector (ms) | Keyword (ms) | Rerank (ms) | Samples |",
            "|---------|-------------|-----|-------------------|-------------|--------------|-------------|---------|",
        ]
        for r in results:
            label = r.config.label
            comp = r.per_component_latency_ms
            lines.append(
                f"| {label} | {r.precision_at_5:.4f} | {r.mrr:.4f} | {r.avg_latency_ms:.1f} "
                f"| {comp.get('vector', 0):.1f} | {comp.get('keyword', 0):.1f} "
                f"| {comp.get('rerank', 0):.1f} | {r.sample_count} |"
            )

        # Add marginal contribution table
        if len(results) >= 2:
            baseline = results[0]
            lines.append("")
            lines.append("## Marginal Contribution vs Baseline")
            lines.append("")
            lines.append("| Variant | Δ Precision@5 | Δ MRR | Δ Latency (ms) |")
            lines.append("|---------|---------------|-------|----------------|")
            for r in results[1:]:
                d_prec = r.precision_at_5 - baseline.precision_at_5
                d_mrr = r.mrr - baseline.mrr
                d_lat = r.avg_latency_ms - baseline.avg_latency_ms
                lines.append(
                    f"| {r.config.label} | {d_prec:+.4f} | {d_mrr:+.4f} | {d_lat:+.1f} |"
                )

        return "\n".join(lines)
