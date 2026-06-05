#!/usr/bin/env python
"""CLI tool to run retrieval ablation studies.

Usage::

    python -m benchmarks.run_ablation_study \\
        --samples tests/fixtures/qa_eval_samples.json \\
        --output ablation_report.md

    # Only run specific variants
    python -m benchmarks.run_ablation_study \\
        --variants baseline,no_rerank,no_hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.qa_eval import QAEvalSample, load_qa_samples
from core.ablation import AblationStudy, RetrievalPipelineConfig


_VARIANT_MAP = {
    "baseline": RetrievalPipelineConfig(),
    "no_rewrite": RetrievalPipelineConfig(enable_rewrite=False),
    "no_hybrid": RetrievalPipelineConfig(enable_hybrid_search=False),
    "no_rerank": RetrievalPipelineConfig(enable_rerank=False),
    "no_rewrite+no_rerank": RetrievalPipelineConfig(enable_rewrite=False, enable_rerank=False),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run retrieval ablation study.")
    parser.add_argument(
        "--samples",
        default=str(_PROJECT_ROOT / "tests" / "fixtures" / "qa_eval_samples.json"),
        help="Path to QA evaluation samples JSON file.",
    )
    parser.add_argument("--limit", type=int, default=3, help="Top-K docs to retrieve.")
    parser.add_argument("--score-threshold", type=float, default=0.7, help="Min similarity score.")
    parser.add_argument("--output", help="Write Markdown report to file instead of stdout.")
    parser.add_argument(
        "--variants",
        help="Comma-separated variant names: baseline,no_rewrite,no_hybrid,no_rerank,no_rewrite+no_rerank",
    )
    args = parser.parse_args(argv)

    # Load samples
    samples = load_qa_samples(args.samples)
    if not samples:
        print("No evaluation samples found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(samples)} evaluation samples.")

    # Initialize collection (requires DB)
    try:
        from db.vector_db_manager import VectorDbManager
        from config import CHILD_COLLECTION
        mgr = VectorDbManager()
        mgr.create_collection(CHILD_COLLECTION)
        collection = mgr.get_collection(CHILD_COLLECTION)
    except Exception as exc:
        print(f"Failed to connect to vector DB: {exc}", file=sys.stderr)
        print("Make sure PostgreSQL is running and the collection exists.", file=sys.stderr)
        sys.exit(1)

    # Build variant configs
    if args.variants:
        variant_names = [v.strip() for v in args.variants.split(",")]
        configs = []
        for name in variant_names:
            cfg = _VARIANT_MAP.get(name)
            if cfg is None:
                print(f"Unknown variant: {name}. Available: {', '.join(_VARIANT_MAP)}", file=sys.stderr)
                sys.exit(1)
            configs.append(cfg)
    else:
        configs = AblationStudy.STANDARD_CONFIGS

    # Run study
    study = AblationStudy(collection, limit=args.limit, score_threshold=args.score_threshold)
    results = []
    for cfg in configs:
        print(f"Running variant: {cfg.label} ...")
        result = study.run_variant(samples, cfg)
        results.append(result)

    # Generate report
    report = AblationStudy.format_comparison_report(results)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
