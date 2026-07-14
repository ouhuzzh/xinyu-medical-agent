import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.resume_benchmarks import (
    InMemoryHybridBenchmarkCollection,
    KeywordBenchmarkEmbeddings,
    build_isolated_medical_corpora,
    evaluate_medical_rag_benchmark,
    evaluate_offline_answer_benchmark,
    evaluate_memory_token_benchmark,
    load_medical_rag_benchmark_samples,
    load_memory_benchmark_samples,
    load_offline_answer_benchmark_samples,
)
from benchmarks.evaluate_acceptance_report import build_acceptance_report


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]

# These benchmarks need the bundled NHC/WHO corpus under markdown_docs/.
_HAS_BENCHMARK_DOCS = bool(
    list((REPO_ROOT / "markdown_docs").glob("who-*.md"))
    or list((REPO_ROOT / "markdown_docs").glob("nhc-*.md"))
)
_BENCHMARK_DOC_SKIP = unittest.skipUnless(
    _HAS_BENCHMARK_DOCS, "needs markdown_docs benchmark corpus (who-*.md / nhc-*.md)"
)


class ResumeBenchmarkTests(unittest.TestCase):
    def test_memory_benchmark_returns_positive_average_reduction(self):
        samples = load_memory_benchmark_samples(FIXTURES_DIR / "memory_benchmark_samples.json")
        report = evaluate_memory_token_benchmark(samples)

        self.assertEqual(report["summary"]["sample_count"], 4)
        self.assertGreater(report["summary"]["avg_token_reduction_rate"], 0.0)
        self.assertGreater(report["summary"]["p95_token_reduction_rate"], 0.0)

    @_BENCHMARK_DOC_SKIP
    def test_medical_rag_benchmark_shows_precision_uplift(self):
        doc_paths = sorted((REPO_ROOT / "markdown_docs").glob("who-*.md")) + sorted((REPO_ROOT / "markdown_docs").glob("nhc-*.md"))
        baseline_docs, optimized_docs = build_isolated_medical_corpora(doc_paths)
        embeddings = KeywordBenchmarkEmbeddings()
        baseline_collection = InMemoryHybridBenchmarkCollection(baseline_docs, embeddings)
        optimized_collection = InMemoryHybridBenchmarkCollection(optimized_docs, embeddings)
        samples = load_medical_rag_benchmark_samples(FIXTURES_DIR / "medical_rag_benchmark_samples.json")

        report = evaluate_medical_rag_benchmark(samples, baseline_collection, optimized_collection)

        self.assertEqual(report["summary"]["sample_count"], 10)
        self.assertGreater(report["summary"]["optimized_precision_at_5"], report["summary"]["baseline_precision_at_5"])
        self.assertGreaterEqual(report["summary"]["optimized_recall_at_5"], report["summary"]["baseline_recall_at_5"])

    @_BENCHMARK_DOC_SKIP
    def test_acceptance_report_builds_without_live_qa(self):
        report = build_acceptance_report(include_live_qa=False)

        self.assertIn("summary", report)
        self.assertIn("memory_benchmark", report)
        self.assertIn("retrieval_benchmark", report)
        self.assertIsNone(report["live_answer_eval"])
        self.assertGreater(report["summary"]["memory_token_reduction_avg"], 0.0)
        self.assertGreater(
            report["summary"]["retrieval_precision_at_5_optimized"],
            report["summary"]["retrieval_precision_at_5_baseline"],
        )

    @_BENCHMARK_DOC_SKIP
    def test_offline_answer_benchmark_shows_answer_score_uplift(self):
        doc_paths = sorted((REPO_ROOT / "markdown_docs").glob("who-*.md")) + sorted((REPO_ROOT / "markdown_docs").glob("nhc-*.md"))
        baseline_docs, optimized_docs = build_isolated_medical_corpora(doc_paths)
        embeddings = KeywordBenchmarkEmbeddings()
        baseline_collection = InMemoryHybridBenchmarkCollection(baseline_docs, embeddings)
        optimized_collection = InMemoryHybridBenchmarkCollection(optimized_docs, embeddings)
        samples = load_offline_answer_benchmark_samples(FIXTURES_DIR / "offline_answer_benchmark_samples.json")

        report = evaluate_offline_answer_benchmark(samples, baseline_collection, optimized_collection)

        self.assertEqual(report["summary"]["sample_count"], 11)
        self.assertGreater(report["summary"]["optimized_avg_answer_score"], report["summary"]["baseline_avg_answer_score"])
        self.assertGreater(report["summary"]["answer_score_uplift"], 0.0)


if __name__ == "__main__":
    unittest.main()
