"""Tests for ablation study framework."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from core.ablation import (
    AblationResult,
    AblationStudy,
    RetrievalPipelineConfig,
    compute_mrr,
)
from core.qa_eval import QAEvalSample, RetrievalEvalResult


class TestRetrievalPipelineConfig(unittest.TestCase):

    def test_baseline_label(self):
        cfg = RetrievalPipelineConfig()
        self.assertEqual(cfg.label, "baseline")

    def test_no_rewrite_label(self):
        cfg = RetrievalPipelineConfig(enable_rewrite=False)
        self.assertEqual(cfg.label, "no_rewrite")

    def test_no_hybrid_label(self):
        cfg = RetrievalPipelineConfig(enable_hybrid_search=False)
        self.assertEqual(cfg.label, "no_hybrid")

    def test_no_rerank_label(self):
        cfg = RetrievalPipelineConfig(enable_rerank=False)
        self.assertEqual(cfg.label, "no_rerank")

    def test_combined_label(self):
        cfg = RetrievalPipelineConfig(enable_rewrite=False, enable_rerank=False)
        self.assertEqual(cfg.label, "no_rewrite+no_rerank")


class TestComputeMRR(unittest.TestCase):

    def _make_result(self, sample_id, snippets):
        return RetrievalEvalResult(
            sample_id=sample_id, question="", category="general",
            difficulty="medium", transcript_turns=[],
            route_primary_intent="", route_secondary_intent="",
            route_decision_source="", route_reason="",
            route_hit=True, secondary_route_hit=True,
            preferred_source_layers=[], confidence_bucket="high",
            retrieval_score=0.9, answer_score=None, overall_score=0.9,
            retrieval_relevance_hit=True, evidence_sufficient=True,
            grounding_violation_detected=False,
            top_source_type="", top_source="",
            matched_retrieval_keywords=[], missing_retrieval_keywords=[],
            matched_answer_keywords=[], missing_answer_keywords=[],
            matched_safety_keywords=[], missing_safety_keywords=[],
            clarification_detected=False, patient_friendly_detected=False,
            no_evidence_answer_detected=False,
            safety_score=None, tone_score=None,
            source_type_hit=True, source_contains_hit=True,
            no_evidence_detected=False,
            retrieved_sources=[], retrieved_source_types=[],
            answer_text="", snippets=snippets,
        )

    def test_mrr_rank1(self):
        """First snippet contains keyword → MRR = 1.0."""
        samples = [QAEvalSample(sample_id="s1", question="q",
                                expected_retrieval_keywords=["高血压"])]
        results = [self._make_result("s1", ["高血压的预防和治疗", "其他内容"])]
        self.assertAlmostEqual(compute_mrr(results, samples), 1.0)

    def test_mrr_rank2(self):
        """Second snippet contains keyword → MRR = 0.5."""
        samples = [QAEvalSample(sample_id="s1", question="q",
                                expected_retrieval_keywords=["高血压"])]
        results = [self._make_result("s1", ["无关内容", "高血压注意事项"])]
        self.assertAlmostEqual(compute_mrr(results, samples), 0.5)

    def test_mrr_no_match(self):
        """No snippet matches → MRR = 0.0."""
        samples = [QAEvalSample(sample_id="s1", question="q",
                                expected_retrieval_keywords=["高血压"])]
        results = [self._make_result("s1", ["无关1", "无关2"])]
        self.assertAlmostEqual(compute_mrr(results, samples), 0.0)

    def test_mrr_multiple_samples(self):
        """Average across multiple samples."""
        samples = [
            QAEvalSample(sample_id="s1", question="q", expected_retrieval_keywords=["高血压"]),
            QAEvalSample(sample_id="s2", question="q", expected_retrieval_keywords=["糖尿病"]),
        ]
        results = [
            self._make_result("s1", ["高血压注意事项"]),  # rank 1 → 1.0
            self._make_result("s2", ["无关", "糖尿病饮食指南"]),  # rank 2 → 0.5
        ]
        self.assertAlmostEqual(compute_mrr(results, samples), 0.75)

    def test_mrr_empty(self):
        self.assertEqual(compute_mrr([], []), 0.0)


class TestAblationStudyFormatReport(unittest.TestCase):

    def test_format_comparison_report(self):
        results = [
            AblationResult(
                config=RetrievalPipelineConfig(),
                precision_at_5=0.82, mrr=0.71, avg_latency_ms=245.0,
                per_component_latency_ms={"vector": 45.0, "keyword": 12.0, "rerank": 78.0, "total": 135.0},
                sample_count=10,
            ),
            AblationResult(
                config=RetrievalPipelineConfig(enable_rerank=False),
                precision_at_5=0.70, mrr=0.58, avg_latency_ms=167.0,
                per_component_latency_ms={"vector": 46.0, "keyword": 13.0, "rerank": 0.0, "total": 59.0},
                sample_count=10,
            ),
        ]
        report = AblationStudy.format_comparison_report(results)
        self.assertIn("baseline", report)
        self.assertIn("no_rerank", report)
        self.assertIn("Marginal Contribution", report)
        self.assertIn("0.82", report)
        self.assertIn("0.70", report)


class TestAblationStudyRunVariant(unittest.TestCase):

    @patch("core.ablation.RetrievalQualityEvaluator")
    def test_run_variant_returns_result(self, MockEvaluator):
        mock_eval = MagicMock()
        mock_result = RetrievalEvalResult(
            sample_id="s1", question="q", category="general",
            difficulty="medium", transcript_turns=[],
            route_primary_intent="", route_secondary_intent="",
            route_decision_source="", route_reason="",
            route_hit=True, secondary_route_hit=True,
            preferred_source_layers=[], confidence_bucket="high",
            retrieval_score=0.9, answer_score=None, overall_score=0.9,
            retrieval_relevance_hit=True, evidence_sufficient=True,
            grounding_violation_detected=False,
            top_source_type="", top_source="",
            matched_retrieval_keywords=["高血压"], missing_retrieval_keywords=[],
            matched_answer_keywords=[], missing_answer_keywords=[],
            matched_safety_keywords=[], missing_safety_keywords=[],
            clarification_detected=False, patient_friendly_detected=False,
            no_evidence_answer_detected=False,
            safety_score=None, tone_score=None,
            source_type_hit=True, source_contains_hit=True,
            no_evidence_detected=False,
            retrieved_sources=[], retrieved_source_types=[],
            answer_text="", snippets=["高血压预防"],
            retrieval_latency_ms=100.0, vector_search_latency_ms=45.0,
            keyword_search_latency_ms=12.0, rerank_latency_ms=43.0,
        )
        mock_eval.evaluate_sample.return_value = mock_result
        MockEvaluator.return_value = mock_eval

        study = AblationStudy(collection=MagicMock())
        samples = [QAEvalSample(sample_id="s1", question="高血压怎么办")]
        result = study.run_variant(samples, RetrievalPipelineConfig())

        self.assertIsInstance(result, AblationResult)
        self.assertEqual(result.config.label, "baseline")
        self.assertEqual(result.sample_count, 1)
        self.assertGreater(result.precision_at_5, 0)
        # MRR may be 0 if sample doesn't have expected_retrieval_keywords,
        # just verify the result is well-formed
        self.assertGreaterEqual(result.mrr, 0.0)


if __name__ == "__main__":
    unittest.main()
