import json
import sys
import unittest
from pathlib import Path

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "project"))

from core.qa_eval import RetrievalQualityEvaluator, load_qa_samples  # noqa: E402
from benchmarks.evaluate_qa_quality import _render_markdown_report  # noqa: E402


FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
ANSWER_FIXTURE = json.loads((FIXTURES_DIR / "qa_eval_answers.json").read_text(encoding="utf-8"))


class FakeCollection:
    def __init__(self, docs_by_source=None):
        self.docs_by_source = docs_by_source or {}
        self.calls = []

    def similarity_search(self, query, k=4, score_threshold=None, source_types=None, rerank=True):
        if "月球低重力综合征" in query:
            return []
        self.calls.append(
            {
                "query": query,
                "k": k,
                "score_threshold": score_threshold,
                "source_types": list(source_types or []),
                "rerank": rerank,
            }
        )
        if source_types:
            return list(self.docs_by_source.get(source_types[0], []))
        merged = []
        for docs in self.docs_by_source.values():
            merged.extend(docs)
        return merged[:k]

    def rerank_candidates(self, query, candidates, top_n):
        return candidates[:top_n]


class QAEvaluationTests(unittest.TestCase):
    def test_load_qa_samples_parses_realistic_fixture(self):
        samples = load_qa_samples(FIXTURES_DIR / "qa_eval_samples.json")

        self.assertEqual(len(samples), 21)
        self.assertEqual(samples[0].sample_id, "patient-hypertension-symptoms")
        self.assertTrue(samples[1].must_not_clarify)
        self.assertEqual(samples[1].category, "follow_up")
        self.assertTrue(samples[1].transcript_turns)
        self.assertEqual(samples[10].expected_primary_intent, "cancel_appointment")
        self.assertEqual(samples[10].expected_secondary_intent, "medical_rag")
        self.assertEqual(samples[9].expected_no_evidence, True)

    def test_retrieval_evaluator_scores_source_type_and_keywords(self):
        collection = FakeCollection(
            docs_by_source={
                "patient_education": [
                    Document(
                        page_content="高血压常见症状包括头痛头晕，生活方式管理包括低盐饮食和规律复诊。",
                        metadata={
                            "parent_id": "p1",
                            "source": "patient_hypertension_education.pdf",
                            "source_type": "patient_education",
                            "score": 0.92,
                        },
                    )
                ],
                "public_health": [
                    Document(
                        page_content="流感预防要关注疫苗、传播和通风。",
                        metadata={
                            "parent_id": "p2",
                            "source": "public_health_flu_prevention.pdf",
                            "source_type": "public_health",
                            "score": 0.81,
                        },
                    )
                ],
                "clinical_guideline": [
                    Document(
                        page_content="高血压诊疗方案要结合剂量和治疗目标。",
                        metadata={
                            "parent_id": "p3",
                            "source": "clinical_guideline_hypertension_treatment.pdf",
                            "source_type": "clinical_guideline",
                            "score": 0.8,
                        },
                    )
                ],
            }
        )
        sample = load_qa_samples(FIXTURES_DIR / "qa_eval_samples.json")[0]
        evaluator = RetrievalQualityEvaluator(collection, limit=3)

        result = evaluator.evaluate_sample(sample)

        self.assertGreaterEqual(result.retrieval_score, 0.95)
        self.assertEqual(result.top_source_type, "patient_education")
        self.assertEqual(result.route_primary_intent, "medical_rag")
        self.assertIn("生活方式", result.matched_retrieval_keywords)

    def test_answer_quality_detects_clarification_penalty(self):
        collection = FakeCollection(
            docs_by_source={
                "patient_education": [
                    Document(
                        page_content="高血压症状和低盐饮食建议。",
                        metadata={"source": "patient_hypertension_education.pdf", "source_type": "patient_education", "score": 0.9},
                    )
                ]
            }
        )
        sample = load_qa_samples(FIXTURES_DIR / "qa_eval_samples.json")[0]
        evaluator = RetrievalQualityEvaluator(collection, limit=1)

        result = evaluator.evaluate_sample(sample, answer_text="请问你的年龄多大？高血压可能会头晕。")

        self.assertIsNotNone(result.answer_score)
        self.assertTrue(result.clarification_detected)
        self.assertLess(result.answer_score, 0.6)

    def test_fixture_answers_produce_safety_and_tone_scores(self):
        samples = load_qa_samples(FIXTURES_DIR / "qa_eval_samples.json")
        docs_by_source = {
            "patient_education": [
                Document(
                    page_content="高血压症状、低盐饮食、复诊、胸痛和呼吸困难时尽快就医。",
                    metadata={"source": "patient_hypertension_education.pdf", "source_type": "patient_education", "score": 0.92},
                )
            ],
            "public_health": [
                Document(
                    page_content="流感传播、疫苗、通风、老年人、婴幼儿、孕妇、筛查和健康管理。",
                    metadata={"source": "public_health_flu_prevention.pdf", "source_type": "public_health", "score": 0.93},
                )
            ],
            "clinical_guideline": [
                Document(
                    page_content="诊疗方案、剂量、治疗目标、复诊、不良反应，以及更易懂的患者解释。",
                    metadata={"source": "clinical_guideline_hypertension_treatment.pdf", "source_type": "clinical_guideline", "score": 0.94},
                )
            ],
        }
        evaluator = RetrievalQualityEvaluator(FakeCollection(docs_by_source=docs_by_source), limit=3)

        report = evaluator.evaluate_samples(samples, answer_provider=lambda sample: ANSWER_FIXTURE.get(sample.sample_id))

        self.assertGreaterEqual(report["summary"]["avg_answer_score"], 0.8)
        self.assertGreaterEqual(report["summary"]["avg_safety_score"], 0.8)
        self.assertGreaterEqual(report["summary"]["avg_tone_score"], 0.8)
        self.assertGreaterEqual(report["summary"]["route_hit_rate"], 0.8)
        self.assertGreaterEqual(report["summary"]["retrieval_relevance_hit_rate"], 0.8)
        self.assertGreaterEqual(report["summary"]["evidence_sufficiency_pass_rate"], 0.6)
        self.assertGreater(report["summary"]["patient_friendly_rate"], 0.0)
        self.assertGreater(report["summary"]["no_evidence_answer_rate"], 0.0)
        no_evidence_result = next(item for item in report["results"] if item["sample_id"] == "no-evidence-rare-condition")
        self.assertTrue(no_evidence_result["no_evidence_answer_detected"])
        self.assertTrue(no_evidence_result["grounding_violation_detected"] is False)
        seek_care_result = next(item for item in report["results"] if item["sample_id"] == "patient-hypertension-seek-care")
        self.assertEqual(seek_care_result["safety_score"], 1.0)
        self.assertTrue(seek_care_result["patient_friendly_detected"])

    def test_evaluate_samples_aggregates_summary(self):
        samples = load_qa_samples(FIXTURES_DIR / "qa_eval_samples.json")
        docs_by_source = {
            "patient_education": [
                Document(
                    page_content="高血压症状和低盐饮食建议。",
                    metadata={"source": "patient_hypertension_education.pdf", "source_type": "patient_education", "score": 0.92},
                )
            ],
            "public_health": [
                Document(
                    page_content="流感传播风险和疫苗预防。",
                    metadata={"source": "public_health_flu_prevention.pdf", "source_type": "public_health", "score": 0.93},
                )
            ],
            "clinical_guideline": [
                Document(
                    page_content="诊疗方案要关注剂量调整和复诊目标。",
                    metadata={"source": "clinical_guideline_hypertension_treatment.pdf", "source_type": "clinical_guideline", "score": 0.94},
                )
            ],
        }
        evaluator = RetrievalQualityEvaluator(FakeCollection(docs_by_source=docs_by_source), limit=3)

        report = evaluator.evaluate_samples(samples)

        self.assertEqual(report["summary"]["sample_count"], 21)
        self.assertGreaterEqual(report["summary"]["avg_retrieval_score"], 0.72)
        self.assertEqual(len(report["results"]), 21)
        self.assertIn("patient_education", report["summary"]["by_category"])
        self.assertIn("hard", report["summary"]["by_difficulty"])
        self.assertIn("patient_education", report["summary"]["by_top_source_type"])
        self.assertIn("route_hit_rate", report["summary"])
        self.assertIn("compound_request_handling_rate", report["summary"])
        self.assertIn("retrieval_relevance_hit_rate", report["summary"])
        self.assertIn("evidence_sufficiency_pass_rate", report["summary"])
        self.assertIn("grounding_violation_rate", report["summary"])
        no_evidence_result = next(item for item in report["results"] if item["sample_id"] == "no-evidence-rare-condition")
        self.assertTrue(no_evidence_result["no_evidence_detected"])
        self.assertIsInstance(report["summary"]["low_scoring_samples"], list)

    def test_markdown_report_includes_grouped_sections(self):
        samples = load_qa_samples(FIXTURES_DIR / "qa_eval_samples.json")
        docs_by_source = {
            "patient_education": [
                Document(
                    page_content="高血压症状和低盐饮食建议，以及胸痛呼吸困难时尽快就医。",
                    metadata={"source": "patient_hypertension_education.pdf", "source_type": "patient_education", "score": 0.92},
                )
            ],
            "public_health": [
                Document(
                    page_content="流感传播、疫苗、通风、老年人和孕妇风险。",
                    metadata={"source": "public_health_flu_prevention.pdf", "source_type": "public_health", "score": 0.93},
                )
            ],
            "clinical_guideline": [
                Document(
                    page_content="诊疗方案、剂量、治疗目标、复诊与不良反应。",
                    metadata={"source": "clinical_guideline_hypertension_treatment.pdf", "source_type": "clinical_guideline", "score": 0.94},
                )
            ],
        }
        evaluator = RetrievalQualityEvaluator(FakeCollection(docs_by_source=docs_by_source), limit=3)

        report = evaluator.evaluate_samples(samples, answer_provider=lambda sample: ANSWER_FIXTURE.get(sample.sample_id))
        report["summary"]["appointment_skill_metrics"] = {
            "sample_count": 12,
            "required_confirmation_rate": 0.5,
            "candidate_exposure_rate": 0.3,
            "final_action_distribution": {"prepare_appointment": 6},
        }
        report["summary"]["route_log_metrics"] = {
            "sample_count": 20,
            "compound_request_rate": 0.4,
            "pending_resume_rate": 0.25,
            "deferred_question_rate": 0.2,
        }
        markdown = _render_markdown_report(report)

        self.assertIn("# QA Retrieval Quality Report", markdown)
        self.assertIn("## By Category", markdown)
        self.assertIn("## Low Scoring Samples", markdown)
        self.assertIn("Average safety score", markdown)
        self.assertIn("Route hit rate", markdown)
        self.assertIn("Grounding violation rate", markdown)
        self.assertIn("## Appointment Skill Metrics", markdown)
        self.assertIn("## Route Log Metrics", markdown)
        self.assertIn("### patient-hypertension-symptoms", markdown)


if __name__ == "__main__":
    unittest.main()
