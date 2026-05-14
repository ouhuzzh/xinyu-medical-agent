from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.qa_eval import RetrievalQualityEvaluator, load_qa_samples
from db.appointment_skill_log_store import AppointmentSkillLogStore
from db.route_log_store import RouteLogStore
from db.retrieval_log_store import RetrievalLogStore
from db.vector_db_manager import VectorDbManager


DEFAULT_SAMPLES_PATH = REPO_ROOT / "tests" / "fixtures" / "qa_eval_samples.json"
DEFAULT_ANSWERS_PATH = REPO_ROOT / "tests" / "fixtures" / "qa_eval_answers.json"


def _load_answer_map(path: str | None) -> dict:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    raise ValueError("Answer file must be a JSON object mapping sample id to answer text.")


def _print_text_report(report: dict):
    summary = report["summary"]
    print("QA Evaluation Summary")
    print(f"- sample_count: {summary['sample_count']}")
    print(f"- avg_retrieval_score: {summary['avg_retrieval_score']}")
    print(f"- avg_overall_score: {summary['avg_overall_score']}")
    if summary.get("avg_answer_score") is not None:
        print(f"- avg_answer_score: {summary['avg_answer_score']}")
    if summary.get("avg_safety_score") is not None:
        print(f"- avg_safety_score: {summary['avg_safety_score']}")
    if summary.get("avg_tone_score") is not None:
        print(f"- avg_tone_score: {summary['avg_tone_score']}")
    print(f"- pass_rate_085: {summary['pass_rate_085']}")
    print(f"- route_hit_rate: {summary.get('route_hit_rate')}")
    print(f"- secondary_route_hit_rate: {summary.get('secondary_route_hit_rate')}")
    print(f"- compound_request_handling_rate: {summary.get('compound_request_handling_rate')}")
    print(f"- patient_friendly_rate: {summary.get('patient_friendly_rate')}")
    print(f"- no_evidence_answer_rate: {summary.get('no_evidence_answer_rate')}")
    print(f"- retrieval_relevance_hit_rate: {summary.get('retrieval_relevance_hit_rate')}")
    print(f"- evidence_sufficiency_pass_rate: {summary.get('evidence_sufficiency_pass_rate')}")
    print(f"- grounding_violation_rate: {summary.get('grounding_violation_rate')}")
    if summary.get("appointment_skill_metrics"):
        print(f"- appointment_skill_metrics: {summary['appointment_skill_metrics']}")
    if summary.get("route_log_metrics"):
        print(f"- route_log_metrics: {summary['route_log_metrics']}")
    print("")
    for item in report["results"]:
        print(f"[{item['sample_id']}] {item['question']}")
        print(f"  category={item['category']} difficulty={item['difficulty']}")
        print(f"  overall={item['overall_score']} retrieval={item['retrieval_score']} answer={item['answer_score']}")
        print(f"  route={item['route_primary_intent']} secondary={item['route_secondary_intent']} source={item['route_decision_source']} confidence={item['confidence_bucket']}")
        print(f"  top_source_type={item['top_source_type']} top_source={item['top_source']}")
        print(f"  preferred_layers={', '.join(item['preferred_source_layers'])}")
        if item["transcript_turns"]:
            print("  transcript_preview=" + " | ".join(item["transcript_turns"][:3]))
        if item["matched_retrieval_keywords"] or item["missing_retrieval_keywords"]:
            print(
                "  retrieval_keywords="
                f"matched={item['matched_retrieval_keywords']} missing={item['missing_retrieval_keywords']}"
            )
        if item["answer_score"] is not None:
            print(
                "  answer_keywords="
                f"matched={item['matched_answer_keywords']} missing={item['missing_answer_keywords']}"
            )
            print(
                "  safety_keywords="
                f"matched={item['matched_safety_keywords']} missing={item['missing_safety_keywords']}"
            )
            print(f"  clarification_detected={item['clarification_detected']}")
            print(f"  patient_friendly_detected={item['patient_friendly_detected']} safety_score={item['safety_score']} tone_score={item['tone_score']}")
            print(f"  grounding_violation_detected={item['grounding_violation_detected']}")
        print("")
    print("Grouped Summary")
    for group_name, values in report["summary"].get("by_category", {}).items():
        print(
            f"- category:{group_name} sample_count={values['sample_count']} "
            f"avg_retrieval={values['avg_retrieval_score']} avg_overall={values['avg_overall_score']}"
        )
    if report["summary"].get("low_scoring_samples"):
        print("")
        print("Low Scoring Samples")
        for item in report["summary"]["low_scoring_samples"]:
            print(
                f"- {item['sample_id']} category={item['category']} difficulty={item['difficulty']} "
                f"overall={item['overall_score']} retrieval={item['retrieval_score']} top_source_type={item['top_source_type']}"
            )


def _render_markdown_report(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# QA Retrieval Quality Report",
        "",
        "## Summary",
        f"- Sample count: {summary['sample_count']}",
        f"- Average retrieval score: {summary['avg_retrieval_score']}",
        f"- Average overall score: {summary['avg_overall_score']}",
        f"- Pass rate (>=0.85): {summary['pass_rate_085']}",
        f"- Clarification rate: {summary.get('clarification_rate')}",
        f"- No-evidence rate: {summary.get('no_evidence_rate')}",
        f"- Source type hit rate: {summary.get('source_type_hit_rate')}",
        f"- Route hit rate: {summary.get('route_hit_rate')}",
        f"- Secondary route hit rate: {summary.get('secondary_route_hit_rate')}",
        f"- Compound request handling rate: {summary.get('compound_request_handling_rate')}",
        f"- Patient-friendly rate: {summary.get('patient_friendly_rate')}",
        f"- No-evidence answer rate: {summary.get('no_evidence_answer_rate')}",
        f"- Retrieval relevance hit rate: {summary.get('retrieval_relevance_hit_rate')}",
        f"- Evidence sufficiency pass rate: {summary.get('evidence_sufficiency_pass_rate')}",
        f"- Grounding violation rate: {summary.get('grounding_violation_rate')}",
    ]
    if summary.get("avg_answer_score") is not None:
        lines.append(f"- Average answer score: {summary['avg_answer_score']}")
    if summary.get("avg_safety_score") is not None:
        lines.append(f"- Average safety score: {summary['avg_safety_score']}")
    if summary.get("avg_tone_score") is not None:
        lines.append(f"- Average tone score: {summary['avg_tone_score']}")
    if summary.get("appointment_skill_metrics"):
        lines.extend(
            [
                "",
                "## Appointment Skill Metrics",
                "",
                f"- Sample count: {summary['appointment_skill_metrics'].get('sample_count')}",
                f"- Required confirmation rate: {summary['appointment_skill_metrics'].get('required_confirmation_rate')}",
                f"- Candidate exposure rate: {summary['appointment_skill_metrics'].get('candidate_exposure_rate')}",
                f"- Final action distribution: {summary['appointment_skill_metrics'].get('final_action_distribution')}",
            ]
        )
    if summary.get("route_log_metrics"):
        lines.extend(
            [
                "",
                "## Route Log Metrics",
                "",
                f"- Sample count: {summary['route_log_metrics'].get('sample_count')}",
                f"- Compound request rate: {summary['route_log_metrics'].get('compound_request_rate')}",
                f"- Pending resume rate: {summary['route_log_metrics'].get('pending_resume_rate')}",
                f"- Deferred question rate: {summary['route_log_metrics'].get('deferred_question_rate')}",
            ]
        )

    def append_group(title: str, payload: dict):
        if not payload:
            return
        lines.extend(["", f"## {title}", "", "| Group | Sample Count | Avg Retrieval | Avg Overall | Pass Rate 0.85 |", "| --- | ---: | ---: | ---: | ---: |"])
        for group_name, values in payload.items():
            lines.append(
                f"| {group_name or 'unspecified'} | {values['sample_count']} | {values['avg_retrieval_score']} | "
                f"{values['avg_overall_score']} | {values['pass_rate_085']} |"
            )

    append_group("By Category", summary.get("by_category", {}))
    append_group("By Difficulty", summary.get("by_difficulty", {}))
    append_group("By Top Source Type", summary.get("by_top_source_type", {}))

    low_scoring = summary.get("low_scoring_samples") or []
    if low_scoring:
        lines.extend(["", "## Low Scoring Samples", "", "| Sample | Category | Difficulty | Overall | Retrieval | Top Source Type |", "| --- | --- | --- | ---: | ---: | --- |"])
        for item in low_scoring:
            lines.append(
                f"| {item['sample_id']} | {item['category']} | {item['difficulty']} | {item['overall_score']} | "
                f"{item['retrieval_score']} | {item['top_source_type'] or 'n/a'} |"
            )

    lines.extend(["", "## Sample Details", ""])
    for item in report["results"]:
        lines.extend(
            [
                f"### {item['sample_id']}",
                f"- Question: {item['question']}",
                f"- Category: {item['category']}",
                f"- Difficulty: {item['difficulty']}",
                f"- Overall / Retrieval / Answer: {item['overall_score']} / {item['retrieval_score']} / {item['answer_score']}",
                f"- Preferred layers: {', '.join(item['preferred_source_layers'])}",
                f"- Route: {item['route_primary_intent']} / {item['route_secondary_intent'] or 'n/a'} ({item['route_decision_source']})",
                f"- Route reason: {item['route_reason']}",
                f"- Confidence bucket: {item['confidence_bucket']}",
                f"- Retrieval relevance hit: {item['retrieval_relevance_hit']}",
                f"- Evidence sufficient: {item['evidence_sufficient']}",
                f"- Grounding violation detected: {item['grounding_violation_detected']}",
                f"- Top source: {item['top_source_type'] or 'n/a'} / {item['top_source'] or 'n/a'}",
                f"- Clarification detected: {item['clarification_detected']}",
                f"- Patient friendly detected: {item['patient_friendly_detected']}",
                f"- Safety / tone score: {item['safety_score']} / {item['tone_score']}",
                f"- No evidence detected: {item['no_evidence_detected']}",
                f"- No-evidence answer detected: {item['no_evidence_answer_detected']}",
            ]
        )
        if item["transcript_turns"]:
            lines.append("- Transcript context:")
            lines.extend([f"  - {turn}" for turn in item["transcript_turns"]])
        if item["matched_retrieval_keywords"] or item["missing_retrieval_keywords"]:
            lines.append(
                f"- Retrieval keywords: matched={item['matched_retrieval_keywords']} missing={item['missing_retrieval_keywords']}"
            )
        if item["matched_answer_keywords"] or item["missing_answer_keywords"]:
            lines.append(
                f"- Answer keywords: matched={item['matched_answer_keywords']} missing={item['missing_answer_keywords']}"
            )
        if item["matched_safety_keywords"] or item["missing_safety_keywords"]:
            lines.append(
                f"- Safety keywords: matched={item['matched_safety_keywords']} missing={item['missing_safety_keywords']}"
            )
        lines.append("")
    return "\n".join(lines)


def _safe_operational_metrics() -> dict:
    metrics = {}
    try:
        metrics["route_log_metrics"] = RouteLogStore().summarize_recent(limit=200)
    except Exception:
        metrics["route_log_metrics"] = None
    try:
        metrics["retrieval_log_metrics"] = RetrievalLogStore().summarize_recent(limit=200)
    except Exception:
        metrics["retrieval_log_metrics"] = None
    try:
        metrics["appointment_skill_metrics"] = AppointmentSkillLogStore().summarize_recent(limit=200)
    except Exception:
        metrics["appointment_skill_metrics"] = None
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate retrieval and optional answer quality on a realistic QA sample set.")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES_PATH), help="Path to QA sample JSON.")
    parser.add_argument("--answers", help="Optional JSON object mapping sample ids to answer text for answer scoring.")
    parser.add_argument("--fixture-answers", action="store_true", help="Use the bundled realistic answer fixture set.")
    parser.add_argument("--limit", type=int, default=3, help="Top-k retrieval depth per sample.")
    parser.add_argument("--score-threshold", type=float, default=0.7, help="Similarity threshold.")
    parser.add_argument("--json", action="store_true", help="Print JSON report instead of text.")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown report instead of text.")
    parser.add_argument("--output", help="Optional path to write the rendered report.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Exit with status 1 if avg overall score is below this threshold.")
    args = parser.parse_args(argv)

    vector_db = VectorDbManager()
    vector_db.create_collection(config.CHILD_COLLECTION)
    stats = vector_db.get_collection_stats()
    if stats["child_chunks"] == 0:
        print("Knowledge base is empty. Import or index documents before running QA evaluation.", file=sys.stderr)
        return 1

    samples = load_qa_samples(args.samples)
    answer_path = str(DEFAULT_ANSWERS_PATH) if args.fixture_answers and not args.answers else args.answers
    answer_map = _load_answer_map(answer_path)
    evaluator = RetrievalQualityEvaluator(
        vector_db.get_collection(config.CHILD_COLLECTION),
        limit=args.limit,
        score_threshold=args.score_threshold,
    )
    report = evaluator.evaluate_samples(samples, answer_provider=lambda sample: answer_map.get(sample.sample_id))
    operational_metrics = _safe_operational_metrics()
    for key, value in operational_metrics.items():
        if value:
            report["summary"][key] = value

    if args.json:
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
    elif args.markdown:
        rendered = _render_markdown_report(report)
    else:
        rendered = None
        _print_text_report(report)

    if rendered is not None:
        print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if rendered is None:
            rendered = _render_markdown_report(report)
        output_path.write_text(rendered, encoding="utf-8")

    if report["summary"]["avg_overall_score"] < args.min_score:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
