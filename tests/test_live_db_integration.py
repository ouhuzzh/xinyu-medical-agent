import sys
import shutil
import tempfile
import threading
import unittest
import uuid
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "project"))

import psycopg  # noqa: E402
import config  # noqa: E402
from core.document_manager import DocumentManager  # noqa: E402
from core.qa_eval import RetrievalQualityEvaluator, load_qa_samples  # noqa: E402
from core.rag_system import RAGSystem  # noqa: E402
from db.appointment_skill_log_store import AppointmentSkillLogStore  # noqa: E402
from db.document_ids import build_document_no  # noqa: E402
from db.import_task_store import ImportTaskStore  # noqa: E402
from db.route_log_store import RouteLogStore  # noqa: E402
from db.vector_db_manager import PgVectorCollection, VectorDbManager  # noqa: E402
from memory.summary_store import SummaryStore  # noqa: E402
from rag_agent.tools import ToolFactory, reset_retrieval_context, set_retrieval_context  # noqa: E402
from services.appointment_service import AppointmentService  # noqa: E402

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


class KeywordEmbeddings:
    KEYWORDS = (
        "高血压",
        "症状",
        "低盐饮食",
        "生活方式",
        "胸痛",
        "呼吸困难",
        "流感",
        "疫苗",
        "传播",
        "通风",
        "老年人",
        "孕妇",
        "聚集性发热",
        "筛查",
        "诊疗方案",
        "剂量",
        "治疗目标",
        "复诊",
        "不良反应",
        "更易懂",
    )

    def _vector(self, text: str):
        normalized = str(text or "").lower()
        values = [float(normalized.count(keyword.lower())) for keyword in self.KEYWORDS]
        values += [0.0] * (config.VECTOR_DIMENSION - len(values))
        return values

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, query):
        return self._vector(query)


def _db_available() -> bool:
    try:
        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            connect_timeout=1,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("select 1")
                cur.fetchone()
        return True
    except Exception:
        return False


@unittest.skipUnless(_db_available(), "PostgreSQL is unavailable for live integration tests.")
class LiveDatabaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary_store = SummaryStore()
        cls.import_task_store = ImportTaskStore()
        cls.route_log_store = RouteLogStore()
        cls.appointment_skill_log_store = AppointmentSkillLogStore()
        cls.appointment_service = AppointmentService()
        cls.vector_db = VectorDbManager()

    def tearDown(self):
        if hasattr(self, "thread_id"):
            self._cleanup_thread(self.thread_id)
        if hasattr(self, "thread_ids"):
            for thread_id in self.thread_ids:
                self._cleanup_thread(thread_id)
        if hasattr(self, "temp_markdown_dir"):
            shutil.rmtree(self.temp_markdown_dir, ignore_errors=True)
        if hasattr(self, "document_nos"):
            for document_no in self.document_nos:
                self._cleanup_document(document_no)
        if hasattr(self, "document_no"):
            self._cleanup_document(self.document_no)
        if hasattr(self, "retrieval_log_thread_id"):
            self._cleanup_retrieval_logs(self.retrieval_log_thread_id)
        if hasattr(self, "route_log_thread_id"):
            self._cleanup_route_logs(self.route_log_thread_id)
        if hasattr(self, "appointment_skill_thread_id"):
            self._cleanup_appointment_skill_logs(self.appointment_skill_thread_id)
        if hasattr(self, "quota_restore"):
            self._restore_schedule_quota(*self.quota_restore)

    def _cleanup_thread(self, thread_id: str):
        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("select patient_id from chat_sessions where thread_id = %s", (thread_id,))
                row = cur.fetchone()
                patient_id = row[0] if row else None
                if patient_id:
                    cur.execute("delete from appointment_logs where appointment_id in (select id from appointments where patient_id = %s)", (patient_id,))
                    cur.execute("delete from appointments where patient_id = %s", (patient_id,))
                cur.execute("delete from retrieval_logs where thread_id = %s", (thread_id,))
                cur.execute("delete from chat_session_summaries where thread_id = %s", (thread_id,))
                cur.execute("delete from chat_sessions where thread_id = %s", (thread_id,))
                if patient_id:
                    cur.execute("delete from patients where id = %s", (patient_id,))
            conn.commit()

    def _cleanup_document(self, document_no: str):
        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("delete from documents where document_no = %s", (document_no,))
            conn.commit()

    def _cleanup_retrieval_logs(self, thread_id: str):
        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("delete from retrieval_logs where thread_id = %s", (thread_id,))
            conn.commit()

    def _cleanup_route_logs(self, thread_id: str):
        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("delete from route_logs where thread_id = %s", (thread_id,))
            conn.commit()

    def _cleanup_appointment_skill_logs(self, thread_id: str):
        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("delete from appointment_skill_logs where thread_id = %s", (thread_id,))
            conn.commit()

    def _set_schedule_quota(self, schedule_id: int, quota: int):
        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("update doctor_schedules set quota_available = %s where id = %s", (quota, schedule_id))
            conn.commit()

    def _restore_schedule_quota(self, schedule_id: int, quota: int):
        self._set_schedule_quota(schedule_id, quota)

    def _find_future_schedule(self):
        for day_offset in range(0, 4):
            target_day = date.today() + timedelta(days=day_offset)
            schedule = self.appointment_service.find_available_schedule("呼吸内科", target_day, "morning", "张医生")
            if schedule:
                return schedule
        self.skipTest("No demo appointment schedule available in the next 4 days.")

    def test_summary_store_round_trip(self):
        self.thread_id = f"live-summary-{uuid.uuid4().hex[:12]}"
        self.summary_store.save_summary(self.thread_id, "第一次摘要", 2)
        self.summary_store.save_summary(self.thread_id, "第二次摘要", 4)

        saved = self.summary_store.get_summary(self.thread_id)

        self.assertEqual(saved, "第二次摘要")
        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select count(*) from chat_session_summaries
                    where thread_id = %s and summary_type = 'long_term'
                    """,
                    (self.thread_id,),
                )
                count = cur.fetchone()[0]
        self.assertEqual(count, 1)

    def test_appointment_service_live_booking_and_cancellation(self):
        self.thread_id = f"live-appointment-{uuid.uuid4().hex[:12]}"
        schedule = self._find_future_schedule()

        booking = self.appointment_service.create_appointment(
            self.thread_id,
            "呼吸内科",
            schedule["schedule_date"],
            schedule["time_slot"],
            "张医生",
        )
        if booking is None:
            self.skipTest("Selected live demo schedule became unavailable before booking.")
        self.assertEqual(booking["department"], "呼吸内科")

        candidates = self.appointment_service.find_candidate_appointments(
            self.thread_id,
            appointment_no=booking["appointment_no"],
        )
        self.assertEqual(len(candidates), 1)

        cancelled = self.appointment_service.cancel_appointment(self.thread_id, candidates[0]["appointment_id"])
        self.assertIsNotNone(cancelled)
        self.assertEqual(cancelled["status"], "cancelled")

    def test_appointment_service_concurrent_booking_does_not_oversell(self):
        schedule = self._find_future_schedule()
        self.thread_ids = [f"live-concurrent-{uuid.uuid4().hex[:10]}", f"live-concurrent-{uuid.uuid4().hex[:10]}"]

        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("select quota_available from doctor_schedules where id = %s", (schedule["schedule_id"],))
                original_quota = cur.fetchone()[0]
                cur.execute(
                    """
                    select count(*) from appointments
                    where schedule_id = %s and status = 'booked'
                    """,
                    (schedule["schedule_id"],),
                )
                original_booked_count = cur.fetchone()[0]
        self.quota_restore = (schedule["schedule_id"], original_quota)
        self._set_schedule_quota(schedule["schedule_id"], 1)

        results = []
        errors = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                result = self.appointment_service.create_appointment(
                    thread_id,
                    schedule["department_name"],
                    schedule["schedule_date"],
                    schedule["time_slot"],
                    schedule["doctor_name"],
                )
                with lock:
                    results.append(result)
            except Exception as exc:  # pragma: no cover - defensive capture for live threads
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(thread_id,)) for thread_id in self.thread_ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertFalse(errors)
        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("select quota_available from doctor_schedules where id = %s", (schedule["schedule_id"],))
                remaining_quota = cur.fetchone()[0]
                cur.execute(
                    """
                    select count(*) from appointments
                    where schedule_id = %s and status = 'booked'
                    """,
                    (schedule["schedule_id"],),
                )
                booked_count = cur.fetchone()[0]

        self.assertGreaterEqual(remaining_quota, 0)
        self.assertLessEqual(booked_count - original_booked_count, 1)

    def test_schema_migrations_install_required_indexes(self):
        self.vector_db.create_collection(config.CHILD_COLLECTION)

        schema_status = self.vector_db.get_schema_status()

        self.assertIn("vector", schema_status["extensions"])
        self.assertIn("pg_trgm", schema_status["extensions"])
        self.assertIn("uq_chat_session_summaries_thread_type", schema_status["indexes"])
        self.assertIn("idx_child_chunks_embedding_cosine", schema_status["indexes"])
        self.assertIn("idx_import_task_logs_created_at", schema_status["indexes"])
        self.assertIn("uq_documents_source_key", schema_status["indexes"])
        self.assertIn("idx_documents_is_active", schema_status["indexes"])
        self.assertIn("idx_route_logs_created_at", schema_status["indexes"])
        self.assertIn("idx_appointment_skill_logs_created_at", schema_status["indexes"])
        self.assertIn("001_summary_dedup_and_indexes", schema_status["versions"])
        self.assertIn("002_child_chunks_vector_index", schema_status["versions"])
        self.assertIn("003_import_task_logs", schema_status["versions"])
        self.assertIn("004_route_logs", schema_status["versions"])
        self.assertIn("005_appointment_skill_and_retrieval_quality", schema_status["versions"])
        self.assertIn("006_knowledge_base_sync", schema_status["versions"])
        self.assertIn("007_request_trace_ids", schema_status["versions"])
        self.assertIn("008_appointment_demo_seed", schema_status["versions"])
        self.assertIn("idx_route_logs_request_id", schema_status["indexes"])
        self.assertIn("idx_retrieval_logs_request_id", schema_status["indexes"])

    def test_demo_appointment_seed_provides_future_schedule(self):
        self.vector_db.create_collection(config.CHILD_COLLECTION)

        schedule = self._find_future_schedule()

        self.assertEqual(schedule["department_name"], "呼吸内科")
        self.assertEqual(schedule["doctor_name"], "张医生")
        self.assertEqual(schedule["time_slot"], "morning")

    def test_import_task_store_round_trip(self):
        self.vector_db.create_collection(config.CHILD_COLLECTION)
        unique_note = f"live-import-{uuid.uuid4().hex[:10]}"
        self.import_task_store.save_event(
            {
                "source": "medlineplus",
                "label": "MedlinePlus Import",
                "status": "completed",
                "downloaded": 2,
                "written": 1,
                "updated": 1,
                "deactivated": 0,
                "unchanged": 1,
                "skipped": 1,
                "failed": 0,
                "index_added": 1,
                "index_skipped": 0,
                "duration_ms": 123.45,
                "note": unique_note,
                "trigger_type": "manual",
                "scope": "official:medlineplus",
                "conversion_details": ["method=pymupdf4llm"],
                "failure_details": [],
            }
        )

        events = self.import_task_store.list_recent(limit=10)

        self.assertTrue(any(item["note"] == unique_note for item in events))

    def test_route_log_store_round_trip(self):
        self.vector_db.create_collection(config.CHILD_COLLECTION)
        self.route_log_thread_id = f"live-route-{uuid.uuid4().hex[:10]}"
        request_id = f"req-{uuid.uuid4().hex[:10]}"
        self.route_log_store.save_log(
            {
                "request_id": request_id,
                "thread_id": self.route_log_thread_id,
                "user_query": "取消刚才那个预约，然后我这个咳嗽还要看吗",
                "primary_intent": "cancel_appointment",
                "secondary_intent": "medical_rag",
                "decision_source": "rule",
                "route_reason": "explicit_cancel_rule+medical_question_rule",
                "had_pending_state": True,
                "extra_metadata": {"topic_focus": "咳嗽"},
            }
        )

        events = self.route_log_store.list_recent(limit=10)
        saved = next(item for item in events if item["thread_id"] == self.route_log_thread_id)

        self.assertEqual(saved["primary_intent"], "cancel_appointment")
        self.assertEqual(saved["request_id"], request_id)
        self.assertEqual(saved["secondary_intent"], "medical_rag")
        self.assertEqual(saved["decision_source"], "rule")
        self.assertTrue(saved["had_pending_state"])

    def test_route_log_store_summary_includes_compound_metrics(self):
        self.vector_db.create_collection(config.CHILD_COLLECTION)
        self.route_log_thread_id = f"live-route-summary-{uuid.uuid4().hex[:10]}"
        self.route_log_store.save_log(
            {
                "thread_id": self.route_log_thread_id,
                "user_query": "帮我挂呼吸内科，另外高血压药还要不要继续吃",
                "primary_intent": "appointment",
                "secondary_intent": "medical_rag",
                "decision_source": "rule",
                "route_reason": "explicit_appointment_rule+medical_question_rule",
                "had_pending_state": False,
                "extra_metadata": {
                    "topic_focus": "高血压",
                    "deferred_user_question": "高血压药还要不要继续吃",
                },
            }
        )

        report = self.route_log_store.build_recent_report(limit=20)

        self.assertGreaterEqual(report["summary"]["sample_count"], 1)
        self.assertIn("appointment", report["summary"]["intent_distribution"])
        self.assertIn("medical_rag", report["summary"]["secondary_intent_distribution"])
        self.assertIn("rule", report["summary"]["decision_source_distribution"])
        self.assertIn("events", report)

    def test_appointment_skill_log_store_round_trip(self):
        self.vector_db.create_collection(config.CHILD_COLLECTION)
        self.appointment_skill_thread_id = f"live-skill-{uuid.uuid4().hex[:10]}"
        self.appointment_skill_log_store.save_log(
            {
                "thread_id": self.appointment_skill_thread_id,
                "skill_mode": "discover_doctor",
                "request_type": "discover_doctor",
                "selected_candidate_count": 2,
                "required_confirmation": False,
                "final_action": "discover_doctor",
                "extra_metadata": {"department": "呼吸内科"},
            }
        )

        summary = self.appointment_skill_log_store.summarize_recent(limit=20)

        self.assertGreaterEqual(summary["sample_count"], 1)
        self.assertIn("discover_doctor", summary["final_action_distribution"])

    def test_auto_index_single_markdown_with_fake_embeddings(self):
        unique_marker = f"livefollowup{uuid.uuid4().hex[:8]}"

        class SingleDocEmbeddings:
            def _vector(self, text: str):
                normalized = str(text or "").lower()
                values = [
                    float(normalized.count("慢性咳嗽")),
                    float(normalized.count(unique_marker)),
                ]
                values += [0.0] * (config.VECTOR_DIMENSION - len(values))
                return values

            def embed_documents(self, texts):
                return [self._vector(text) for text in texts]

            def embed_query(self, query):
                return self._vector(query)

        self.document_no = f"live-doc-{uuid.uuid4().hex[:8]}"
        self.temp_markdown_dir = tempfile.mkdtemp(prefix="live-markdown-")
        fixture_content = (FIXTURES_DIR / "respiratory_guidance.md").read_text(encoding="utf-8")
        markdown_path = Path(self.temp_markdown_dir) / f"{self.document_no}.md"
        markdown_path.write_text(
            fixture_content + "\n\n" + ((f"慢性咳嗽随诊建议 {unique_marker}。\n") * 250),
            encoding="utf-8",
        )

        rag = RAGSystem()
        rag.vector_db.create_collection(rag.collection_name)
        rag.vector_db.get_collection = lambda _: PgVectorCollection(rag.vector_db.conninfo, SingleDocEmbeddings())
        manager = DocumentManager(rag)
        manager.markdown_dir = Path(self.temp_markdown_dir)

        result = manager.index_existing_markdowns(skip_existing=True)
        with mock.patch("core.document_manager.config.MARKDOWN_DIR", self.temp_markdown_dir):
            rag.refresh_knowledge_base_status()
        collection = rag.vector_db.get_collection(rag.collection_name)
        matches = collection.keyword_search(unique_marker, k=5)

        self.assertEqual(result["added"], 1)
        self.assertTrue(any(self.document_no in match.metadata.get("source", "") for match in matches))
        self.assertIn(build_document_no(f"local:{self.document_no}.md"), rag.vector_db.get_indexed_document_nos())
        self.assertEqual(rag.get_knowledge_base_status()["status"], "ready")

    def test_live_qa_eval_fixture_samples_score_expected_sources(self):
        self.temp_markdown_dir = tempfile.mkdtemp(prefix="live-qa-eval-")
        fixture_names = (
            "patient_hypertension_education.md",
            "public_health_flu_prevention.md",
            "clinical_guideline_hypertension_treatment.md",
        )
        for fixture_name in fixture_names:
            source = FIXTURES_DIR / fixture_name
            target = Path(self.temp_markdown_dir) / fixture_name
            shutil.copy(source, target)
        self.document_nos = [Path(name).stem for name in fixture_names]

        rag = RAGSystem()
        rag.vector_db.create_collection(rag.collection_name)
        rag.vector_db.get_collection = lambda _: PgVectorCollection(rag.vector_db.conninfo, KeywordEmbeddings())
        manager = DocumentManager(rag)
        manager.markdown_dir = Path(self.temp_markdown_dir)

        index_result = manager.index_existing_markdowns(skip_existing=True)
        evaluator = RetrievalQualityEvaluator(rag.vector_db.get_collection(rag.collection_name), limit=3, score_threshold=0.0)
        samples = [sample for sample in load_qa_samples(FIXTURES_DIR / "qa_eval_samples.json") if not sample.expected_no_evidence]
        report = evaluator.evaluate_samples(samples)

        self.assertEqual(index_result["processed"], 3)
        self.assertEqual(index_result["added"] + index_result["skipped"], 3)
        self.assertGreaterEqual(report["summary"]["avg_retrieval_score"], 0.78)
        top_types = {item["sample_id"]: item["top_source_type"] for item in report["results"]}
        self.assertEqual(top_types["patient-hypertension-symptoms"], "patient_education")
        self.assertEqual(top_types["public-health-flu-prevention"], "public_health")
        self.assertEqual(top_types["clinical-guideline-hypertension-dosing"], "clinical_guideline")

    def test_retrieval_logs_record_search_context(self):
        self.temp_markdown_dir = tempfile.mkdtemp(prefix="live-log-eval-")
        fixture_name = "patient_hypertension_education.md"
        shutil.copy(FIXTURES_DIR / fixture_name, Path(self.temp_markdown_dir) / fixture_name)
        self.document_nos = [Path(fixture_name).stem]
        self.retrieval_log_thread_id = f"live-retrieval-{uuid.uuid4().hex[:10]}"
        request_id = f"req-{uuid.uuid4().hex[:10]}"

        rag = RAGSystem()
        rag.vector_db.create_collection(rag.collection_name)
        rag.vector_db.get_collection = lambda _: PgVectorCollection(rag.vector_db.conninfo, KeywordEmbeddings())
        manager = DocumentManager(rag)
        manager.markdown_dir = Path(self.temp_markdown_dir)
        manager.index_existing_markdowns(skip_existing=True)

        tool_factory = ToolFactory(rag.vector_db.get_collection(rag.collection_name))
        token = set_retrieval_context(
            thread_id=self.retrieval_log_thread_id,
            original_query="高血压要注意什么",
            request_id=request_id,
        )
        try:
            tool_factory._search_child_chunks("高血压 平时 要注意什么", limit=2)
        finally:
            reset_retrieval_context(token)

        with psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select request_id, query_text, rewritten_query, retrieval_mode, result_count, selected_parent_ids,
                           query_plan, graded_doc_count, sufficiency_result, retry_count, final_confidence_bucket
                    from retrieval_logs
                    where thread_id = %s
                    order by id desc
                    limit 1
                    """,
                    (self.retrieval_log_thread_id,),
                )
                row = cur.fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], request_id)
        self.assertEqual(row[1], "高血压要注意什么")
        self.assertEqual(row[2], "高血压 平时 要注意什么")
        self.assertIn("layered", row[3])
        self.assertGreaterEqual(row[4], 1)
        self.assertTrue(row[5])
        self.assertTrue(row[6])
        self.assertGreaterEqual(row[7], 1)
        self.assertTrue(row[8])
        self.assertGreaterEqual(row[9], 0)
        self.assertIn(row[10], ("high", "medium", "low", "no_evidence"))


if __name__ == "__main__":
    unittest.main()
