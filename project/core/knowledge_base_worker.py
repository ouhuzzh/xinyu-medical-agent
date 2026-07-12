"""Standalone knowledge-base worker runtime.

The API process owns interactive chat requests.  This module owns automatic
knowledge-base bootstrap and scheduled source synchronization so those jobs do
not depend on an API process or run once per API replica.
"""

from __future__ import annotations

import logging
import signal
import threading

import config


logger = logging.getLogger(__name__)


class KnowledgeBaseRuntime:
    """Minimal service container required by knowledge-base maintenance jobs."""

    def __init__(self, collection_name: str = config.CHILD_COLLECTION):
        from core.document_chunker import DocumentChuncker
        from core.document_manager import DocumentManager
        from core.knowledge_base_supervisor import KnowledgeBaseSupervisor
        from db.import_task_store import ImportTaskStore
        from db.parent_store_manager import ParentStoreManager
        from db.vector_db_manager import VectorDbManager

        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.import_task_store = ImportTaskStore()
        self.chunker = DocumentChuncker()
        self.document_manager = DocumentManager(self)
        self._startup_steps: dict[str, dict] = {}
        self.knowledge_base_supervisor = KnowledgeBaseSupervisor(self)

    def prepare(self) -> None:
        self.vector_db.create_collection(self.collection_name)
        self.knowledge_base_supervisor.refresh_status()

    def _set_startup_step(self, key: str, state: str, message: str) -> None:
        self._startup_steps[key] = {"state": state, "message": message}

    def is_ready(self) -> bool:
        return True


class KnowledgeBaseJobRunner:
    """Execute knowledge-base jobs with the existing database lock contract."""

    def __init__(self, runtime: KnowledgeBaseRuntime | None = None):
        self.runtime = runtime or KnowledgeBaseRuntime()
        self.runtime.prepare()

    def bootstrap(self) -> dict:
        from core.knowledge_base_sync import KnowledgeBaseSyncService

        sync_service = KnowledgeBaseSyncService(
            self.runtime,
            self.runtime.document_manager.markdown_dir,
        )
        lock_conn = sync_service._try_advisory_lock("knowledge_base_sync")
        if lock_conn is None:
            logger.info("Knowledge-base bootstrap skipped because another job owns the lock.")
            return {"status": "skipped_locked"}

        try:
            self.runtime.knowledge_base_supervisor._bootstrap_knowledge_base()
            return self.runtime.knowledge_base_supervisor.get_status()
        finally:
            sync_service._release_advisory_lock(lock_conn, "knowledge_base_sync")

    def sync_all(self) -> list[dict]:
        results = self.runtime.document_manager.sync_all_sources(trigger_type="scheduler")
        events = []
        for result in results:
            event = result.to_event()
            self.runtime.knowledge_base_supervisor.record_import_event(event)
            events.append(event)
        self.runtime.knowledge_base_supervisor.refresh_status()
        return events

    def close(self) -> None:
        from db.connection import close_connection_pool

        close_connection_pool()


class KnowledgeBaseWorker:
    """Long-running scheduler with dependency injection for deterministic tests."""

    def __init__(
        self,
        runner: KnowledgeBaseJobRunner | None = None,
        *,
        runner_factory=KnowledgeBaseJobRunner,
        bootstrap_on_start: bool = config.KB_WORKER_BOOTSTRAP_ON_START,
        sync_enabled: bool = config.KB_WORKER_SYNC_ENABLED,
        sync_interval_seconds: float | None = None,
    ):
        self._runner = runner
        self._runner_factory = runner_factory
        self.bootstrap_on_start = bootstrap_on_start
        self.sync_enabled = sync_enabled
        self.sync_interval_seconds = sync_interval_seconds or max(
            int(config.KB_SYNC_INTERVAL_HOURS), 1
        ) * 3600

    def _get_runner(self) -> KnowledgeBaseJobRunner:
        """Create the expensive DB/embedding runtime only when work is enabled."""
        if self._runner is None:
            self._runner = self._runner_factory()
        return self._runner

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        try:
            if self.bootstrap_on_start:
                logger.info("Running knowledge-base bootstrap in worker.")
                self._get_runner().bootstrap()

            if not self.sync_enabled:
                logger.info("Knowledge-base scheduler is disabled; worker is standing by.")
                while not stop_event.wait(3600):
                    pass
                return

            logger.info(
                "Knowledge-base scheduler started with interval_seconds=%s.",
                self.sync_interval_seconds,
            )
            while not stop_event.wait(self.sync_interval_seconds):
                try:
                    events = self._get_runner().sync_all()
                    logger.info("Knowledge-base scheduled sync completed with %s events.", len(events))
                except Exception:
                    logger.exception("Knowledge-base scheduled sync failed; next run remains scheduled.")
        finally:
            if self._runner is not None:
                self._runner.close()


def run_worker() -> None:
    logging.basicConfig(
        level=getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    stop_event = threading.Event()

    def _request_stop(signum, frame):
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    KnowledgeBaseWorker().run_forever(stop_event)
