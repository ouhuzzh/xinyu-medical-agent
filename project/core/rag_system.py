"""RAG system orchestrator — manages the full lifecycle of the agentic RAG pipeline.

Responsibilities:
    - Initialize database schema, LLM models, vector collections, and tools
    - Compile the LangGraph agent graph with checkpointer
    - Bootstrap the knowledge base from markdown_docs/
    - Schedule periodic KB sync from official sources (MedlinePlus, NHC, WHO)
    - Track startup status and KB status for health monitoring
    - Provide thread-level checkpoint reset
"""

import uuid
import logging
import threading
import time
from datetime import datetime
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from db.import_task_store import ImportTaskStore
from core.document_chunker import DocumentChuncker
from memory.redis_memory import RedisSessionMemory
from memory.summary_store import SummaryStore
from model_factory import get_chat_model
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from core.observability import Observability
from services.appointment_service import AppointmentService


logger = logging.getLogger(__name__)

class RAGSystem:

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.import_task_store = ImportTaskStore()
        self.chunker = DocumentChuncker()
        self.session_memory = RedisSessionMemory()
        self.summary_store = SummaryStore()
        self.appointment_service = AppointmentService()
        self.observability = Observability()
        self.document_manager = None
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT
        self._initialize_lock = threading.Lock()
        self._bootstrap_lock = threading.Lock()
        self._bootstrap_thread = None
        self._sync_thread = None
        self._initialize_thread = None
        self._startup_status = {
            "state": "not_started",
            "message": "等待系统初始化。",
            "last_error": "",
            "steps": {},
        }
        self._knowledge_base_status = {
            "status": "not_checked",
            "message": "尚未检查知识库状态。",
            "last_error": "",
            "stats": {
                "local_markdown_files": 0,
                "documents": 0,
                "inactive_documents": 0,
                "parent_chunks": 0,
                "child_chunks": 0,
                "last_bootstrap_result": "",
                "last_sync_result": "",
                "recent_imports": [],
            },
        }

    def _set_startup_step(self, key, state, message):
        step = self._startup_status["steps"].get(key, {})
        if state == "running":
            step["started_at"] = time.perf_counter()
        elif step.get("started_at") is not None:
            step["elapsed_ms"] = round((time.perf_counter() - step["started_at"]) * 1000, 2)
        step["state"] = state
        step["message"] = message
        self._startup_status["steps"][key] = step

    def _set_startup_status(self, state, message, last_error=""):
        self._startup_status["state"] = state
        self._startup_status["message"] = message
        self._startup_status["last_error"] = last_error

    def get_system_status(self):
        degraded_components = []
        memory_status = self.session_memory.status_info()
        if memory_status.get("degraded"):
            degraded_components.append(memory_status["component"])
        return {
            "state": self._startup_status["state"],
            "message": self._startup_status["message"],
            "last_error": self._startup_status["last_error"],
            "steps": {key: value.copy() for key, value in self._startup_status["steps"].items()},
            "degraded_components": degraded_components,
        }

    def get_knowledge_base_status(self):
        recent_imports = list(self._knowledge_base_status["stats"].get("recent_imports", []))
        if not recent_imports:
            import_task_store = getattr(self, "import_task_store", None)
            if import_task_store is None:
                return {
                    "status": self._knowledge_base_status["status"],
                    "message": self._knowledge_base_status["message"],
                    "last_error": self._knowledge_base_status["last_error"],
                    "stats": {
                        **dict(self._knowledge_base_status["stats"]),
                        "recent_imports": [],
                    },
                }
            try:
                recent_imports = import_task_store.list_recent(config.RECENT_IMPORT_TASK_LIMIT)
                self._knowledge_base_status["stats"]["recent_imports"] = recent_imports
            except Exception:
                logger.warning("Failed to load recent import tasks", exc_info=True)
                recent_imports = []
        return {
            "status": self._knowledge_base_status["status"],
            "message": self._knowledge_base_status["message"],
            "last_error": self._knowledge_base_status["last_error"],
            "stats": {
                **dict(self._knowledge_base_status["stats"]),
                "recent_imports": recent_imports,
            },
        }

    def is_ready(self):
        return self.agent_graph is not None and self._startup_status["state"] == "ready"

    def get_readiness_message(self):
        status = self.get_system_status()
        if status["state"] == "failed":
            detail = f" 失败原因：{status['last_error']}" if status["last_error"] else ""
            return f"系统初始化失败，暂时无法处理请求。{detail}"
        if status["state"] != "ready":
            return f"系统正在准备中：{status['message']}"
        return ""

    def _update_knowledge_base_status(self, status, message, *, last_error=None, stats=None):
        self._knowledge_base_status["status"] = status
        self._knowledge_base_status["message"] = message
        if last_error is not None:
            self._knowledge_base_status["last_error"] = last_error
        if stats:
            if "recent_imports" not in stats:
                stats["recent_imports"] = list(self._knowledge_base_status["stats"].get("recent_imports", []))
            self._knowledge_base_status["stats"].update(stats)

    def record_import_event(self, event: dict):
        history = list(self._knowledge_base_status["stats"].get("recent_imports", []))
        payload = dict(event)
        payload.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        try:
            self.import_task_store.save_event(payload)
        except Exception:
            logger.warning("Failed to persist import event", exc_info=True)
            pass
        history.insert(0, payload)
        self._knowledge_base_status["stats"]["recent_imports"] = history[:config.RECENT_IMPORT_TASK_LIMIT]
        scope = payload.get("scope") or payload.get("source") or ""
        self._knowledge_base_status["stats"]["last_sync_result"] = (
            f"{payload.get('label', scope)} | 新增 {payload.get('written', 0)} | "
            f"更新 {payload.get('updated', 0)} | 下线 {payload.get('deactivated', 0)} | "
            f"未变化 {payload.get('unchanged', 0)}"
        )

    def refresh_knowledge_base_status(self):
        from core.document_manager import DocumentManager

        doc_manager = getattr(self, "document_manager", None) or DocumentManager(self)
        local_stats = doc_manager.get_local_document_stats()
        db_stats = self.vector_db.get_collection_stats()
        stats = {
            **local_stats,
            **db_stats,
            "last_bootstrap_result": self._knowledge_base_status["stats"].get("last_bootstrap_result", ""),
            "last_sync_result": self._knowledge_base_status["stats"].get("last_sync_result", ""),
            "recent_imports": list(self._knowledge_base_status["stats"].get("recent_imports", [])),
        }
        import_task_store = getattr(self, "import_task_store", None)
        if import_task_store is not None:
            try:
                stats["recent_imports"] = import_task_store.list_recent(config.RECENT_IMPORT_TASK_LIMIT)
            except Exception:
                logger.warning("Failed to refresh recent import tasks", exc_info=True)

        current_status = self._knowledge_base_status["status"]
        if current_status == "building":
            self._knowledge_base_status["stats"].update(stats)
            return self.get_knowledge_base_status()

        if stats["local_markdown_files"] == 0 and stats["child_chunks"] == 0:
            self._update_knowledge_base_status("no_documents", "尚无可索引文档。", stats=stats, last_error="")
        elif stats["documents"] < stats["local_markdown_files"]:
            self._update_knowledge_base_status("pending_rebuild", "检测到本地文档未完成索引，等待后台补建。", stats=stats, last_error="")
        elif stats["child_chunks"] > 0:
            self._update_knowledge_base_status("ready", "知识库可检索。", stats=stats, last_error="")
        else:
            self._update_knowledge_base_status("pending_rebuild", "知识库缺少可检索内容，等待后台补建。", stats=stats, last_error="")
        return self.get_knowledge_base_status()

    def _bootstrap_knowledge_base(self):
        from core.document_manager import DocumentManager

        with self._bootstrap_lock:
            doc_manager = getattr(self, "document_manager", None) or DocumentManager(self)
            local_stats = doc_manager.get_local_document_stats()
            if local_stats["local_markdown_files"] == 0:
                self._set_startup_step("knowledge_base_bootstrap", "completed", "当前没有本地文档，无需补建。")
                self._update_knowledge_base_status(
                    "no_documents",
                    "尚无可索引文档。",
                    stats={**local_stats, **self.vector_db.get_collection_stats()},
                    last_error="",
                )
                return

            self._update_knowledge_base_status(
                "building",
                "正在后台补建知识库索引，请稍候。",
                stats={**local_stats, **self.vector_db.get_collection_stats()},
                last_error="",
            )
            self._set_startup_step("knowledge_base_bootstrap", "running", "正在后台补建知识库。")
            try:
                result = doc_manager.index_existing_markdowns(skip_existing=True)
                self.refresh_knowledge_base_status()
                self._knowledge_base_status["stats"]["last_bootstrap_result"] = (
                    f"processed={result['processed']}, added={result['added']}, skipped={result['skipped']}"
                )
                if self._knowledge_base_status["status"] == "ready":
                    self._knowledge_base_status["message"] = "知识库已完成后台补建，可正常检索。"
                self._set_startup_step("knowledge_base_bootstrap", "completed", "知识库后台补建完成。")
            except Exception as exc:
                logger.exception("Knowledge base bootstrap failed")
                self._set_startup_step("knowledge_base_bootstrap", "failed", f"知识库后台补建失败：{exc}")
                self._update_knowledge_base_status(
                    "failed",
                    "知识库后台补建失败。",
                    stats={**local_stats, **self.vector_db.get_collection_stats()},
                    last_error=str(exc),
                )

    def start_knowledge_base_bootstrap(self):
        if not config.AUTO_BOOTSTRAP_KNOWLEDGE_BASE:
            return
        status = self.refresh_knowledge_base_status()
        if status["status"] not in {"pending_rebuild"}:
            return
        if self._bootstrap_thread and self._bootstrap_thread.is_alive():
            return
        self._bootstrap_thread = threading.Thread(
            target=self._bootstrap_knowledge_base,
            name="kb-bootstrap",
            daemon=True,
        )
        self._bootstrap_thread.start()

    def _knowledge_base_sync_loop(self):
        interval_seconds = max(int(config.KB_SYNC_INTERVAL_HOURS), 1) * 3600
        while True:
            time.sleep(interval_seconds)
            try:
                doc_manager = getattr(self, "document_manager", None)
                if doc_manager is None:
                    from core.document_manager import DocumentManager

                    doc_manager = DocumentManager(self)
                results = doc_manager.sync_all_sources(trigger_type="scheduler")
                for result in results:
                    self.record_import_event(result.to_event())
                self.refresh_knowledge_base_status()
            except Exception as exc:
                logger.exception("Scheduled knowledge base sync failed")
                self._knowledge_base_status["last_error"] = str(exc)
                self._knowledge_base_status["stats"]["last_sync_result"] = f"后台同步失败：{exc}"

    def start_knowledge_base_sync_scheduler(self):
        if not config.ENABLE_KB_SYNC_SCHEDULER:
            return
        if not self.is_ready():
            return
        if self._sync_thread and self._sync_thread.is_alive():
            return
        self._sync_thread = threading.Thread(
            target=self._knowledge_base_sync_loop,
            name="kb-sync-scheduler",
            daemon=True,
        )
        self._sync_thread.start()

    def start_background_initialize(self):
        if self._initialize_thread and self._initialize_thread.is_alive():
            return
        if self.is_ready():
            return
        self._initialize_thread = threading.Thread(
            target=self.initialize,
            name="rag-system-init",
            daemon=True,
        )
        self._initialize_thread.start()

    def initialize(self):
        with self._initialize_lock:
            if self.agent_graph is not None:
                return

            self._set_startup_status("preparing", "正在检查数据库与模型依赖。")

            try:
                self._set_startup_step("database_check", "running", "检查数据库 schema 和索引。")
                self.vector_db.create_collection(self.collection_name)
                self.session_memory.ensure_ready()
                self.refresh_knowledge_base_status()
                self._set_startup_step("database_check", "completed", "数据库 schema 检查完成。")

                self._set_startup_step("model_init", "running", "初始化聊天模型。")
                from llm_tiered_router import TieredLLMRouter
                llm_router = TieredLLMRouter.from_env()
                llm = llm_router.get_llm("default")
                self._set_startup_step("model_init", "completed", "聊天模型初始化完成。")

                self._set_startup_step("graph_compile", "running", "构建代理图。")
                collection = self.vector_db.get_collection(self.collection_name)
                tools = ToolFactory(collection).create_tools()

                # Register skills (if enabled)
                if getattr(config, "SKILLS_ENABLED", False):
                    from skills.registry import get_skill_registry
                    from skills.greeting_skill import GreetingSkill
                    from skills.medical_rag_skill import MedicalRagSkill
                    registry = get_skill_registry()
                    registry.register(GreetingSkill())
                    registry.register(MedicalRagSkill())
                    logger.info("Skill plugin framework enabled: %d skills registered", len(registry.skills))

                self.agent_graph = create_agent_graph(llm, tools, appointment_service=self.appointment_service, llm_router=llm_router)
                self._set_startup_step("graph_compile", "completed", "代理图已就绪。")

                self._set_startup_step("knowledge_base_bootstrap", "completed", "知识库状态检查完成。")
                self._set_startup_status("ready", "系统已就绪。")
            except Exception as exc:
                logger.exception("RAG system initialization failed")
                self._set_startup_status("failed", "系统初始化失败。", last_error=str(exc))
                for key in ("database_check", "model_init", "graph_compile"):
                    step = self._startup_status["steps"].get(key)
                    if step and step.get("state") == "running":
                        self._set_startup_step(key, "failed", f"失败：{exc}")
                raise

    def get_config(self, thread_id=None):
        cfg = {"configurable": {"thread_id": thread_id or self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self, thread_id=None):
        old_thread_id = thread_id or self.thread_id
        try:
            self.agent_graph.checkpointer.delete_thread(old_thread_id)
        except Exception as e:
            logger.warning("Could not delete thread %s: %s", old_thread_id, e)
        self.session_memory.clear_session(old_thread_id)
        self.summary_store.clear_session(old_thread_id)
        if thread_id is None:
            self.thread_id = str(uuid.uuid4())
