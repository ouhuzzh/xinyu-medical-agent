import threading

import config
from api.session_locks import SessionLockRegistry
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager
from core.rag_system import RAGSystem
from db.audit_log_store import AuditLogStore
from db.chat_session_store import ChatSessionStore
from db.schema_guard import EmbeddingSchemaGuard
from db.user_store import UserStore


class ApiContainer:
    def __init__(self):
        self.rag_system = RAGSystem()
        if config.APP_ENV == "development":
            self.rag_system.start_background_initialize()
        else:
            self.rag_system.initialize()
        self.container = self.rag_system.container
        self.chat_interface = ChatInterface(self.rag_system)
        self.document_manager = DocumentManager(self.rag_system)
        self.rag_system.document_manager = self.document_manager
        self.chat_sessions = ChatSessionStore()
        self.user_store = UserStore()
        self.audit_log = AuditLogStore()
        self.thread_locks = SessionLockRegistry(self.rag_system.session_memory)
        self.schema_guard = EmbeddingSchemaGuard()
        self.schema_guard.assert_compatible()

    def get_thread_lock(self, thread_id: str):
        return self.thread_locks.get_lock(thread_id)


_container: ApiContainer | None = None
_container_lock = threading.Lock()


def get_container() -> ApiContainer:
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = ApiContainer()
    return _container


def set_container_for_tests(container):
    global _container
    _container = container
