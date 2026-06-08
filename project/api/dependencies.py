import threading

import config
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager
from core.rag_system import RAGSystem
from db.audit_log_store import AuditLogStore
from db.chat_session_store import ChatSessionStore
from db.user_store import UserStore


class ThreadLockRegistry:
    def __init__(self):
        self._locks = {}
        self._guard = threading.Lock()

    def get_lock(self, thread_id: str):
        with self._guard:
            lock = self._locks.get(thread_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[thread_id] = lock
            return lock


class ApiContainer:
    def __init__(self):
        self.rag_system = RAGSystem()
        if config.APP_ENV == "development":
            self.rag_system.start_background_initialize()
        else:
            self.rag_system.initialize()
        self.chat_interface = ChatInterface(self.rag_system)
        self.document_manager = DocumentManager(self.rag_system)
        self.chat_sessions = ChatSessionStore()
        self.user_store = UserStore()
        self.audit_log = AuditLogStore()
        self.thread_locks = ThreadLockRegistry()

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

