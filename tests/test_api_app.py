import sys
import threading
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

import api.auth as auth_module  # noqa: E402
from api.app import create_app  # noqa: E402
from api.dependencies import set_container_for_tests  # noqa: E402


ADMIN_HEADERS = {"Authorization": "Bearer demo-admin-token"}
USER_HEADERS = {"Authorization": "Bearer demo-user-token"}
OTHER_USER_HEADERS = {"Authorization": "Bearer other-user-token"}


class FakeSessionMemory:
    def __init__(self):
        self.messages = {}

    def get_recent_messages(self, thread_id):
        return self.messages.get(
            thread_id,
            [
                HumanMessage(content="你好"),
                AIMessage(content=f"你好，我记得当前会话是 {thread_id}"),
            ],
        )


class FakeRagSystem:
    def __init__(self):
        self.session_memory = FakeSessionMemory()
        self.cleared = []
        self.user_mcp_pool = self

    def get_system_status(self):
        return {
            "state": "ready",
            "message": "系统已就绪。",
            "last_error": "",
            "steps": {"graph_compile": {"state": "completed"}},
            "degraded_components": [],
        }

    def get_knowledge_base_status(self):
        return {
            "status": "ready",
            "message": "知识库可检索。",
            "last_error": "",
            "stats": {
                "documents": 2,
                "child_chunks": 12,
                "recent_imports": [
                    {
                        "source": "local",
                        "label": "本地文档同步",
                        "status": "completed_with_failures",
                        "written": 1,
                        "updated": 0,
                        "deactivated": 0,
                        "unchanged": 0,
                        "failed": 1,
                        "duration_ms": 123.4,
                        "trigger_type": "manual",
                        "scope": "local",
                        "conversion_details": ["demo.txt: method=plain_text_fallback"],
                        "failure_details": ["bad.pdf: parse failed"],
                        "timestamp": "2026-04-26 12:00:00",
                    }
                ],
            },
        }

    def reset_thread(self, thread_id=None):
        self.cleared.append(thread_id)

    def record_import_event(self, event):
        self.last_import_event = event

    def refresh_knowledge_base_status(self):
        return self.get_knowledge_base_status()

    def backend_name(self):
        return "in_process"


class FakeChatInterface:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.calls = []

    def chat(self, message, history, reveal_diagnostics=False, thread_id=None):
        self.calls.append(
            {
                "message": message,
                "history": history,
                "reveal_diagnostics": reveal_diagnostics,
                "thread_id": thread_id,
            }
        )
        yield [{"role": "assistant", "content": "正在整理回答"}]
        yield [{"role": "assistant", "content": f"回答：{message}"}]

    def clear_session(self, thread_id=None):
        self.rag_system.reset_thread(thread_id)


class FakeSchemaGuard:
    def backend_name(self):
        return "postgres"

    def get_health(self):
        return {
            "status": "ok",
            "message": "Embedding vector dimensions match configuration.",
            "expected_dimension": 1024,
            "actual_dimensions": {
                "child_chunks.embedding": 1024,
                "user_memories.embedding": 1024,
                "episodic_memories.embedding": 1024,
                "reflection_memories.embedding": 1024,
            },
            "errors": [],
        }


class FakeSyncResult:
    source = "nhc"
    label = "国家卫健委同步"
    written = 1
    updated = 0
    deactivated = 0
    unchanged = 0
    status = "completed"

    def to_event(self):
        return {
            "source": self.source,
            "label": self.label,
            "status": self.status,
            "written": self.written,
            "updated": self.updated,
            "deactivated": self.deactivated,
            "unchanged": self.unchanged,
            "failed": 0,
        }


class FakeDocumentManager:
    def __init__(self, temp_dir):
        self.markdown_dir = Path(temp_dir)
        (self.markdown_dir / "guide.md").write_text("# Guide\n", encoding="utf-8")
        self.uploaded_paths = []
        self.synced = []
        self.sync_locked = False

    def get_markdown_paths(self):
        return sorted(self.markdown_dir.glob("*.md"))

    def get_document_inventory(self):
        path = self.markdown_dir / "guide.md"
        return [
            {
                "name": path.name,
                "file_type": "md",
                "size_bytes": path.stat().st_size,
                "modified_at": path.stat().st_mtime,
                "title": "Guide",
                "source_name": "本地文档",
                "source_type": "local_document",
                "source_key": "local:guide.md",
                "sync_status": "active",
                "is_active": True,
                "freshness_bucket": "current",
                "original_url": "",
            }
        ]

    def add_documents_with_report(self, paths):
        self.uploaded_paths = [Path(path).name for path in paths]
        return {
            "processed": len(paths),
            "added": len(paths),
            "updated": 0,
            "unchanged": 0,
            "deactivated": 0,
            "skipped": 0,
            "failed": 0,
            "sync_event": {
                "source": "local",
                "label": "本地文档同步",
                "status": "completed",
                "written": len(paths),
                "updated": 0,
                "deactivated": 0,
                "unchanged": 0,
                "failed": 0,
            },
        }

    def sync_official_source(self, source, limit=10, trigger_type="manual"):
        self.synced.append({"source": source, "limit": limit, "trigger_type": trigger_type})
        result = FakeSyncResult()
        if self.sync_locked:
            result.status = "skipped_locked"
        return result

    def get_official_source_coverage(self):
        return [
            {
                "source": "nhc",
                "label": "国家卫健委",
                "manifest_count": 4,
                "local_file_count": 1,
                "coverage_note": "测试覆盖度说明",
            }
        ]


class FakeChatSessionStore:
    def __init__(self):
        self.counter = 0
        self.sessions = {}

    def create_session(self, owner_user_id):
        self.counter += 1
        thread_id = f"thread-{self.counter}"
        self.sessions[thread_id] = {"thread_id": thread_id, "owner_user_id": owner_user_id, "status": "active"}
        return thread_id

    def get_session(self, thread_id):
        return self.sessions.get(thread_id)

    def assign_owner_if_missing(self, thread_id, owner_user_id):
        session = self.sessions.get(thread_id)
        if session and not session.get("owner_user_id"):
            session["owner_user_id"] = owner_user_id
            return True
        return False


class FakeContainer:
    def __init__(self, temp_dir):
        self.rag_system = FakeRagSystem()
        self.chat_interface = FakeChatInterface(self.rag_system)
        self.document_manager = FakeDocumentManager(temp_dir)
        self.chat_sessions = FakeChatSessionStore()
        self._thread_locks = {}
        self._thread_lock_guard = threading.Lock()
        self.thread_locks = self
        self.schema_guard = FakeSchemaGuard()

    def backend_name(self):
        return "in_process"

    def get_thread_lock(self, thread_id):
        with self._thread_lock_guard:
            lock = self._thread_locks.get(thread_id)
            if lock is None:
                lock = threading.Lock()
                self._thread_locks[thread_id] = lock
            return lock


class ApiAppTests(unittest.TestCase):
    def setUp(self):
        auth_module._rate_limiter = auth_module.InMemoryRateLimiter()
        auth_module._login_lockout = auth_module.LoginLockoutTracker()
        self.tmp = TemporaryDirectory()
        self.container = FakeContainer(self.tmp.name)
        self.container.rag_system.session_memory.messages["thread-existing"] = [
            HumanMessage(content="hi"),
            AIMessage(content="owned thread"),
        ]
        self.container.chat_sessions.sessions["thread-existing"] = {
            "thread_id": "thread-existing",
            "owner_user_id": "demo-user",
            "status": "active",
        }
        self.container.chat_sessions.sessions["thread-other"] = {
            "thread_id": "thread-other",
            "owner_user_id": "other-user",
            "status": "active",
        }
        set_container_for_tests(self.container)
        self.client = TestClient(create_app())

    def tearDown(self):
        set_container_for_tests(None)
        self.tmp.cleanup()

    def test_requires_bearer_token_for_api_routes(self):
        response = self.client.get("/api/system/status")

        self.assertEqual(response.status_code, 401)

    def test_invalid_bearer_token_is_rejected(self):
        response = self.client.get(
            "/api/system/status",
            headers={"Authorization": "Bearer invalid-token"},
        )

        self.assertEqual(response.status_code, 401)

    def test_regular_user_cannot_access_admin_document_routes(self):
        response = self.client.get("/api/documents/status", headers=USER_HEADERS)

        self.assertEqual(response.status_code, 403)

    def test_create_session_reuses_owned_thread_id(self):
        response = self.client.post(
            "/api/chat/session",
            json={"thread_id": "thread-existing"},
            headers=USER_HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["thread_id"], "thread-existing")

    def test_create_session_generates_new_thread_for_unowned_id(self):
        response = self.client.post(
            "/api/chat/session",
            json={"thread_id": "thread-other"},
            headers=USER_HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.json()["thread_id"], "thread-other")

    def test_system_status_includes_current_user_and_knowledge_base_status(self):
        response = self.client.get("/api/system/status", headers=ADMIN_HEADERS)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["state"], "ready")
        self.assertEqual(data["knowledge_base"]["status"], "ready")
        self.assertEqual(data["knowledge_base"]["stats"]["documents"], 2)
        self.assertEqual(data["current_user"]["role"], "admin")
        self.assertEqual(data["runtime_backends"]["session_lock_backend"], "in_process")
        self.assertIn(data["runtime_backends"]["rate_limit_backend"], {"in_process", "redis"})
        self.assertIn(data["runtime_backends"]["login_lockout_backend"], {"in_process", "redis"})
        self.assertEqual(data["runtime_backends"]["mcp_pool_backend"], "in_process")
        self.assertEqual(data["runtime_backends"]["schema_guard_backend"], "postgres")
        self.assertEqual(data["schema_health"]["status"], "ok")
        self.assertEqual(data["schema_health"]["expected_dimension"], 1024)

    def test_chat_history_returns_visible_messages_for_owner(self):
        response = self.client.get(
            "/api/chat/history",
            params={"thread_id": "thread-existing"},
            headers=USER_HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["thread_id"], "thread-existing")
        self.assertEqual([item["role"] for item in data["messages"]], ["user", "assistant"])

    def test_chat_history_blocks_other_users_thread(self):
        response = self.client.get(
            "/api/chat/history",
            params={"thread_id": "thread-other"},
            headers=USER_HEADERS,
        )

        self.assertEqual(response.status_code, 403)

    def test_clear_session_uses_requested_thread_id(self):
        response = self.client.post(
            "/api/chat/clear",
            json={"thread_id": "thread-existing"},
            headers=USER_HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.container.rag_system.cleared, ["thread-existing"])

    def test_chat_stream_emits_session_message_and_final_events(self):
        with self.client.stream(
            "POST",
            "/api/chat/stream",
            json={"thread_id": "thread-existing", "message": "高血压要注意什么"},
            headers=USER_HEADERS,
        ) as response:
            body = response.read().decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: session", body)
        self.assertIn("event: message", body)
        self.assertIn("event: final", body)
        self.assertNotIn("event: error", body)
        self.assertIn("thread-existing", body)
        self.assertEqual(self.container.chat_interface.calls[0]["thread_id"], "thread-existing")

    def test_chat_stream_blocks_other_users_thread(self):
        response = self.client.post(
            "/api/chat/stream",
            json={"thread_id": "thread-other", "message": "test"},
            headers=USER_HEADERS,
        )

        self.assertEqual(response.status_code, 403)

    def test_documents_status_list_and_tasks_are_user_facing_for_admin(self):
        status_response = self.client.get("/api/documents/status", headers=ADMIN_HEADERS)
        list_response = self.client.get("/api/documents/list", headers=ADMIN_HEADERS)
        tasks_response = self.client.get("/api/documents/tasks", headers=ADMIN_HEADERS)

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(tasks_response.status_code, 200)
        self.assertEqual(list_response.json()["documents"][0]["name"], "guide.md")
        self.assertEqual(list_response.json()["documents"][0]["source_key"], "local:guide.md")
        self.assertEqual(list_response.json()["documents"][0]["source_type"], "local_document")
        task = tasks_response.json()["tasks"][0]
        self.assertEqual(task["source"], "local")
        self.assertEqual(task["failed"], 1)
        self.assertEqual(task["conversion_details"], ["demo.txt: method=plain_text_fallback"])
        self.assertEqual(task["failure_details"], ["bad.pdf: parse failed"])
        self.assertEqual(status_response.json()["source_coverage"][0]["source"], "nhc")

    def test_documents_sources_returns_official_source_coverage(self):
        response = self.client.get("/api/documents/sources", headers=ADMIN_HEADERS)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"][0]["manifest_count"], 4)

    def test_documents_upload_records_import_event(self):
        response = self.client.post(
            "/api/documents/upload",
            files=[("files", ("new-guide.md", b"# New Guide\n", "text/markdown"))],
            headers=ADMIN_HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("已处理 1 个文件", data["message"])
        self.assertEqual(self.container.document_manager.uploaded_paths, ["new-guide.md"])
        self.assertEqual(self.container.rag_system.last_import_event["source"], "local")

    def test_documents_upload_rejects_too_many_files(self):
        files = [
            ("files", (f"file-{index}.md", b"# Demo\n", "text/markdown"))
            for index in range(6)
        ]
        response = self.client.post("/api/documents/upload", files=files, headers=ADMIN_HEADERS)

        self.assertEqual(response.status_code, 400)
        self.assertIn("单次最多上传", response.json()["detail"])

    def test_documents_upload_rejects_unsupported_extension(self):
        response = self.client.post(
            "/api/documents/upload",
            files=[("files", ("payload.exe", b"nope", "application/octet-stream"))],
            headers=ADMIN_HEADERS,
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("不支持的文件类型", response.json()["detail"])

    def test_documents_sync_official_uses_document_manager(self):
        response = self.client.post(
            "/api/documents/sync-official",
            json={"source": "nhc", "limit": 2},
            headers=ADMIN_HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("官方同步完成", response.json()["message"])
        self.assertEqual(
            self.container.document_manager.synced,
            [{"source": "nhc", "limit": 2, "trigger_type": "manual"}],
        )

    def test_documents_sync_returns_conflict_when_locked(self):
        self.container.document_manager.sync_locked = True

        response = self.client.post(
            "/api/documents/sync-official",
            json={"source": "nhc", "limit": 2},
            headers=ADMIN_HEADERS,
        )

        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
