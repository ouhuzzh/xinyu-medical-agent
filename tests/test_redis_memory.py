import sys
import unittest
from unittest import mock

sys.path.insert(0, r"D:\nageoffer\agentic-rag-for-dummies\project")

import config  # noqa: E402
from memory import redis_memory as redis_memory_module  # noqa: E402
from memory.redis_memory import RedisSessionMemory  # noqa: E402


class _FailingRedisClient:
    def ping(self):
        raise RuntimeError("redis down")


class _FailingRedisModule:
    @staticmethod
    def Redis(*args, **kwargs):
        return _FailingRedisClient()


class RedisSessionMemoryFallbackTests(unittest.TestCase):
    def test_in_process_fallback_keeps_recent_messages_and_state(self):
        memory = RedisSessionMemory()
        memory._enabled = False
        thread_id = "thread-fallback"

        count = memory.append_exchange(thread_id, "你好", "你好，我在。")
        memory.set_state(thread_id, {"intent": "medical_rag", "recommended_department": "全科医学科"})

        messages = memory.get_recent_messages(thread_id)
        state = memory.get_state(thread_id)

        self.assertEqual(count, 2)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "你好")
        self.assertEqual(messages[1].content, "你好，我在。")
        self.assertEqual(state["recommended_department"], "全科医学科")

    def test_clear_session_removes_fallback_data(self):
        memory = RedisSessionMemory()
        memory._enabled = False
        thread_id = "thread-clear"
        memory.append_exchange(thread_id, "A", "B")
        memory.set_state(thread_id, {"intent": "appointment"})

        memory.clear_session(thread_id)

        self.assertEqual(memory.get_recent_messages(thread_id), [])
        self.assertEqual(memory.get_state(thread_id), {})

    def test_status_info_marks_degraded_in_development_when_redis_is_down(self):
        with mock.patch.object(redis_memory_module, "redis", _FailingRedisModule), \
                mock.patch.object(config, "APP_ENV", "development"), \
                mock.patch.object(config, "REDIS_ENABLED", True):
            memory = RedisSessionMemory()

            status = memory.status_info()

        self.assertTrue(status["degraded"])
        self.assertEqual(status["mode"], "memory_fallback")

    def test_ensure_ready_fails_outside_development_when_redis_is_down(self):
        with mock.patch.object(redis_memory_module, "redis", _FailingRedisModule), \
                mock.patch.object(config, "APP_ENV", "production"), \
                mock.patch.object(config, "REDIS_ENABLED", True):
            memory = RedisSessionMemory()

            with self.assertRaises(RuntimeError):
                memory.ensure_ready()


if __name__ == "__main__":
    unittest.main()
