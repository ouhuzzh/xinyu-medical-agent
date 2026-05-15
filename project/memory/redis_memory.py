import json
import logging
import config
from langchain_core.messages import HumanMessage, AIMessage

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency fallback
    redis = None


logger = logging.getLogger(__name__)


class RedisSessionMemory:
    def __init__(self):
        self._enabled = bool(redis) and config.REDIS_ENABLED
        self._client = None
        self._checked = False
        self._fallback_messages = {}
        self._fallback_state = {}
        self._degraded_reason = ""
        self._mode = "disabled" if not config.REDIS_ENABLED else "redis"
        self._require_external_redis = config.REDIS_ENABLED and config.APP_ENV != "development"

    def _messages_key(self, thread_id: str) -> str:
        return f"chat:session:{thread_id}:recent_messages"

    def _state_key(self, thread_id: str) -> str:
        return f"chat:session:{thread_id}:state"

    def _get_client(self):
        if not self._enabled:
            return None
        if self._checked:
            return self._client

        self._checked = True
        try:
            self._client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True,
            )
            self._client.ping()
            self._mode = "redis"
            self._degraded_reason = ""
        except Exception as e:
            if self._require_external_redis:
                raise RuntimeError(f"Redis is required but unavailable: {e}") from e
            logger.warning("Redis unavailable, memory cache degraded to in-process fallback: %s", e)
            self._client = None
            self._enabled = False
            self._mode = "memory_fallback"
            self._degraded_reason = str(e)
        return self._client

    def ensure_ready(self):
        if not config.REDIS_ENABLED:
            return
        self._get_client()

    def status_info(self):
        if not config.REDIS_ENABLED:
            return {
                "component": "redis_memory",
                "mode": "disabled",
                "degraded": False,
                "message": "Redis memory is disabled by configuration.",
            }
        try:
            client = self._get_client()
        except RuntimeError as exc:
            return {
                "component": "redis_memory",
                "mode": "unavailable",
                "degraded": True,
                "message": str(exc),
            }
        if client is not None:
            return {
                "component": "redis_memory",
                "mode": self._mode,
                "degraded": False,
                "message": "Redis session memory is available.",
            }
        return {
            "component": "redis_memory",
            "mode": self._mode,
            "degraded": True,
            "message": self._degraded_reason or "Redis is unavailable; using in-process fallback memory.",
        }

    @staticmethod
    def _serialize_messages(messages):
        return json.dumps(messages, ensure_ascii=False)

    @staticmethod
    def _deserialize_messages(raw):
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except Exception:
            logger.warning("Failed to deserialize Redis message history", exc_info=True)
            return []
        messages = []
        for item in data:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def get_recent_messages(self, thread_id: str):
        client = self._get_client()
        if not client:
            return self._deserialize_messages(self._fallback_messages.get(thread_id))
        raw = client.get(self._messages_key(thread_id))
        return self._deserialize_messages(raw)

    def recent_message_count(self, thread_id: str) -> int:
        return len(self.get_recent_messages(thread_id))

    def append_exchange(self, thread_id: str, user_message: str, assistant_message: str) -> int:
        client = self._get_client()
        existing = self.get_recent_messages(thread_id)
        serialized = []
        for msg in existing:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            serialized.append({"role": role, "content": msg.content})

        serialized.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )

        max_messages = max(config.SHORT_TERM_WINDOW_SIZE * 2, 2)
        serialized = serialized[-max_messages:]
        if not client:
            self._fallback_messages[thread_id] = self._serialize_messages(serialized)
            return len(serialized)
        client.setex(
            self._messages_key(thread_id),
            config.REDIS_TTL_SECONDS,
            self._serialize_messages(serialized),
        )
        return len(serialized)

    def get_state(self, thread_id: str):
        client = self._get_client()
        if not client:
            return dict(self._fallback_state.get(thread_id) or {})
        raw = client.get(self._state_key(thread_id))
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("Failed to deserialize Redis session state", exc_info=True)
            return {}

    def set_state(self, thread_id: str, state: dict):
        client = self._get_client()
        if not client:
            self._fallback_state[thread_id] = dict(state or {})
            return
        client.setex(
            self._state_key(thread_id),
            config.REDIS_TTL_SECONDS,
            json.dumps(state, ensure_ascii=False),
        )

    def clear_session(self, thread_id: str):
        client = self._get_client()
        if not client:
            self._fallback_messages.pop(thread_id, None)
            self._fallback_state.pop(thread_id, None)
            return
        client.delete(self._messages_key(thread_id), self._state_key(thread_id))
