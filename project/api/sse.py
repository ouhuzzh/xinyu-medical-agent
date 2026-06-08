import logging
from typing import Iterable

from api.dependencies import get_container
from api.schemas import ChatSseEvent


logger = logging.getLogger(__name__)


def event_payload(event: ChatSseEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def visible_assistant_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk.strip()
    if not isinstance(chunk, list):
        return ""
    for item in reversed(chunk):
        if not isinstance(item, dict):
            continue
        if item.get("role") != "assistant":
            continue
        content = str(item.get("content") or "").strip()
        if content:
            return content
    return ""


def stream_chat_events(thread_id: str, message: str) -> Iterable[str]:
    container = get_container()
    final_content = ""
    yield event_payload(ChatSseEvent(type="session", thread_id=thread_id, content=thread_id))
    yield event_payload(ChatSseEvent(type="status", thread_id=thread_id, content="thinking"))
    lock = container.get_thread_lock(thread_id)
    acquired = lock.acquire(timeout=120)
    if not acquired:
        yield event_payload(
            ChatSseEvent(
                type="app-error",
                thread_id=thread_id,
                content="会话繁忙，请稍后再试。",
                error="thread_lock_timeout",
                done=True,
            )
        )
        return
    try:
        for chunk in container.chat_interface.chat(
            message,
            [],
            reveal_diagnostics=False,
            thread_id=thread_id,
        ):
            content = visible_assistant_text(chunk)
            if not content:
                continue
            final_content = content
            yield event_payload(ChatSseEvent(type="message", thread_id=thread_id, content=content))
    except Exception as exc:
        logger.exception("API chat stream failed for thread_id=%s", thread_id)
        yield event_payload(
            ChatSseEvent(
                type="app-error",
                thread_id=thread_id,
                content="聊天服务暂时不可用，请稍后再试。",
                error=str(exc),
                done=True,
            )
        )
    else:
        yield event_payload(ChatSseEvent(type="final", thread_id=thread_id, content=final_content, done=True))
    finally:
        lock.release()
