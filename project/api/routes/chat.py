from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
from api.auth import (
    AuthenticatedUser,
    enforce_chat_rate_limit,
    ensure_owned_session,
    require_current_user,
)
from api.dependencies import get_container
from api.schemas import (
    ChatHistoryResponse,
    ChatMessage,
    ChatSessionItem,
    ChatSessionListResponse,
    ChatStreamRequest,
    ClearSessionRequest,
    ClearSessionResponse,
    CompressSessionRequest,
    CompressSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    DeleteSessionRequest,
    DeleteSessionResponse,
    RenameSessionRequest,
    RenameSessionResponse,
)
from api.sse import stream_chat_events


router = APIRouter()


def _message_from_langchain(message) -> ChatMessage | None:
    content = str(getattr(message, "content", "") or "").strip()
    if not content:
        return None
    if isinstance(message, HumanMessage):
        return ChatMessage(role="user", content=content)
    if isinstance(message, AIMessage):
        return ChatMessage(role="assistant", content=content)
    if isinstance(message, SystemMessage):
        return ChatMessage(role="system", content=content)
    return None


def _format_timestamp(value) -> str:
    if value is None:
        return ""
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat(timespec="seconds")
    return str(value)


def _session_visible_messages(container, thread_id: str) -> list[ChatMessage]:
    messages = []
    try:
        for item in container.rag_system.session_memory.get_recent_messages(thread_id):
            converted = _message_from_langchain(item)
            if converted:
                messages.append(converted)
    except Exception:
        return []
    return messages


def _session_is_empty(container, thread_id: str) -> bool:
    return not _session_visible_messages(container, thread_id)


def _session_title(container, session: dict) -> str:
    explicit_title = str(session.get("title") or "").strip()
    if explicit_title:
        return explicit_title
    for converted in _session_visible_messages(container, session["thread_id"]):
        if converted.role == "user":
            title = converted.content.strip()
            return title[:32] + ("..." if len(title) > 32 else "")
    return "新会话"


def _find_reusable_empty_session(container, user_id: str) -> str:
    for item in container.chat_sessions.list_sessions(user_id, limit=100):
        if str(item.get("title") or "").strip():
            continue
        if _session_is_empty(container, item["thread_id"]):
            return item["thread_id"]
    return ""


@router.post("/api/chat/session", response_model=CreateSessionResponse)
def create_session(
    request: Request,
    payload: CreateSessionRequest | None = Body(default=None),
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "chat_session_create"
    container = get_container()
    existing_thread_id = (payload.thread_id if payload else None) or ""
    if existing_thread_id:
        session = container.chat_sessions.get_session(existing_thread_id)
        if session and session.get("owner_user_id") == current_user.user_id and session.get("status") == "active":
            thread_id = existing_thread_id
        elif session and not session.get("owner_user_id"):
            container.chat_sessions.assign_owner_if_missing(existing_thread_id, current_user.user_id)
            thread_id = existing_thread_id
        else:
            thread_id = container.chat_sessions.create_session(current_user.user_id)
    else:
        thread_id = _find_reusable_empty_session(container, current_user.user_id)
        if not thread_id:
            thread_id = container.chat_sessions.create_session(current_user.user_id)
    request.state.thread_id = thread_id
    return CreateSessionResponse(thread_id=thread_id)


@router.get("/api/chat/sessions", response_model=ChatSessionListResponse)
def list_sessions(
    request: Request,
    limit: int = Query(default=30, ge=1, le=100),
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "chat_session_list"
    container = get_container()
    sessions = container.chat_sessions.list_sessions(current_user.user_id, limit=limit)
    visible_sessions = []
    has_empty_session = False
    for item in sessions:
        is_untitled_empty = not str(item.get("title") or "").strip() and _session_is_empty(container, item["thread_id"])
        if is_untitled_empty:
            if has_empty_session:
                continue
            has_empty_session = True
        visible_sessions.append(item)
    return ChatSessionListResponse(
        sessions=[
            ChatSessionItem(
                thread_id=item["thread_id"],
                title=_session_title(container, item),
                status=item.get("status") or "active",
                created_at=_format_timestamp(item.get("created_at")),
                updated_at=_format_timestamp(item.get("updated_at")),
            )
            for item in visible_sessions
        ]
    )


@router.post("/api/chat/session/rename", response_model=RenameSessionResponse)
def rename_session(
    request: Request,
    payload: RenameSessionRequest,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "chat_session_rename"
    request.state.thread_id = payload.thread_id
    ensure_owned_session(payload.thread_id, current_user)
    container = get_container()
    title = payload.title.strip()
    if not container.chat_sessions.update_session_title(payload.thread_id, current_user.user_id, title):
        raise HTTPException(status_code=404, detail="会话不存在或不可修改。")
    return RenameSessionResponse(thread_id=payload.thread_id, title=title)


@router.post("/api/chat/session/delete", response_model=DeleteSessionResponse)
def delete_session(
    request: Request,
    payload: DeleteSessionRequest,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "chat_session_delete"
    request.state.thread_id = payload.thread_id
    ensure_owned_session(payload.thread_id, current_user)
    container = get_container()
    if not container.chat_sessions.archive_session(payload.thread_id, current_user.user_id):
        raise HTTPException(status_code=404, detail="会话不存在或不可删除。")
    return DeleteSessionResponse(thread_id=payload.thread_id)


@router.get("/api/chat/history", response_model=ChatHistoryResponse)
def chat_history(
    request: Request,
    thread_id: str = Query(..., min_length=1),
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "chat_history"
    request.state.thread_id = thread_id
    container = get_container()
    ensure_owned_session(thread_id, current_user)
    messages = []
    messages = _session_visible_messages(container, thread_id)
    return ChatHistoryResponse(thread_id=thread_id, messages=messages)


@router.post("/api/chat/clear", response_model=ClearSessionResponse)
def clear_chat(
    request: Request,
    payload: ClearSessionRequest,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "chat_clear"
    request.state.thread_id = payload.thread_id
    container = get_container()
    ensure_owned_session(payload.thread_id, current_user)
    container.chat_interface.clear_session(payload.thread_id)
    touch_session = getattr(container.chat_sessions, "touch_session", None)
    if callable(touch_session):
        touch_session(payload.thread_id)
    return ClearSessionResponse(thread_id=payload.thread_id)


@router.post("/api/chat/compress", response_model=CompressSessionResponse)
def compress_chat(
    request: Request,
    payload: CompressSessionRequest,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "chat_compress"
    request.state.thread_id = payload.thread_id
    ensure_owned_session(payload.thread_id, current_user)
    container = get_container()

    from core.context_compression import ContextCompressionService

    service = ContextCompressionService()
    result = service.compress_thread(
        session_memory=container.rag_system.session_memory,
        summary_store=container.rag_system.summary_store,
        thread_id=payload.thread_id,
        preserve_recent_turns=config.RECENT_CONTEXT_TURNS,
    )

    touch_session = getattr(container.chat_sessions, "touch_session", None)
    if callable(touch_session):
        touch_session(payload.thread_id)

    return CompressSessionResponse(
        thread_id=payload.thread_id,
        compressed=result["compressed"],
        preserved_count=result["preserved_count"],
        summary_length=result["summary_length"],
    )


@router.post("/api/chat/stream")
def chat_stream_post(
    request: Request,
    payload: ChatStreamRequest,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "chat_stream"
    request.state.thread_id = payload.thread_id
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    ensure_owned_session(payload.thread_id, current_user)
    enforce_chat_rate_limit(current_user)
    container = get_container()
    touch_session = getattr(container.chat_sessions, "touch_session", None)
    if callable(touch_session):
        touch_session(payload.thread_id)
    return StreamingResponse(
        stream_chat_events(payload.thread_id, payload.message.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
