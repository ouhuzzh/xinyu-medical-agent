from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
    ChatStreamRequest,
    ClearSessionRequest,
    ClearSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
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
        if session and session.get("owner_user_id") == current_user.user_id:
            thread_id = existing_thread_id
        elif session and not session.get("owner_user_id"):
            container.chat_sessions.assign_owner_if_missing(existing_thread_id, current_user.user_id)
            thread_id = existing_thread_id
        else:
            thread_id = container.chat_sessions.create_session(current_user.user_id)
    else:
        thread_id = container.chat_sessions.create_session(current_user.user_id)
    request.state.thread_id = thread_id
    return CreateSessionResponse(thread_id=thread_id)


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
    for item in container.rag_system.session_memory.get_recent_messages(thread_id):
        converted = _message_from_langchain(item)
        if converted:
            messages.append(converted)
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
    return ClearSessionResponse(thread_id=payload.thread_id)


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
    return StreamingResponse(
        stream_chat_events(payload.thread_id, payload.message.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
