from __future__ import annotations
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class CreateSessionRequest(BaseModel):
    thread_id: str | None = None


class CreateSessionResponse(BaseModel):
    thread_id: str


class ChatHistoryResponse(BaseModel):
    thread_id: str
    messages: list[ChatMessage] = Field(default_factory=list)


class ClearSessionRequest(BaseModel):
    thread_id: str


class ClearSessionResponse(BaseModel):
    thread_id: str
    cleared: bool = True


class ChatStreamRequest(BaseModel):
    thread_id: str
    message: str


class KnowledgeBaseStatusResponse(BaseModel):
    status: str
    message: str
    last_error: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)


class CurrentUserResponse(BaseModel):
    user_id: str
    role: Literal["user", "admin"]
    username: str = ""


class SystemStatusResponse(BaseModel):
    state: str
    message: str
    last_error: str = ""
    steps: dict[str, Any] = Field(default_factory=dict)
    degraded_components: list[str] = Field(default_factory=list)
    runtime_backends: dict[str, str] = Field(default_factory=dict)
    schema_health: dict[str, Any] = Field(default_factory=dict)
    current_user: CurrentUserResponse
    knowledge_base: KnowledgeBaseStatusResponse


class ChatSseEvent(BaseModel):
    type: Literal["session", "status", "message", "final", "app-error"]
    thread_id: str
    content: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    done: bool = False
    error: str = ""


class DocumentItem(BaseModel):
    name: str
    file_type: str = "md"
    size_bytes: int = 0
    modified_at: str = ""
    title: str = ""
    source_name: str = ""
    source_type: str = ""
    source_key: str = ""
    sync_status: str = ""
    is_active: bool = True
    freshness_bucket: str = ""
    original_url: str = ""


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem] = Field(default_factory=list)


class DocumentTaskItem(BaseModel):
    source: str = ""
    label: str = ""
    status: str = "completed"
    timestamp: str = ""
    downloaded: int = 0
    written: int = 0
    updated: int = 0
    deactivated: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    index_added: int = 0
    index_skipped: int = 0
    duration_ms: float = 0
    note: str = ""
    trigger_type: str = "manual"
    scope: str = ""
    conversion_details: list[str] = Field(default_factory=list)
    failure_details: list[str] = Field(default_factory=list)


class DocumentTaskListResponse(BaseModel):
    tasks: list[DocumentTaskItem] = Field(default_factory=list)


class DocumentSourceCoverageResponse(BaseModel):
    sources: list[dict[str, Any]] = Field(default_factory=list)


class DocumentStatusResponse(BaseModel):
    knowledge_base: KnowledgeBaseStatusResponse
    recent_tasks: list[DocumentTaskItem] = Field(default_factory=list)
    source_coverage: list[dict[str, Any]] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    message: str
    report: dict[str, Any] = Field(default_factory=dict)


class OfficialSyncRequest(BaseModel):
    source: Literal["medlineplus", "nhc", "who"]
    limit: int = Field(default=10, ge=1, le=50)


class OfficialSyncResponse(BaseModel):
    message: str
    result: dict[str, Any] = Field(default_factory=dict)


# --- Auth schemas ---

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64, description="用户名，支持中英文、数字、下划线")
    password: str = Field(..., min_length=6, description="密码")
    display_name: str = Field("", max_length=128, description="显示名称")

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class RefreshRequest(BaseModel):
    refresh_token: str

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=6)

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class UserProfileResponse(BaseModel):
    user_id: str
    username: str
    display_name: str
    role: str
