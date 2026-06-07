"""Memory observability routes — view and manage user memories."""

from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import AuthenticatedUser, require_current_user
from api.dependencies import get_container
from pydantic import BaseModel, Field


router = APIRouter()


class MemoryItemResponse(BaseModel):
    memory_type: str
    content: str
    importance: int
    source_thread_id: str = ""


class MyMemoriesResponse(BaseModel):
    memories: list[MemoryItemResponse] = Field(default_factory=list)
    total: int = 0


@router.get("/api/memory/my-memories", response_model=MyMemoriesResponse)
def my_memories(
    request: Request,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    """Return all memories for the currently authenticated user."""
    request.state.route_type = "memory_list"
    container = get_container()

    user_id = current_user.username or current_user.user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="无法识别当前用户。")

    try:
        all_memories = container.rag_system.user_memory_store.get_memories_for_user(user_id)
    except Exception:
        raise HTTPException(status_code=500, detail="读取记忆失败，请稍后再试。")

    result = []
    for m in all_memories:
        result.append(MemoryItemResponse(
            memory_type=m.get("memory_type", "fact"),
            content=m.get("content", ""),
            importance=m.get("importance", 5),
            source_thread_id=m.get("source_thread_id", "") or "",
        ))

    return MyMemoriesResponse(memories=result, total=len(result))


@router.delete("/api/memory/{memory_id}")
def delete_memory(
    memory_id: int,
    request: Request,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    """Delete a specific memory by ID (must belong to current user)."""
    request.state.route_type = "memory_delete"
    container = get_container()

    user_id = current_user.username or current_user.user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="无法识别当前用户。")

    # Verify ownership before deleting
    all_memories = container.rag_system.user_memory_store.get_memories_for_user(user_id)
    owned_ids = {m.get("id") for m in all_memories}
    if memory_id not in owned_ids:
        raise HTTPException(status_code=404, detail="记忆不存在或不属于当前用户。")

    try:
        # Soft-delete: set importance to 0
        container.rag_system.user_memory_store._update_importance(memory_id, 0)
    except Exception:
        raise HTTPException(status_code=500, detail="删除记忆失败，请稍后再试。")

    return {"deleted": True, "memory_id": memory_id}
