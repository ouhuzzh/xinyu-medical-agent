from fastapi import APIRouter, Depends, Request

from api.auth import AuthenticatedUser, require_current_user
from api.dependencies import get_container
from api.schemas import CurrentUserResponse, KnowledgeBaseStatusResponse, SystemStatusResponse


router = APIRouter()


@router.get("/api/health")
def health(
    request: Request,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "health"
    return {"ok": True, "user_id": current_user.user_id, "role": current_user.role}


@router.get("/api/system/status", response_model=SystemStatusResponse)
def system_status(
    request: Request,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "system_status"
    container = get_container()
    system = container.rag_system.get_system_status()
    knowledge = container.rag_system.get_knowledge_base_status()
    return SystemStatusResponse(
        state=system["state"],
        message=system["message"],
        last_error=system.get("last_error") or "",
        steps=system.get("steps") or {},
        degraded_components=system.get("degraded_components") or [],
        current_user=CurrentUserResponse(user_id=current_user.user_id, role=current_user.role),
        knowledge_base=KnowledgeBaseStatusResponse(
            status=knowledge["status"],
            message=knowledge["message"],
            last_error=knowledge.get("last_error") or "",
            stats=knowledge.get("stats") or {},
        ),
    )
