import logging

from fastapi import APIRouter, Depends, Request

from api.auth import AuthenticatedUser, get_auth_runtime_status, require_current_user
from api.dependencies import get_container
from api.schemas import CurrentUserResponse, KnowledgeBaseStatusResponse, SystemStatusResponse


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/api/health")
def health(
    request: Request,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "health"
    return {"ok": True, "user_id": current_user.user_id, "role": current_user.role}


@router.get("/api/healthz")
def healthz(request: Request):
    request.state.route_type = "health"
    return {"ok": True}


@router.get("/api/system/status", response_model=SystemStatusResponse)
def system_status(
    request: Request,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "system_status"
    try:
        container = get_container()
    except Exception as exc:
        logger.exception("System status requested while API container is unavailable")
        return SystemStatusResponse(
            state="failed",
            message="系统初始化失败。",
            last_error=str(exc),
            steps={},
            degraded_components=["api_container"],
            runtime_backends={
                "session_lock_backend": "unknown",
                "mcp_pool_backend": "unknown",
                "schema_guard_backend": "unknown",
                **get_auth_runtime_status(),
            },
            schema_health={
                "status": "unknown",
                "message": "API container is unavailable; schema guard did not run.",
                "expected_dimension": 0,
                "actual_dimensions": {},
                "errors": [str(exc)],
            },
            current_user=CurrentUserResponse(
                user_id=current_user.user_id, role=current_user.role, username=current_user.username or ""
            ),
            knowledge_base=KnowledgeBaseStatusResponse(
                status="failed",
                message="知识库状态不可用。",
                last_error=str(exc),
                stats={},
            ),
        )
    system = container.rag_system.get_system_status()
    knowledge = container.rag_system.get_knowledge_base_status()
    session_lock_backend = "unknown"
    thread_locks = getattr(container, "thread_locks", None)
    if thread_locks is not None:
        backend_name = getattr(thread_locks, "backend_name", None)
        if callable(backend_name):
            session_lock_backend = str(backend_name())
    mcp_pool_backend = "unknown"
    user_mcp_pool = getattr(container.rag_system, "user_mcp_pool", None)
    if user_mcp_pool is not None:
        backend_name = getattr(user_mcp_pool, "backend_name", None)
        if callable(backend_name):
            mcp_pool_backend = str(backend_name())
    schema_guard_backend = "unknown"
    schema_health = {
        "status": "unknown",
        "message": "Schema guard is not configured.",
        "expected_dimension": 0,
        "actual_dimensions": {},
        "errors": [],
    }
    schema_guard = getattr(container, "schema_guard", None)
    if schema_guard is not None:
        backend_name = getattr(schema_guard, "backend_name", None)
        if callable(backend_name):
            schema_guard_backend = str(backend_name())
        get_health = getattr(schema_guard, "get_health", None)
        if callable(get_health):
            try:
                schema_health = dict(get_health(refresh=True))
            except TypeError:
                schema_health = dict(get_health())
    runtime_backends = {
        "session_lock_backend": session_lock_backend,
        "mcp_pool_backend": mcp_pool_backend,
        "schema_guard_backend": schema_guard_backend,
        **get_auth_runtime_status(),
    }
    return SystemStatusResponse(
        state=system["state"],
        message=system["message"],
        last_error=system.get("last_error") or "",
        steps=system.get("steps") or {},
        degraded_components=system.get("degraded_components") or [],
        runtime_backends=runtime_backends,
        schema_health=schema_health,
        current_user=CurrentUserResponse(
            user_id=current_user.user_id, role=current_user.role, username=current_user.username or ""
        ),
        knowledge_base=KnowledgeBaseStatusResponse(
            status=knowledge["status"],
            message=knowledge["message"],
            last_error=knowledge.get("last_error") or "",
            stats=knowledge.get("stats") or {},
        ),
    )


@router.get("/api/system/llm-status")
def llm_status(
    request: Request,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    """Return circuit-breaker status for all LLM providers (admin only)."""
    request.state.route_type = "llm_status"
    if current_user.role != "admin":
        return {"error": "Admin access required"}
    container = get_container()
    rag = container.rag_system
    if rag.agent_graph is None:
        return {"status": "not_initialized"}
    # Retrieve the router from the RAG system's initialize() scope
    try:
        from llm_tiered_router import TieredLLMRouter
        router = TieredLLMRouter.from_env()
        return router.get_status()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
