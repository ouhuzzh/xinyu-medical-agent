"""Hospital MCP binding API."""

from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import AuthenticatedUser, require_current_user, _client_ip
from api.dependencies import get_container
from pydantic import BaseModel, Field

router = APIRouter()


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or ""


class AddCredentialRequest(BaseModel):
    hospital_code: str = Field(..., min_length=1)
    token: str = Field(..., min_length=1)
    label: str = ""


class DeleteCredentialRequest(BaseModel):
    hospital_code: str = Field(..., min_length=1)


class TestConnectionRequest(BaseModel):
    hospital_code: str = Field(..., min_length=1)


@router.get("/api/hospitals/list")
def list_hospitals(request: Request, current_user: AuthenticatedUser = Depends(require_current_user)):
    request.state.route_type = "hospitals_list"
    container = get_container()
    hospitals = container.rag_system.mcp_server_registry.list_active()
    return {
        "hospitals": [
            {"code": h["code"], "name": h["name"],
             "description": h.get("description", ""),
             "auth_type": h.get("auth_type", "bearer")}
            for h in hospitals
        ]
    }


@router.get("/api/hospitals/credentials")
def list_credentials(request: Request, current_user: AuthenticatedUser = Depends(require_current_user)):
    request.state.route_type = "hospitals_credentials"
    container = get_container()
    creds = container.rag_system.user_mcp_credential_store.list_for_user(current_user.user_id)
    return {
        "credentials": [
            {"hospital_code": c["hospital_code"], "label": c.get("label", ""),
             "last_used_at": str(c.get("last_used_at") or ""),
             "last_health_status": c.get("last_health_status", "unknown"),
             "last_health_at": str(c.get("last_health_at") or "")}
            for c in creds
        ]
    }


@router.post("/api/hospitals/credentials/add")
def add_credential(
    request: Request,
    payload: AddCredentialRequest,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "hospitals_credentials_add"
    container = get_container()

    # Validate hospital exists
    hospital = container.rag_system.mcp_server_registry.get_by_code(payload.hospital_code)
    if not hospital:
        raise HTTPException(status_code=404, detail=f"医院 {payload.hospital_code} 不在平台支持列表中。")
    if not hospital.get("is_active"):
        raise HTTPException(status_code=400, detail=f"医院 {payload.hospital_code} 暂不可用。")

    try:
        container.rag_system.user_mcp_credential_store.save_credential(
            user_id=current_user.user_id,
            hospital_code=payload.hospital_code,
            plain_token=payload.token,
            label=payload.label,
        )
    except ValueError as e:
        container.audit_log.record(
            action="mcp_credential_save",
            actor_user_id=current_user.user_id,
            actor_username=current_user.username,
            target_type="hospital",
            target_id=payload.hospital_code,
            client_ip=_client_ip(request),
            request_id=_request_id(request),
            success=False,
            detail={"error": str(e)},
        )
        raise HTTPException(status_code=400, detail=str(e))

    # Invalidate pool so next turn picks up new tools
    container.rag_system.user_mcp_pool.invalidate(current_user.user_id)
    container.audit_log.record(
        action="mcp_credential_save",
        actor_user_id=current_user.user_id,
        actor_username=current_user.username,
        target_type="hospital",
        target_id=payload.hospital_code,
        client_ip=_client_ip(request),
        request_id=_request_id(request),
        success=True,
        detail={"label": payload.label or ""},
    )

    return {"message": f"已绑定 {hospital['name']}。"}


@router.post("/api/hospitals/credentials/delete")
def delete_credential(
    request: Request,
    payload: DeleteCredentialRequest,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "hospitals_credentials_delete"
    container = get_container()
    deleted = container.rag_system.user_mcp_credential_store.delete_credential(
        current_user.user_id, payload.hospital_code
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="未找到该绑定记录。")
    container.rag_system.user_mcp_pool.invalidate(current_user.user_id)
    container.audit_log.record(
        action="mcp_credential_delete",
        actor_user_id=current_user.user_id,
        actor_username=current_user.username,
        target_type="hospital",
        target_id=payload.hospital_code,
        client_ip=_client_ip(request),
        request_id=_request_id(request),
        success=True,
    )
    return {"message": "已解除绑定。"}


@router.post("/api/hospitals/credentials/test")
def test_connection(
    request: Request,
    payload: TestConnectionRequest,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "hospitals_credentials_test"
    container = get_container()
    rag = container.rag_system

    # Just trigger a pool rebuild and see if it succeeds
    rag.user_mcp_pool.invalidate(current_user.user_id)
    _ = rag.user_mcp_pool.get_tools_for_user(current_user.user_id)
    failed = rag.user_mcp_pool.get_failed_hospitals(current_user.user_id)

    if payload.hospital_code in failed:
        return {"ok": False, "error": failed[payload.hospital_code]}
    return {"ok": True}
