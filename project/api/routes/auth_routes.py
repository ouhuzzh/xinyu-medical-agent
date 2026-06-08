"""Auth routes — user registration, login, token refresh, profile."""

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from api.auth import (
    AuthenticatedUser,
    assert_login_not_locked,
    enforce_auth_rate_limit,
    record_login_failure,
    record_login_success,
    require_current_user,
)
from api.dependencies import get_container
from api.jwt_utils import create_token_pair, decode_token, create_access_token
from api.schemas import (
    ChangePasswordRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserProfileResponse,
)

router = APIRouter()


@router.post("/api/auth/register", response_model=TokenResponse)
def register(request: Request, payload: RegisterRequest):
    request.state.route_type = "auth_register"
    enforce_auth_rate_limit(request)
    container = get_container()
    try:
        user = container.user_store.create_user(
            username=payload.username.strip(),
            password=payload.password,
            display_name=payload.display_name.strip(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    tokens = create_token_pair(
        user_id=str(user["id"]),
        username=user["username"],
        role=user["role"],
    )
    request.state.user_id = str(user["id"])
    return TokenResponse(**tokens)


@router.post("/api/auth/login", response_model=TokenResponse)
def login(request: Request, payload: LoginRequest):
    request.state.route_type = "auth_login"
    enforce_auth_rate_limit(request)
    username = payload.username.strip()
    # Reject early if the account is currently locked out — avoids the bcrypt
    # round-trip and gives the attacker no signal about credentials.
    assert_login_not_locked(username)
    container = get_container()
    user = container.user_store.verify_password(
        username=username,
        password=payload.password,
    )
    if user is None:
        record_login_failure(username)
        raise HTTPException(status_code=401, detail="用户名或密码不正确。")

    record_login_success(username)
    tokens = create_token_pair(
        user_id=str(user["id"]),
        username=user["username"],
        role=user["role"],
    )
    request.state.user_id = str(user["id"])
    return TokenResponse(**tokens)


@router.post("/api/auth/refresh", response_model=TokenResponse)
def refresh_token(request: Request, payload: RefreshRequest):
    request.state.route_type = "auth_refresh"
    enforce_auth_rate_limit(request)
    decoded = decode_token(payload.refresh_token)
    if decoded is None or decoded.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Refresh token 无效或已过期。")

    # Re-verify user still exists and is active
    container = get_container()
    user = container.user_store.get_user_by_username(decoded["username"])
    if not user or not user.get("is_active"):
        raise HTTPException(status_code=401, detail="用户不存在或已被禁用。")

    tokens = create_token_pair(
        user_id=str(user["id"]),
        username=user["username"],
        role=user["role"],
    )
    request.state.user_id = str(user["id"])
    return TokenResponse(**tokens)


@router.get("/api/auth/profile", response_model=UserProfileResponse)
def get_profile(request: Request, current_user: AuthenticatedUser = Depends(require_current_user)):
    request.state.route_type = "auth_profile"
    container = get_container()
    user = container.user_store.get_user_by_username(current_user.username) if current_user.username else None

    display_name = current_user.username
    username = current_user.username
    if user is not None:
        display_name = user.get("display_name", "") or username

    return UserProfileResponse(
        user_id=current_user.user_id,
        username=username,
        display_name=display_name,
        role=current_user.role,
    )


@router.post("/api/auth/change-password")
def change_password(
    request: Request,
    payload: ChangePasswordRequest,
    current_user: AuthenticatedUser = Depends(require_current_user),
):
    request.state.route_type = "auth_change_password"
    container = get_container()
    try:
        container.user_store.change_password(
            user_id=int(current_user.user_id),
            old_password=payload.old_password,
            new_password=payload.new_password,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": "密码修改成功。"}
