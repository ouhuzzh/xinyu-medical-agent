"""Authentication module — JWT-first with static token backward compatibility.

Auth flow:
1. Bearer Token is checked against JWT tokens first (primary)
2. Falls back to static API_AUTH_TOKENS (legacy/dev mode)
3. JWT tokens carry user_id, username, role in the payload
4. Static tokens map to user_id/role from config
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status

import config
from api.dependencies import get_container
from api.runtime_guards import (
    InMemoryRateLimiter,
    LoginLockoutTracker,
    RedisLoginLockoutTracker,
    RedisRateLimiter,
    build_login_lockout_tracker,
    build_rate_limiter,
    get_runtime_guard_backends,
)


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    role: str
    token: str
    username: str = ""


_rate_limiter = build_rate_limiter()
_login_lockout = build_login_lockout_tracker()


def _client_ip(request: Request) -> str:
    """Best-effort client IP extraction; honours X-Forwarded-For for proxy setups."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    client = request.client
    return client.host if client else "anonymous"


def enforce_auth_rate_limit(request: Request):
    """Per-IP throttle for unauthenticated auth endpoints."""
    _rate_limiter.check(
        bucket="auth",
        key=_client_ip(request),
        limit=config.API_RATE_LIMIT_AUTH_PER_MINUTE,
    )


def assert_login_not_locked(username: str):
    _login_lockout.assert_not_locked(username)


def record_login_failure(username: str):
    _login_lockout.record_failure(username)


def record_login_success(username: str):
    _login_lockout.record_success(username)


def get_auth_runtime_status() -> dict[str, str]:
    return get_runtime_guard_backends()


def _auth_error(detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED):
    headers = {"WWW-Authenticate": "Bearer"} if status_code == status.HTTP_401_UNAUTHORIZED else None
    raise HTTPException(status_code=status_code, detail=detail, headers=headers)


def _authenticate_jwt(token: str) -> AuthenticatedUser | None:
    """Try to authenticate via JWT.  Returns AuthenticatedUser or None.

    Also rejects tokens issued before the user's last password change
    (password_changed_at), so changing your password invalidates all
    existing sessions.
    """
    from api.jwt_utils import decode_token, token_issued_after
    payload = decode_token(token)
    if payload is None:
        return None
    if payload.get("type") != "access":
        return None
    user_id = str(payload.get("user_id", "")).strip()
    username = str(payload.get("username", "")).strip()
    role = str(payload.get("role", "user")).strip().lower()
    if not user_id or role not in ("user", "admin"):
        return None

    # H1: reject tokens issued before the last password change
    try:
        container = get_container()
        user = container.user_store.get_user_by_username(username)
        if user:
            pwd_changed = user.get("password_changed_at")
            if pwd_changed and not token_issued_after(payload, pwd_changed):
                return None
    except Exception:
        # Best-effort — if we can't reach the DB, still allow the token
        pass

    return AuthenticatedUser(user_id=user_id, role=role, token=token, username=username)


def _authenticate_static_token(token: str) -> AuthenticatedUser | None:
    """Try to authenticate via static API_AUTH_TOKENS (legacy/dev mode)."""
    auth_record = config.API_AUTH_TOKENS.get(token)
    if not isinstance(auth_record, dict):
        return None
    user_id = str(auth_record.get("user_id") or "").strip()
    role = str(auth_record.get("role") or "user").strip().lower()
    if not user_id or role not in ("user", "admin"):
        return None
    return AuthenticatedUser(user_id=user_id, role=role, token=token, username=user_id)


def require_current_user(
    request: Request,
    authorization: str | None = Header(default=None),
) -> AuthenticatedUser:
    """Authenticate user — JWT first, then static token fallback."""
    if not authorization:
        _auth_error("缺少 Bearer Token。")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        _auth_error("Bearer Token 格式无效。")

    token = token.strip()

    # Try JWT first
    user = _authenticate_jwt(token)
    if user is not None:
        request.state.user_id = user.user_id
        request.state.user_role = user.role
        return user

    # Fallback to static token
    user = _authenticate_static_token(token)
    if user is not None:
        request.state.user_id = user.user_id
        request.state.user_role = user.role
        return user

    _auth_error("Token 无效或已过期。")


def require_admin_user(
    request: Request,
    current_user: AuthenticatedUser = Depends(require_current_user),
) -> AuthenticatedUser:
    if current_user.role != "admin":
        _auth_error("当前账号无权访问管理员接口。", status_code=status.HTTP_403_FORBIDDEN)
    request.state.user_id = current_user.user_id
    request.state.user_role = current_user.role
    return current_user


def enforce_user_rate_limit(current_user: AuthenticatedUser, bucket: str, limit: int):
    _rate_limiter.check(bucket=bucket, key=current_user.user_id, limit=limit)


def enforce_chat_rate_limit(current_user: AuthenticatedUser):
    enforce_user_rate_limit(current_user, "chat", config.API_RATE_LIMIT_CHAT_PER_MINUTE)


def enforce_upload_rate_limit(current_user: AuthenticatedUser):
    enforce_user_rate_limit(current_user, "upload", config.API_RATE_LIMIT_UPLOADS_PER_MINUTE)


def enforce_sync_rate_limit(current_user: AuthenticatedUser):
    enforce_user_rate_limit(current_user, "sync", config.API_RATE_LIMIT_SYNCS_PER_MINUTE)


def ensure_owned_session(thread_id: str, current_user: AuthenticatedUser):
    container = get_container()
    session = container.chat_sessions.get_session(thread_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在。")
    if not session.get("owner_user_id"):
        container.chat_sessions.assign_owner_if_missing(thread_id, current_user.user_id)
        session = container.chat_sessions.get_session(thread_id)
    if not session or session.get("owner_user_id") != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="当前账号无权访问该会话。")
    return session
