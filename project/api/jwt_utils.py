"""JWT token utilities — create and verify access/refresh tokens.

Uses PyJWT with HS256 algorithm.  Access tokens are short-lived (24h by default),
refresh tokens are long-lived (30 days).  Both carry user_id, username, and role.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import config

logger = logging.getLogger(__name__)


def create_access_token(data: Dict[str, Any]) -> str:
    """Create a JWT access token."""
    import jwt
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({
        "exp": expire,
        "iat": now,
        "type": "access",
    })
    return jwt.encode(to_encode, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a JWT refresh token."""
    import jwt
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=config.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({
        "exp": expire,
        "iat": now,
        "type": "refresh",
    })
    return jwt.encode(to_encode, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify a JWT token.  Returns None on any failure."""
    import jwt
    try:
        payload = jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.debug("Invalid token")
        return None


def token_issued_after(token_payload: dict, timestamp) -> bool:
    """Return True if the token was issued AFTER the given datetime.

    If iat is missing (legacy token), returns True — the check degrades
    gracefully rather than locking out existing sessions.
    """
    iat = token_payload.get("iat")
    if iat is None:
        return True  # legacy tokens: allow
    if isinstance(iat, (int, float)):
        iat = datetime.fromtimestamp(iat, tz=timezone.utc)
    if timestamp and isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    if timestamp and timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return iat > timestamp


def create_token_pair(user_id: int, username: str, role: str) -> Dict[str, str]:
    """Create both access and refresh tokens for a user."""
    data = {"user_id": user_id, "username": username, "role": role}
    return {
        "access_token": create_access_token(data),
        "refresh_token": create_refresh_token(data),
        "token_type": "bearer",
    }
