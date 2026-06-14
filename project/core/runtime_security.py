from __future__ import annotations

import logging
import re
from typing import Any, Mapping

import config


logger = logging.getLogger(__name__)

_INSECURE_JWT_DEFAULT = "change-me-in-production-please"
_FERNET_KEY_RE = re.compile(r"[A-Za-z0-9_-]{43}=")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_fernet_key_list(value: str) -> bool:
    keys = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return bool(keys) and all(_FERNET_KEY_RE.fullmatch(key) for key in keys)


def collect_security_issues_from_settings(settings: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    app_env = str(settings.get("APP_ENV", "development") or "development").strip().lower()
    jwt_secret_key = str(settings.get("JWT_SECRET_KEY", "") or "").strip()
    checkpoint_signing_key = str(settings.get("CHECKPOINT_SIGNING_KEY", "") or "").strip()
    encryption_keys = str(settings.get("MCP_TOKEN_ENCRYPTION_KEYS", "") or "").strip()
    legacy_encryption_key = str(settings.get("MCP_TOKEN_ENCRYPTION_KEY", "") or "").strip()
    user_memory_enabled = _as_bool(settings.get("USER_MEMORY_ENABLED", True))
    user_memory_encrypt_pii = _as_bool(settings.get("USER_MEMORY_ENCRYPT_PII", True))
    mcp_enabled = _as_bool(settings.get("MCP_ENABLED", True))

    if not jwt_secret_key:
        errors.append("JWT_SECRET_KEY is not set.")
    elif jwt_secret_key == _INSECURE_JWT_DEFAULT and app_env != "development":
        errors.append(
            f"JWT_SECRET_KEY is the default insecure value '{_INSECURE_JWT_DEFAULT}' outside development."
        )
    elif jwt_secret_key == _INSECURE_JWT_DEFAULT:
        warnings.append(
            "JWT_SECRET_KEY is using the default development value."
        )

    if app_env == "production":
        if len(jwt_secret_key) < 32:
            errors.append("JWT_SECRET_KEY should be at least 32 characters in production.")
        if len(checkpoint_signing_key) < 32:
            errors.append("CHECKPOINT_SIGNING_KEY should be at least 32 characters in production.")
        if user_memory_enabled and not user_memory_encrypt_pii:
            errors.append(
                "USER_MEMORY_ENABLED=true requires USER_MEMORY_ENCRYPT_PII=true in production."
            )

        encryption_required = user_memory_enabled or user_memory_encrypt_pii or mcp_enabled
        if encryption_required:
            key_source = encryption_keys or legacy_encryption_key
            if not key_source:
                errors.append(
                    "MCP_TOKEN_ENCRYPTION_KEY(S) must be set when MCP or user memory encryption is enabled in production."
                )
            elif not _looks_like_fernet_key_list(encryption_keys or legacy_encryption_key):
                errors.append("MCP_TOKEN_ENCRYPTION_KEY(S) must contain valid Fernet keys.")

    return errors, warnings


def collect_runtime_security_issues() -> tuple[list[str], list[str]]:
    return collect_security_issues_from_settings(
        {
            "APP_ENV": config.APP_ENV,
            "JWT_SECRET_KEY": config.JWT_SECRET_KEY,
            "CHECKPOINT_SIGNING_KEY": config.CHECKPOINT_SIGNING_KEY,
            "MCP_TOKEN_ENCRYPTION_KEYS": config.MCP_TOKEN_ENCRYPTION_KEYS,
            "MCP_TOKEN_ENCRYPTION_KEY": config.MCP_TOKEN_ENCRYPTION_KEY,
            "USER_MEMORY_ENABLED": config.USER_MEMORY_ENABLED,
            "USER_MEMORY_ENCRYPT_PII": config.USER_MEMORY_ENCRYPT_PII,
            "MCP_ENABLED": config.MCP_ENABLED,
        }
    )


def validate_runtime_security_or_raise() -> None:
    errors, warnings = collect_runtime_security_issues()
    for warning in warnings:
        logger.warning(warning)
    if errors:
        raise RuntimeError(" | ".join(errors))
