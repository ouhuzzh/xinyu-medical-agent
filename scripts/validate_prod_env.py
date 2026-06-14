"""Validate `.env.docker.prod.local` before starting production Docker.

Usage:
    python scripts/validate_prod_env.py .env.docker.prod.local
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core.runtime_security import collect_security_issues_from_settings


PLACEHOLDERS = {
    "",
    "replace-with-a-long-random-secret",
    "replace-with-a-different-long-random-secret",
    "replace-with-a-strong-db-password",
    "replace-with-fernet-key-list",
}

REQUIRED = (
    "APP_DOMAIN",
    "API_DOMAIN",
    "PUBLIC_API_BASE_URL",
    "API_CORS_ORIGINS",
    "JWT_SECRET_KEY",
    "CHECKPOINT_SIGNING_KEY",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "ACTIVE_LLM_PROVIDER",
    "ACTIVE_EMBEDDING_PROVIDER",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "VECTOR_DIMENSION",
    "MCP_TOKEN_ENCRYPTION_KEYS",
    "USER_MEMORY_ENCRYPT_PII",
)


def _load_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in PLACEHOLDERS or "example.com" in lowered or lowered.startswith("replace-")
def validate(env: dict[str, str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for key in REQUIRED:
        value = env.get(key, "")
        if not value:
            errors.append(f"{key} is required.")
        elif _is_placeholder(value):
            errors.append(f"{key} still uses a placeholder value.")

    if env.get("PUBLIC_API_BASE_URL", "").startswith("http://"):
        errors.append("PUBLIC_API_BASE_URL must use https:// in production.")
    if "localhost" in env.get("API_CORS_ORIGINS", "") or "127.0.0.1" in env.get("API_CORS_ORIGINS", ""):
        errors.append("API_CORS_ORIGINS must not contain localhost values in production.")
    if env.get("APP_DOMAIN") == env.get("API_DOMAIN"):
        warnings.append("APP_DOMAIN and API_DOMAIN are identical; current Caddyfile expects separate domains.")

    if len(env.get("JWT_SECRET_KEY", "")) < 32:
        errors.append("JWT_SECRET_KEY should be at least 32 characters.")
    if len(env.get("CHECKPOINT_SIGNING_KEY", "")) < 32:
        errors.append("CHECKPOINT_SIGNING_KEY should be at least 32 characters.")
    if len(env.get("POSTGRES_PASSWORD", "")) < 16:
        errors.append("POSTGRES_PASSWORD should be at least 16 characters.")

    security_errors, security_warnings = collect_security_issues_from_settings(env)
    errors.extend(item for item in security_errors if item not in errors)
    warnings.extend(item for item in security_warnings if item not in warnings)

    api_tokens = env.get("API_AUTH_TOKENS_JSON", "{}")
    try:
        decoded = json.loads(api_tokens or "{}")
        if not isinstance(decoded, dict):
            errors.append("API_AUTH_TOKENS_JSON must decode to a JSON object.")
        if any("demo" in str(token).lower() for token in decoded):
            errors.append("API_AUTH_TOKENS_JSON must not contain demo tokens in production.")
    except json.JSONDecodeError as exc:
        errors.append(f"API_AUTH_TOKENS_JSON is not valid JSON: {exc}")

    if env.get("ACTIVE_EMBEDDING_PROVIDER") == "huggingface_local" and env.get("INSTALL_LOCAL_ML") != "true":
        errors.append("huggingface_local requires INSTALL_LOCAL_ML=true in Docker builds.")

    return errors, warnings


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else ".env.docker.prod.local")
    if not path.exists():
        print(f"ERROR: {path} does not exist.")
        return 1

    errors, warnings = validate(_load_env(path))
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print(f"OK: {path} looks ready for production Docker startup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
