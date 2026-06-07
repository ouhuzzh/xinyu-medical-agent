"""Token encryption utilities for MCP hospital credentials.

Uses Fernet (symmetric, AES-128-CBC + HMAC-SHA256) from the cryptography library.

The encryption key is loaded from MCP_TOKEN_ENCRYPTION_KEY env var. In development,
falls back to a deterministic dev key derived from the JWT secret — convenient for
local testing but NOT safe for production. Production deployments MUST set
MCP_TOKEN_ENCRYPTION_KEY to a freshly generated Fernet key.

Generate a production key:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import base64
import hashlib
import logging
import threading

import config

logger = logging.getLogger(__name__)

_fernet_cache = None
_fernet_lock = threading.Lock()


def _derive_dev_key() -> bytes:
    """Derive a deterministic 32-byte key from the JWT secret for dev convenience."""
    seed = (config.JWT_SECRET_KEY or "dev-token-key-please-override").encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    return base64.urlsafe_b64encode(digest)


def _get_fernet():
    global _fernet_cache
    if _fernet_cache is not None:
        return _fernet_cache
    with _fernet_lock:
        if _fernet_cache is not None:
            return _fernet_cache
        from cryptography.fernet import Fernet
        key_str = (config.MCP_TOKEN_ENCRYPTION_KEY or "").strip()
        if key_str:
            key = key_str.encode("utf-8")
        else:
            if config.APP_ENV == "production":
                raise RuntimeError(
                    "MCP_TOKEN_ENCRYPTION_KEY must be set in production."
                )
            logger.warning(
                "MCP_TOKEN_ENCRYPTION_KEY not set; using derived dev key. UNSAFE for production."
            )
            key = _derive_dev_key()
        _fernet_cache = Fernet(key)
        return _fernet_cache


def encrypt_token(plain_token: str) -> str:
    """Encrypt a plaintext token.  Returns a base64 ciphertext string."""
    if not plain_token:
        return ""
    fernet = _get_fernet()
    return fernet.encrypt(plain_token.encode("utf-8")).decode("utf-8")


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt a ciphertext token.  Returns '' on any failure."""
    if not encrypted_token:
        return ""
    try:
        fernet = _get_fernet()
        return fernet.decrypt(encrypted_token.encode("utf-8")).decode("utf-8")
    except Exception:
        logger.warning("Failed to decrypt MCP token", exc_info=True)
        return ""


def mask_token(plain_or_encrypted: str, prefix_len: int = 4, suffix_len: int = 4) -> str:
    s = plain_or_encrypted or ""
    if len(s) <= prefix_len + suffix_len:
        return "****"
    return f"{s[:prefix_len]}***{s[-suffix_len:]}"
