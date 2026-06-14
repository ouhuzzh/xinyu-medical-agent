"""Symmetric encryption with key rotation support.

Backed by Fernet (AES-128-CBC + HMAC-SHA256).  Supports key rotation via
MultiFernet: the FIRST key in the list is used for new encryption; ALL keys
are tried in order on decryption.  This lets you roll a compromised key
without re-encrypting all existing ciphertexts in a single transaction:

  1. Generate new key, prepend it to ``CRYPTO_KEYS``:  "new,old"
  2. Deploy — new writes use the new key, old reads still work
  3. Background job decrypts + re-encrypts existing rows (or wait for natural churn)
  4. Drop the old key from ``CRYPTO_KEYS`` once nothing decrypts with it

Two purposes the same primitive serves:
  - MCP credential tokens (``encrypt_token`` / ``decrypt_token``)
  - User PII memories (``encrypt_pii`` / ``decrypt_pii``) — same key list,
    separate function names for grep-ability and future split (e.g., a KMS-
    backed envelope encryption for PII).

Generate a key:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import base64
import os
import hashlib
import logging
import re
import threading
from typing import List

import config

logger = logging.getLogger(__name__)

_crypto_cache = None
_crypto_lock = threading.Lock()
_FERNET_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+=*")


def _derive_dev_key() -> bytes:
    """Derive a deterministic dev-only Fernet key.

    Uses a dedicated secret (MCP_TOKEN_DEV_SECRET) if available; otherwise
    derives from a fixed dev-only seed.  NEVER derived from JWT_SECRET_KEY —
    that would mean compromising the JWT secret also decrypts all PII.
    """
    dev_secret = os.environ.get("MCP_TOKEN_DEV_SECRET", "").strip()
    if not dev_secret:
        # Fixed dev-only seed — clearly unsafe, but isolated from JWT
        dev_secret = "xinyu-medical-agent-dev-only-key-do-not-use-in-prod"
    digest = hashlib.sha256(dev_secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _collect_keys() -> List[bytes]:
    """Return ordered list of Fernet keys (newest first).

    Reads ``MCP_TOKEN_ENCRYPTION_KEYS`` (comma-separated, preferred) or the
    legacy single ``MCP_TOKEN_ENCRYPTION_KEY``.  Falls back to a derived dev
    key when neither is set (and we're not in production).
    """
    raw_list = (config.MCP_TOKEN_ENCRYPTION_KEYS or "").strip()
    if raw_list:
        keys = [k.strip().encode("utf-8") for k in raw_list.split(",") if k.strip()]
        if keys:
            return keys

    single = (config.MCP_TOKEN_ENCRYPTION_KEY or "").strip()
    if single:
        return [single.encode("utf-8")]

    if config.APP_ENV == "production":
        raise RuntimeError(
            "MCP_TOKEN_ENCRYPTION_KEY(S) must be set in production. "
            "Generate one: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
    logger.warning(
        "No encryption keys configured; using derived dev key. UNSAFE for production."
    )
    return [_derive_dev_key()]


def _get_crypto():
    """Lazy-init the MultiFernet (or single Fernet) instance."""
    global _crypto_cache
    if _crypto_cache is not None:
        return _crypto_cache
    with _crypto_lock:
        if _crypto_cache is not None:
            return _crypto_cache
        from cryptography.fernet import Fernet, MultiFernet
        keys = _collect_keys()
        fernets = [Fernet(k) for k in keys]
        # MultiFernet works for any non-empty list — for a single key it's
        # equivalent to plain Fernet but keeps the call sites uniform.
        _crypto_cache = MultiFernet(fernets) if len(fernets) > 1 else fernets[0]
        return _crypto_cache


def _reset_cache_for_tests():
    """Test helper — invalidate the cached MultiFernet so config patches take effect."""
    global _crypto_cache
    with _crypto_lock:
        _crypto_cache = None


def looks_like_fernet_token(value: str) -> bool:
    token = str(value or "").strip()
    return (
        len(token) >= 80
        and token.startswith("gAAAAA")
        and _FERNET_TOKEN_RE.fullmatch(token) is not None
    )


def encrypt_token(plain_token: str) -> str:
    """Encrypt a plaintext token.  Returns a base64 ciphertext string."""
    if not plain_token:
        return ""
    return _get_crypto().encrypt(plain_token.encode("utf-8")).decode("utf-8")


def _decrypt_token(encrypted_token: str, *, log_failures: bool) -> str:
    if not encrypted_token:
        return ""
    try:
        return _get_crypto().decrypt(encrypted_token.encode("utf-8")).decode("utf-8")
    except Exception:
        if log_failures:
            logger.warning("Failed to decrypt token; returning empty result.")
            logger.debug("Token decryption failure details", exc_info=True)
        return ""


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt a ciphertext token.  Returns '' on any failure."""
    return _decrypt_token(encrypted_token, log_failures=True)


def try_decrypt_token(encrypted_token: str) -> str:
    """Decrypt a ciphertext token without emitting warning logs on failure."""
    return _decrypt_token(encrypted_token, log_failures=False)


# --- PII helpers (medical memories, user-private text) ---
# Same crypto today; separate names so we can swap in envelope encryption /
# KMS later without touching call sites.

def encrypt_pii(plaintext: str) -> str:
    return encrypt_token(plaintext)


def decrypt_pii(ciphertext: str) -> str:
    return decrypt_token(ciphertext)


def try_decrypt_pii(ciphertext: str) -> str:
    return try_decrypt_token(ciphertext)


def mask_token(plain_or_encrypted: str, prefix_len: int = 4, suffix_len: int = 4) -> str:
    s = plain_or_encrypted or ""
    if len(s) <= prefix_len + suffix_len:
        return "****"
    return f"{s[:prefix_len]}***{s[-suffix_len:]}"
