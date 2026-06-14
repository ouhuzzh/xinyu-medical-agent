"""Generate production secrets for Docker deployment.

Prints values that can be copied into `.env.docker.prod.local`.
"""

from __future__ import annotations

import base64
import os
import secrets


def _fernet_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def main() -> int:
    print("JWT_SECRET_KEY=" + secrets.token_urlsafe(48))
    print("CHECKPOINT_SIGNING_KEY=" + secrets.token_urlsafe(48))
    print("MCP_TOKEN_ENCRYPTION_KEYS=" + _fernet_key())
    print("POSTGRES_PASSWORD=" + secrets.token_urlsafe(32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
