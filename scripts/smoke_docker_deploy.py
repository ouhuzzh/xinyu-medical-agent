"""Small post-deploy smoke test for Docker deployments.

Environment variables:
    FRONTEND_URL      defaults to http://localhost:8080
    API_BASE_URL      defaults to http://localhost:8000
    API_AUTH_TOKEN    optional; when set, also checks /api/system/status
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request


def _get(url: str, *, token: str = "", timeout: float = 10.0) -> tuple[int, str]:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def main() -> int:
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:8080").rstrip("/")
    api_base_url = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
    api_token = os.environ.get("API_AUTH_TOKEN", "").strip()

    checks = [
        ("frontend", frontend_url),
        ("api_healthz", f"{api_base_url}/api/healthz"),
    ]
    if api_token:
        checks.append(("api_system_status", f"{api_base_url}/api/system/status"))

    failed = False
    for name, url in checks:
        try:
            status, body = _get(url, token=api_token if name == "api_system_status" else "")
        except Exception as exc:
            print(f"FAIL {name}: {exc}")
            failed = True
            continue
        print(f"OK {name}: HTTP {status}")
        if name == "api_system_status":
            payload = json.loads(body)
            print(
                "   state={state} schema={schema}".format(
                    state=payload.get("state"),
                    schema=(payload.get("schema_health") or {}).get("status"),
                )
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
