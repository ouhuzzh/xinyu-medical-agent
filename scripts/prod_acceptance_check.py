"""Post-deploy acceptance checks for a production stack.

Usage:
    python scripts/prod_acceptance_check.py
    python scripts/prod_acceptance_check.py --chat-smoke

Environment variables:
    FRONTEND_URL       defaults to https://medical.example.com
    API_BASE_URL       defaults to https://api.medical.example.com
    API_AUTH_TOKEN     optional; enables authenticated checks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def _request(
    url: str,
    *,
    method: str = "GET",
    token: str = "",
    payload: dict[str, Any] | None = None,
    timeout: float = 20.0,
) -> tuple[int, str]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def _request_json(*args, **kwargs) -> tuple[int, dict[str, Any]]:
    status, body = _request(*args, **kwargs)
    return status, json.loads(body)


def _parse_sse_events(body: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current["data"] = line.split(":", 1)[1].strip()
    if current:
        events.append(current)
    return events


def _check_frontend(frontend_url: str) -> list[str]:
    _, body = _request(frontend_url)
    if "<html" not in body.lower():
        raise RuntimeError(f"Frontend at {frontend_url} did not return HTML.")
    return [f"frontend OK: {frontend_url}"]


def _check_public_api(api_base_url: str) -> list[str]:
    _, body = _request(f"{api_base_url}/api/healthz")
    payload = json.loads(body)
    if payload != {"ok": True}:
        raise RuntimeError(f"/api/healthz returned unexpected payload: {payload}")
    return [f"api_healthz OK: {api_base_url}/api/healthz"]


def _check_authenticated_api(api_base_url: str, token: str) -> list[str]:
    _, payload = _request_json(f"{api_base_url}/api/system/status", token=token)
    state = payload.get("state")
    schema_status = (payload.get("schema_health") or {}).get("status")
    knowledge_status = (payload.get("knowledge_base") or {}).get("status")
    if state not in {"ready", "degraded"}:
        raise RuntimeError(f"/api/system/status returned state={state!r}")
    if schema_status not in {"ok", "degraded"}:
        raise RuntimeError(f"/api/system/status returned schema_health.status={schema_status!r}")
    lines = [
        f"api_system_status OK: state={state}",
        f"schema_health={schema_status}",
    ]
    if knowledge_status:
        lines.append(f"knowledge_base={knowledge_status}")
    return lines


def _run_chat_smoke(api_base_url: str, token: str, message: str) -> list[str]:
    _, session_payload = _request_json(
        f"{api_base_url}/api/chat/session",
        method="POST",
        token=token,
        payload={},
    )
    thread_id = session_payload.get("thread_id", "").strip()
    if not thread_id:
        raise RuntimeError("chat session response did not include thread_id")

    _, stream_body = _request(
        f"{api_base_url}/api/chat/stream",
        method="POST",
        token=token,
        payload={"thread_id": thread_id, "message": message},
        timeout=120.0,
    )
    events = _parse_sse_events(stream_body)
    event_names = [item.get("event", "") for item in events]
    if "app-error" in event_names:
        raise RuntimeError("chat stream produced an app-error event")
    if "final" not in event_names:
        raise RuntimeError("chat stream did not produce a final event")
    return [
        f"chat_session OK: thread_id={thread_id}",
        f"chat_stream OK: events={','.join(event_names)}",
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend-url", default=os.environ.get("FRONTEND_URL", "https://medical.example.com"))
    parser.add_argument("--api-base-url", default=os.environ.get("API_BASE_URL", "https://api.medical.example.com"))
    parser.add_argument("--api-auth-token", default=os.environ.get("API_AUTH_TOKEN", "").strip())
    parser.add_argument("--chat-smoke", action="store_true", help="Run a real chat session smoke test.")
    parser.add_argument("--chat-message", default="你好，请用一句话确认系统工作正常。")
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    failures: list[str] = []
    infos: list[str] = []

    checks = [
        lambda: _check_frontend(args.frontend_url),
        lambda: _check_public_api(args.api_base_url),
    ]
    if args.api_auth_token:
        checks.append(lambda: _check_authenticated_api(args.api_base_url, args.api_auth_token))
        if args.chat_smoke:
            checks.append(lambda: _run_chat_smoke(args.api_base_url, args.api_auth_token, args.chat_message))
    elif args.chat_smoke:
        failures.append("--chat-smoke requires API_AUTH_TOKEN.")

    for check in checks:
        try:
            infos.extend(check())
        except (RuntimeError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            failures.append(str(exc))

    for line in infos:
        print(f"OK: {line}")
    for line in failures:
        print(f"FAIL: {line}")

    if failures:
        return 1
    print("OK: production acceptance checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
