#!/usr/bin/env python
"""End-to-end MCP demo script.

Walks through the full flow:
  1. Start mock MCP server (spawn subprocess)
  2. Seed hospital registry
  3. Bind user credential
  4. Build MCP pool & load tools
  5. Call a tool like a real user
  6. Shut down server

Usage:
    python scripts/demo_mcp_flow.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

# --- Config ---
MOCK_PORT = 8091
MOCK_HOSPITAL_CODE = "xiehe"
MOCK_HOSPITAL_NAME = "北京协和医院"
MOCK_TOKEN = "demo-xiehe-token-12345"


def red(s):  return f"\033[91m{s}\033[0m"
def green(s): return f"\033[92m{s}\033[0m"
def bold(s): return f"\033[1m{s}\033[0m"


def step(n, title):
    print(f"\n{bold(f'--- Step {n}: {title} ---')}")


def main():
    server_proc = None
    try:
        # --------------------------------------------------------------
        # Step 1: Start mock MCP server
        # --------------------------------------------------------------
        step(1, "Start Mock MCP Server")
        server_script = Path(__file__).resolve().parent / "mock_hospital_mcp_server.py"
        if not server_script.exists():
            print(red(f"Mock server script not found at {server_script}"))
            return 1
        server_proc = subprocess.Popen(
            [sys.executable, str(server_script), "--port", str(MOCK_PORT), "--hospital", MOCK_HOSPITAL_NAME],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(2)
        if server_proc.poll() is not None:
            print(red(f"Server crashed:\n{server_proc.stdout.read() if server_proc.stdout else ''}"))
            return 1
        print(green(f"✓ Mock server running on port {MOCK_PORT}"))

        # --------------------------------------------------------------
        # Step 2: Seed hospital in registry
        # --------------------------------------------------------------
        step(2, "Seed Hospital Registry")
        from mcp_integration.hospital_registry import HospitalRegistry
        reg = HospitalRegistry()
        rid = reg.upsert_hospital(
            code=MOCK_HOSPITAL_CODE,
            name=MOCK_HOSPITAL_NAME,
            mcp_url=f"http://127.0.0.1:{MOCK_PORT}/mcp",
            description="Mock hospital for MCP demo",
            auth_type="bearer",
        )
        hospital = reg.get_by_code(MOCK_HOSPITAL_CODE)
        print(green(f"✓ Hospital '{hospital['name']}' registered (code={MOCK_HOSPITAL_CODE}, url={hospital['mcp_url']})"))

        # --------------------------------------------------------------
        # Step 3: Bind user credential
        # --------------------------------------------------------------
        step(3, "Bind User Credential")
        from mcp_integration.user_hospital_store import UserHospitalStore
        store = UserHospitalStore()
        cid = store.save_credential(
            user_id="demo-user",           # static token userId
            hospital_code=MOCK_HOSPITAL_CODE,
            plain_token=MOCK_TOKEN,
            label="Demo user's 北京协和 token",
        )
        creds = store.list_for_user("demo-user")
        print(green(f"✓ Bound {len(creds)} hospital(s) for demo-user"))
        for c in creds:
            print(f"  - {c['hospital_code']}: {c['label']} (last_used={c.get('last_used_at', 'never')})")

        # --------------------------------------------------------------
        # Step 4: Build MCP pool and load tools
        # --------------------------------------------------------------
        step(4, "Build MCP Pool & Load Tools")
        from mcp_integration.user_mcp_pool import UserMCPPool
        pool = UserMCPPool(reg, store)
        tools = pool.get_tools_for_user("demo-user")
        connected = pool.get_connected_hospitals("demo-user")
        failed = pool.get_failed_hospitals("demo-user")
        print(green(f"✓ Connected to: {connected}"))
        if failed:
            print(red(f"  Failed: {failed}"))
        print(green(f"✓ Loaded {len(tools)} tools:"))
        for t in tools:
            print(f"  • {t.name} — {t.description[:80]}")

        # --------------------------------------------------------------
        # Step 5: Call a tool directly
        # --------------------------------------------------------------
        step(5, "Call a Tool Directly")
        if not tools:
            print(red("No tools loaded — cannot demonstrate calls."))
            return 1
        list_deps = next((t for t in tools if "list_departments" in t.name), None)
        if list_deps is None:
            print(red("'list_departments' tool not found"))
            return 1
        raw_result = list_deps.invoke({"query": "心内科"})
        print(green(f"✓ Called '{list_deps.name}'"))
        print(f"  Result: {json.dumps(raw_result, ensure_ascii=False, indent=2)}")

        # --------------------------------------------------------------
        # Step 6: Shutdown
        # --------------------------------------------------------------
        step(6, "Shutdown")
        print(green("✓ All steps passed."))
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(red(f"\n✗ Error: {type(e).__name__}: {e}"))
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if server_proc is not None:
            print("Stopping mock server...")
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            print("Server stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
