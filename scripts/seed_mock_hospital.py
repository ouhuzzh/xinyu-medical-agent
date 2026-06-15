#!/usr/bin/env python
"""Register mock hospitals in the database.

Run AFTER the DB is up and migrations are applied.

Usage:
    python scripts/seed_mock_hospital.py
    python scripts/seed_mock_hospital.py --docker-desktop
    python scripts/seed_mock_hospital.py --mcp-host 192.168.1.23
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from mcp_integration.mcp_server_registry import MCPServerRegistry


def _build_hospitals(mcp_host: str) -> list[dict[str, str]]:
    return [
        {
            "code": "xiehe",
            "name": "北京协和医院",
            "description": "模拟医院 - 本地 Mock MCP server，用于演示 MCP 集成",
            "mcp_url": f"http://{mcp_host}:8001/mcp",
            "auth_type": "bearer",
        },
        {
            "code": "renji",
            "name": "上海仁济医院",
            "description": "模拟医院 - 本地 Mock MCP server，用于演示 MCP 集成",
            "mcp_url": f"http://{mcp_host}:8002/mcp",
            "auth_type": "bearer",
        },
    ]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mcp-host",
        default=os.environ.get("MOCK_MCP_HOST", "127.0.0.1"),
        help="Host name that the API process should use to reach mock MCP servers.",
    )
    parser.add_argument(
        "--docker-desktop",
        action="store_true",
        help="Use host.docker.internal so Docker Desktop containers can reach host mock servers.",
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    mcp_host = "host.docker.internal" if args.docker_desktop else args.mcp_host
    registry = MCPServerRegistry()

    for h in _build_hospitals(mcp_host):
        rid = registry.upsert_hospital(
            code=h["code"],
            name=h["name"],
            mcp_url=h["mcp_url"],
            description=h["description"],
            auth_type=h["auth_type"],
        )
        print(f"[OK] {h['name']} registered (id={rid}, url={h['mcp_url']})")

    print("\nNext:")
    print("  xiehe mock: python scripts/mock_hospital_mcp_server.py --port=8001 --hospital=北京协和医院")
    print("  renji mock: python scripts/mock_hospital_mcp_server.py --port=8002 --hospital=上海仁济医院")
    print("  demo token: demo-xiehe-token-12345")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
