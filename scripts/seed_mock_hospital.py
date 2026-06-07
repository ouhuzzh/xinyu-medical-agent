#!/usr/bin/env python
"""Register mock hospitals in the database.

Run AFTER the DB is up and migrations are applied.

Usage:
    python scripts/seed_mock_hospital.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from mcp_integration.hospital_registry import HospitalRegistry

registry = HospitalRegistry()

HOSPITALS = [
    {
        "code": "xiehe",
        "name": "北京协和医院",
        "description": "模拟医院——本地 Mock MCP server，用于演示 MCP 集成",
        "mcp_url": "http://127.0.0.1:8001/mcp",
        "auth_type": "bearer",
    },
    {
        "code": "renji",
        "name": "上海仁济医院",
        "description": "模拟医院——本地 Mock MCP server，用于演示 MCP 集成",
        "mcp_url": "http://127.0.0.1:8002/mcp",
        "auth_type": "bearer",
    },
]

for h in HOSPITALS:
    rid = registry.upsert_hospital(
        code=h["code"],
        name=h["name"],
        mcp_url=h["mcp_url"],
        description=h["description"],
        auth_type=h["auth_type"],
    )
    print(f"✅ {h['name']} registered (id={rid})")
