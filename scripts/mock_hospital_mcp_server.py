#!/usr/bin/env python
"""Mock Hospital MCP Server — pure stdlib, zero dependencies.

Implements MCP JSON-RPC 2.0 over HTTP on a configurable port.
6 tools: list_departments, list_doctors, get_available_slots,
book_appointment, cancel_appointment, list_my_appointments.

Usage:
    python scripts/mock_hospital_mcp_server.py [--port=8001] [--hospital=协和]

The client must send a valid MCP initialize request first (without session-id
requirement), then tools/list and tools/call will work.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import date, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

# ---------------------------------------------------------------------------
# In-memory data
# ---------------------------------------------------------------------------

DEPARTMENTS = [
    {"code": "neike",         "name": "内科"},
    {"code": "waike",         "name": "外科"},
    {"code": "xinneike",      "name": "心内科"},
    {"code": "shenjingneike", "name": "神经内科"},
    {"code": "pifuke",        "name": "皮肤科"},
    {"code": "erke",          "name": "儿科"},
    {"code": "jizhenke",      "name": "急诊科"},
]

DOCTOR_POOL = [
    {"name": "张医生", "title": "主任医师"},
    {"name": "李医生", "title": "副主任医师"},
    {"name": "王医生", "title": "主治医师"},
    {"name": "陈医生", "title": "主任医师"},
    {"name": "刘医生", "title": "住院医师"},
    {"name": "赵医生", "title": "副主任医师"},
]

SCHEDULES: dict[str, dict] = {}   # slot_id -> {...}
APPOINTMENTS: dict[str, dict] = {}  # appointment_no -> {...}


def _seed() -> None:
    if SCHEDULES:
        return
    today = date.today()
    for dept in DEPARTMENTS:
        for doc in DOCTOR_POOL[:3]:
            for day_offset in range(7):
                d = today + timedelta(days=day_offset)
                for slot in ("morning", "afternoon"):
                    sid = f"{dept['code']}_{doc['name']}_{d}_{slot}"
                    SCHEDULES[sid] = {
                        "schedule_id": sid,
                        "department": dept["name"],
                        "department_code": dept["code"],
                        "doctor_name": doc["name"],
                        "doctor_title": doc["title"],
                        "schedule_date": d.isoformat(),
                        "time_slot": slot,
                        "quota_total": 5,
                        "quota_available": 5,
                    }


# ---------------------------------------------------------------------------
# Tool definitions (MCP-compliant JSON Schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "list_departments",
        "description": "查询医院科室列表。可选按名称或代码关键词筛选。",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "按科室名称或代码筛选，可选"}},
        },
    },
    {
        "name": "list_doctors",
        "description": "查询某个科室的医生列表。",
        "inputSchema": {
            "type": "object",
            "properties": {"department": {"type": "string", "description": "科室名称，必填"}},
            "required": ["department"],
        },
    },
    {
        "name": "get_available_slots",
        "description": "查询某科室/医生/日期的可用号源。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "department": {"type": "string", "description": "科室名称，必填"},
                "schedule_date": {"type": "string", "description": "日期 YYYY-MM-DD，可选"},
                "doctor_name": {"type": "string", "description": "医生姓名，可选"},
                "time_slot": {"type": "string", "description": "morning / afternoon，可选"},
            },
            "required": ["department"],
        },
    },
    {
        "name": "book_appointment",
        "description": "确认预约挂号。成功后返回 appointment_no。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "department": {"type": "string", "description": "科室，必填"},
                "date": {"type": "string", "description": "日期 YYYY-MM-DD，必填"},
                "time_slot": {"type": "string", "description": "morning / afternoon，必填"},
                "doctor_name": {"type": "string", "description": "医生姓名，可选"},
                "patient_name": {"type": "string", "description": "患者姓名，默认 AI患者"},
            },
            "required": ["department", "date", "time_slot"],
        },
    },
    {
        "name": "cancel_appointment",
        "description": "按 appointment_no 取消预约。",
        "inputSchema": {
            "type": "object",
            "properties": {"appointment_no": {"type": "string", "description": "预约号，必填"}},
            "required": ["appointment_no"],
        },
    },
    {
        "name": "list_my_appointments",
        "description": "列出某患者的所有预约。",
        "inputSchema": {
            "type": "object",
            "properties": {"patient_name": {"type": "string", "description": "患者姓名，默认 AI患者"}},
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

def _exec_list_departments(args: dict) -> dict:
    _seed()
    q = (args.get("query") or "").strip()
    filtered = [{"code": d["code"], "name": d["name"]}
                for d in DEPARTMENTS
                if not q or q in d["name"] or q in d["code"]]
    return {"departments": filtered, "count": len(filtered)}


def _exec_list_doctors(args: dict) -> dict:
    _seed()
    dept = args["department"]
    seen = set()
    docs = []
    for s in SCHEDULES.values():
        if s["department"] == dept and s["doctor_name"] not in seen:
            seen.add(s["doctor_name"])
            docs.append({"name": s["doctor_name"], "title": s["doctor_title"]})
    return {"department": dept, "doctors": docs, "count": len(docs)}


def _exec_get_available_slots(args: dict) -> dict:
    _seed()
    results = []
    for s in SCHEDULES.values():
        if s["department"] != args["department"]:
            continue
        if args.get("schedule_date") and s["schedule_date"] != args["schedule_date"]:
            continue
        if args.get("doctor_name") and s["doctor_name"] != args["doctor_name"]:
            continue
        if args.get("time_slot") and s["time_slot"] != args["time_slot"]:
            continue
        if s["quota_available"] <= 0:
            continue
        results.append(s)
    return {"slots": results, "count": len(results)}


def _exec_book_appointment(args: dict) -> dict:
    _seed()
    slots = _exec_get_available_slots(args)["slots"]
    if not slots:
        return {"error": f"无可用号源", "appointment_no": ""}
    target = slots[0]
    sid = target["schedule_id"]
    SCHEDULES[sid]["quota_available"] = max(0, SCHEDULES[sid]["quota_available"] - 1)
    apt_no = f"APT-{uuid.uuid4().hex[:8].upper()}"
    APPOINTMENTS[apt_no] = {
        "appointment_no": apt_no,
        "department": args["department"],
        "doctor_name": target["doctor_name"],
        "date": target["schedule_date"],
        "time_slot": target["time_slot"],
        "patient_name": args.get("patient_name", "AI患者"),
    }
    return APPOINTMENTS[apt_no]


def _exec_cancel_appointment(args: dict) -> dict:
    apt_no = args["appointment_no"]
    if apt_no in APPOINTMENTS:
        del APPOINTMENTS[apt_no]
        return {"cancelled": True, "appointment_no": apt_no}
    return {"error": f"预约号 {apt_no} 不存在", "cancelled": False}


def _exec_list_my_appointments(args: dict) -> dict:
    patient = args.get("patient_name", "AI患者")
    results = [a for a in APPOINTMENTS.values() if a["patient_name"] == patient]
    return {"appointments": results, "count": len(results)}


_EXEC_MAP = {
    "list_departments":      _exec_list_departments,
    "list_doctors":          _exec_list_doctors,
    "get_available_slots":   _exec_get_available_slots,
    "book_appointment":      _exec_book_appointment,
    "cancel_appointment":    _exec_cancel_appointment,
    "list_my_appointments":  _exec_list_my_appointments,
}


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 dispatcher
# ---------------------------------------------------------------------------

def _handle(body: dict) -> dict:
    method = body.get("method", "")
    rid = body.get("id")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "1.0",
            "serverInfo": {"name": "mock-hospital", "version": "1.0"},
        }}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = body.get("params", {})
        name = params.get("name", "")
        tool_args = params.get("arguments", {}) or {}
        if name not in _EXEC_MAP:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Tool not found: {name}"}}
        try:
            result = _EXEC_MAP[name](tool_args)
            return {"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
            }}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(e)}}

    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Unknown method: {method}"}}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/mcp":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            resp = _handle(body)
        except Exception as e:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
        data = json.dumps(resp, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="Mock Hospital MCP Server")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--hospital", type=str, default="北京协和医院")
    args = p.parse_args()

    srv = HTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"[{args.hospital}] Mock MCP Server")
    print(f"   MCP endpoint: http://127.0.0.1:{args.port}/mcp")
    print(f"   Press Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
