#!/usr/bin/env python
"""Mock Hospital MCP Server — for demo and testing.

Starts a local MCP streamable-HTTP server on http://127.0.0.1:8001/mcp
that simulates a hospital's appointment system: departments, doctors,
available time slots, booking, cancellation, and appointment listing.

Usage:
    pip install mcp fastmcp uvicorn
    python scripts/mock_hospital_mcp_server.py [--port=8001] [--hospital=协和]

IMPORTANT — keep this script self-contained (no imports from project/).
The TRPC FastMCP package needs to be active in the Python environment to run this.
"""

from __future__ import annotations

import argparse
import uuid
from datetime import date, timedelta
from typing import Any

# --- In-memory "database" ---
# You can replace this with real PostgreSQL calls in production.

DOCTORS: dict[str, list[dict]] = {}   # department_name → [{name, title}]
SCHEDULES: dict[str, dict] = {}       # slot_id → {doctor, date, time_slot, quota}
APPOINTMENTS: dict[str, dict] = {}    # appointment_id → {department, doctor, date, slot, patient}

DEPARTMENTS = [
    {"name": "内科", "code": "neike"},
    {"name": "外科", "code": "waike"},
    {"name": "心内科", "code": "xinneike"},
    {"name": "神经内科", "code": "shenjingneike"},
    {"name": "皮肤科", "code": "pifuke"},
    {"name": "儿科", "code": "erke"},
    {"name": "急诊科", "code": "jizhenke"},
]

DOCTOR_POOL = [
    {"name": "张医生", "title": "主任医师"},
    {"name": "李医生", "title": "副主任医师"},
    {"name": "王医生", "title": "主治医师"},
    {"name": "陈医生", "title": "主任医师"},
    {"name": "刘医生", "title": "住院医师"},
    {"name": "赵医生", "title": "副主任医师"},
]


def _seed_data():
    """Generate fake schedules for the next 7 days."""
    if DOCTORS:
        return  # Already seeded
    today = date.today()
    for dept in DEPARTMENTS:
        DOCTORS[dept["name"]] = [
            {"name": pool["name"], "title": pool["title"]}
            for pool in DOCTOR_POOL
        ]
        for i, doc in enumerate(DOCTOR_POOL[:3]):  # 3 docs per dept
            for day_offset in range(7):
                d = today + timedelta(days=day_offset)
                for slot in ("morning", "afternoon"):
                    sid = f"{dept['code']}_{doc['name']}_{d}_{slot}"
                    SCHEDULES[sid] = {
                        "schedule_id": sid,
                        "department": dept["name"],
                        "doctor_name": doc["name"],
                        "doctor_title": doc["title"],
                        "schedule_date": d.isoformat(),
                        "time_slot": slot,
                        "quota_total": 3,
                        "quota_available": 3,
                    }


# --- FastMCP application ---

def _build_app(hospital_name: str, port: int):
    """Build the MCP server app."""

    # Try fastmcp first (newer), fall back to mcp.server
    try:
        from fastmcp import FastMCP
        mcp = FastMCP(
            name=f"{hospital_name} Mock Booking",
            host="127.0.0.1",
            port=port,
            log_level="ERROR",
        )
        MCP_READY = True
    except ImportError:
        MCP_READY = False

    if not MCP_READY:
        raise ImportError(
            "Neither 'fastmcp' nor a suitable MCP server package is available. "
            "Install with: pip install fastmcp"
        )

    # ------------------------------------------------------------------
    # Tools (exposed to the LLM via MCP)
    # ------------------------------------------------------------------

    @mcp.tool()
    def list_departments(query: str = "") -> dict[str, Any]:
        """List available departments. Optionally filter by name keyword with the query parameter."""
        _seed_data()
        filtered = [
            {"code": d["code"], "name": d["name"]}
            for d in DEPARTMENTS
            if not query or query.strip() in d["name"]
        ]
        return {"departments": filtered, "count": len(filtered)}

    @mcp.tool()
    def list_doctors(department: str) -> dict[str, Any]:
        """List doctors in a department."""
        _seed_data()
        docs = DOCTORS.get(department, [])
        return {"department": department, "doctors": docs, "count": len(docs)}

    @mcp.tool()
    def get_available_slots(
        department: str,
        schedule_date: str = "",
        doctor_name: str = "",
        time_slot: str = "",
    ) -> dict[str, Any]:
        """Query available appointment slots for a department/doctor/date/time-slot combination."""
        _seed_data()
        results = []
        for sid, s in SCHEDULES.items():
            if s["department"] != department:
                continue
            if schedule_date and s["schedule_date"] != schedule_date:
                continue
            if doctor_name and s["doctor_name"] != doctor_name:
                continue
            if time_slot and s["time_slot"] != time_slot:
                continue
            if s["quota_available"] <= 0:
                continue
            results.append({k: v for k, v in s.items()})
        return {"slots": results, "count": len(results)}

    @mcp.tool()
    def book_appointment(
        department: str,
        date: str,
        time_slot: str,
        doctor_name: str = "",
        patient_name: str = "AI患者",
    ) -> dict[str, Any]:
        """Confirm an appointment booking.  Returns appointment_no on success."""
        _seed_data()
        slots = get_available_slots(
            department=department,
            schedule_date=date,
            doctor_name=doctor_name,
            time_slot=time_slot,
        )
        if not slots["slots"]:
            return {"error": f"No available slots for {department} on {date} {time_slot}", "appointment_no": ""}
        target = slots["slots"][0]
        sid = target["schedule_id"]
        SCHEDULES[sid]["quota_available"] = max(0, SCHEDULES[sid]["quota_available"] - 1)
        apt_id = str(uuid.uuid4())
        APPOINTMENTS[apt_id] = {
            "appointment_id": apt_id,
            "appointment_no": f"APT-{uuid.uuid4().hex[:8].upper()}",
            "department": department,
            "doctor_name": target["doctor_name"],
            "date": target["schedule_date"],
            "time_slot": target["time_slot"],
            "patient_name": patient_name,
        }
        return APPOINTMENTS[apt_id]

    @mcp.tool()
    def cancel_appointment(appointment_no: str) -> dict[str, Any]:
        """Cancel an appointment by appointment_no."""
        for aid, apt in APPOINTMENTS.items():
            if apt["appointment_no"] == appointment_no:
                del APPOINTMENTS[aid]
                return {"cancelled": True, "appointment_no": appointment_no}
        return {"error": f"Appointment {appointment_no} not found", "cancelled": False}

    @mcp.tool()
    def list_my_appointments(patient_name: str = "AI患者") -> dict[str, Any]:
        """List appointments for a patient."""
        results = [apt for apt in APPOINTMENTS.values() if apt.get("patient_name") == patient_name]
        return {"appointments": results, "count": len(results)}

    return mcp


def main():
    parser = argparse.ArgumentParser(description="Mock Hospital MCP Server")
    parser.add_argument("--port", type=int, default=8001, help="Listen port (default 8001)")
    parser.add_argument("--hospital", type=str, default="北京协和医院", help="Hospital display name")
    args = parser.parse_args()

    mcp = _build_app(hospital_name=args.hospital, port=args.port)
    print(f"🏥 {args.hospital} Mock MCP Server")
    print(f"   MCP endpoint: http://127.0.0.1:{args.port}/mcp")
    print(f"   SSE endpoint: http://127.0.0.1:{args.port}/sse")
    print(f"   Start with:  python scripts/mock_hospital_mcp_server.py --port={args.port}")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
