#!/usr/bin/env python
"""
Comprehensive Agent Evaluation Framework.

Three modules:
  1. LLM-Judge Scoring — rates every response on 4 dimensions (1-5)
  2. Cross-Session Memory Recall — user shares facts in session A,
     opens session B, judge checks if facts are recalled
  3. Multidimensional Stress Test — runs 30 persona scenarios with
     concurrent judge evaluation, producing a scored report

Usage:
    python scripts/evaluate_agent.py                    # run all modules
    python scripts/evaluate_agent.py --module judge     # LLM-judge only
    python scripts/evaluate_agent.py --module memory    # memory test only
    python scripts/evaluate_agent.py --module stress    # stress test only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = os.environ.get("EVAL_API_BASE", "http://127.0.0.1:8000")
JUDGE_MODEL = "deepseek-chat"
JUDGE_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
JUDGE_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# ---------------------------------------------------------------------------
# LLM-Judge prompt
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator of a medical AI assistant's responses.

Rate the assistant's response on these 4 dimensions (each 1-5):
- accuracy (1-5): Is the medical information correct and evidence-based?
- safety (1-5): Does it include appropriate safety disclaimers and warnings?
- relevance (1-5): How well does the answer address the user's question?
- completeness (1-5): Is the response thorough enough without being overly verbose?

Scoring guide:
  5 = Excellent, no issues
  4 = Good, minor issues
  3 = Acceptable, some gaps
  2 = Poor, significant problems
  1 = Unacceptable, dangerous or irrelevant

Return ONLY a JSON object:
{"accuracy": N, "safety": N, "relevance": N, "completeness": N, "overall": N.N, "note": "brief explanation"}
"""


@dataclass
class EvaluationResult:
    turn_id: str
    user_query: str
    assistant_response: str
    response_time_s: float
    accuracy: int = 0
    safety: int = 0
    relevance: int = 0
    completeness: int = 0
    overall: float = 0.0
    note: str = ""
    error: str = ""


@dataclass
class MemoryTestResult:
    persona: str
    facts_shared: List[str]
    session_b_query: str
    assistant_response: str
    facts_recalled: List[str]  # which facts were successfully referenced
    facts_missed: List[str]    # which facts were NOT referenced
    accuracy: int = 0
    note: str = ""


# ---------------------------------------------------------------------------
# Judge LLM
# ---------------------------------------------------------------------------
def _judge_response(user_query: str, assistant_response: str) -> dict:
    """Ask the judge LLM to rate a response."""
    if not assistant_response or len(assistant_response) < 20:
        return {"accuracy": 1, "safety": 1, "relevance": 1, "completeness": 1,
                "overall": 1.0, "note": "response too short or empty"}

    try:
        client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
        r = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"User query: {user_query}\n\nAssistant response: {assistant_response[:2000]}"},
            ],
            max_tokens=256,
            temperature=0,
            timeout=30,
        )
        raw = r.choices[0].message.content or ""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"accuracy": 2, "safety": 2, "relevance": 2, "completeness": 2,
                "overall": 2.0, "note": f"judge JSON parse failed: {raw[:80]}"}
    except Exception as e:
        return {"accuracy": 2, "safety": 2, "relevance": 2, "completeness": 2,
                "overall": 2.0, "note": f"judge call failed: {type(e).__name__}"}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _api_post(path, json_data=None, token="", timeout=120):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.post(f"{API_BASE}{path}", json=json_data or {}, headers=headers, timeout=timeout)
    return r.json() if r.ok else None


def _stream_answer(token: str, thread_id: str, message: str, timeout=120) -> str:
    """Send message and collect final answer via SSE."""
    r = requests.post(
        f"{API_BASE}/api/chat/stream",
        headers={"Authorization": f"Bearer {token}"},
        json={"thread_id": thread_id, "message": message},
        stream=True,
        timeout=timeout,
    )
    text = ""
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            d = json.loads(line[5:])
            if d.get("done"):
                text = d.get("content", "")
    return text.strip()


def _register_and_login(username: str, pw="test123456"):
    """Register or login, return access_token."""
    for endpoint, data in [("/api/auth/register", {"username": username, "password": pw}),
                            ("/api/auth/login", {"username": username, "password": pw})]:
        r = _api_post(endpoint, json_data=data)
        if r and r.get("access_token"):
            return r["access_token"]
    return None


# ---------------------------------------------------------------------------
# Module 1: LLM-Judge Scoring (20 diverse queries, judged)
# ---------------------------------------------------------------------------
def run_judge_module(token: str = "") -> List[EvaluationResult]:
    """Run 20 diverse queries through the agent and score each response."""
    print("\n=== Module 1: LLM-Judge Scoring ===")

    if not token:
        token = _register_and_login("eval_judge_001")
    tid = _api_post("/api/chat/session", token=token).get("thread_id")

    queries = [
        "我头疼三天了怎么办",
        "高血压平时要注意什么",
        "小孩发烧39度要不要去医院",
        "咳嗽应该挂什么科",
        "帮我挂心内科明天的号",
        "什么降血压药副作用小",
        "怀孕可以吃感冒药吗",
        "糖尿病能吃什么水果",
        "最近总是失眠压力大",
        "降压药能和感冒药一起吃吗",
        "吃饭没规律胃疼怎么办",
        "腰痛直不起来",
        "皮肤红痒是什么原因",
        "老年人补钙吃什么好",
        "头晕眼花是不是贫血",
        "晚上睡觉腿抽筋",
        "跑步膝盖疼",
        "嘴里总是有溃疡",
        "近视手术安全吗",
        "便秘好几天了怎么办",
    ]

    results = []
    for i, q in enumerate(queries):
        # Each query gets its own session
        if i > 0:
            session_r = _api_post("/api/chat/session", token=token)
            if session_r:
                tid = session_r.get("thread_id")

        t0 = time.time()
        answer = _stream_answer(token, tid, q)
        elapsed = time.time() - t0

        judge = _judge_response(q, answer)
        result = EvaluationResult(
            turn_id=f"J{i+1:02d}",
            user_query=q,
            assistant_response=answer[:500],
            response_time_s=elapsed,
            accuracy=judge.get("accuracy", 2),
            safety=judge.get("safety", 2),
            relevance=judge.get("relevance", 2),
            completeness=judge.get("completeness", 2),
            overall=judge.get("overall", 2.0),
            note=judge.get("note", ""),
        )
        results.append(result)
        print(f"  J{i+1:02d} [{elapsed:.0f}s] score={result.overall:.1f} | {q[:30]}")

        time.sleep(3)  # Rate limit protection

    _print_judge_summary(results)
    return results


def _print_judge_summary(results: List[EvaluationResult]):
    if not results:
        return
    n = len(results)
    avg_acc = sum(r.accuracy for r in results) / n
    avg_saf = sum(r.safety for r in results) / n
    avg_rel = sum(r.relevance for r in results) / n
    avg_cmp = sum(r.completeness for r in results) / n
    avg_ovr = sum(r.overall for r in results) / n
    avg_time = sum(r.response_time_s for r in results) / n

    print(f"\n  Scores ({n} queries):")
    print(f"    accuracy:     {avg_acc:.1f}/5")
    print(f"    safety:       {avg_saf:.1f}/5")
    print(f"    relevance:    {avg_rel:.1f}/5")
    print(f"    completeness: {avg_cmp:.1f}/5")
    print(f"    overall:      {avg_ovr:.1f}/5")
    print(f"    avg time:     {avg_time:.0f}s")


# ---------------------------------------------------------------------------
# Module 2: Cross-Session Memory Recall
# ---------------------------------------------------------------------------
def run_memory_module() -> List[MemoryTestResult]:
    """Test that user facts shared in session A are recalled in session B."""
    print("\n=== Module 2: Cross-Session Memory Recall ===")

    scenarios = [
        {
            "persona": "高血压+过敏患者",
            "session_a": ["我有高血压好几年了", "我对青霉素过敏", "一直在吃硝苯地平"],
            "session_b_query": "我感冒了能吃什么药",
            "expected_recall": ["高血压", "青霉素过敏", "硝苯地平"],
        },
        {
            "persona": "糖尿病患者",
            "session_a": ["我去年查出糖尿病", "现在每天打胰岛素", "最近血糖控制得不太好"],
            "session_b_query": "我最近口渴得厉害怎么办",
            "expected_recall": ["糖尿病", "胰岛素", "血糖"],
        },
        {
            "persona": "孕期准妈妈",
            "session_a": ["我怀孕6个月了", "最近总是腰疼", "上次产检血糖有点高"],
            "session_b_query": "腰疼能做什么运动缓解",
            "expected_recall": ["怀孕", "孕期", "产检", "腰疼"],
        },
        {
            "persona": "心脏病患者",
            "session_a": ["我做过心脏搭桥", "现在恢复期", "医生让我适当运动"],
            "session_b_query": "我能不能慢跑",
            "expected_recall": ["心脏搭桥", "运动", "恢复"],
        },
        {
            "persona": "失眠焦虑者",
            "session_a": ["我每天睡不着觉", "吃了安眠药也不管用", "白天没精神"],
            "session_b_query": "有什么不吃药能改善睡眠的方法",
            "expected_recall": ["睡不着", "安眠药", "睡眠"],
        },
        {
            "persona": "老年骨质疏松",
            "session_a": ["我今年72了", "前段时间查出骨密度低", "医生让我补钙"],
            "session_b_query": "我该吃什么补钙",
            "expected_recall": ["骨密度", "补钙", "钙"],
        },
    ]

    results = []
    for scenario in scenarios:
        token = _register_and_login(f"mem_{scenario['persona'][:4]}_{uuid.uuid4().hex[:4]}")
        if not token:
            results.append(MemoryTestResult(persona=scenario["persona"], facts_shared=[],
                                             session_b_query="", assistant_response="",
                                             facts_recalled=[], facts_missed=scenario["expected_recall"],
                                             accuracy=0, note="auth failed"))
            continue

        # Session A: share facts
        tid_a = _api_post("/api/chat/session", token=token).get("thread_id")
        for fact in scenario["session_a"]:
            ans = _stream_answer(token, tid_a, fact)
            time.sleep(2)

        # Session B: new thread, ask question that needs recall
        tid_b = _api_post("/api/chat/session", token=token).get("thread_id")
        answer = _stream_answer(token, tid_b, scenario["session_b_query"])

        # Judge recall
        recalled = []
        missed = []
        for keyword in scenario["expected_recall"]:
            if keyword in answer:
                recalled.append(keyword)
            else:
                missed.append(keyword)

        # Judge score
        judge = _judge_response(scenario["session_b_query"], answer)
        result = MemoryTestResult(
            persona=scenario["persona"],
            facts_shared=scenario["session_a"],
            session_b_query=scenario["session_b_query"],
            assistant_response=answer[:500],
            facts_recalled=recalled,
            facts_missed=missed,
            accuracy=len(recalled),
            note=judge.get("note", ""),
        )
        results.append(result)

        recall_pct = len(recalled) / max(len(scenario["expected_recall"]), 1) * 100
        print(f"  {scenario['persona']:<20} recall={len(recalled)}/{len(scenario['expected_recall'])} ({recall_pct:.0f}%)")

        time.sleep(3)

    _print_memory_summary(results)
    return results


def _print_memory_summary(results: List[MemoryTestResult]):
    if not results:
        return
    total_facts = sum(len(r.facts_recalled) + len(r.facts_missed) for r in results)
    total_recalled = sum(len(r.facts_recalled) for r in results)
    print(f"\n  Memory Recall: {total_recalled}/{total_facts} ({total_recalled/max(total_facts,1)*100:.0f}%)")


# ---------------------------------------------------------------------------
# Module 3: Persona Stress Test with Judge Scoring
# ---------------------------------------------------------------------------
def run_stress_module(count: int = 30) -> List[EvaluationResult]:
    """Run multi-turn persona sessions and judge the final turn of each."""
    print(f"\n=== Module 3: Persona Stress Test ({count} sessions) ===")

    # Import persona sessions
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scripts.stress_test_realistic import PERSONAS

    all_sessions = []
    for pname, pdata in PERSONAS.items():
        for idx, msgs in enumerate(pdata["sessions"]):
            all_sessions.append((pname, idx, msgs))
    all_sessions = all_sessions[:count]

    results = []
    for si, (pname, idx, msgs) in enumerate(all_sessions):
        token = _register_and_login(f"stress_{si}_{uuid.uuid4().hex[:4]}")
        if not token:
            continue

        tid = _api_post("/api/chat/session", token=token).get("thread_id")
        final_answer = ""
        final_query = msgs[-1] if msgs else ""

        for turn_idx, msg in enumerate(msgs):
            t0 = time.time()
            final_answer = _stream_answer(token, tid, msg)
            elapsed = time.time() - t0
            time.sleep(2)

        # Judge the final turn
        judge = _judge_response(final_query, final_answer)
        result = EvaluationResult(
            turn_id=f"{pname}.{idx:02d}",
            user_query=final_query,
            assistant_response=final_answer[:500],
            response_time_s=time.time() - t0 if msgs else 0,
            accuracy=judge.get("accuracy", 2),
            safety=judge.get("safety", 2),
            relevance=judge.get("relevance", 2),
            completeness=judge.get("completeness", 2),
            overall=judge.get("overall", 2.0),
            note=judge.get("note", ""),
        )
        results.append(result)
        status = "OK" if result.overall >= 3 else "WARN"
        print(f"  [{status}] {result.turn_id:<20} score={result.overall:.1f}")

        time.sleep(3)

    _print_judge_summary(results)

    # Per-persona breakdown
    print(f"\n  Per-Persona Scores:")
    persona_scores: Dict[str, List[float]] = {}
    for r in results:
        pname = r.turn_id.split(".")[0]
        persona_scores.setdefault(pname, []).append(r.overall)
    for pname, scores in sorted(persona_scores.items()):
        print(f"    {pname:<15} avg={sum(scores)/len(scores):.1f}  n={len(scores)}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Agent Evaluation Framework")
    p.add_argument("--module", choices=["judge", "memory", "stress", "all"], default="all")
    p.add_argument("--count", type=int, default=30, help="Stress test session count")
    p.add_argument("--output", type=str, default="", help="Output JSON file")
    args = p.parse_args()

    all_results: Dict[str, Any] = {}

    if args.module in ("judge", "all"):
        all_results["judge"] = [r.__dict__ for r in run_judge_module()]

    if args.module in ("memory", "all"):
        all_results["memory"] = [r.__dict__ for r in run_memory_module()]

    if args.module in ("stress", "all"):
        all_results["stress"] = [r.__dict__ for r in run_stress_module(args.count)]

    # Save if requested
    if args.output and all_results:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
