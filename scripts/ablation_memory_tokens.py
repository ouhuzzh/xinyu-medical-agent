"""Ablation study for memory token reduction — extended version.

9 samples across 5/10/20/30/40-turn gradients, multiple categories.
Groups results by turn count and category for defensible analysis.
"""

import json
from pathlib import Path

import tiktoken
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "scripts" / "ablation_data" / "memory_samples_extended.json"
SHORT_TERM_WINDOW_SIZE = 12


def _estimate_tokens(messages: list) -> int:
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(str(getattr(m, "content", "") or ""))) for m in messages)


def _history_messages(turns):
    msgs = []
    for line in turns:
        line = line.strip()
        if line.startswith("User: "):
            msgs.append(HumanMessage(content=line[6:]))
        elif line.startswith("Assistant: "):
            msgs.append(AIMessage(content=line[10:]))
    return msgs


def _build_state_messages(state):
    if not state:
        return []
    parts = []
    for key in ("intent", "risk_level", "pending_clarification", "clarification_target",
                "topic_focus", "deferred_user_question", "secondary_intent",
                "recommended_department", "appointment_context", "appointment_skill_mode",
                "appointment_candidates", "pending_action_type", "pending_action_payload",
                "pending_confirmation_id", "pending_candidates"):
        val = state.get(key)
        if val and val != "" and val != [] and val != {}:
            parts.append(f"{key.replace('_', ' ').capitalize()}: {val}")
    return [SystemMessage(content="Conversation state context:\n" + "\n".join(parts))] if parts else []


def run():
    with open(DATA_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    window = SHORT_TERM_WINDOW_SIZE * 2
    rows = []

    for s in samples:
        history = _history_messages(s["history_turns"])
        n_turns = len(s["history_turns"])
        summary = s.get("conversation_summary", "")
        state = s.get("session_state", {})
        q = s["current_question"]

        # BASELINE: all history
        baseline = _estimate_tokens([*history, HumanMessage(content=q)])

        # FULL_OPTIMIZED
        full_msgs = []
        if summary:
            full_msgs.append(SystemMessage(content=f"Conversation summary:\n{summary}"))
        full_msgs.extend(_build_state_messages(state))
        full_msgs.extend(history[-window:])
        full_msgs.append(HumanMessage(content=q))
        full = _estimate_tokens(full_msgs)

        # NO_SUMMARY
        ns = [*_build_state_messages(state), *history[-window:], HumanMessage(content=q)]
        no_summary = _estimate_tokens(ns)

        # NO_WINDOW
        nw = []
        if summary:
            nw.append(SystemMessage(content=f"Conversation summary:\n{summary}"))
        nw.extend(_build_state_messages(state))
        nw.extend(history)
        nw.append(HumanMessage(content=q))
        no_window = _estimate_tokens(nw)

        # NO_STATE
        nst = []
        if summary:
            nst.append(SystemMessage(content=f"Conversation summary:\n{summary}"))
        nst.extend(history[-window:])
        nst.append(HumanMessage(content=q))
        no_state = _estimate_tokens(nst)

        # NO_COMPRESSION (verbatim summary)
        verbatim = "\n".join(s["history_turns"])
        nc = [SystemMessage(content=f"Conversation summary:\n{verbatim}"),
              *_build_state_messages(state), *history[-window:], HumanMessage(content=q)]
        no_compress = _estimate_tokens(nc)

        reduction = baseline - full
        rate = reduction / baseline if baseline > 0 else 0

        rows.append({
            "id": s["id"], "category": s["category"], "turns": n_turns,
            "baseline": baseline, "full": full, "no_summary": no_summary,
            "no_window": no_window, "no_state": no_state, "no_compress": no_compress,
            "reduction": reduction, "rate": round(rate, 4),
            "summary_delta": full - no_summary,   # positive = summary adds tokens
            "window_delta": no_window - full,       # positive = window saves tokens
            "state_delta": full - no_state,         # positive = state adds tokens
            "compression_delta": no_compress - full, # positive = compression saves tokens
        })

    # ── Print per-sample detail ──
    print("=" * 80)
    print("ABLATION STUDY: Memory Token Reduction (Extended)")
    print("=" * 80)
    print(f"\nSamples: {len(rows)}   Window: {window} msgs   Tokenizer: tiktoken cl100k_base\n")

    print(f"{'ID':<35} {'Cat':<20} {'Turns':>5} {'Base':>6} {'Full':>6} {'Rate':>7}")
    print("-" * 80)
    for r in rows:
        print(f"{r['id']:<35} {r['category']:<20} {r['turns']:>5} {r['baseline']:>6} {r['full']:>6} {r['rate']:>6.1%}")

    # ── Group by turn count bucket ──
    buckets = {"5": [], "10": [], "20": [], "30": [], "40": []}
    for r in rows:
        t = r["turns"]
        if t <= 6:
            buckets["5"].append(r)
        elif t <= 12:
            buckets["10"].append(r)
        elif t <= 22:
            buckets["20"].append(r)
        elif t <= 32:
            buckets["30"].append(r)
        else:
            buckets["40"].append(r)

    def _avg(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 1) if rows else 0

    print("\n" + "=" * 80)
    print("GROUP BY CONVERSATION LENGTH")
    print("=" * 80)
    print(f"\n{'Bucket':>8} {'N':>3} {'AvgBase':>8} {'AvgFull':>8} {'AvgRate':>8} {'WindowΔ':>9} {'CompressΔ':>10} {'SummaryΔ':>9} {'StateΔ':>7}")
    print("-" * 80)
    for bucket_name, bucket_rows in buckets.items():
        if not bucket_rows:
            continue
        n = len(bucket_rows)
        print(f"{bucket_name:>8} {n:>3} {_avg(bucket_rows,'baseline'):>8} {_avg(bucket_rows,'full'):>8} "
              f"{_avg(bucket_rows,'rate'):>7.1%} {_avg(bucket_rows,'window_delta'):>+9.0f} "
              f"{_avg(bucket_rows,'compression_delta'):>+10.0f} {_avg(bucket_rows,'summary_delta'):>+9.0f} "
              f"{_avg(bucket_rows,'state_delta'):>+7.0f}")

    # ── Group by category ──
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)

    print("\n" + "=" * 80)
    print("GROUP BY SCENARIO CATEGORY")
    print("=" * 80)
    print(f"\n{'Category':<25} {'N':>3} {'AvgTurns':>9} {'AvgBase':>8} {'AvgFull':>8} {'AvgRate':>8}")
    print("-" * 80)
    for cat, cat_rows in sorted(cats.items()):
        n = len(cat_rows)
        print(f"{cat:<25} {n:>3} {_avg(cat_rows,'turns'):>9.0f} {_avg(cat_rows,'baseline'):>8} "
              f"{_avg(cat_rows,'full'):>8} {_avg(cat_rows,'rate'):>7.1%}")

    # ── Overall ──
    n = len(rows)
    overall_rate = round(sum(r["rate"] for r in rows) / n, 4) if n else 0
    print(f"\n{'OVERALL':<25} {n:>3} {_avg(rows,'turns'):>9.0f} {_avg(rows,'baseline'):>8} "
          f"{_avg(rows,'full'):>8} {overall_rate:>7.1%}")

    # ── Component contribution ──
    print("\n" + "=" * 80)
    print("PER-COMPONENT CONTRIBUTION (across all samples)")
    print("=" * 80)
    total_saved = _avg(rows, "reduction")
    for name, key in [("Windowed recent context", "window_delta"),
                       ("LLM compression vs verbatim", "compression_delta"),
                       ("Conversation summary", "summary_delta"),
                       ("State messages (overhead)", "state_delta")]:
        val = _avg(rows, key)
        pct = (abs(val) / total_saved * 100) if total_saved > 0 and val > 0 else 0
        direction = "saves" if val > 0 else "adds"
        print(f"  {name:<35} {val:>+7.0f} tokens ({direction})  "
              f"{'('+f'{pct:.0f}% of savings)' if val > 0 else '(overhead)'}")

    # Save
    out = REPO_ROOT / "scripts" / "ablation_memory_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "buckets": {k: len(v) for k, v in buckets.items()}, "overall_rate": overall_rate},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run()
