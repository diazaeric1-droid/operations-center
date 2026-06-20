"""Evals — measure LLM answer quality instead of eyeballing it.

"Build evals/testing frameworks for LLM systems" shows up in most AI-engineer
JDs. This is the simplest honest version: a fixed test set, assertion-based
scoring (must-include terms, must-AVOID terms, length), and per-provider latency
— run the SAME eval across every model and get a comparison table.

It catches real failures automatically. The `must_avoid` groundwater terms below
would have flagged the wrong "aquifer recharge / rainfall" answer a generalist
model gave to an oil-well question — the exact bug we hit by hand.

    python examples/eval_harness.py            # eval every provider whose key is set
    python examples/eval_harness.py groq gemini  # only these

Runs key-free against a deterministic stub so the harness itself is testable.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM = "You are an oil & gas operations assistant. Answer in 1–2 sentences."

# Each case: the question, terms the answer SHOULD contain, terms it must NOT
# (domain hallucinations), and a length cap. Assertion-based — no judge needed.
EVAL_SET = [
    {"id": "esp_underload",
     "q": "Why does an ESP pump trip on underload?",
     "must_include": ["fluid"],
     "must_avoid": ["rainfall", "aquifer", "groundwater", "water table"],
     "max_chars": 400},
    {"id": "water_increase",
     "q": "Why might an oil well produce more water over time?",
     "must_include": ["reservoir"],
     "must_avoid": ["rainfall", "groundwater", "water table", "drinking"],
     "max_chars": 400},
    {"id": "separator_highlevel",
     "q": "What causes a separator to shut a well in on high level?",
     "must_include": ["level"],
     "must_avoid": ["rainfall", "aquifer"],
     "max_chars": 400},
]


# --- the system under test: a model call (swap in any answer_fn) --------------
def model_answer(provider: str, question: str) -> str:
    if provider == "stub":
        return _stub(question)
    from langgraph_rag.providers import chat
    return chat(question, provider=provider, system=SYSTEM, max_tokens=1024)


def _stub(question: str) -> str:
    q = question.lower()
    if "underload" in q:
        return "Insufficient fluid over the pump lowers motor load, so the drive trips the ESP."
    if "water" in q:
        return "As the reservoir depletes, water encroaches toward the wellbore, raising water cut."
    return "A high liquid level trips the separator to protect downstream equipment."


def pick_all_available() -> list[str]:
    from langgraph_rag.providers import available
    return [name for name, ok in available().items() if ok] or ["stub"]


# --- scoring ------------------------------------------------------------------
def score_case(case: dict, answer: str, latency: float) -> dict:
    a = answer.lower()
    inc = [k for k in case["must_include"] if k in a]
    coverage = len(inc) / max(len(case["must_include"]), 1)
    violations = [k for k in case["must_avoid"] if k in a]
    len_ok = len(answer) <= case["max_chars"]
    passed = coverage >= 0.5 and not violations and len_ok
    return {"id": case["id"], "passed": passed, "coverage": coverage,
            "violations": violations, "len_ok": len_ok,
            "chars": len(answer), "latency": round(latency, 2), "answer": answer}


def run_eval(providers: list[str],
             answer_fn: Callable[[str, str], str] = model_answer) -> dict:
    """{provider: [case score, ...]} — the same eval set on every provider."""
    out: dict[str, list[dict]] = {}
    for prov in providers:
        rows = []
        for case in EVAL_SET:
            t = time.time()
            try:
                ans = answer_fn(prov, case["q"])
            except Exception as e:  # a provider erroring is itself a result
                ans = f"[ERROR: {type(e).__name__}]"
            rows.append(score_case(case, ans, time.time() - t))
        out[prov] = rows
    return out


def report(results: dict) -> None:
    print(f"\n{'provider':12s} {'pass':>7s} {'avg cov':>8s} "
          f"{'violations':>11s} {'avg lat':>8s}")
    print("  " + "-" * 50)
    for prov, rows in results.items():
        n = len(rows)
        passed = sum(r["passed"] for r in rows)
        cov = sum(r["coverage"] for r in rows) / n
        viol = sum(len(r["violations"]) for r in rows)
        lat = sum(r["latency"] for r in rows) / n
        print(f"{prov:12s} {passed}/{n:<5d} {cov:7.0%} {viol:>11d} {lat:7.2f}s")
    # surface the actual failures — the point of an eval is to SEE them
    print()
    for prov, rows in results.items():
        for r in rows:
            if not r["passed"]:
                why = (f"avoided-term {r['violations']}" if r["violations"]
                       else "missing required term" if r["coverage"] < 0.5
                       else "too long")
                print(f"  ✗ {prov}/{r['id']}: {why}")
                print(f"      “{r['answer'][:90]}…”")


if __name__ == "__main__":
    providers = sys.argv[1:] or pick_all_available()
    print(f"Eval set: {len(EVAL_SET)} cases · providers: {', '.join(providers)}")
    report(run_eval(providers))
