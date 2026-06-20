"""Evals — measure LLM answer quality instead of eyeballing it.

"Build evals/testing frameworks for LLM systems" shows up in most AI-engineer
JDs. This harness combines the three building blocks a real one uses:

  1. ASSERTION checks  — must-include / must-AVOID terms (word-boundary matched)
                         + length. Exact and cheap.
  2. LLM-as-JUDGE      — a model grades each answer 1–5 on a rubric (quality the
                         assertions can't see). The judge is blind to which
                         provider wrote the answer (removes *label* bias).
  3. MULTI-SAMPLING    — run each case N times at temperature>0 and report a PASS
                         RATE, so a model that's wrong 1-in-3 can't look perfect
                         by being asked once.

Run the same eval across every provider for an apples-to-apples table:

    python examples/eval_harness.py                       # every provider w/ a key
    python examples/eval_harness.py groq gemini --runs 3  # 3 samples each
    python examples/eval_harness.py groq --runs 3 --judge gemini --temp 0.7

LIMITATIONS (an eval is only as honest as its disclosure):
  * It's a 3-case SMOKE SET — illustrative, not statistically meaningful.
  * The judge is UNVALIDATED against human ground truth, and a SINGLE judge can
    self-prefer (favor its own family's style) even when blind to labels — so
    the judge column is indicative, not authoritative. Use a judge from a
    different model family than the system under test (the harness warns if you
    judge a provider with itself).
  * Assertion term-matching is word-boundary but still approximate.
  * Temperature trade-off: --temp 0 is reproducible but makes multi-sampling
    only catch transient API errors, not model nondeterminism; the default 0.7
    lets --runs surface real flakiness.

OBSERVED (a real run, documented as a teaching case): on "why might an oil well
produce more water?", Groq repeatedly answered with *groundwater/"water table"*
framing — pass-rate 0% on the must_avoid check — yet Gemini-as-judge scored it
~5/5. The hard assertion caught a domain hallucination the LLM judge missed.
That disagreement is the whole reason to run assertions AND a judge: neither
alone is sufficient.

Runs key-free against a deterministic stub (model AND judge) so it's testable.
NOTE: free tiers rate-limit (~15–30 req/min) — runs × cases × providers (× judge)
adds up; keep --runs small on free keys.
"""
from __future__ import annotations

import os
import re
import sys
import time
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM = "You are an oil & gas operations assistant. Answer in 1–2 sentences."
DEFAULT_TEMP = 0.7   # >0 so multi-sampling can actually surface model flakiness

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
def model_answer(provider: str, question: str, temperature: float = DEFAULT_TEMP) -> str:
    if provider == "stub":
        return _stub(question)
    from langgraph_rag.providers import chat
    return chat(question, provider=provider, system=SYSTEM, max_tokens=1024,
                temperature=temperature)


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


# --- LLM-as-judge -------------------------------------------------------------
JUDGE_SYSTEM = (
    "You are a strict petroleum-engineering grader. Score how well the ANSWER "
    "answers the QUESTION, from 1 (wrong or irrelevant) to 5 (correct, relevant, "
    "concise). A confident but WRONG answer (e.g. groundwater/rainfall for an "
    "oil-well question) scores 1. Reply EXACTLY:  SCORE: <1-5> | REASON: <short>")


def judge_answer(judge_provider: str, question: str, answer: str) -> dict:
    if judge_provider == "stub":
        return _stub_judge(answer)
    from langgraph_rag.providers import chat
    raw = chat(f"QUESTION: {question}\nANSWER: {answer}", provider=judge_provider,
               system=JUDGE_SYSTEM, max_tokens=1024)
    return _parse_judge(raw)


def _parse_judge(raw: str) -> dict:
    """Parse 'SCORE: n | REASON: ...' robustly. Case-insensitive; never raises;
    score is None when unparseable or out of 1–5 (None != a default 3)."""
    m = re.search(r"SCORE:\s*([1-5])\b", raw, re.I)
    score = int(m.group(1)) if m else None
    rm = re.split(r"REASON:", raw, maxsplit=1, flags=re.I)
    reason = (rm[1] if len(rm) > 1 else raw).strip()[:140]
    return {"score": score, "reason": reason}


def _stub_judge(answer: str) -> dict:
    a = answer.lower()
    bad = any(t in a for t in ("rainfall", "aquifer", "groundwater", "water table"))
    return {"score": 1 if bad else 4,
            "reason": "stub: domain hallucination" if bad else "stub: on-domain"}


# --- assertion scoring (one run) ----------------------------------------------
def _has_term(text: str, term: str) -> bool:
    """Word-boundary match (so 'level' doesn't match 'levelheaded')."""
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def score_case(case: dict, answer: str, latency: float,
               errored: Optional[bool] = None) -> dict:
    # infra failure = the answer_fn RAISED (run_case passes that ground truth).
    # Fall back to the text prefix only for direct callers that don't pass it.
    if errored is None:
        errored = answer.startswith("[ERROR:")
    a = answer.lower()
    inc = [k for k in case["must_include"] if _has_term(a, k)]
    coverage = len(inc) / max(len(case["must_include"]), 1)
    violations = [k for k in case["must_avoid"] if _has_term(a, k)]
    len_ok = len(answer) <= case["max_chars"]
    passed = (not errored) and coverage >= 0.5 and not violations and len_ok
    return {"id": case["id"], "passed": passed, "coverage": coverage,
            "violations": violations, "len_ok": len_ok, "errored": errored,
            "chars": len(answer), "latency": round(latency, 2), "answer": answer}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


# --- one case, N samples, optional judge --------------------------------------
def run_case(case: dict, provider: str, answer_fn: Callable[[str, str], str],
             judge_provider: Optional[str], n_runs: int) -> dict:
    runs = []
    for _ in range(max(1, n_runs)):
        t = time.time()
        raised = False
        try:
            ans = answer_fn(provider, case["q"])
        except Exception as e:  # a provider erroring IS a result (infra, not quality)
            ans, raised = f"[ERROR: {type(e).__name__}: {e}]", True
        s = score_case(case, ans, time.time() - t, errored=raised)
        if judge_provider and not s["errored"]:   # don't waste the judge on error rows
            try:
                j = judge_answer(judge_provider, case["q"], ans)
            except Exception as e:  # noqa: BLE001
                j = {"score": None, "reason": f"judge error: {type(e).__name__}"}
            s["judge_score"], s["judge_reason"] = j["score"], j["reason"]
        runs.append(s)

    n = len(runs)
    ok = [r for r in runs if not r["errored"]]    # runs we can actually score for quality
    lats = [r["latency"] for r in runs]
    agg = {
        "id": case["id"], "n_runs": n, "errors": n - len(ok),
        # quality metrics ignore errored runs — a 429 is not a wrong answer.
        # pass_rate is None when ALL runs errored (couldn't evaluate quality at all).
        "pass_rate": (sum(r["passed"] for r in ok) / len(ok)) if ok else None,
        "passed": bool(ok) and all(r["passed"] for r in ok),
        "coverage": (sum(r["coverage"] for r in ok) / len(ok)) if ok else 0.0,
        "violations": sorted({v for r in ok for v in r["violations"]}),
        "avg_latency": round(sum(lats) / n, 2),
        "p95_latency": round(_percentile(lats, 0.95), 2),
        "answer": runs[-1]["answer"], "runs": runs,
    }
    if judge_provider:
        js = [r["judge_score"] for r in runs if r.get("judge_score") is not None]
        agg["judge_graded"] = len(js)             # how many runs the judge actually graded
        agg["judge_attempts"] = len(ok)           # only non-error runs are sent to the judge
        agg["judge_score"] = round(sum(js) / len(js), 2) if js else None
    return agg


def self_judging(providers: list[str], judge: Optional[str]) -> bool:
    """True when the judge is also one of the graded providers (self-preference risk)."""
    return bool(judge) and judge in providers


def run_eval(providers: list[str],
             answer_fn: Optional[Callable[[str, str], str]] = None,
             n_runs: int = 1, judge: Optional[str] = None,
             temperature: float = DEFAULT_TEMP) -> dict:
    """{provider: [case aggregate, ...]} — the same eval set on every provider.

    Each row carries id/passed/coverage/violations/pass_rate plus latency stats
    (and judge_score/judge_graded when judge is set). n_runs>1 reports a pass RATE
    per case; judge=<provider> adds a 1–5 LLM-as-judge score. answer_fn defaults to
    model_answer at `temperature`; pass your own (provider, question)->str to eval
    an agent instead.
    """
    fn = answer_fn or (lambda p, q: model_answer(p, q, temperature))
    return {prov: [run_case(c, prov, fn, judge, n_runs) for c in EVAL_SET]
            for prov in providers}


def report(results: dict, judge: Optional[str] = None) -> None:
    has_judge = judge is not None
    head = f"\n{'provider':12s} {'pass-rate':>10s} {'avg cov':>8s} {'viol':>5s} {'err':>4s}"
    head += f" {'judge':>9s}" if has_judge else ""
    head += f" {'p95 lat':>8s}"
    print(head)
    print("  " + "-" * (72 if has_judge else 60))
    for prov, rows in results.items():
        scored = [r for r in rows if r["pass_rate"] is not None]   # drop all-errored cases
        prs = [r["pass_rate"] for r in scored]
        pr_str = f"{sum(prs) / len(prs):9.0%}" if prs else f"{'—':>9s}"
        cov = sum(r["coverage"] for r in scored) / len(scored) if scored else 0.0
        viol = sum(len(r["violations"]) for r in rows)
        errs = sum(r.get("errors", 0) for r in rows)
        p95 = max(r["p95_latency"] for r in rows)
        line = f"{prov:12s} {pr_str} {cov:8.0%} {viol:5d} {errs:4d}"
        if has_judge:
            js = [r["judge_score"] for r in rows if r.get("judge_score") is not None]
            graded = sum(r.get("judge_graded", 0) for r in rows)
            attempts = sum(r.get("judge_attempts", 0) for r in rows)
            # surface judge COVERAGE so a 1-of-N parsed score can't masquerade clean
            line += (f" {sum(js) / len(js):4.1f}({graded}/{attempts})" if js
                     else f" {'n/a':>9s}")
        line += f" {p95:7.2f}s"
        print(line)

    # surface failures, distinguishing INFRA errors (⚠) from QUALITY failures (✗)
    print()
    for prov, rows in results.items():
        for r in rows:
            if r["pass_rate"] is None:
                print(f"  ⚠ {prov}/{r['id']}: {r['errors']}/{r['n_runs']} runs ERRORED "
                      "(infrastructure, e.g. rate-limit — NOT a model quality failure)")
                print(f"      “{r['answer'][:90]}…”")
            elif r["pass_rate"] < 1.0:
                why = (f"avoided-term {r['violations']}" if r["violations"]
                       else "missing required term" if r["coverage"] < 0.5
                       else "too long")
                print(f"  ✗ {prov}/{r['id']}: pass-rate {r['pass_rate']:.0%} ({why})")
                print(f"      “{r['answer'][:90]}…”")
            elif r.get("errors", 0):   # some runs errored but the survivors all passed
                print(f"  · {prov}/{r['id']}: {r['errors']}/{r['n_runs']} runs errored "
                      f"(infra); pass-rate over the {r['n_runs'] - r['errors']} survivor(s)")


def _parse_argv(argv: list[str]):
    judge, runs, temp, provs = None, 1, DEFAULT_TEMP, []
    i = 0
    while i < len(argv):
        if argv[i] in ("--judge", "--runs", "--temp"):
            if i + 1 >= len(argv):
                raise SystemExit(f"{argv[i]} requires a value")
            val = argv[i + 1]
            if argv[i] == "--judge":
                judge = val
            elif argv[i] == "--runs":
                runs = int(val)
            else:
                temp = float(val)
            i += 2
        else:
            provs.append(argv[i]); i += 1
    return provs, runs, judge, temp


if __name__ == "__main__":
    provs, runs, judge, temp = _parse_argv(sys.argv[1:])
    providers = provs or pick_all_available()
    if self_judging(providers, judge):
        print(f"⚠ self-judging: {judge} grades its own answers — self-preference "
              f"bias likely; prefer a judge from a different model family.")
    print(f"Eval set: {len(EVAL_SET)} cases × {runs} run(s) @ temp {temp} · "
          f"providers: {', '.join(providers)}"
          + (f" · judge: {judge}" if judge else ""))
    report(run_eval(providers, n_runs=runs, judge=judge, temperature=temp),
           judge=judge)
