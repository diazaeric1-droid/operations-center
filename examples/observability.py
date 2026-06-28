"""Observability demo — trace LLM calls across providers; see tokens, latency, $.

Answers the same question on each provider you have a key for, traced, then prints
a per-call table + a summary. This is the cost/latency half of "measure everything"
(evals are the quality half) — and it's how you justify a routing decision with
data: "Groq is 4× faster and 10× cheaper than the premium model for this step."

    python examples/observability.py "why does an ESP trip on underload?"
    python examples/observability.py --demo     # synthetic numbers, no keys needed

Costs are ESTIMATES from approximate list prices (free tiers really cost $0); the
dollar figure is "what this would cost at paid scale". Token counts are real when
the provider reports usage.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_Q = "Why does an ESP pump trip on underload?"


def run_workload(question: str, providers: list[str], max_tokens: int = 1024) -> list:
    """Answer `question` on each provider, traced. Returns the TraceEvents.

    A brief system prompt keeps answers short (so cost stays low) while a generous
    max_tokens leaves "thinking" models like gemini-2.5-flash room — a tight cap
    truncates them and makes the cost comparison apples-to-oranges.
    """
    from langgraph_rag import tracing
    brief = "Answer in 1-2 sentences."
    with tracing.trace() as events:
        for p in providers:
            try:
                tracing.chat_traced(question, provider=p, label="answer",
                                    system=brief, max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001 — a missing key/rate-limit is fine
                print(f"  {p}: skipped — {type(e).__name__}: {str(e)[:60]}")
    return events


def _demo_events() -> list:
    """Synthetic events so the report renders with no keys."""
    from langgraph_rag.tracing import TraceEvent, cost_usd
    rows = [
        ("groq", "llama-3.3-70b-versatile", 60, 90, 0.31),
        ("gemini", "gemini-2.5-flash", 58, 150, 2.90),
        ("claude", "claude-sonnet-4-6", 58, 120, 3.40),
        ("openai", "gpt-4o-mini", 58, 110, 1.05),
    ]
    return [TraceEvent(p, m, pt, ct, lat, cost_usd(m, pt, ct), label="answer")
            for p, m, pt, ct, lat in rows]


if __name__ == "__main__":
    from langgraph_rag import tracing
    from langgraph_rag.providers import available

    args = [a for a in sys.argv[1:] if a != "--demo"]
    demo = "--demo" in sys.argv[1:]
    question = " ".join(args) or DEFAULT_Q
    providers = [n for n, ok in available().items() if ok]

    if demo or not providers:
        if not demo:
            print("(no provider keys set — showing synthetic --demo numbers)")
        tracing.print_report(_demo_events())
    else:
        print(f"Tracing one answer per provider: {', '.join(providers)}\n"
              f"Q: {question}")
        tracing.print_report(run_workload(question, providers))
