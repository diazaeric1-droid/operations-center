"""Model routing — one agent, a DIFFERENT model per step.

The senior move beyond "pick one model": route each step to the right model.
Cheap/fast models do the grunt work (classify, first draft); a premium model
does the step where quality matters (the final polish). You save cost and latency
without giving up quality on the part that counts.

Pipeline (a LangGraph graph; each node calls its own provider):

    classify  ─ cheap  ─►  draft  ─ cheap  ─►  polish  ─ premium  ─►  END

Each node prints which model handled it and how long it took, so you can SEE the
routing (and why cheap models are worth using for the easy steps — they're fast).

Run:
    python examples/model_routing.py "why might a well start making more water?"

Providers are auto-chosen from the keys you have set:
    cheap   = first of  groq / gemini / github / openrouter
    premium = first of  claude / gemini / groq
…with a no-key stub fallback so it always runs.
"""
from __future__ import annotations

import operator
import os
import sys
import time
from typing import Annotated, Optional, TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_Q = "Why might an oil well start producing more water over time?"


# --- model call: real provider or a no-key stub ------------------------------
def call(provider: str, prompt: str, system: Optional[str] = None) -> str:
    if provider == "stub":
        return _stub(prompt)
    from langgraph_rag.providers import chat
    return chat(prompt, provider=provider, system=system, max_tokens=1024)


def _stub(prompt: str) -> str:
    p = prompt.lower()
    if "one-word" in p or "category" in p:
        return "reservoir"
    if "polish" in p or "rewrite" in p:
        return ("Rising water cut usually means the reservoir is watering out — "
                "water encroaching toward the wellbore as pressure depletes.")
    return "Water increases as the reservoir depletes and water moves toward the well."


def pick(prefs: list[str]) -> str:
    """First provider in `prefs` whose key is set, else 'stub'."""
    from langgraph_rag.providers import available
    av = available()
    for p in prefs:
        if av.get(p):
            return p
    return "stub"


# --- graph state + nodes ------------------------------------------------------
class State(TypedDict):
    question: str
    cheap: str                 # provider for the grunt steps
    premium: str               # provider for the final step
    category: str
    draft: str
    answer: str
    trace: Annotated[list, operator.add]


def _timed(provider: str, prompt: str, system: Optional[str] = None):
    t = time.time()
    out = call(provider, prompt, system=system)
    return out, time.time() - t


def _classify(state: State) -> dict:
    out, dt = _timed(state["cheap"],
                     f"Reply with ONE-WORD category for this oil & gas question: "
                     f"{state['question']}")
    cat = out.split()[0].strip(".,:") if out.split() else "general"
    return {"category": cat,
            "trace": [f"classify [{state['cheap']}, {dt:.1f}s] → {cat}"]}


def _draft(state: State) -> dict:
    out, dt = _timed(state["cheap"],
                     f"Briefly answer this {state['category']} question in 1–2 "
                     f"sentences: {state['question']}")
    return {"draft": out, "trace": [f"draft    [{state['cheap']}, {dt:.1f}s] "
                                    f"→ {len(out)} chars"]}


def _polish(state: State) -> dict:
    out, dt = _timed(
        state["premium"],
        f"Polish this draft into a clear, accurate 1–2 sentence answer. "
        f"Question: {state['question']}\nDraft: {state['draft']}",
        system="You are a precise petroleum-engineering editor.")
    return {"answer": out, "trace": [f"polish   [{state['premium']}, {dt:.1f}s] "
                                    f"→ final answer"]}


def build():
    from langgraph.graph import StateGraph, START, END
    g = StateGraph(State)
    g.add_node("classify", _classify)
    g.add_node("draft", _draft)
    g.add_node("polish", _polish)
    g.add_edge(START, "classify")
    g.add_edge("classify", "draft")
    g.add_edge("draft", "polish")     # each edge hands off to a node with its own model
    g.add_edge("polish", END)
    return g.compile()


def run(question: str, cheap: Optional[str] = None,
        premium: Optional[str] = None) -> State:
    cheap = cheap or pick(["groq", "gemini", "github", "openrouter", "openai"])
    premium = premium or pick(["claude", "gemini", "groq", "github", "openrouter"])
    app = build()
    return app.invoke({"question": question, "cheap": cheap, "premium": premium,
                       "category": "", "draft": "", "answer": "", "trace": []})


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or DEFAULT_Q
    c = pick(["groq", "gemini", "github", "openrouter", "openai"])
    p = pick(["claude", "gemini", "groq", "github", "openrouter"])
    print(f"QUESTION: {q}")
    print(f"ROUTING — grunt steps: {c}   |   final polish: {p}\n")
    final = run(q, cheap=c, premium=p)
    for line in final["trace"]:
        print("  •", line)
    print(f"\nCATEGORY: {final['category']}")
    print(f"DRAFT  ({c}): {final['draft']}")
    print(f"FINAL  ({p}): {final['answer']}")
