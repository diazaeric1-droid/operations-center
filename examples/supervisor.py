"""Multi-agent supervisor — a router agent dispatches to specialist agents.

The pattern every "agentic" job posting means by *multi-agent*: a **supervisor**
reads the task and routes it to the right **specialist worker**, each of which is
its own agent with a focused system prompt (and can run on its own model). It's
the fix for the failure you just saw — a generalist model gave a *groundwater*
answer to an oil-well question; a domain specialist with the right prompt won't.

    START → supervisor ─┬─► production_expert ─► END
                        ├─► facilities_expert ─► END
                        └─► reservoir_expert  ─► END

The supervisor's routing is a conditional edge (the BRANCH); each expert is a
separate node (a separate "agent"). Supervisor runs on a cheap/fast model; the
experts on the best available. Runs key-free via a stub fallback.

    python examples/supervisor.py "my ESP keeps tripping on underload, why?"
"""
from __future__ import annotations

import operator
import os
import sys
from typing import Annotated, Optional, TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_Q = "My ESP keeps tripping on underload — what's likely going on?"

SPECIALISTS = {
    "production": "You are a PRODUCTION ENGINEER. Focus on wells, artificial lift "
                  "(ESP, rod pump, gas lift), and well performance.",
    "facilities": "You are a FACILITIES ENGINEER. Focus on surface equipment — "
                  "separators, treaters, compressors, tanks, LACT units.",
    "reservoir": "You are a RESERVOIR ENGINEER. Focus on subsurface behavior — "
                 "pressure, water encroachment, depletion, recovery.",
}
# keyword fallback so the supervisor (and the stub) can route with no LLM.
_HINTS = {
    "production": ("esp", "pump", "rod", "lift", "underload", "vsd", "well", "tubing"),
    "facilities": ("separator", "treater", "compressor", "tank", "lact", "facility",
                   "battery", "valve"),
    "reservoir": ("water", "pressure", "reservoir", "depletion", "aquifer", "gor",
                  "decline", "injection"),
}


def _route_by_keyword(q: str) -> str:
    ql = q.lower()
    best, score = "production", 0
    for name, words in _HINTS.items():
        s = sum(w in ql for w in words)
        if s > score:
            best, score = name, s
    return best


def call(provider: str, prompt: str, system: Optional[str] = None) -> str:
    if provider == "stub":
        return f"[{(system or '').split('.')[0]}] {prompt[:60]}… — diagnosis here."
    from langgraph_rag.providers import chat
    return chat(prompt, provider=provider, system=system, max_tokens=1024)


def pick(prefs: list[str]) -> str:
    from langgraph_rag.providers import available
    av = available()
    return next((p for p in prefs if av.get(p)), "stub")


# --- state + nodes ------------------------------------------------------------
class State(TypedDict):
    question: str
    supervisor_model: str
    expert_model: str
    route: str
    answer: str
    trace: Annotated[list, operator.add]


def _supervisor(state: State) -> dict:
    """Pick the specialist. Try the LLM, validate, fall back to keywords."""
    prov = state["supervisor_model"]
    choice = _route_by_keyword(state["question"])
    if prov != "stub":
        out = call(prov, f"Route this question to ONE specialist — "
                         f"production, facilities, or reservoir. Reply with only "
                         f"that word.\nQuestion: {state['question']}").lower()
        choice = next((s for s in SPECIALISTS if s in out), choice)
    return {"route": choice,
            "trace": [f"supervisor [{prov}] → routed to the {choice} specialist"]}


def _make_expert(name: str):
    def expert(state: State) -> dict:
        out = call(state["expert_model"], state["question"], system=SPECIALISTS[name])
        return {"answer": out,
                "trace": [f"{name}_expert [{state['expert_model']}] → answered"]}
    return expert


def _route(state: State) -> str:
    return state["route"]


def build():
    from langgraph.graph import StateGraph, START, END
    g = StateGraph(State)
    g.add_node("supervisor", _supervisor)
    for name in SPECIALISTS:
        g.add_node(f"{name}_expert", _make_expert(name))
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", _route,
                            {n: f"{n}_expert" for n in SPECIALISTS})
    for name in SPECIALISTS:
        g.add_edge(f"{name}_expert", END)
    return g.compile()


def run(question: str, supervisor_model: Optional[str] = None,
        expert_model: Optional[str] = None) -> State:
    supervisor_model = supervisor_model or pick(["groq", "gemini", "github", "openai"])
    expert_model = expert_model or pick(["claude", "gemini", "groq", "github"])
    return build().invoke({
        "question": question, "supervisor_model": supervisor_model,
        "expert_model": expert_model, "route": "", "answer": "", "trace": []})


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or DEFAULT_Q
    sm = pick(["groq", "gemini", "github", "openai"])
    em = pick(["claude", "gemini", "groq", "github"])
    print(f"QUESTION: {q}")
    print(f"MODELS — supervisor: {sm}   |   specialists: {em}\n")
    final = run(q, supervisor_model=sm, expert_model=em)
    for line in final["trace"]:
        print("  •", line)
    print(f"\nSPECIALIST: {final['route']}")
    print(f"ANSWER: {final['answer']}")
