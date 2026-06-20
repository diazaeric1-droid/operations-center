"""Same task, two ways, on ANY model — see when you need LangGraph, and see the
model-agnostic pattern in action.

Task: write a one-line summary that must be <= LIMIT characters.

  (a) PROMPT mode  — one model call. If the result is too long, you're stuck:
      a single call has no way to react to its own output.
  (b) GRAPH mode   — a LangGraph graph that checks the length and LOOPS
      (ask → check → "make it shorter" → ask …) until the constraint is met.

The model is swappable — the SAME graph runs on Claude, Gemini, Groq, GitHub
Models (GPT-4o), OpenRouter, or OpenAI. That's the "provider-agnostic" skill:
the agent's logic doesn't care which model answers.

Run it:
    python examples/prompt_vs_graph.py                  # both modes, auto-pick a model
    python examples/prompt_vs_graph.py graph gemini     # the loop, on Gemini
    python examples/prompt_vs_graph.py both groq         # both, on Groq (Llama)
    python examples/prompt_vs_graph.py prompt stub       # no key needed (deterministic)

Provider defaults to "auto" (first one whose API key is set; else a no-key stub
so it always runs). Keys: see langgraph_rag/providers.py.
"""
from __future__ import annotations

import operator
import sys
from typing import Annotated, TypedDict

TASK = "Explain why an ESP pump trips on underload."
LIMIT = 80   # the summary must fit in this many characters


# --- "the model": one function, any provider ---------------------------------
def call_model(instruction: str, provider: str) -> str:
    if provider == "stub":
        return _stub(instruction)
    from langgraph_rag.providers import chat        # the model-agnostic layer
    return chat(instruction, provider=provider, max_tokens=120)


def _stub(instruction: str) -> str:
    """No-key stand-in: returns a shorter answer the more insistently you ask."""
    variants = [
        "An ESP trips on underload when there is too little fluid over the pump "
        "and the drive shuts it down to protect the motor",
        "ESP underload trip: too little fluid over the pump, the drive shuts it "
        "down to protect the motor",
        "ESP underload: low fluid over the pump, drive trips to protect the motor",
        "Underload = low fluid over pump; drive trips the motor",
    ]
    return variants[min(instruction.lower().count("shorter"), len(variants) - 1)]


def resolve_provider(arg: str | None) -> str:
    """'auto' (or no arg) -> first provider with a key set, else the stub."""
    if arg and arg != "auto":
        return arg
    from langgraph_rag.providers import first_available
    return first_available() or "stub"


# --- (a) PROMPT mode: one call, no recourse ----------------------------------
def run_prompt(provider: str = "stub") -> tuple[str, bool]:
    out = call_model(f"In one short sentence: {TASK}", provider)
    ok = len(out) <= LIMIT
    print("\n[a] PROMPT — one model call")
    print(f"    \"{out}\"")
    print(f"    {len(out)} chars · " +
          ("fits ✓" if ok else f"OVER {LIMIT} ✗ — a single prompt can't react to "
                               "its own output"))
    return out, ok


# --- (b) GRAPH mode: loop until the constraint is met ------------------------
class State(TypedDict):
    instruction: str
    draft: str
    rounds: int
    provider: str                          # which model to call (passed in state)
    trace: Annotated[list, operator.add]   # reducer: append, don't overwrite


def _draft(state: State) -> dict:
    out = call_model(state["instruction"], state["provider"])
    return {"draft": out,
            "trace": [f"round {state['rounds']}: {len(out)} chars — \"{out[:46]}…\""]}


def _too_long(state: State) -> str:           # the conditional edge (BRANCH)
    if len(state["draft"]) <= LIMIT or state["rounds"] >= 4:
        return "done"
    return "shorten"


def _shorten(state: State) -> dict:
    return {"instruction": state["instruction"] + " Make it shorter.",
            "rounds": state["rounds"] + 1,
            "trace": [f"  → {len(state['draft'])} > {LIMIT}: ask for a shorter one"]}


def build():
    from langgraph.graph import StateGraph, START, END
    g = StateGraph(State)
    g.add_node("draft", _draft)
    g.add_node("shorten", _shorten)
    g.add_edge(START, "draft")
    g.add_conditional_edges("draft", _too_long, {"shorten": "shorten", "done": END})
    g.add_edge("shorten", "draft")            # the CYCLE
    return g.compile()


def run_graph(provider: str = "stub") -> State:
    app = build()
    final = app.invoke({"instruction": f"In one short sentence: {TASK}",
                        "draft": "", "rounds": 0, "provider": provider, "trace": []})
    print("\n[b] GRAPH — loop until it fits")
    for line in final["trace"]:
        print("   ", line)
    print(f"    FINAL: \"{final['draft']}\" ({len(final['draft'])} chars · fits ✓)")
    return final


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args else "both"                 # prompt | graph | both
    provider = resolve_provider(args[1] if len(args) > 1 else "auto")
    print(f"TASK: {TASK}\nCONSTRAINT: <= {LIMIT} chars   (model provider: {provider})")
    if mode in ("prompt", "both"):
        run_prompt(provider)
    if mode in ("graph", "both"):
        run_graph(provider)
