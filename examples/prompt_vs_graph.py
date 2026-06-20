"""Same task, two ways — so you can SEE when you need LangGraph and when you don't.

Task: write a one-line summary that must be <= LIMIT characters.

  (a) PROMPT mode  — one model call. If the result is too long, you're stuck:
      a single call has no way to react to its own output.
  (b) GRAPH mode   — a LangGraph graph that checks the length and LOOPS
      (ask → check → "make it shorter" → ask …) until the constraint is met.

The only difference is control flow. A prompt is a function call; a graph is a
little program with a branch and a loop. The "toggle" between them is just an
``if`` at the bottom of this file — there is no LangGraph switch to flip.

Run it:
    python examples/prompt_vs_graph.py            # runs both
    python examples/prompt_vs_graph.py prompt     # just the one-call version
    python examples/prompt_vs_graph.py graph      # just the looping version

With ANTHROPIC_API_KEY set it makes real Claude calls; without one it uses a tiny
deterministic stand-in so the *structure* runs with zero setup.
"""
from __future__ import annotations

import operator
import os
import sys
from typing import Annotated, TypedDict

TASK = "Explain why an ESP pump trips on underload."
LIMIT = 80   # the summary must fit in this many characters


# --- "the model": one function both modes call -------------------------------
# THIS is what "a prompt" is — a single call that maps text in -> text out.
def call_model(instruction: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        import anthropic
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model="claude-sonnet-4-6", max_tokens=120,
            messages=[{"role": "user", "content": instruction}])
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    return _stub(instruction)


def _stub(instruction: str) -> str:
    """No-key stand-in: returns a shorter answer the more insistently you ask."""
    variants = [
        "An ESP trips on underload when there is too little fluid over the pump "
        "and the drive shuts it down to protect the motor",                      # ~117
        "ESP underload trip: too little fluid over the pump, the drive shuts it "
        "down to protect the motor",                                             # ~94
        "ESP underload: low fluid over the pump, drive trips to protect the motor",  # ~71
        "Underload = low fluid over pump; drive trips the motor",                # ~53
    ]
    return variants[min(instruction.lower().count("shorter"), len(variants) - 1)]


# --- (a) PROMPT mode: one call, no recourse ----------------------------------
def run_prompt() -> tuple[str, bool]:
    out = call_model(f"In one short sentence: {TASK}")
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
    trace: Annotated[list, operator.add]   # reducer: append, don't overwrite


def _draft(state: State) -> dict:
    out = call_model(state["instruction"])
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


def run_graph() -> State:
    app = build()
    final = app.invoke({"instruction": f"In one short sentence: {TASK}",
                        "draft": "", "rounds": 0, "trace": []})
    print("\n[b] GRAPH — loop until it fits")
    for line in final["trace"]:
        print("   ", line)
    print(f"    FINAL: \"{final['draft']}\" ({len(final['draft'])} chars · fits ✓)")
    return final


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"   # <- the only "toggle"
    print(f"TASK: {TASK}\nCONSTRAINT: <= {LIMIT} chars   "
          f"(model: {'Claude' if os.environ.get('ANTHROPIC_API_KEY') else 'no-key stub'})")
    if mode in ("prompt", "both"):
        run_prompt()
    if mode in ("graph", "both"):
        run_graph()
