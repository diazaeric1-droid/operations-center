"""A LangGraph reference flow — intervention approval — built to LEARN the
patterns an AI-engineer role expects: cycles, branches, durable persistence, and
human-in-the-loop. It is deliberately a teaching artifact (heavily commented),
separate from the production agentic-RAG flow in graph.py.

The agent decides what to do with a candidate workover:

    START → assess ─┬─(missing data)──► gather ──┐  ← CYCLE (enrich, re-assess)
                    │                            │
                    │                            ▼
                    │                         assess
                    ├─(negative value)──────► auto_reject ──► END   ┐
                    ├─(cheap + positive)────► auto_approve ──► END   ├ BRANCH
                    └─(costly / borderline)─► human_review           ┘
                                                  │  ← HUMAN-IN-THE-LOOP
                                              interrupt(); wait…
                                                  │
                                                  ▼
                                              authorize ──► END

Why each piece matters in an interview:
  * **Branch** — `route_after_assess` returns one of four next nodes; a chain
    can't do this, a graph routes on state.
  * **Cycle** — `gather → assess` loops to enrich missing inputs (capped), the
    thing prompt-chains fundamentally cannot express.
  * **Durable persistence** — compiled with a SqliteSaver, the paused state is
    written to disk, so an approval can wait *days* and survive a process
    restart (proved in the __main__ demo by rebuilding the graph from the same
    db file before resuming).
  * **Human-in-the-loop** — `human_review` calls `interrupt()`, which pauses the
    whole graph and hands control back to the caller; you resume by invoking
    `Command(resume=<decision>)` on the same thread_id.
"""
from __future__ import annotations

import operator
import sqlite3
from typing import Annotated, Optional, TypedDict

# Illustrative thresholds (NOT certified economics — see the real risked-NPV
# engine for that; this flow is about the LangGraph control structure).
HORIZON_DAYS = 365
AUTO_APPROVE_COST = 25_000.0   # at/under this, a positive job needs no human
DEFAULT_RISK = 0.40            # what `gather` fills in when risk is unknown


class ApprovalState(TypedDict):
    well_id: str
    deferred_usd_day: float          # $/day currently being deferred
    failure_risk: float              # 0..1 (<=0 means "unknown" -> gather)
    intervention_cost: float
    gather_attempts: int
    assessment: str
    risked_value: float
    human_decision: Optional[str]    # filled on resume ("approve"/"reject")
    outcome: str
    trace: Annotated[list, operator.add]


# --- nodes --------------------------------------------------------------------
def assess(state: ApprovalState) -> dict:
    risk = state["failure_risk"]
    protect = max(risk, 0.0) * state["deferred_usd_day"] * HORIZON_DAYS
    rv = protect - state["intervention_cost"]
    txt = (f"risk {risk:.0%} × ${state['deferred_usd_day']:,.0f}/day × {HORIZON_DAYS}d "
           f"− ${state['intervention_cost']:,.0f} cost = ${rv:,.0f} risked value")
    return {"risked_value": rv, "assessment": txt,
            "trace": [f"assess: {txt}"]}


def gather(state: ApprovalState) -> dict:
    """Enrich missing inputs, then the graph loops back to assess (the cycle)."""
    return {"failure_risk": DEFAULT_RISK,
            "gather_attempts": state["gather_attempts"] + 1,
            "trace": [f"gather: failure_risk was unknown → filled {DEFAULT_RISK:.0%} "
                      "(would query the ESP risk model); re-assessing"]}


def auto_approve(state: ApprovalState) -> dict:
    return {"outcome": "auto-approved",
            "trace": ["auto_approve: cheap job with positive risked value — "
                      "no human needed"]}


def auto_reject(state: ApprovalState) -> dict:
    return {"outcome": "rejected",
            "trace": ["auto_reject: non-positive risked value — intervening "
                      "destroys value"]}


def human_review(state: ApprovalState) -> dict:
    """PAUSE here for a human. interrupt() suspends the graph and returns its
    payload to the caller; execution resumes (re-running this node) when the
    caller invokes Command(resume=<decision>)."""
    from langgraph.types import interrupt
    decision = interrupt({
        "well_id": state["well_id"],
        "risked_value": state["risked_value"],
        "intervention_cost": state["intervention_cost"],
        "assessment": state["assessment"],
        "question": "Approve this workover? (approve / reject)",
    })
    return {"human_decision": str(decision),
            "trace": [f"human_review: human said '{decision}'"]}


def authorize(state: ApprovalState) -> dict:
    approved = str(state.get("human_decision", "")).strip().lower().startswith("a")
    return {"outcome": "approved by human" if approved else "rejected by human",
            "trace": [f"authorize: {'APPROVED' if approved else 'REJECTED'} "
                      "by reviewer"]}


# --- the conditional router (the BRANCH + the CYCLE entry) ---------------------
def route_after_assess(state: ApprovalState) -> str:
    if state["failure_risk"] <= 0.0 and state["gather_attempts"] < 1:
        return "gather"            # CYCLE: unknown risk -> enrich, then re-assess
    if state["risked_value"] <= 0.0:
        return "auto_reject"
    if state["intervention_cost"] <= AUTO_APPROVE_COST:
        return "auto_approve"
    return "human_review"          # costly + positive -> needs a human


# --- graph builder ------------------------------------------------------------
def build_approval_graph(checkpointer=None):
    """Compile the graph. Pass a SqliteSaver for DURABLE persistence; defaults to
    in-memory (MemorySaver)."""
    from langgraph.graph import StateGraph, START, END
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    g = StateGraph(ApprovalState)
    for name, fn in (("assess", assess), ("gather", gather),
                     ("auto_approve", auto_approve), ("auto_reject", auto_reject),
                     ("human_review", human_review), ("authorize", authorize)):
        g.add_node(name, fn)
    g.add_edge(START, "assess")
    g.add_conditional_edges("assess", route_after_assess, {
        "gather": "gather", "auto_reject": "auto_reject",
        "auto_approve": "auto_approve", "human_review": "human_review"})
    g.add_edge("gather", "assess")            # the cycle
    g.add_edge("auto_approve", END)
    g.add_edge("auto_reject", END)
    g.add_edge("human_review", "authorize")
    g.add_edge("authorize", END)
    return g.compile(checkpointer=checkpointer)


def sqlite_saver(path: str):
    """A durable, file-backed checkpointer (survives process restarts)."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def _initial(well_id: str, deferred_usd_day: float, intervention_cost: float,
             failure_risk: float) -> ApprovalState:
    return {"well_id": well_id, "deferred_usd_day": deferred_usd_day,
            "failure_risk": failure_risk, "intervention_cost": intervention_cost,
            "gather_attempts": 0, "assessment": "", "risked_value": 0.0,
            "human_decision": None, "outcome": "", "trace": []}


def is_interrupted(app, config) -> bool:
    """True when the graph is paused awaiting a human (its next node is set)."""
    return bool(app.get_state(config).next)


if __name__ == "__main__":   # python -m langgraph_rag.approval_agent
    from langgraph.types import Command

    def show(title, state):
        print(f"\n=== {title} ===")
        for t in state["trace"]:
            print("  •", t)
        print("  OUTCOME:", state["outcome"] or "(paused — awaiting human)")

    # 1) auto-approve: cheap job, clearly positive
    app = build_approval_graph()
    cfg = {"configurable": {"thread_id": "demo-cheap"}}
    s = app.invoke(_initial("well_021", 1800, 18_000, 0.55), cfg)
    show("Auto-approve (cheap + positive)", s)

    # 2) auto-reject: cost dwarfs the protected value
    s = app.invoke(_initial("well_007", 90, 60_000, 0.20),
                   {"configurable": {"thread_id": "demo-reject"}})
    show("Auto-reject (negative risked value)", s)

    # 3) cycle: risk unknown -> gather -> re-assess
    s = app.invoke(_initial("well_055", 1200, 15_000, 0.0),
                   {"configurable": {"thread_id": "demo-cycle"}})
    show("Cycle (unknown risk → gather → re-assess)", s)

    # 4) DURABLE human-in-the-loop: costly job pauses for approval, and the paused
    #    state survives a "restart" (a fresh graph built from the same sqlite db).
    db = "/tmp/ops_approvals_demo.sqlite"
    import os
    if os.path.exists(db):
        os.remove(db)
    app1 = build_approval_graph(sqlite_saver(db))
    cfg = {"configurable": {"thread_id": "AFE-9921"}}
    s1 = app1.invoke(_initial("well_030", 2100, 140_000, 0.62), cfg)
    print("\n=== Human-in-the-loop (durable) ===")
    for t in s1["trace"]:
        print("  •", t)
    irq = s1.get("__interrupt__")
    print("  PAUSED at human_review — interrupt payload:",
          irq[0].value["question"] if irq else "(none)")

    # --- simulate a process restart: brand-new graph object, same db file ---
    app2 = build_approval_graph(sqlite_saver(db))
    print("  …(process restarts; rebuilt graph from the same db)…")
    print("  still paused?", is_interrupted(app2, cfg))
    s2 = app2.invoke(Command(resume="approve"), cfg)   # resume with the decision
    for t in s2["trace"]:
        print("  •", t)
    print("  OUTCOME:", s2["outcome"])
