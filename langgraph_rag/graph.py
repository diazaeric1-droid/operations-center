"""The LangGraph agentic-RAG state machine over rag.engine.NoteSearchEngine.

    START → retrieve → grade ─(weak)→ rewrite → retrieve → …   (cycle, max-capped)
                          └─(good / max iters)→ generate → END

LangGraph is imported lazily inside build_graph() so this module imports without
the optional extra. The nodes close over an injected ``engine`` (anything with
``.retrieve()`` + ``.synthesize()``), which makes the control flow unit-testable
with a fake engine and no vector DB.
"""
from __future__ import annotations

import operator
from dataclasses import asdict
from typing import Annotated, Optional, TypedDict

from . import grading


def deps_available() -> tuple[bool, str]:
    try:
        import langgraph  # noqa: F401
        return True, ""
    except ImportError:
        return False, "pip install -r requirements-langgraph.txt"


class AgenticRAGState(TypedDict):
    """The shared state that flows through every node."""
    original_query: str                 # what the user asked (used for synthesis)
    query: str                          # current (possibly rewritten) retrieval query
    cause: Optional[str]
    top_k: int
    anthropic_key: Optional[str]
    notes: list                         # retrieved notes as plain dicts
    relevant: bool
    iterations: int                     # how many rewrites so far
    max_iterations: int
    answer: str
    used_llm: bool
    trace: Annotated[list, operator.add]   # step log (reducer appends)


def build_graph(engine, max_iterations: int = 2):
    """Compile the agentic-RAG graph for a given retrieval engine."""
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver

    def retrieve(state: AgenticRAGState) -> dict:
        hits = engine.retrieve(state["query"], top_k=state["top_k"],
                               cause=state["cause"])
        notes = [asdict(h) for h in hits]
        return {"notes": notes,
                "trace": [f"retrieve(round {state['iterations']}): "
                          f"\"{state['query'][:64]}\" → {len(notes)} notes, "
                          f"top score {grading.best_score(notes):.2f}"]}

    def grade(state: AgenticRAGState) -> dict:
        rel = grading.grade_relevance(state["original_query"], state["notes"],
                                      state["anthropic_key"])
        return {"relevant": rel,
                "trace": [f"grade: {'relevant' if rel else 'WEAK — will rewrite'}"]}

    def rewrite(state: AgenticRAGState) -> dict:
        nq = grading.rewrite_query(state["query"], state["notes"],
                                   state["anthropic_key"])
        return {"query": nq, "iterations": state["iterations"] + 1,
                "trace": [f"rewrite → \"{nq[:64]}\""]}

    def generate(state: AgenticRAGState) -> dict:
        from rag.engine import RetrievedNote
        hits = [RetrievedNote(**n) for n in state["notes"]]
        ans = engine.synthesize(state["original_query"], hits,
                                anthropic_key=state["anthropic_key"])
        return {"answer": ans.text, "used_llm": ans.used_llm,
                "trace": [f"generate: {'LLM' if ans.used_llm else 'extractive'} "
                          f"answer from {len(hits)} notes"]}

    def route_after_grade(state: AgenticRAGState) -> str:
        if state["relevant"] or state["iterations"] >= state["max_iterations"]:
            return "generate"
        return "rewrite"

    g = StateGraph(AgenticRAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("rewrite", "retrieve")          # the self-correction cycle
    g.add_edge("generate", END)
    return g.compile(checkpointer=MemorySaver())


def run(query: str, cause: Optional[str] = None, top_k: int = 6,
        anthropic_key: Optional[str] = None, max_iterations: int = 2,
        engine=None, thread_id: str = "agentic-rag") -> dict:
    """Build + run the graph for one query. Returns the final state (incl. the
    answer, the source notes, and the step trace)."""
    if engine is None:
        from rag.engine import NoteSearchEngine
        engine = NoteSearchEngine()
    app = build_graph(engine, max_iterations=max_iterations)
    init: AgenticRAGState = {
        "original_query": query, "query": query, "cause": cause, "top_k": top_k,
        "anthropic_key": anthropic_key, "notes": [], "relevant": False,
        "iterations": 0, "max_iterations": max_iterations, "answer": "",
        "used_llm": False, "trace": [],
    }
    return app.invoke(init, config={"configurable": {"thread_id": thread_id}})


if __name__ == "__main__":   # CLI demo: python -m langgraph_rag.graph "your query"
    import sys
    q = " ".join(sys.argv[1:]) or "the gas buyer cut our pipeline takeaway"
    final = run(q)
    print(f"\nQUERY: {q}\n")
    print("TRACE:")
    for line in final["trace"]:
        print("  •", line)
    print(f"\nANSWER ({'LLM' if final['used_llm'] else 'extractive'}):")
    print(final["answer"])
