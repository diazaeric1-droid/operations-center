"""Loss Accounting · Note Search (RAG) — semantic search over operator notes.

The keyword reason-code classifier (Causes & Pareto) answers "what bucket is
this note?". This page answers the questions a keyword match can't: it embeds
every operator note into a pgvector index and retrieves by *meaning*, then has
Claude synthesize a cited answer (BYOK; a deterministic extractive rollup runs
without a key).

Stack on display: LlamaIndex + pgvector + local fastembed embeddings + Claude.
The feature is fully optional — if the RAG extras (requirements-rag.txt) or the
vector DB are absent, the page explains how to bring them up and stops cleanly,
so the core console is never coupled to this heavier retrieval stack.
"""
from __future__ import annotations

import streamlit as st

import product_theme as pt
import theme

from views import _common as c

_CAUSES = [
    "(any cause)", "artificial_lift", "surface_facility", "power",
    "gathering_thirdparty", "wellbore", "planned", "weather", "reservoir",
]
_EXAMPLES = [
    "slow ESP failure, gradual underload rather than an instant trip",
    "freeze-offs that lasted more than a couple of days",
    "midstream curtailment where they cut our gas takeaway",
    "separator or treater emulsion upsets",
]


@st.cache_resource(show_spinner="Loading the embedding model…")
def _engine():
    """One engine per process (loads the fastembed model once)."""
    from rag.engine import NoteSearchEngine
    return NoteSearchEngine()


def render() -> None:
    c.ensure_state()

    pt.masthead("ops", "Note Search (RAG)",
                "Ask the operator-note log in plain language. Semantic retrieval "
                "over a pgvector index, with a cited, LLM-narrated answer.")
    c.page_purpose(
        "**The question this page answers: what did we already learn about this "
        "problem, in our own operator notes?**\n\n"
        "- **When:** any time a well's behaviour looks familiar — before you "
        "re-diagnose, search whether pumpers/engineers already wrote it up.\n"
        "- **Headline read:** the ranked note excerpts with similarity scores; "
        "every claim in the narrated answer cites the note it came from.\n"
        "- **Next:** confirm the well's current state on **Surveillance** or "
        "**Well 360**, then act through the **Action Chain**.")

    # --- guard: optional extras + the vector DB ------------------------------
    from rag.engine import deps_available
    from rag import store

    ok, fix = deps_available()
    if not ok:
        pt.section("Extended install required")
        st.info(
            "Semantic search needs the optional RAG extras (kept out of the core "
            "app to keep it light). From the repo root:\n\n"
            "```bash\npip install -r requirements-rag.txt\n```")
        st.caption(f"Missing: `{fix}`")
        return

    db_ok, detail = store.ping()
    if not db_ok:
        pt.section("Vector database unavailable")
        st.warning(
            "The pgvector store isn't reachable. Bring up the local one:\n\n"
            "```bash\ndocker compose -f docker-compose.rag.yml up -d\n```\n\n"
            "Or point `OPS_PG_DSN` at your Postgres+pgvector (e.g. AWS RDS).")
        st.caption(f"Tried: `{detail}`")
        return

    eng = _engine()
    n_indexed = eng.index_size()

    pt.context_bar([
        ("Vector store", f"pgvector · {detail}"),
        ("Embeddings", "fastembed BAAI/bge-small (local, 384-d)"),
        ("Synthesis", "Claude (BYOK)" if st.session_state.get("anthropic_key")
         else "extractive (no key)"),
        ("Indexed notes", f"{n_indexed:,}"),
    ])

    # --- index lifecycle -----------------------------------------------------
    if n_indexed == 0:
        pt.section("Build the index")
        st.write("The operator-note corpus isn't embedded yet. Build it once "
                 "(~10s); it persists in pgvector across restarts.")
        if st.button("Build index", type="primary"):
            with st.spinner("Embedding the operator-note corpus into pgvector…"):
                eng.reset_table()
                cnt = eng.build_index()
            st.success(f"Indexed {cnt:,} operator notes.")
            st.rerun()
        return

    # --- query ---------------------------------------------------------------
    pt.section("Ask the note log")
    q = st.text_input(
        "Question or description", key="rag_query",
        placeholder=_EXAMPLES[0],
        help="Plain language — retrieval is by meaning, not keywords.")
    cols = st.columns([3, 1, 1])
    with cols[0]:
        st.caption("Try: " + " · ".join(f"*{e}*" for e in _EXAMPLES[:3]))
    with cols[1]:
        cause = st.selectbox("Filter cause", _CAUSES, index=0)
    with cols[2]:
        top_k = st.slider("Results", 3, 12, 6)

    from langgraph_rag.graph import deps_available as _lg_deps
    lg_ok = _lg_deps()[0]
    agentic = st.checkbox(
        "Self-correcting (agentic) — grade retrieval & rewrite weak queries",
        value=False, disabled=not lg_ok,
        help=("Runs a LangGraph loop: retrieve → grade relevance → if weak, "
              "rewrite the query and retry (max 2) → synthesize. The trace shows "
              "each step." if lg_ok else
              "Install the optional extra to enable: "
              "pip install -r requirements-langgraph.txt"))

    with st.expander("Rebuild index (re-embed the corpus)"):
        if st.button("Rebuild from scratch"):
            with st.spinner("Re-embedding…"):
                eng.reset_table()
                cnt = eng.build_index()
            st.success(f"Re-indexed {cnt:,} notes.")
            st.rerun()

    if not q:
        pt.empty_state("Enter a question above to search the operator-note log.")
        theme.references(["deferment"])
        return

    cause_arg = None if cause == _CAUSES[0] else cause
    key = st.session_state.get("anthropic_key")
    trace = None
    if agentic and lg_ok:
        from langgraph_rag.graph import run as lg_run
        from rag.engine import Answer, RetrievedNote
        with st.spinner("Agentic RAG: retrieve → grade → rewrite → synthesize…"):
            final = lg_run(q, cause=cause_arg, top_k=top_k, anthropic_key=key,
                           max_iterations=2, engine=eng)
        ans = Answer(final["answer"], final["used_llm"],
                     [RetrievedNote(**n) for n in final["notes"]])
        trace = final["trace"]
    else:
        with st.spinner("Retrieving + synthesizing…"):
            ans = eng.answer(q, top_k=top_k, cause=cause_arg, anthropic_key=key)

    if trace:
        n_rw = sum(t.startswith("rewrite") for t in trace)
        with st.expander(f"🔁 Self-correction trace — {len(trace)} steps · "
                         f"{n_rw} rewrite(s)", expanded=True):
            for t in trace:
                st.markdown(f"- {t}")
            st.caption("LangGraph state machine: retrieve → grade → (rewrite → "
                       "retrieve)* → generate. Rewrites fire only when the top "
                       "retrieval score is below the relevance bar.")

    pt.section("Answer")
    if ans.used_llm:
        st.markdown(ans.text)
        st.caption("Synthesized by Claude from the retrieved notes only, with "
                   "[n] citations into the table below.")
    else:
        st.code(ans.text, language=None)
        st.caption("Deterministic extractive rollup (no API key). Add an "
                   "Anthropic key in the sidebar for a narrated, cited answer.")

    pt.section("Retrieved notes", "Ranked by semantic similarity to the query.")
    if ans.sources:
        import pandas as pd
        df = pd.DataFrame([{
            "#": i, "Score": h.score, "Well": h.well_id, "Cause": h.cause,
            "Start": h.start_date, "Days": h.duration_days,
            "Deferred bbl": h.deferred_bbl, "Operator Note": h.note,
        } for i, h in enumerate(ans.sources, 1)])
        st.dataframe(df, width="stretch", hide_index=True)
        theme.source_note(
            "Notes embedded with BAAI/bge-small-en-v1.5 (local), retrieved from a "
            "pgvector index by cosine similarity; the LLM answer is grounded ONLY "
            "in these rows. Corpus = the synthetic reason-coded event log.")
    theme.references(["deferment"])
