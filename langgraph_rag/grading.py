"""Grade retrieval relevance + rewrite weak queries — the graph's 'brain'.

Pure Python (no LangGraph, no torch): deterministic by default so the agentic
flow runs with no API key, LLM-enhanced when a key is supplied. Kept separate
from graph.py so this decision logic is unit-tested on its own.
"""
from __future__ import annotations

# A note set is "relevant" when its best cosine similarity clears this bar.
# Calibrated to the fastembed/bge-small scale on this corpus: sharp, on-vocabulary
# queries land ~0.72–0.80 and pass one-shot; softer / lay-worded queries land
# ~0.62–0.68 and get a domain-vocabulary rewrite that reliably lifts them over the
# bar (measured: e.g. 0.63 → 0.71). Set the bar where the rewrite earns its keep.
RELEVANCE_THRESHOLD = 0.68

# Lay-term -> field-vocabulary expansions. On a weak retrieval the deterministic
# rewriter appends the matching field terms, nudging the embedding toward the
# operator-note vocabulary (the whole reason a rewrite can rescue a bad query).
_EXPANSIONS: dict[str, str] = {
    "power": "substation breaker transformer outage electrical lost power",
    "electric": "substation breaker transformer outage electrical",
    "outage": "lost power substation breaker grid",
    "pump": "esp underload vsd intake rod pump worn no fluid",
    "esp": "underload vsd fault intake motor amps",
    "gas": "midstream curtailment compressor gas plant line pressure takeaway",
    "midstream": "gas plant curtailment takeaway line pressure",
    "freeze": "frozen freeze off winter ice cold weather",
    "cold": "freeze frozen winter ice",
    "scale": "paraffin wax wellbore restriction cleanout fill",
    "wax": "paraffin scale hot oil wellbore",
    "separator": "treater emulsion facility high level dump valve compressor",
    "facility": "separator treater compressor tank battery",
    "water": "watering out water cut reservoir loading",
    "decline": "pressure depletion reservoir declining inflow",
}


def best_score(notes: list[dict]) -> float:
    return max((float(n.get("score", 0.0)) for n in notes), default=0.0)


def grade_relevance(query: str, notes: list[dict],
                    anthropic_key: str | None = None,
                    model: str = "claude-sonnet-4-6") -> bool:
    """Are the retrieved notes good enough to answer the query?

    Deterministic: best retrieval score ≥ threshold. With a key, asks Claude for
    a yes/no relevance judgment, falling back to the score rule on any error.
    """
    if not notes:
        return False
    if anthropic_key:
        try:
            return _llm_grade(query, notes, anthropic_key, model)
        except Exception:  # noqa: BLE001 — never fail the graph on an API hiccup
            pass
    return best_score(notes) >= RELEVANCE_THRESHOLD


def rewrite_query(query: str, notes: list[dict],
                  anthropic_key: str | None = None,
                  model: str = "claude-sonnet-4-6") -> str:
    """Produce a better retrieval query after a weak round.

    Deterministic: append field-vocabulary expansions for any lay terms present
    (or a generic operations context if none match). With a key, ask Claude to
    rephrase. Always returns a query DIFFERENT enough to change retrieval.
    """
    if anthropic_key:
        try:
            rewritten = _llm_rewrite(query, anthropic_key, model)
            if rewritten and rewritten.strip().lower() != query.strip().lower():
                return rewritten.strip()
        except Exception:  # noqa: BLE001
            pass
    q = query.lower()
    adds = [exp for term, exp in _EXPANSIONS.items() if term in q]
    if not adds:
        adds.append("production downtime curtailment operator note")
    # de-dup tokens we already have, keep it compact
    extra = " ".join(dict.fromkeys(" ".join(adds).split()))
    return f"{query} {extra}".strip()


# --- LLM backends (used only when a key is supplied) --------------------------
def _client(key: str):
    import anthropic
    return anthropic.Anthropic(api_key=key)


def _llm_grade(query: str, notes: list[dict], key: str, model: str) -> bool:
    ctx = "\n".join(f"- {n.get('note', '')}" for n in notes[:6])
    resp = _client(key).messages.create(
        model=model, max_tokens=8,
        system=("You judge whether retrieved oil & gas operator notes are "
                "relevant enough to answer a question. Reply ONLY 'yes' or 'no'."),
        messages=[{"role": "user",
                   "content": f"Question: {query}\n\nNotes:\n{ctx}\n\nRelevant?"}])
    ans = "".join(b.text for b in resp.content if b.type == "text").strip().lower()
    return ans.startswith("y")


def _llm_rewrite(query: str, key: str, model: str) -> str:
    resp = _client(key).messages.create(
        model=model, max_tokens=60,
        system=("You rewrite a search query to better match an oil & gas "
                "operator's downtime/curtailment note log. Reply with ONLY the "
                "rewritten query, no preamble."),
        messages=[{"role": "user", "content": query}])
    return "".join(b.text for b in resp.content if b.type == "text").strip()
