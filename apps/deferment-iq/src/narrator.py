"""Base-management review writer. The analytics are deterministic; the LLM only
narrates them in Senior-PE / asset-manager voice. With no key, a fully templated
review is rendered, so the app never depends on an API key.
"""
from __future__ import annotations

import json
import os
from datetime import date

from dotenv import load_dotenv


class MissingAPIKey(RuntimeError):
    """Raised when an LLM review is requested without a key/client."""


SYSTEM_PROMPT = """You are a Senior Production Engineer presenting the weekly base-management review to an asset VP. You are given deterministic deferment analytics (lost barrels vs. each well's potential, by cause). Write a tight one-page markdown review:

1. **# Base-Management Review — {date}**
2. **## Bottom line** — 2-3 sentences: total deferred $/day at risk, production efficiency (uptime %), and the single biggest lever. Lead with the money.
3. **## Where the barrels are going** — the Pareto: name the top 2-3 causes with their $ and share; call out how much is RECOVERABLE vs. planned/reservoir (not recoverable).
4. **## Worst offenders** — the top 3 wells by deferred $ and their dominant cause, each with a concrete next step + owner (lease operator / field foreman / on-call engineer) and a deadline.
5. **## Data quality** — if capture rate < 90%, flag the unclassified (uncaptured) deferment $ as a reason-coding gap to close.
6. **## The ask** — the recovery opportunity $ and the 1-2 actions that capture most of it.

Use the numbers verbatim. Terse, specific, decision-ready. First character must be '#'."""


def render_review_markdown(kpis: dict, pareto, top, recovery: dict, brief_date: str | None = None) -> str:
    """Deterministic templated review (no LLM) — also the no-key fallback."""
    brief_date = brief_date or date.today().isoformat()
    L = [f"# Base-Management Review — {brief_date}", "", "## Bottom line"]
    L.append(
        f"~{kpis.get('deferred_bopd_avg', 0):,.0f} BOPD deferred "
        f"(~${kpis.get('deferred_usd', 0) / max(kpis.get('period_days', 1), 1):,.0f}/day); "
        f"production efficiency **{kpis.get('uptime_pct', 0):.1f}%**, "
        f"**{kpis.get('pct_deferred', 0):.1f}%** of potential lost. "
        f"Recoverable opportunity: **${recovery.get('recoverable_usd', 0):,.0f}**.")
    L += ["", "## Where the barrels are going"]
    if len(pareto):
        for _, r in pareto.head(4).iterrows():
            tag = "recoverable" if r["recoverable"] else ("planned" if r["planned"] else "not recoverable")
            L.append(f"- **{r['label']}** — ${r['deferred_usd']:,.0f} ({r['pct_of_total']:.0f}%, {tag})")
    else:
        L.append("- No deferment in the period — fleet at full potential.")
    L += ["", "## Worst offenders", "", "| Well | Deferred $ | Dominant cause | Uptime |",
          "|---|---|---|---|"]
    for _, r in top.head(5).iterrows():
        L.append(f"| {r['well_id']} | ${r['deferred_usd']:,.0f} | {r['top_cause']} | {r['uptime_pct']:.0f}% |")
    cap = kpis.get("capture_rate_pct", 100.0)
    if cap < 90:
        L += ["", "## Data quality",
              f"Capture rate **{cap:.0f}%** — ${recovery.get('unclassified_usd', 0):,.0f} of deferment is "
              f"**unclassified** (no reason code). Close the coding gap before trusting the Pareto."]
    L += ["", "## The ask",
          f"Capturing the recoverable **${recovery.get('recoverable_usd', 0):,.0f}** starts with the top "
          f"cause above — assign it today."]
    return "\n".join(L)


def write_review(kpis: dict, pareto, top, recovery: dict, brief_date: str | None = None,
                 model: str | None = None, client=None) -> str:
    """LLM-narrated review. Raises MissingAPIKey if no client and no key in env."""
    load_dotenv()
    if client is None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKey("No ANTHROPIC_API_KEY — use render_review_markdown() for the "
                                "deterministic review, or provide a key.")
        from anthropic import Anthropic
        client = Anthropic(api_key=key)
    model = model or os.environ.get("MODEL", "claude-sonnet-4-6")
    brief_date = brief_date or date.today().isoformat()

    payload = {
        "kpis": kpis,
        "pareto": pareto.head(6).to_dict("records") if len(pareto) else [],
        "top_wells": top.head(6).to_dict("records") if len(top) else [],
        "recovery": recovery,
    }
    user = f"Date: {brief_date}\n\nDeferment analytics:\n{json.dumps(payload, indent=2, default=str)}\n\nWrite the review."
    resp = client.messages.create(model=model, max_tokens=1600,
                                  system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    h = text.find("\n#")
    if h > 0 and not text.lstrip().startswith("#"):
        text = text[h:].lstrip()
    return text
