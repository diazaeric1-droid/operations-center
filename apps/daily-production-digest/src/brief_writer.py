"""Claude-powered morning brief writer. Takes the deterministic fleet summary +
anomaly list and produces a one-page markdown brief in Senior PE voice.

Detection stays deterministic — the LLM only narrates. If no API key is present
(public demo, CI without a secret), ``render_brief_markdown`` produces a fully
deterministic brief from the same data, so the pipeline never just crashes.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import date

from dotenv import load_dotenv

from .anomaly_detector import Anomaly


class MissingAPIKey(RuntimeError):
    """Raised when an LLM brief is requested without ANTHROPIC_API_KEY set."""


SYSTEM_PROMPT = """You are a Senior Production Engineer writing the daily morning brief for an asset team's 6:30am standup. You're given:

- Today's date
- Fleet-wide summary stats (total BOPD, water cut, average runtime)
- A list of anomalies detected overnight by deterministic Python rules, ranked by severity then by DEFERRED $/day (money first)
- (Optional) A list of TRACKED EVENTS from a persistent state machine: each has a state (NEW / ONGOING / RESOLVED), how many days it has been open (duration_days), its CUMULATIVE deferred bbl/$ over the whole event, and today's deferred snapshot. An ONGOING event is a previously-reported problem that is STILL abnormal — report it as continuing (day N), not as brand-new. A RESOLVED event recovered back to normal — give it one closing-out mention.

Write a one-page markdown brief in this exact structure:

1. **# Daily Production Brief — {date}** (top heading)
2. **## Bottom Line** — 2-3 sentence executive summary. Lead with the worst news AND the total deferred $/day at risk.
3. **## Field Status** — 3-bullet recap of fleet KPIs (BOPD, water cut, runtime)
4. **## Top Priorities** — Numbered list of HIGH-severity anomalies, each with: well, what happened (1 sentence citing the evidence incl. deferred bbl/$ where present), action owner & deadline. If no HIGH items, say "No HIGH-priority anomalies — fleet is stable."
5. **## Watch List** — MEDIUM-severity anomalies as a compact table (Well, Category, Headline, Action)
6. **## Data Quality / Acknowledged** — note any comms-loss / metering-dropout flags and any acknowledged (known/planned) items that were suppressed from priorities.
7. **## Ongoing & Resolved Events** — ONLY if tracked events are supplied. Multi-day ONGOING events as a compact table (Well, Type, State, Day N, cumulative deferred bbl/$, today's deferral) so a still-down well stays visible every morning; then a closing-out line for each RESOLVED event. Skip this section entirely if no events are supplied or none are multi-day/resolved.
8. **## Closing** — One sentence either reassuring (if stable) or escalating (if multiple HIGH items)

Style:
- Write the way a Staff Production Engineer talks to an Ops Manager — terse, specific, no hedging, no fluff
- Use the evidence numbers verbatim from the anomaly data — never round or generalize
- For an ONGOING event, make clear it is a CONTINUATION ("day N of an open outage"), citing the cumulative deferred bbl/$ — never re-report it as a fresh discovery.
- Action items must have an owner role (lease operator, field foreman, on-call engineer) and a deadline
- Never invent anomalies not in the input. Never reference wells not in the input.
- **First character of your response must be `#` — no preamble.**
"""


def render_brief_markdown(summary: dict, anomalies: list[Anomaly],
                          brief_date: str | None = None,
                          events: list | None = None) -> str:
    """Deterministic morning brief (no LLM) — used as the no-API-key fallback and
    as the committed sample. Same data the LLM narrates, just templated.

    ``events`` (optional) is the list of live state-machine ``Event`` objects from
    ``event_store.update_events`` for ``brief_date``. When provided, the brief gains
    an **Ongoing & Resolved Events** section that shows still-open events with their
    running DURATION and CUMULATIVE deferred bbl/$ — so a multi-day outage keeps
    appearing every morning (with day N of N) instead of vanishing once it ages out
    of the stateless detector's lookback window, and a just-recovered well gets one
    "closing out" mention. When ``events`` is None the brief is byte-identical to
    the pre-state-machine output (back-compat for callers / the committed sample)."""
    brief_date = brief_date or date.today().isoformat()
    active = [a for a in anomalies if not a.acknowledged]
    acked = [a for a in anomalies if a.acknowledged]
    highs = [a for a in active if a.severity == "HIGH"]
    meds = [a for a in active if a.severity == "MEDIUM"]
    dq = [a for a in active if a.category in ("comms_loss", "meter_dropout")]
    total_deferred_bopd = sum(a.deferred_bopd for a in active)
    total_deferred_usd = sum(a.deferred_usd_per_day for a in active)

    L = [f"# Daily Production Brief — {brief_date}", ""]
    L.append("## Bottom Line")
    if highs:
        L.append(f"{len(highs)} HIGH-priority well(s) overnight; "
                 f"~{total_deferred_bopd:.0f} BOPD (${total_deferred_usd:,.0f}/day) deferred and at risk. "
                 f"{len(meds)} on the watch list.")
    else:
        L.append("No HIGH-priority anomalies — fleet is stable. "
                 f"{len(meds)} item(s) on the watch list.")
    L += ["", "## Field Status",
          f"- Total oil: **{summary.get('total_bopd', 0):.0f} BOPD** across {summary.get('well_count', 0)} wells",
          f"- Water cut: **{summary.get('water_cut_pct', 0):.0f}%**",
          f"- Avg runtime: **{summary.get('avg_runtime_pct', 0):.1f}%**", ""]

    L.append("## Top Priorities")
    if highs:
        for i, a in enumerate(highs, 1):
            defer = (f" — deferring ~{a.deferred_bopd:.0f} BOPD (${a.deferred_usd_per_day:,.0f}/day)"
                     if a.deferred_bopd > 0 else "")
            L.append(f"{i}. **{a.well_id}** — {a.headline}{defer}. _Action:_ {a.recommended_action}")
    else:
        L.append("No HIGH-priority anomalies — fleet is stable.")
    L.append("")

    L.append("## Watch List")
    if meds:
        L += ["| Well | Category | Headline | Action |", "|---|---|---|---|"]
        for a in meds:
            L.append(f"| {a.well_id} | {a.category} | {a.headline} | {a.recommended_action} |")
    else:
        L.append("Nothing on the watch list.")
    L.append("")

    if dq or acked:
        L.append("## Data Quality / Acknowledged")
        for a in dq:
            L.append(f"- ⚠️ **{a.well_id}** — {a.headline} (verify before dispatching).")
        for a in acked:
            L.append(f"- 🔕 **{a.well_id}** ({a.category}) suppressed — acknowledged / known event.")
        L.append("")

    L += _render_events_section(events)

    L.append("## Closing")
    L.append("Multiple HIGH items — escalate at standup." if len(highs) > 1
             else ("One HIGH item to close out today." if highs else "Fleet stable; routine monitoring."))
    return "\n".join(L)


def _render_events_section(events: list | None) -> list[str]:
    """Render the **Ongoing & Resolved Events** block from the state machine.

    Ongoing (NEW/ONGOING) events that have lasted more than one day are the whole
    point of the state store: they show *day N* and the *cumulative* deferred
    bbl/$ so a still-down well never drops out of the brief. Just-RESOLVED events
    get one closing-out line. Returns [] (no section) when there are no events or
    the caller didn't pass an event list, so the brief shape is unchanged for the
    stateless path."""
    if not events:
        return []
    from .event_store import NEW, ONGOING, RESOLVED  # local import avoids a cycle

    ongoing = [e for e in events if e.state in (NEW, ONGOING) and not e.acknowledged]
    # "Ongoing" worth a dedicated callout = lasting beyond its first day (day 1 is
    # already covered by Top Priorities / Watch List as a fresh anomaly).
    multi_day = [e for e in ongoing if e.duration_days > 1]
    resolved = [e for e in events if e.state == RESOLVED and not e.acknowledged]
    if not multi_day and not resolved:
        return []

    L = ["## Ongoing & Resolved Events"]
    if multi_day:
        L += ["| Well | Type | State | Day | Cumulative deferred | Today | Note |",
              "|---|---|---|---|---|---|---|"]
        for e in multi_day:
            cum = (f"~{e.deferred_bopd:.0f} bbl (${e.deferred_usd:,.0f})"
                   if e.deferred_bopd > 0 else "—")
            today = (f"{e.last_deferred_bopd:.0f} BOPD" if e.last_deferred_bopd > 0 else "—")
            L.append(f"| {e.well_id} | {e.event_type} | {e.state} | "
                     f"day {e.duration_days} | {cum} | {today} | {e.headline} |")
    for e in resolved:
        span = f"{e.duration_days}-day" if e.duration_days > 1 else "1-day"
        cum = (f" ~{e.deferred_bopd:.0f} bbl (${e.deferred_usd:,.0f}) deferred over the event."
               if e.deferred_bopd > 0 else "")
        L.append(f"- ✅ **{e.well_id}** ({e.event_type}) — {span} event RESOLVED.{cum} {e.headline}")
    L.append("")
    return L


def write_brief(
    summary: dict,
    anomalies: list[Anomaly],
    brief_date: str | None = None,
    model: str | None = None,
    client=None,
    events: list | None = None,
) -> str:
    """LLM-narrated brief. ``events`` (optional) is the live state-machine event
    list (from ``event_store.update_events``); when present it is handed to the
    model so ONGOING multi-day events are narrated as continuations (day N, with
    cumulative deferred $) rather than re-discovered, and RESOLVED ones get a
    closing-out mention."""
    load_dotenv()
    if client is None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKey(
                "ANTHROPIC_API_KEY is not set. Use render_brief_markdown() for a "
                "deterministic brief, or set the key to get the LLM-narrated version.")
        from anthropic import Anthropic
        client = Anthropic(api_key=key)
    # Honor the MODEL env var that .env.example documents; fall back to the default.
    model = model or os.environ.get("MODEL", "claude-sonnet-4-6")

    brief_date = brief_date or date.today().isoformat()
    anomaly_dicts = [{**asdict(a)} for a in anomalies]

    events_block = ""
    if events:
        event_dicts = [{**asdict(e)} for e in events]
        events_block = (
            f"\nTracked events from the persistent state machine ({len(event_dicts)} live):\n"
            f"{json.dumps(event_dicts, indent=2)}\n"
        )

    user_prompt = (
        f"Date: {brief_date}\n\n"
        f"Fleet summary:\n{json.dumps(summary, indent=2)}\n\n"
        f"Anomalies detected overnight ({len(anomalies)} total):\n"
        f"{json.dumps(anomaly_dicts, indent=2)}\n"
        f"{events_block}\n"
        "Write the morning brief."
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")

    # Belt-and-suspenders: strip any preamble before the first markdown header.
    first_header = text.find("\n#")
    if first_header > 0 and not text.lstrip().startswith("#"):
        text = text[first_header:].lstrip()
    return text
