"""Main entry point: run once per morning. Cron, GitHub Actions, and Streamlit
all call into this same function. Outputs to briefs/YYYY-MM-DD.md."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from .anomaly_detector import load_acknowledgements, scan_fleet
from .brief_writer import MissingAPIKey, render_brief_markdown, write_brief
from .data_loader import fleet_summary, load_fleet
from .event_store import DEFAULT_EVENT_DB, EventStore, update_events


DEFAULT_DATA_DIR = "data/synthetic/fleet"
BRIEFS_DIR = Path("briefs")
ACK_PATH = "acknowledged.yml"


def run(data_dir: str = DEFAULT_DATA_DIR, brief_date: str | None = None,
        verbose: bool = False, event_db: str = DEFAULT_EVENT_DB) -> Path:
    """Generate today's brief and persist to disk. Returns the brief's path.

    Drives the persistent event state machine (``event_db``) so a multi-day outage
    is reported as ONGOING every morning — with running duration + cumulative
    deferred bbl/$ — instead of vanishing once it ages out of the stateless
    detector's lookback window, and a just-recovered well gets one closing-out
    mention before dropping off."""
    console = Console()
    brief_date = brief_date or date.today().isoformat()

    if verbose:
        console.print(f"[bold cyan]Loading fleet from {data_dir}...[/]")
    fleet = load_fleet(data_dir)
    if not fleet:
        raise RuntimeError(f"No wells found in {data_dir}. Run data/synthetic/generate_fleet.py first.")

    summary = fleet_summary(fleet)
    if verbose:
        console.print(f"[bold]Fleet:[/] {summary['well_count']} wells · "
                      f"{summary['total_bopd']:.0f} BOPD · {summary['water_cut_pct']:.0f}% WC")

    if verbose:
        console.print("[bold cyan]Scanning for anomalies...[/]")
    acknowledged = load_acknowledgements(ACK_PATH)
    anomalies = scan_fleet(fleet, acknowledged=acknowledged)
    if verbose:
        sev_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for a in anomalies:
            sev_counts[a.severity] += 1
        deferred = sum(a.deferred_usd_per_day for a in anomalies if not a.acknowledged)
        console.print(f"[bold]Anomalies:[/] {sev_counts['HIGH']} HIGH · {sev_counts['MEDIUM']} MEDIUM · "
                      f"{sev_counts['LOW']} LOW · ${deferred:,.0f}/day deferred")

    # Advance the persistent event state machine one day. This is what gives the
    # brief memory across runs: ongoing events keep appearing (with duration +
    # cumulative deferral) even after the stateless scan above goes quiet.
    store = EventStore(event_db)
    try:
        events = update_events(store, fleet, as_of=brief_date,
                               acknowledged=acknowledged)
    finally:
        store.close()
    if verbose:
        from .event_store import NEW, ONGOING, RESOLVED
        open_n = sum(1 for e in events if e.state in (NEW, ONGOING))
        res_n = sum(1 for e in events if e.state == RESOLVED)
        console.print(f"[bold]Events:[/] {open_n} open (NEW/ONGOING) · {res_n} just-resolved")

    # Detection is deterministic; the LLM only narrates. With no API key we still
    # emit a real (templated) brief instead of crashing.
    try:
        if verbose:
            console.print("[bold cyan]Writing brief (LLM)...[/]")
        brief_md = write_brief(summary, anomalies, brief_date=brief_date, events=events)
    except MissingAPIKey:
        if verbose:
            console.print("[yellow]No ANTHROPIC_API_KEY — writing deterministic brief.[/]")
        brief_md = render_brief_markdown(summary, anomalies, brief_date=brief_date,
                                         events=events)

    BRIEFS_DIR.mkdir(exist_ok=True)
    out_path = BRIEFS_DIR / f"{brief_date}.md"
    out_path.write_text(brief_md)
    if verbose:
        console.print(f"\n[bold green]Wrote {out_path}[/]\n")
        console.print(Markdown(brief_md))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run the daily production digest.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", default=None, help="Override brief date (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run(data_dir=args.data_dir, brief_date=args.date, verbose=args.verbose)


if __name__ == "__main__":
    main()
