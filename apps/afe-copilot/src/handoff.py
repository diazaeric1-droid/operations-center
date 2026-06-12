"""Pipeline handoff (stage 3 of 3): turn a ``WellDiagnosis`` into a complete AFE.

    Daily Production Digest  ──WellAlert──▶  ESP Failure-Risk Agent  ──WellDiagnosis──▶  [AFE Copilot]

``render_afe_markdown`` assembles a full, decision-ready AFE **deterministically**
(cost DB + economics + risk register + authority routing) with NO LLM call, so the
end-to-end pipeline runs with zero API keys. Pass ``--llm`` to instead route through
the richer Claude drafter (``drafter.run_drafter``) when a key is available.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .cost_db import cost_rollup, lookup_cost_template
from .economics import compute_economics, price_sensitivity
from .models import AFEDiagnosis
from .risk_register import lookup_risks
from .tracker import required_approver


def _fmt(x: float) -> str:
    return f"${x:,.0f}"


def render_afe_markdown(diag: dict, working_interest: float = 1.0,
                        net_revenue_interest: float = 0.80,
                        realized_price: float = 70.0) -> str:
    """Deterministic AFE markdown from a WellDiagnosis dict (validated via
    AFEDiagnosis.from_pe_copilot)."""
    d = AFEDiagnosis.from_pe_copilot(diag)
    items = lookup_cost_template(d.intervention)
    roll = cost_rollup(d.intervention)
    total = roll["total"]
    approver = required_approver(total)
    risks = lookup_risks(d.intervention)
    risk_score = diag.get("esp_risk_score")
    mode = diag.get("suspected_mode")

    econ = compute_economics(
        total, d.incremental_rate_bopd, d.expected_uplift_decline_per_yr,
        realized_price_per_bbl=realized_price,
        working_interest=working_interest, net_revenue_interest=net_revenue_interest)
    deck = price_sensitivity(
        total, d.incremental_rate_bopd, uplift_decline_per_yr=d.expected_uplift_decline_per_yr,
        working_interest=working_interest, net_revenue_interest=net_revenue_interest)

    L: list[str] = []
    L.append("# AUTHORIZATION FOR EXPENDITURE")
    L.append(f"*{d.field} · Well {d.well_id} · {d.intervention.replace('_', ' ').title()}*")
    L.append("")
    L.append("## AFE Header")
    L += ["| Field | Detail |", "|---|---|",
          "| **AFE Number** | TBD — assigned by Finance at approval |",
          f"| **Well / API** | {d.well_id} / {d.api_number} |",
          f"| **Field / Operator** | {d.field} / {d.operator} |",
          f"| **Requested By** | {d.requested_by} |",
          f"| **Date Prepared** | {date.today().isoformat()} |",
          f"| **Routed To (authority limit)** | {approver} |", ""]

    L.append("## Executive Summary")
    # Only add an explicit risk sentence if the diagnosis text doesn't already carry one.
    risk_txt = (f" ESP 30-day failure risk **{risk_score:.0%}**."
                if isinstance(risk_score, (int, float)) and "failure risk" not in d.primary_diagnosis
                else "")
    L.append(
        f"{d.well_id} — {d.primary_diagnosis}{risk_txt} Recommended intervention: "
        f"**{d.intervention.replace('_', ' ')}** at an estimated **{_fmt(total)}** "
        f"(routes to **{approver}**). Estimated uplift **+{d.incremental_rate_bopd:.0f} BOPD**, "
        f"gross NPV@10% **{_fmt(econ.npv_10pct_usd)}**, net-to-operator NPV "
        f"**{_fmt(econ.net_npv_10pct_usd)}** (WI {working_interest:.0%}/NRI {net_revenue_interest:.0%}), "
        f"payout **{'—' if econ.payout_months == float('inf') else f'{econ.payout_months:.0f} mo'}**.")
    L.append("")

    L.append("## Cost Breakdown")
    L += ["| Category | Description | Qty | Unit | Unit Cost | Total | Class |",
          "|---|---|---|---|---|---|---|"]
    for it in items:
        L.append(f"| {it.category} | {it.description} | {it.qty:g} | {it.unit} | "
                 f"{_fmt(it.unit_cost_usd)} | {_fmt(it.total_usd)} | {it.cost_class} |")
    L.append(f"| | | | | **AFE TOTAL** | **{_fmt(total)}** | |")
    L.append("")
    L.append(f"**Tangible (capitalized):** {_fmt(roll['tangible'])} · "
             f"**Intangible (IDC, expensed):** {_fmt(roll['intangible'])} · "
             f"**Contingency:** {_fmt(roll['contingency'])}")
    L.append("")

    L.append("## Economics")
    L += ["| Metric | Value |", "|---|---|",
          f"| Total AFE cost | {_fmt(total)} |",
          f"| Incremental rate | +{d.incremental_rate_bopd:.0f} BOPD |",
          f"| First-year incremental | {econ.incremental_first_year_bbl:,.0f} bbl |",
          f"| Gross NPV @ 10% | {_fmt(econ.npv_10pct_usd)} |",
          f"| Net NPV to operator (WI {working_interest:.0%}/NRI {net_revenue_interest:.0%}) | {_fmt(econ.net_npv_10pct_usd)} |",
          f"| Payout | {'—' if econ.payout_months == float('inf') else f'{econ.payout_months:.0f} mo'} |",
          f"| Cost per incremental bbl | {_fmt(econ.dollars_per_incremental_bbl)} |", ""]
    L.append("**Price-deck sensitivity (gross NPV @ 10%):**")
    L += ["| Realized $/bbl | Gross NPV | Net NPV | Payout |", "|---|---|---|---|"]
    for row in deck:
        pm = "—" if row["payout_months"] == float("inf") else f"{row['payout_months']:.0f} mo"
        L.append(f"| {_fmt(row['realized_price'])} | {_fmt(row['npv_usd'])} | {_fmt(row['net_npv_usd'])} | {pm} |")
    L.append("")

    L.append("## Risk Register")
    if risks:
        L += ["| Category | Risk | Likelihood | Consequence | Mitigation |", "|---|---|---|---|---|"]
        for r in risks:
            L.append(f"| {r.category} | {r.description} | {r.likelihood} | {r.consequence} | {r.mitigation} |")
    else:
        L.append("_No standard risks on file for this intervention._")
    L.append("")

    L.append("## Approvals")
    L.append(f"This AFE's value ({_fmt(total)}) requires sign-off up to **{approver}** "
             "per delegation-of-authority limits (PE < $50k · Eng Mgr < $250k · "
             "Ops Mgr < $1MM · VP above).")
    L += ["", "| Role | Name | Signature | Date |", "|---|---|---|---|",
          "| Prepared By | (auto — ESP Failure-Risk Agent) | | |",
          f"| {approver} | | | |", ""]
    L.append("> Auto-drafted by the PE pipeline (digest → ESP → AFE). Cost overruns "
             "exceeding 10% of AFE total require a supplemental AFE.")
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description="Render an AFE from a WellDiagnosis JSON.")
    parser.add_argument("--input", required=True, help="WellDiagnosis JSON path")
    parser.add_argument("--out", default=None, help="Write AFE markdown here (default: stdout)")
    parser.add_argument("--docx", default=None, help="Also write a .docx to this path")
    parser.add_argument("--wi", type=float, default=1.0, help="Working interest")
    parser.add_argument("--nri", type=float, default=0.80, help="Net revenue interest")
    parser.add_argument("--price", type=float, default=70.0, help="Realized price $/bbl")
    parser.add_argument("--llm", action="store_true",
                        help="Use the Claude drafter instead of the deterministic renderer (needs API key)")
    args = parser.parse_args()

    diag = json.loads(Path(args.input).read_text())

    if args.llm:
        from .drafter import MissingAPIKey, run_drafter
        try:
            md = run_drafter(AFEDiagnosis.from_pe_copilot(diag))
        except MissingAPIKey:
            print("No API key — falling back to the deterministic renderer.")
            md = render_afe_markdown(diag, working_interest=args.wi,
                                     net_revenue_interest=args.nri, realized_price=args.price)
    else:
        md = render_afe_markdown(diag, working_interest=args.wi,
                                 net_revenue_interest=args.nri, realized_price=args.price)

    if args.out:
        Path(args.out).write_text(md)
        print(f"Wrote AFE for {diag.get('well_id')} → {args.out}")
    else:
        print(md)

    if args.docx:
        from .docx_builder import build_docx
        build_docx(md, args.docx, AFEDiagnosis.from_pe_copilot(diag))
        print(f"Wrote {args.docx}")


if __name__ == "__main__":
    main()
