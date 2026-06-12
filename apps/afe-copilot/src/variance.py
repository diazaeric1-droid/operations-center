"""Variance analysis: actual cost vs. AFE, with breakdown by category, vendor, rig,
and AFE-supplement flagging against a policy overrun threshold."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


# Policy: an actual that exceeds its AFE by more than this requires a supplemental
# AFE before further spend (the boilerplate every AFE footer cites — made real here).
SUPPLEMENT_THRESHOLD_PCT = 10.0


@dataclass
class VarianceSummary:
    n_afes: int
    total_afe_usd: float
    total_actual_usd: float
    overall_variance_pct: float
    over_budget_count: int
    worst_offender_category: str | None
    worst_offender_overrun_usd: float          # ranked by $ overrun, not %
    worst_offender_pct: float | None           # None when the category was unbudgeted
    unbudgeted_categories: list[str] = field(default_factory=list)
    supplement_required_afes: list[str] = field(default_factory=list)


def analyze_variance(afe_df: pd.DataFrame, actuals_df: pd.DataFrame) -> VarianceSummary:
    """Compute portfolio variance, worst-offender category, and AFE-supplement flags.

    afe_df columns: afe_number, category, line_total_usd
    actuals_df columns: afe_number, category, actual_usd
    """
    merged = afe_df.merge(actuals_df, on=["afe_number", "category"], how="outer").fillna(0)
    merged["variance_usd"] = merged["actual_usd"] - merged["line_total_usd"]
    # pct is NA when there was no AFE budget for that line (a fully UNBUDGETED actual).
    merged["variance_pct"] = (merged["variance_usd"]
                              / merged["line_total_usd"].replace(0, pd.NA)) * 100

    by_afe = merged.groupby("afe_number").agg(
        afe_total=("line_total_usd", "sum"),
        actual_total=("actual_usd", "sum"),
    )
    by_afe["overrun_usd"] = by_afe["actual_total"] - by_afe["afe_total"]
    by_afe["pct"] = (by_afe["overrun_usd"] / by_afe["afe_total"].replace(0, pd.NA)) * 100

    by_cat = merged.groupby("category").agg(
        afe_total=("line_total_usd", "sum"),
        actual_total=("actual_usd", "sum"),
    )
    by_cat["overrun_usd"] = by_cat["actual_total"] - by_cat["afe_total"]
    by_cat["pct"] = (by_cat["overrun_usd"] / by_cat["afe_total"].replace(0, pd.NA)) * 100

    # Rank worst offender by ABSOLUTE $ overrun (what a VP cares about), so a 100%-
    # unbudgeted category — the most important case — is NOT dropped (the old
    # dropna() silently hid these). Only consider categories that actually overran.
    overruns = by_cat[by_cat["overrun_usd"] > 0].sort_values("overrun_usd", ascending=False)
    if not overruns.empty:
        worst_cat = str(overruns.index[0])
        worst_overrun = float(overruns["overrun_usd"].iloc[0])
        worst_pct_raw = overruns["pct"].iloc[0]
        worst_pct = None if pd.isna(worst_pct_raw) else float(worst_pct_raw)
    else:
        worst_cat, worst_overrun, worst_pct = None, 0.0, None

    unbudgeted = [str(c) for c in by_cat.index[(by_cat["afe_total"] == 0)
                                               & (by_cat["actual_total"] > 0)]]

    supplement_afes = [str(a) for a in by_afe.index[by_afe["pct"] > SUPPLEMENT_THRESHOLD_PCT]]

    total_afe = float(by_afe["afe_total"].sum())
    total_actual = float(by_afe["actual_total"].sum())
    overall_pct = ((total_actual - total_afe) / total_afe * 100) if total_afe else 0.0

    return VarianceSummary(
        n_afes=int(by_afe.shape[0]),
        total_afe_usd=total_afe,
        total_actual_usd=total_actual,
        overall_variance_pct=float(overall_pct),
        over_budget_count=int((by_afe["overrun_usd"] > 0).sum()),
        worst_offender_category=worst_cat,
        worst_offender_overrun_usd=worst_overrun,
        worst_offender_pct=worst_pct,
        unbudgeted_categories=unbudgeted,
        supplement_required_afes=supplement_afes,
    )


def demo_variance_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic AFE-vs-actuals for two closed-out AFEs — used by the dashboard so
    the Variance tab is populated out-of-the-box (incl. an unbudgeted 'Fishing' line
    and a >10% overrun that should trip a supplement)."""
    afe = pd.DataFrame([
        ("AFE-2026-0042", "Workover rig", 72_000),
        ("AFE-2026-0042", "Coiled tubing unit", 84_000),
        ("AFE-2026-0042", "Acid system", 38_000),
        ("AFE-2026-0042", "Flowback / disposal", 31_000),
        ("AFE-2026-0047", "Workover rig", 54_000),
        ("AFE-2026-0047", "Downhole pump", 5_200),
        ("AFE-2026-0047", "Rod inspection", 12_000),
    ], columns=["afe_number", "category", "line_total_usd"])
    actuals = pd.DataFrame([
        ("AFE-2026-0042", "Workover rig", 90_000),       # rig ran long (+25%)
        ("AFE-2026-0042", "Coiled tubing unit", 84_000),
        ("AFE-2026-0042", "Acid system", 41_500),
        ("AFE-2026-0042", "Flowback / disposal", 33_000),
        ("AFE-2026-0042", "Fishing", 28_500),            # UNBUDGETED — must not be hidden
        ("AFE-2026-0047", "Workover rig", 51_000),
        ("AFE-2026-0047", "Downhole pump", 5_200),
        ("AFE-2026-0047", "Rod inspection", 11_200),
    ], columns=["afe_number", "category", "actual_usd"])
    return afe, actuals
