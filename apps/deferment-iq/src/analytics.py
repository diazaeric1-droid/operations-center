"""Base-management analytics over the daily deferment table — the views an asset VP
actually reviews: KPIs, the deferment waterfall (volume bridge), a Pareto of $ lost by
cause, the worst-offender wells, MTTR by cause, and the recovery opportunity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .reason_codes import label_for, suggested_action


def _vol(daily: pd.DataFrame, rate_col: str, vol_col: str) -> pd.Series:
    """Volume (bbl) column, with a back-compat fallback to a rate column.

    The cadence-aware engine emits explicit calendar-day volume columns
    (``potential_vol`` / ``actual_vol``). Older callers / hand-built frames may only
    carry the per-record rate (``potential`` / ``bopd``); for daily data a span is one
    day so the rate equals the volume, which keeps those frames working.
    """
    return daily[vol_col] if vol_col in daily.columns else daily[rate_col]


def fleet_kpis(daily: pd.DataFrame, price_per_bbl: float = 70.0) -> dict:
    if daily.empty:
        return {}
    pot = float(_vol(daily, "potential", "potential_vol").sum())
    act = float(_vol(daily, "bopd", "actual_vol").sum())
    deferred = float(daily["total_def"].sum())
    loss = daily[daily["total_def"] > 1e-6]
    captured = float(loss[loss["reason_key"] != "unclassified"]["total_def"].sum())
    n_days = daily["date"].nunique()
    # Calendar days the period actually spans (cadence-aware): for monthly data this is
    # ~30× the row count, so deferred-per-day is barrels/day, not barrels/month.
    period_days = (float(daily.groupby("well_id")["span_days"].sum().max())
                   if "span_days" in daily.columns else float(n_days))
    period_days = max(period_days, 1.0)
    return {
        "n_wells": int(daily["well_id"].nunique()),
        "period_days": int(round(period_days)),
        "potential_bbl": pot,
        "actual_bbl": act,
        "deferred_bbl": deferred,
        "deferred_usd": deferred * float(price_per_bbl),
        "uptime_pct": (act / pot * 100.0) if pot > 0 else 100.0,   # production efficiency
        "pct_deferred": (deferred / pot * 100.0) if pot > 0 else 0.0,
        "capture_rate_pct": (captured / deferred * 100.0) if deferred > 0 else 100.0,
        "deferred_bopd_avg": deferred / period_days,
    }


def pareto_by_cause(daily: pd.DataFrame) -> pd.DataFrame:
    loss = daily[daily["total_def"] > 1e-6]
    if loss.empty:
        return pd.DataFrame(columns=["reason_key", "label", "deferred_bbl", "deferred_usd",
                                     "pct_of_total", "cum_pct", "recoverable", "planned"])
    g = loss.groupby("reason_key").agg(
        deferred_bbl=("total_def", "sum"),
        deferred_usd=("deferred_usd", "sum"),
        recoverable=("recoverable", "max"),
        planned=("planned", "max"),
    ).sort_values("deferred_usd", ascending=False).reset_index()
    g["label"] = g["reason_key"].map(label_for)
    tot = g["deferred_usd"].sum()
    g["pct_of_total"] = g["deferred_usd"] / tot * 100.0 if tot else 0.0
    g["cum_pct"] = g["pct_of_total"].cumsum()
    return g[["reason_key", "label", "deferred_bbl", "deferred_usd",
              "pct_of_total", "cum_pct", "recoverable", "planned"]]


def top_wells(daily: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    loss = daily[daily["total_def"] > 1e-6]
    if loss.empty:
        return pd.DataFrame(columns=["well_id", "deferred_bbl", "deferred_usd", "top_cause", "uptime_pct"])
    d = daily.copy()
    d["_pot_vol"] = _vol(d, "potential", "potential_vol")
    d["_act_vol"] = _vol(d, "bopd", "actual_vol")
    by_well = d.groupby("well_id").agg(
        deferred_bbl=("total_def", "sum"), deferred_usd=("deferred_usd", "sum"),
        potential=("_pot_vol", "sum"), actual=("_act_vol", "sum")).reset_index()
    # dominant cause per well
    cause = (loss.groupby(["well_id", "reason_key"])["deferred_usd"].sum()
             .reset_index().sort_values("deferred_usd", ascending=False)
             .drop_duplicates("well_id").set_index("well_id")["reason_key"])
    by_well["top_cause"] = by_well["well_id"].map(cause).map(lambda k: label_for(k) if isinstance(k, str) else "—")
    by_well["uptime_pct"] = np.where(by_well["potential"] > 0,
                                     by_well["actual"] / by_well["potential"] * 100.0, 100.0)
    return (by_well.sort_values("deferred_usd", ascending=False).head(n)
            [["well_id", "deferred_bbl", "deferred_usd", "top_cause", "uptime_pct"]]
            .reset_index(drop=True))


def waterfall(daily: pd.DataFrame) -> list[dict]:
    """Volume bridge: gross potential → minus each cause (planned first) → actual."""
    if daily.empty:
        return []
    pot = float(_vol(daily, "potential", "potential_vol").sum())
    act = float(_vol(daily, "bopd", "actual_vol").sum())
    steps = [{"label": "Gross potential", "value": pot, "kind": "total"}]
    par = pareto_by_cause(daily)
    # show planned losses first (expected), then unplanned biggest-first
    par = pd.concat([par[par["planned"]], par[~par["planned"]]])
    for _, r in par.iterrows():
        steps.append({"label": r["label"], "value": -float(r["deferred_bbl"]), "kind": "loss"})
    steps.append({"label": "Actual produced", "value": act, "kind": "total"})
    return steps


def mttr_by_cause(events_classified: pd.DataFrame) -> pd.DataFrame:
    """Mean-time-to-restore per cause from the event log (duration in days)."""
    if events_classified.empty or "reason_key" not in events_classified.columns:
        return pd.DataFrame(columns=["reason_key", "label", "n_events", "mttr_days", "total_event_days"])
    ev = events_classified.copy()
    ev["dur"] = (pd.to_datetime(ev["end_date"]) - pd.to_datetime(ev["start_date"])).dt.days + 1
    g = ev.groupby("reason_key").agg(n_events=("dur", "size"),
                                     mttr_days=("dur", "mean"),
                                     total_event_days=("dur", "sum")).reset_index()
    g["label"] = g["reason_key"].map(label_for)
    return g.sort_values("total_event_days", ascending=False)[
        ["reason_key", "label", "n_events", "mttr_days", "total_event_days"]].reset_index(drop=True)


def recovery_opportunity(daily: pd.DataFrame) -> dict:
    """Actionable opportunity = deferred $ in RECOVERABLE causes (excludes planned work
    and reservoir/watering-out, which you can't get back)."""
    if daily.empty:
        return {"recoverable_bbl": 0.0, "recoverable_usd": 0.0, "unclassified_usd": 0.0}
    loss = daily[daily["total_def"] > 1e-6]
    rec = loss[loss["recoverable"]]
    return {
        "recoverable_bbl": float(rec["total_def"].sum()),
        "recoverable_usd": float(rec["deferred_usd"].sum()),
        "unclassified_usd": float(loss[loss["reason_key"] == "unclassified"]["deferred_usd"].sum()),
    }


RECOVERY_QUEUE_COLUMNS = [
    "well_id", "cause", "reason_key", "recoverable_bbl", "recoverable_usd",
    "mttr_days", "priority_score", "suggested_action",
]


def recovery_queue(daily: pd.DataFrame, events: pd.DataFrame | None = None,
                   oil_price: float = 70.0) -> pd.DataFrame:
    """Prioritized recovery work-queue: one actionable item per (well, recoverable cause).

    Converts the deferment analytics from "where are barrels lost" into "what to do next,
    what it's worth, who acts" — the Quantify→Authorize handoff. Only RECOVERABLE causes
    are included; planned work and reservoir/watering-out (and unclassified) are excluded
    because those barrels can't be recovered by an intervention.

    Scoring
    -------
    Items are ranked by ``priority_score = recoverable_usd / max(mttr_days, 1)`` — value
    per day-to-restore, so a quick high-$ win outranks a slow one of similar value (a
    barrels-per-day-of-effort proxy). ``mttr_days`` is the mean-time-to-restore for that
    cause from the event log (falls back to 1.0 day when no event history is available),
    so the divisor never inflates or zeroes the score. Sorted by ``priority_score`` desc.

    Returns columns: well_id, cause (label), reason_key, recoverable_bbl, recoverable_usd,
    mttr_days, priority_score, suggested_action.
    """
    if daily is None or daily.empty:
        return pd.DataFrame(columns=RECOVERY_QUEUE_COLUMNS)
    loss = daily[(daily["total_def"] > 1e-6) & daily["recoverable"]]
    if loss.empty:
        return pd.DataFrame(columns=RECOVERY_QUEUE_COLUMNS)

    q = loss.groupby(["well_id", "reason_key"]).agg(
        recoverable_bbl=("total_def", "sum"),
        recoverable_usd=("deferred_usd", "sum"),
    ).reset_index()

    # MTTR (days) per cause from the event log; default 1.0 day if unavailable.
    mttr_map: dict[str, float] = {}
    if events is not None and len(events):
        m = mttr_by_cause(events)
        if len(m):
            mttr_map = dict(zip(m["reason_key"], m["mttr_days"]))
    q["mttr_days"] = q["reason_key"].map(mttr_map).fillna(1.0).clip(lower=1.0)

    # Re-price defensively if caller passes a different price than `daily` was built with.
    q["priority_score"] = q["recoverable_usd"] / q["mttr_days"]
    q["cause"] = q["reason_key"].map(label_for)
    q["suggested_action"] = q["reason_key"].map(suggested_action)

    q = q.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return q[RECOVERY_QUEUE_COLUMNS]


def deferment_trend(daily: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "deferred_bbl", "potential_bbl"])
    d = daily.copy()
    d["_pot_vol"] = _vol(d, "potential", "potential_vol")
    t = (d.set_index("date").groupby(pd.Grouper(freq=freq))
         .agg(deferred_bbl=("total_def", "sum"), potential_bbl=("_pot_vol", "sum")).reset_index())
    return t
