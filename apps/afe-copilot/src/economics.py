"""Quick economics for AFE — NPV, payout, $/BOE. Mirrors the production-engineer-copilot
economics module but exposed via this repo for standalone use."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import econ_core as _ec


@dataclass
class AFEEconomics:
    treatment_cost_usd: float
    incremental_first_year_bbl: float
    incremental_eur_bbl: float
    npv_10pct_usd: float
    payout_months: float
    dollars_per_incremental_bbl: float
    # Working-interest / net-revenue-interest view (operator's share). When WI=NRI=1
    # these equal the gross figures above.
    working_interest: float = 1.0
    net_revenue_interest: float = 1.0
    net_cost_to_operator_usd: float = 0.0
    net_npv_10pct_usd: float = 0.0


def compute_economics(
    treatment_cost_usd: float,
    incremental_rate_bopd: float,
    uplift_decline_per_yr: float = 0.6,
    horizon_years: int = 5,
    realized_price_per_bbl: float = 65.0,
    opex_per_bbl: float = 12.0,
    discount_rate: float = 0.10,
    working_interest: float = 1.0,
    net_revenue_interest: float = 1.0,
) -> AFEEconomics:
    months = _ec.month_index(horizon_years)
    monthly_rate = _ec.exp_uplift_rate(incremental_rate_bopd, uplift_decline_per_yr, months)
    monthly_vol = monthly_rate * _ec.DAYS_PER_MONTH
    margin_per_bbl = realized_price_per_bbl - opex_per_bbl
    monthly_revenue = monthly_vol * margin_per_bbl

    # TRUE effective-annual discounting: a 10% input means 10% per YEAR, so the
    # monthly factor is (1+r)^(m/12), not (1+r/12)^m (which is 10.47% effective).
    pv = _ec.discounted_pv(monthly_revenue, discount_rate)
    npv = pv - treatment_cost_usd

    payout = _ec.payout_months(monthly_revenue, treatment_cost_usd)

    first_year_bbl = float(monthly_vol[:12].sum())
    eur = float(monthly_vol.sum())
    dollars_per_bbl = treatment_cost_usd / first_year_bbl if first_year_bbl > 0 else float("inf")

    # Operator's net position: it bears WI% of the cost and keeps NRI% of revenue.
    net_cost = treatment_cost_usd * working_interest
    net_pv = _ec.discounted_pv(monthly_revenue * net_revenue_interest, discount_rate)
    net_npv = net_pv - net_cost

    return AFEEconomics(
        treatment_cost_usd=treatment_cost_usd,
        incremental_first_year_bbl=first_year_bbl,
        incremental_eur_bbl=eur,
        npv_10pct_usd=npv,
        payout_months=payout,
        dollars_per_incremental_bbl=dollars_per_bbl,
        working_interest=working_interest,
        net_revenue_interest=net_revenue_interest,
        net_cost_to_operator_usd=net_cost,
        net_npv_10pct_usd=net_npv,
    )


def price_sensitivity(
    treatment_cost_usd: float,
    incremental_rate_bopd: float,
    prices: tuple[float, ...] = (45.0, 55.0, 65.0, 75.0, 85.0),
    **kwargs,
) -> list[dict]:
    """NPV / payout across a realized-price deck — the price-strip row a VP asks for.

    Returns one row per price with NPV, payout months, and $/bbl, holding the rest
    of the assumptions fixed. ``kwargs`` pass through to ``compute_economics`` (decline,
    opex, WI/NRI, etc.).
    """
    rows = []
    for p in prices:
        e = compute_economics(treatment_cost_usd, incremental_rate_bopd,
                              realized_price_per_bbl=p, **kwargs)
        rows.append({
            "realized_price": p,
            "npv_usd": e.npv_10pct_usd,
            "net_npv_usd": e.net_npv_10pct_usd,
            "payout_months": e.payout_months,
            "dollars_per_bbl": e.dollars_per_incremental_bbl,
        })
    return rows


def jib_split(gross_cost_usd: float, partners: dict[str, float]) -> list[dict]:
    """Joint-Interest-Billing preview: allocate a gross AFE cost across partners by
    working interest. ``partners`` maps name -> WI fraction (should sum to ~1.0)."""
    total_wi = sum(partners.values()) or 1.0
    return [
        {"partner": name, "working_interest": wi,
         "net_cost_usd": gross_cost_usd * wi,
         "share_of_afe_pct": 100.0 * wi / total_wi}
        for name, wi in partners.items()
    ]


# ---------- Monte-Carlo NPV ---------------------------------------------------

@dataclass
class MonteCarloResult:
    """Distributional NPV outcome from simulate_economics."""
    n_trials: int
    npv_p10_usd: float          # conservative downside (10th percentile)
    npv_p50_usd: float          # median
    npv_p90_usd: float          # optimistic upside (90th percentile)
    npv_mean_usd: float
    probability_of_payout: float  # P(NPV > 0 AND payout within 24 months)
    tornado: dict[str, dict[str, float]]  # var -> {"low": npv, "high": npv, "swing": abs}
    base_npv_usd: float


def _npv_vectorized(
    treatment_cost_usd: float,
    incremental_rate_bopd,        # scalar or np.ndarray
    uplift_decline_per_yr,        # scalar or np.ndarray
    realized_price_per_bbl,       # scalar or np.ndarray
    horizon_years: int = 5,
    opex_per_bbl: float = 12.0,
    discount_rate: float = 0.10,
):
    """Vectorized NPV reusing the exact deterministic math from compute_economics.

    Each input draw can be a scalar or a 1-D array of length n; broadcasting over
    the monthly time axis yields an (n_draws,) NPV vector (or a scalar). This is
    the same formula as compute_economics, just batched for Monte-Carlo speed.
    """
    months = _ec.month_index(horizon_years)                                    # (T,)
    rate = np.atleast_1d(np.asarray(incremental_rate_bopd, dtype=float))      # (n,)
    decline = np.atleast_1d(np.asarray(uplift_decline_per_yr, dtype=float))   # (n,)
    price = np.atleast_1d(np.asarray(realized_price_per_bbl, dtype=float))    # (n,)

    # exp_uplift_rate batched path -> (n, T)
    monthly_rate = _ec.exp_uplift_rate(rate, decline, months)                  # (n,T)
    monthly_vol = monthly_rate * _ec.DAYS_PER_MONTH                            # (n,T)
    margin_per_bbl = price - opex_per_bbl                                      # (n,)
    monthly_revenue = monthly_vol * margin_per_bbl[:, None]                    # (n,T)

    # discounted_pv handles (n,T) -> (n,) via its axis=-1 sum
    npv = _ec.discounted_pv(monthly_revenue, discount_rate) - treatment_cost_usd
    return npv  # (n,)


def _payout_within(
    treatment_cost_usd: float,
    incremental_rate_bopd: float,
    uplift_decline_per_yr: float,
    realized_price_per_bbl: float,
    months_cap: int,
    horizon_years: int = 5,
    opex_per_bbl: float = 12.0,
) -> np.ndarray:
    """Boolean array: did the (undiscounted) cumulative net revenue recover the
    treatment cost within months_cap, per draw? Mirrors compute_economics payout."""
    months = _ec.month_index(horizon_years)
    rate = np.atleast_1d(np.asarray(incremental_rate_bopd, dtype=float))      # (n,)
    decline = np.atleast_1d(np.asarray(uplift_decline_per_yr, dtype=float))   # (n,)
    price = np.atleast_1d(np.asarray(realized_price_per_bbl, dtype=float))    # (n,)

    # exp_uplift_rate batched -> (n, T)
    monthly_vol = _ec.exp_uplift_rate(rate, decline, months) * _ec.DAYS_PER_MONTH
    monthly_revenue = monthly_vol * (price - opex_per_bbl)[:, None]           # (n,T)

    cap = max(1, min(months_cap, len(months)))   # guard months_cap <= 0
    cumulative = np.cumsum(monthly_revenue, axis=1)             # (n,T)
    # recovered within cap months if cumulative at month `cap` >= cost
    return cumulative[:, cap - 1] >= treatment_cost_usd


def simulate_economics(
    treatment_cost_usd: float,
    incremental_rate_bopd: float,
    uplift_decline_per_yr: float = 0.6,
    realized_price_per_bbl: float = 65.0,
    horizon_years: int = 5,
    opex_per_bbl: float = 12.0,
    discount_rate: float = 0.10,
    n_trials: int = 10_000,
    rate_rel_spread: float = 0.30,        # incremental_rate_bopd ±30% (uniform)
    decline_abs_spread: float = 0.15,     # uplift_decline_per_yr ±0.15 abs (uniform)
    price_sd: float = 12.0,               # realized_price normal sd (~$12)
    payout_cap_months: int = 24,
    seed: int | None = 42,
) -> MonteCarloResult:
    """Monte-Carlo NPV over the three biggest AFE uncertainties.

    Draws ~n_trials samples of:
      - incremental_rate_bopd  ~ Uniform(base*(1-rel), base*(1+rel))
      - uplift_decline_per_yr  ~ Uniform(base-abs, base+abs), clipped to (0, 2)
      - realized_price_per_bbl ~ Normal(base, price_sd), clipped at $1 floor

    Each draw reuses the deterministic NPV math (see _npv_vectorized). Returns
    P10/P50/P90 NPV, P(payout within `payout_cap_months`), and a tornado dict
    holding each variable's NPV swing when moved to its low/high while the others
    sit at base — the classic single-variable sensitivity view.
    """
    rng = np.random.default_rng(seed)

    rate_lo = incremental_rate_bopd * (1 - rate_rel_spread)
    rate_hi = incremental_rate_bopd * (1 + rate_rel_spread)
    rate_draws = rng.uniform(rate_lo, rate_hi, n_trials)

    decline_lo = uplift_decline_per_yr - decline_abs_spread
    decline_hi = uplift_decline_per_yr + decline_abs_spread
    decline_draws = np.clip(rng.uniform(decline_lo, decline_hi, n_trials), 1e-6, 2.0)

    price_draws = np.clip(rng.normal(realized_price_per_bbl, price_sd, n_trials), 1.0, None)

    npvs = _npv_vectorized(
        treatment_cost_usd, rate_draws, decline_draws, price_draws,
        horizon_years=horizon_years, opex_per_bbl=opex_per_bbl, discount_rate=discount_rate,
    )

    paid = _payout_within(
        treatment_cost_usd, rate_draws, decline_draws, price_draws,
        months_cap=payout_cap_months, horizon_years=horizon_years, opex_per_bbl=opex_per_bbl,
    )
    prob_payout = float(np.mean((npvs > 0) & paid))

    p10, p50, p90 = (float(x) for x in np.percentile(npvs, [10, 50, 90]))

    # Tornado: move ONE variable to its low/high (others at base value).
    base = float(_npv_vectorized(
        treatment_cost_usd, incremental_rate_bopd, uplift_decline_per_yr, realized_price_per_bbl,
        horizon_years=horizon_years, opex_per_bbl=opex_per_bbl, discount_rate=discount_rate,
    )[0])

    def _one(rate, decl, price) -> float:
        return float(_npv_vectorized(
            treatment_cost_usd, rate, decl, price,
            horizon_years=horizon_years, opex_per_bbl=opex_per_bbl, discount_rate=discount_rate,
        )[0])

    # use the same low/high anchors as the draws (decline low = lower decline = higher NPV)
    decl_lo_anchor = max(uplift_decline_per_yr - decline_abs_spread, 1e-6)
    decl_hi_anchor = min(uplift_decline_per_yr + decline_abs_spread, 2.0)
    price_lo_anchor = max(realized_price_per_bbl - 1.2816 * price_sd, 1.0)   # ~P10 of normal
    price_hi_anchor = realized_price_per_bbl + 1.2816 * price_sd             # ~P90 of normal

    tornado: dict[str, dict[str, float]] = {}
    for name, lo_vals, hi_vals in (
        ("incremental_rate_bopd",
         (rate_lo, uplift_decline_per_yr, realized_price_per_bbl),
         (rate_hi, uplift_decline_per_yr, realized_price_per_bbl)),
        ("uplift_decline_per_yr",
         (incremental_rate_bopd, decl_hi_anchor, realized_price_per_bbl),   # high decline -> low NPV
         (incremental_rate_bopd, decl_lo_anchor, realized_price_per_bbl)),  # low decline -> high NPV
        ("realized_price_per_bbl",
         (incremental_rate_bopd, uplift_decline_per_yr, price_lo_anchor),
         (incremental_rate_bopd, uplift_decline_per_yr, price_hi_anchor)),
    ):
        low_npv = _one(*lo_vals)
        high_npv = _one(*hi_vals)
        tornado[name] = {
            "low": low_npv,
            "high": high_npv,
            "swing": abs(high_npv - low_npv),
        }

    return MonteCarloResult(
        n_trials=n_trials,
        npv_p10_usd=p10,
        npv_p50_usd=p50,
        npv_p90_usd=p90,
        npv_mean_usd=float(np.mean(npvs)),
        probability_of_payout=prob_payout,
        tornado=tornado,
        base_npv_usd=base,
    )
