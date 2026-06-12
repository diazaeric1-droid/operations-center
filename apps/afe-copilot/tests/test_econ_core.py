"""Tests for econ_core — the suite-wide economics kernel.

These lock the two things that actually drifted across apps: the discount convention
(effective-annual, NOT monthly-compounded) and the risking convention.
"""
import numpy as np
import pytest

from src import econ_core as ec


def test_discount_is_effective_annual_not_monthly_compounded():
    # The whole point of the module: a 10% input discounts a 12-month cash flow by
    # exactly 1.10, NOT 1.1047 (which is what (1 + r/12)**m gives).
    df = ec.discount_factors([12], 0.10)[0]
    assert df == pytest.approx(1.10, abs=1e-12)
    assert df != pytest.approx(1.1047, abs=1e-3)
    # 24 months -> 1.21 exactly under effective-annual.
    assert ec.discount_factors([24], 0.10)[0] == pytest.approx(1.21, abs=1e-12)


def test_discount_factors_monotonic_and_unit_at_t0_equiv():
    df = ec.discount_factors(ec.month_index(5), 0.10)
    assert np.all(np.diff(df) > 0)            # strictly increasing in time
    assert df[0] == pytest.approx((1.10) ** (1 / 12))


def test_arps_exponential_branch():
    months = ec.month_index(1)
    rate = ec.arps_monthly_rate(qi=1000.0, di=0.70, b=0.0, months=months)
    # exact exponential at 12 months: 1000*exp(-0.70)
    assert rate[-1] == pytest.approx(1000.0 * np.exp(-0.70), rel=1e-9)


def test_arps_hyperbolic_above_exponential_tail():
    months = ec.month_index(10)
    hyp = ec.arps_monthly_rate(1000.0, 0.70, 1.0, months)
    exp = ec.arps_monthly_rate(1000.0, 0.70, 0.0, months)
    # hyperbolic declines slower -> tail sits above the exponential
    assert hyp[-1] > exp[-1]


def test_exp_uplift_scalar_and_batched_agree():
    months = ec.month_index(2)
    scalar = ec.exp_uplift_rate(100.0, 0.6, months)
    batched = ec.exp_uplift_rate(np.array([100.0, 100.0]), np.array([0.6, 0.6]), months)
    assert batched.shape == (2, len(months))
    assert np.allclose(batched[0], scalar)


def test_discounted_pv_scalar_and_batched():
    months = ec.month_index(1)
    stream = np.ones(len(months)) * 1000.0
    pv = ec.discounted_pv(stream, 0.10)
    assert isinstance(pv, float)
    batch = np.vstack([stream, 2 * stream])
    pvb = ec.discounted_pv(batch, 0.10)
    assert pvb.shape == (2,)
    assert pvb[1] == pytest.approx(2 * pvb[0])


def test_risked_npv_equals_dry_hole_framing():
    # risked_npv = pc*PV - cost must equal pc*(PV-cost) + (1-pc)*(-cost) for all pc.
    pv, cost = 5_000_000.0, 1_200_000.0
    for pc in (0.0, 0.15, 0.5, 0.85, 1.0):
        unified = ec.risked_npv(pv, cost, pc)
        dry_hole = pc * (pv - cost) + (1 - pc) * (-cost)
        assert unified == pytest.approx(dry_hole, rel=1e-12)


def test_risked_npv_pc_one_is_plain_npv():
    pv, cost = 3_000_000.0, 800_000.0
    assert ec.risked_npv(pv, cost, 1.0) == pytest.approx(ec.npv(pv, cost))


def test_payout_months_basic():
    # $100/month, cost $250 -> recovered at end of month 3.
    mr = np.full(12, 100.0)
    assert ec.payout_months(mr, 250.0) == 3.0
    # never recovers within horizon -> inf
    assert ec.payout_months(mr, 5000.0) == float("inf")


def test_irr_annual_on_known_stream():
    # capex 1.0, then 12 monthly cash flows that PV to >capex at 10% -> positive IRR.
    monthly_cf = np.full(12, 0.10)            # 1.20 nominal over a year vs 1.0 capex
    irr = ec.irr_annual(monthly_cf, 1.0)
    assert irr is not None and irr > 0.0
    # NPV at the returned IRR should be ~0 (within bisection tolerance)
    months = np.arange(1, 13)
    df = (1.0 + irr / 100.0) ** (months / 12.0)
    assert -1.0 + float(np.sum(monthly_cf / df)) == pytest.approx(0.0, abs=1e-2)


def test_irr_none_when_never_economic():
    assert ec.irr_annual(np.full(12, 0.01), 1.0) is None
