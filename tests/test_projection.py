"""Tests for forward net-worth projections."""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf import projection


def _price(annual_growth=0.08, years=8, vol=0.0, seed=0):
    n = years * 252
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    daily = (1 + annual_growth) ** (1 / 252) - 1
    if vol:
        rng = np.random.default_rng(seed)
        rets = rng.normal(daily, vol, n)
    else:
        rets = np.full(n, daily)
    return pd.Series(100 * np.cumprod(1 + rets), index=idx)


def test_ols_drift_recovers_growth():
    s = _price(annual_growth=0.10, years=10, vol=0.0)
    drift = projection.ols_annual_log_drift(s)
    assert drift == np.float64(drift)
    assert abs(drift - 0.10) < 0.01


def test_ols_projection_grows_and_hits_contributions():
    s = _price(annual_growth=0.06, years=6)
    res = projection.project_plan(s, start_value=1000, monthly=100, years=10, method="ols")
    assert res.horizon_years == 10
    assert len(res.fan) == 120
    # Median path exceeds pure contributions when returns are positive.
    assert res.fan["p50"].iloc[-1] > res.fan["invested"].iloc[-1]
    # Total contributed = start + 120 * 100.
    assert res.total_contributed == 1000 + 120 * 100


def test_horizon_capped_at_40_years():
    s = _price()
    res = projection.project_plan(s, start_value=0, monthly=100, years=100, method="ols")
    assert res.horizon_years == 40
    assert len(res.fan) == 40 * 12


def test_monte_carlo_fan_is_ordered():
    s = _price(annual_growth=0.08, years=10, vol=0.01, seed=1)
    res = projection.project_plan(s, start_value=500, monthly=200, years=20,
                                  method="bootstrap", n_sims=500)
    last = res.fan.iloc[-1]
    assert last["p5"] <= last["p25"] <= last["p50"] <= last["p75"] <= last["p95"]
    # Spread must be non-trivial with real volatility.
    assert last["p95"] > last["p5"]


def test_annual_return_override_lifts_median():
    s = _price(annual_growth=0.04, years=10, vol=0.01)
    low = projection.project_plan(s, start_value=0, monthly=100, years=15,
                                  method="bootstrap", annual_return=0.02, n_sims=400)
    high = projection.project_plan(s, start_value=0, monthly=100, years=15,
                                   method="bootstrap", annual_return=0.10, n_sims=400)
    assert high.fan["p50"].iloc[-1] > low.fan["p50"].iloc[-1]


def test_projected_dates_match_horizon():
    s = _price()
    res = projection.project_plan(s, start_value=0, monthly=100, years=5, method="normal",
                                  n_sims=200)
    assert len(res.dates) == 60
    assert res.dates[0] > s.index[-1]
