"""Tests for the DCA backtest and XIRR, using series with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf import strategy


def _daily(values, start="2020-01-01"):
    idx = pd.date_range(start=start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_flat_price_no_profit():
    # Two years of flat prices: end value equals total invested, zero profit.
    s = _daily([100.0] * 730)
    r = strategy.simulate_dca(s, monthly=100.0)
    assert r.n_contributions == 24
    assert r.total_invested == pytest.approx(2400.0)
    assert r.final_value == pytest.approx(2400.0, rel=1e-6)
    assert r.profit == pytest.approx(0.0, abs=1e-6)
    assert r.money_multiple == pytest.approx(1.0, rel=1e-6)
    assert abs(r.xirr) < 0.01  # ~0% money-weighted return on flat prices


def test_rising_price_makes_profit():
    s = _daily(np.linspace(100, 200, 730))  # doubles over ~2 years
    r = strategy.simulate_dca(s, monthly=100.0)
    assert r.final_value > r.total_invested
    assert r.profit > 0
    assert r.money_multiple > 1.0
    assert r.xirr > 0


def test_initial_lump_sum_counted():
    s = _daily([100.0] * 400)
    r = strategy.simulate_dca(s, monthly=100.0, initial=1000.0)
    # 13 monthly buys (~400 days) * 100 + 1000 lump
    assert r.total_invested == pytest.approx(r.n_contributions * 100 + 1000)
    assert r.final_value == pytest.approx(r.total_invested, rel=1e-6)  # flat price


def test_step_up_increases_contributions():
    s = _daily([100.0] * 800)
    flat = strategy.simulate_dca(s, monthly=100.0)
    stepped = strategy.simulate_dca(s, monthly=100.0, annual_step_up=0.10)
    assert stepped.total_invested > flat.total_invested


def test_timeline_aligned_to_prices():
    s = _daily(np.linspace(100, 150, 500))
    r = strategy.simulate_dca(s, monthly=50.0)
    assert list(r.timeline.index) == list(s.index)
    assert (r.timeline["value"] >= 0).all()
    # invested is non-decreasing
    assert (r.timeline["invested"].diff().dropna() >= -1e-9).all()


def test_xirr_matches_known_lump_sum():
    # Single lump sum that doubles in exactly one year -> XIRR ~ 100%.
    dates = pd.Series({pd.Timestamp("2020-01-01"): 1000.0})
    rate = strategy._xirr(dates, final_value=2000.0, final_date=pd.Timestamp("2021-01-01"))
    assert rate == pytest.approx(1.0, abs=0.02)


def test_requires_positive_contribution():
    s = _daily([100.0] * 100)
    with pytest.raises(ValueError):
        strategy.simulate_dca(s, monthly=0.0)
