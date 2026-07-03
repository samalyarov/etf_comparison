"""Unit tests for the analytics layer, using synthetic series with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf import metrics


def _series(values, start="2020-01-01", freq="D"):
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def test_total_return_simple():
    s = _series([100, 110])
    assert metrics.total_return(s) == pytest.approx(0.10)


def test_total_return_needs_two_points():
    assert np.isnan(metrics.total_return(_series([100])))


def test_cagr_doubles_in_one_year():
    # Exactly 365.25 days apart, price doubles -> CAGR ~= 100%.
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")])
    # 365 days; use a 2x over ~1 year and check it's close to 1.0
    s = pd.Series([100.0, 200.0], index=idx)
    assert metrics.cagr(s) == pytest.approx(1.0, abs=0.02)


def test_max_drawdown():
    # Peak 100 -> trough 50 = -50% drawdown, then partial recovery.
    s = _series([100, 120, 60, 80, 120])
    # running max hits 120, trough 60 -> dd = 60/120 - 1 = -0.5
    assert metrics.max_drawdown(s) == pytest.approx(-0.5)


def test_drawdown_series_non_positive():
    s = _series([100, 90, 120, 110])
    dd = metrics.drawdown_series(s)
    assert (dd <= 1e-12).all()
    assert dd.iloc[0] == pytest.approx(0.0)


def test_normalize_to_100():
    s = _series([50, 75, 100])
    n = metrics.normalize_to_100(s)
    assert n.iloc[0] == pytest.approx(100.0)
    assert n.iloc[-1] == pytest.approx(200.0)


def test_volatility_of_constant_is_zero():
    s = _series([100] * 30)
    assert metrics.annualized_volatility(s) == pytest.approx(0.0)


def test_volatility_positive_for_varying():
    rng = np.random.default_rng(42)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    s = _series(prices)
    vol = metrics.annualized_volatility(s)
    assert vol > 0
    # ~1% daily sd annualises to roughly 0.16; wide tolerance for randomness.
    assert 0.10 < vol < 0.25


def test_period_return_1y():
    # Daily series over ~2 years; 1Y return should use price ~365 days back.
    idx = pd.date_range("2022-01-01", periods=800, freq="D")
    s = pd.Series(np.linspace(100, 260, 800), index=idx)
    r = metrics.period_return(s, "1Y")
    assert np.isfinite(r)
    assert r > 0


def test_period_return_insufficient_history_is_nan():
    s = _series([100, 101, 102], start="2024-01-01")  # only 3 days
    assert np.isnan(metrics.period_return(s, "5Y"))


def test_period_return_ytd():
    idx = pd.date_range("2023-11-01", "2024-03-01", freq="D")
    s = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
    r = metrics.period_return(s, "YTD", asof="2024-03-01")
    # Start value ~ price at 2024-01-01, end ~150.
    start_val = s.asof(pd.Timestamp("2024-01-01"))
    expected = s.asof(pd.Timestamp("2024-03-01")) / start_val - 1
    assert r == pytest.approx(expected)


def test_period_returns_returns_all_labels():
    idx = pd.date_range("2015-01-01", periods=2000, freq="D")
    s = pd.Series(np.linspace(100, 300, 2000), index=idx)
    out = metrics.period_returns(s)
    assert set(out.keys()) == set(metrics.STANDARD_PERIODS.keys())


def test_sharpe_zero_rf_positive_trend():
    rng = np.random.default_rng(1)
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 500))
    s = _series(prices)
    sharpe = metrics.sharpe_ratio(s, risk_free=0.0)
    assert np.isfinite(sharpe)


def test_correlation_matrix_perfectly_correlated():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    base = pd.Series(np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 100)), index=idx)
    matrix = pd.DataFrame({"A": base * 10, "B": base * 20})  # identical returns
    corr = metrics.correlation_matrix(matrix)
    assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-6)


def test_summary_after_ter():
    idx = pd.date_range("2020-01-01", periods=500, freq="D")
    s = pd.Series(np.linspace(100, 200, 500), index=idx)
    out = metrics.summary(s, ter=0.002)
    assert out["cagr_after_ter"] == pytest.approx(out["cagr"] - 0.002)
