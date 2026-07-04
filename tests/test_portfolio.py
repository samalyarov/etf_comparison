"""Tests for portfolio blends and rebalancing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf import metrics, portfolio


def _matrix(seed=0, n=800):
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    a = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    b = 100 * np.cumprod(1 + rng.normal(0.0002, 0.006, n))
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_blend_index_between_components():
    P = _matrix()
    blend = portfolio.blend_index(P, {"A": 0.5, "B": 0.5}, rebalance="Q")
    # Blend total return should sit between the two components' returns.
    ra = metrics.total_return(P["A"])
    rb = metrics.total_return(P["B"])
    rblend = metrics.total_return(blend)
    assert min(ra, rb) - 0.02 <= rblend <= max(ra, rb) + 0.02


def test_full_weight_on_one_matches_that_fund():
    P = _matrix()
    blend = portfolio.blend_index(P, {"A": 1.0, "B": 0.0}, rebalance="Q")
    assert metrics.total_return(blend) == np.float64(metrics.total_return(P["A"])).astype(float) \
        or abs(metrics.total_return(blend) - metrics.total_return(P["A"])) < 1e-6


def test_weights_normalised():
    P = _matrix()
    a = portfolio.blend_index(P, {"A": 1, "B": 1}, rebalance="Q")
    b = portfolio.blend_index(P, {"A": 50, "B": 50}, rebalance="Q")
    assert np.allclose(a.values, b.values)


def test_rebalance_comparison_returns_both():
    P = _matrix()
    cmp = portfolio.rebalance_comparison(P, {"A": 0.6, "B": 0.4}, rebalance="Q")
    assert "rebalanced_final" in cmp and "drift_final" in cmp
    assert not cmp["rebalanced"].empty and not cmp["drift"].empty


def test_drift_weights_sum_to_one():
    P = _matrix()
    w = portfolio.blend_weights_drift(P, {"A": 0.5, "B": 0.5})
    assert abs(w.sum() - 1.0) < 1e-9


def test_suggest_low_correlation_picks_requested_count():
    idx = pd.date_range("2018-01-01", periods=500, freq="B")
    rng = np.random.default_rng(3)
    base = rng.normal(0.0003, 0.01, 500)
    cols = {}
    for i in range(5):
        noise = rng.normal(0, 0.008, 500)
        cols[f"F{i}"] = 100 * np.cumprod(1 + base * (i == 0) + noise)
    P = pd.DataFrame(cols, index=idx)
    picks = portfolio.suggest_low_correlation(P, n=3)
    assert len(picks) == 3
    assert len(set(picks)) == 3


def test_empty_weights_returns_empty():
    P = _matrix()
    assert portfolio.blend_index(P, {}).empty
    assert portfolio.blend_index(P, {"A": 0, "B": 0}).empty


# --- Manual import + contribution-only rebalancing ---

def test_parse_positions_csv_and_whitespace():
    txt = "ticker,units\nVWCE.DE, 10\nCSPX.L 5\nIWDA.AS,2.5\n"
    pos = portfolio.parse_positions(txt)
    assert pos["VWCE.DE"] == 10 and pos["CSPX.L"] == 5 and pos["IWDA.AS"] == 2.5


def test_parse_positions_ignores_junk_and_currency():
    pos = portfolio.parse_positions("Symbol,Value\nVWCE.DE,€1000\nbadrow\nCSPX.L,$500")
    assert pos["VWCE.DE"] == 1000 and pos["CSPX.L"] == 500


def test_contribution_rebalance_targets_underweight():
    current = {"A": 8000, "B": 2000}   # 80/20, target 50/50 -> B underweight
    buys = portfolio.contribution_rebalance(current, {"A": 0.5, "B": 0.5}, 2000)
    assert buys["B"] > buys["A"]       # more of the new money goes to the laggard
    assert abs(sum(buys.values()) - 2000) < 1e-6


def test_contribution_rebalance_balanced_splits_by_weight():
    current = {"A": 5000, "B": 5000}   # already at 50/50
    buys = portfolio.contribution_rebalance(current, {"A": 0.5, "B": 0.5}, 1000)
    assert abs(buys["A"] - 500) < 1e-6 and abs(buys["B"] - 500) < 1e-6


def test_contribution_rebalance_sums_to_contribution():
    current = {"A": 1000, "B": 0, "C": 3000}
    buys = portfolio.contribution_rebalance(current, {"A": 0.3, "B": 0.3, "C": 0.4}, 5000)
    assert abs(sum(buys.values()) - 5000) < 1e-6
    assert all(v >= -1e-9 for v in buys.values())
