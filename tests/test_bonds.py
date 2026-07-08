"""Tests for bond-income modelling: yield, cash-out income, and reinvest equivalence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf import bonds


def _prices(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def _dists(pairs):
    """pairs: list of (date_str, amount) -> distributions DataFrame indexed by ex_date."""
    idx = pd.to_datetime([d for d, _ in pairs])
    return pd.DataFrame({"amount": [a for _, a in pairs]}, index=idx)


# --------------------------------------------------------------------------- yield
def test_distribution_yield_ttm_flat_price():
    # Price flat at 100, one €3 coupon in the last year -> 3% yield.
    p = _prices([100.0] * 300)
    d = _dists([(p.index[-30].strftime("%Y-%m-%d"), 3.0)])
    assert bonds.distribution_yield_ttm(p, d) == pytest.approx(0.03)


def test_distribution_yield_excludes_old_coupons():
    p = _prices([100.0] * 400)
    old = p.index[5].strftime("%Y-%m-%d")   # > 1 year before the end
    assert bonds.distribution_yield_ttm(p, _dists([(old, 5.0)])) == 0.0


def test_yield_zero_without_distributions():
    p = _prices([100.0] * 50)
    assert bonds.distribution_yield_ttm(p, None) == 0.0


# --------------------------------------------------------------------------- cash-out income
def test_cashout_accumulates_cash_and_leaves_holding_on_price_path():
    # Flat price 100, initial €10,000 -> 100 units. One €2/share coupon -> €200 cash.
    p = _prices([100.0] * 200)
    d = _dists([(p.index[100].strftime("%Y-%m-%d"), 2.0)])
    sc = bonds.income_scenarios(p, d, initial=10_000.0)
    assert sc.units == pytest.approx(100.0)
    assert sc.total_income == pytest.approx(200.0)
    assert sc.final_cashout_value == pytest.approx(10_000.0)      # price flat
    assert sc.final_cashout_total == pytest.approx(10_200.0)      # + accumulated cash
    # Income is a cumulative step: 0 before the ex-date, 200 from it onward.
    assert sc.cashout_income.iloc[0] == 0.0
    assert sc.cashout_income.iloc[-1] == pytest.approx(200.0)


def test_multiple_coupons_sum():
    p = _prices([100.0] * 300)
    d = _dists([(p.index[50].strftime("%Y-%m-%d"), 1.5),
                (p.index[150].strftime("%Y-%m-%d"), 1.5),
                (p.index[250].strftime("%Y-%m-%d"), 1.5)])
    sc = bonds.income_scenarios(p, d, initial=10_000.0)
    assert sc.total_income == pytest.approx(100.0 * 4.5)  # 100 units * €4.5 total


# --------------------------------------------------------------------------- reinvest path
def test_reinvest_beats_cashout_when_prices_rise():
    # Rising price + a coupon: reinvesting compounds, so it must finish ahead of holding +
    # idle cash of the same coupons.
    p = _prices(list(np.linspace(100.0, 130.0, 250)))
    d = _dists([(p.index[100].strftime("%Y-%m-%d"), 3.0)])
    sc = bonds.income_scenarios(p, d, initial=10_000.0)
    assert sc.final_reinvested > sc.final_cashout_total


def test_reinvest_equals_cashout_total_when_price_flat():
    # With a flat price, reinvesting a coupon buys units at the same price, so the reinvested
    # net worth equals holding value + accumulated cash. No compounding benefit.
    p = _prices([100.0] * 200)
    d = _dists([(p.index[100].strftime("%Y-%m-%d"), 2.0)])
    sc = bonds.income_scenarios(p, d, initial=10_000.0)
    assert sc.final_reinvested == pytest.approx(sc.final_cashout_total)


def test_reinvest_from_price_matches_adj_close_path():
    # The reinvested path fed from a consistently-built adj_close must equal the path
    # reconstructed from price + distributions (they are the same total-return series).
    p = _prices(list(np.linspace(100.0, 120.0, 260)))
    d = _dists([(p.index[80].strftime("%Y-%m-%d"), 2.0),
                (p.index[200].strftime("%Y-%m-%d"), 2.5)])
    adj = bonds.reinvest_from_price(p, d, initial=1.0)  # normalised total-return series
    sc = bonds.income_scenarios(p, d, adj_close=adj, initial=10_000.0)
    reconstructed = bonds.reinvest_from_price(p, d, initial=10_000.0)
    assert np.allclose(sc.reinvested.values, reconstructed.values, rtol=1e-9)


def test_accumulating_fund_has_no_cash_income():
    # No distributions (accumulating): cash-out total tracks the price path; income is zero.
    p = _prices(list(np.linspace(100.0, 110.0, 120)))
    sc = bonds.income_scenarios(p, None, adj_close=p, initial=10_000.0)
    assert sc.total_income == 0.0
    assert sc.final_cashout_total == pytest.approx(sc.final_cashout_value)
    assert sc.dist_yield_ttm == 0.0


def test_empty_price_is_safe():
    sc = bonds.income_scenarios(pd.Series(dtype=float), None, initial=5_000.0)
    assert sc.final_reinvested == 5_000.0
    assert sc.total_income == 0.0
