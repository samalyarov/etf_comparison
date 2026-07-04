"""Tests for the cost & tax model and cost-aware DCA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf import costs, strategy


def test_commission_minimum_applies_to_small_trades():
    m = costs.CommissionModel()
    # A small €500 buy: 0.035% = €0.175, below the €1.25 min -> min applies.
    assert m.trade_commission(500, "EUR") == pytest.approx(1.25)
    # A large €100k buy: 0.035% = €35, above the min.
    assert m.trade_commission(100_000, "EUR") == pytest.approx(35.0)


def test_fx_cost_only_when_currencies_differ():
    m = costs.CommissionModel()
    assert m.buy_cost(10_000, "EUR", "EUR") == m.trade_commission(10_000, "EUR")
    with_fx = m.buy_cost(10_000, "USD", "EUR")
    assert with_fx > m.trade_commission(10_000, "USD")  # extra FX conversion cost


def test_tracking_difference_physical_vs_synthetic():
    phys = costs.estimate_tracking_difference(0.002, "physical")
    synth = costs.estimate_tracking_difference(0.002, "synthetic")
    assert phys < 0.002 < synth
    assert costs.estimate_tracking_difference(None) == 0.0


def test_tco_totals_and_amortisation():
    tco = costs.total_cost_of_ownership(0.002, spread=0.001, fx_bps=0.0002,
                                        holding_years=10, replication="physical")
    assert tco["total_annual"] > tco["tracking_difference"]  # spread+fx add
    # A longer horizon amortises the one-off spread/fx to a smaller annual figure.
    long = costs.total_cost_of_ownership(0.002, spread=0.001, holding_years=40,
                                         replication="physical")
    short = costs.total_cost_of_ownership(0.002, spread=0.001, holding_years=5,
                                          replication="physical")
    assert long["total_annual"] < short["total_annual"]


def test_tax_drag_acc_vs_dist():
    # Distributing: taxed each year; accumulating: deferred (0 drag) unless deemed.
    dist = costs.tax_drag(0.03, "DIST", dividend_tax=0.26)
    assert dist == pytest.approx(0.03 * 0.26)
    assert costs.tax_drag(0.03, "ACC", dividend_tax=0.26) == 0.0
    assert costs.tax_drag(0.03, "ACC", dividend_tax=0.26, deemed_distribution=True) > 0


def test_domicile_note_mentions_treaty():
    assert "15%" in costs.domicile_note("IE")
    assert costs.domicile_note(None)


def test_dca_with_commission_reduces_value():
    s = pd.Series(np.linspace(100, 200, 730),
                  index=pd.date_range("2020-01-01", periods=730, freq="D"))
    gross = strategy.simulate_dca(s, monthly=100.0)
    net = strategy.simulate_dca(s, monthly=100.0, commission=costs.DEFAULT_COMMISSION,
                                currency="USD", account_currency="EUR")
    assert net.total_costs > 0
    assert net.final_value < gross.final_value
    assert net.xirr < gross.xirr  # costs drag the money-weighted return down
