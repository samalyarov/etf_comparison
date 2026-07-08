"""Tests for the Dutch box-3 tax model (current wealth tax + actual-return regime).

Arithmetic is checked against hand-computed examples so the money figures are auditable.
"""

from __future__ import annotations

import pytest

from etf import tax


# --------------------------------------------------------------------------- Box 3 (2026)
def test_box3_tax_hand_computed():
    # €100,000 invested. Base = 100,000 - 59,357 = 40,643.
    # Deemed return = 40,643 * 6.00% = 2,438.58. Tax = 2,438.58 * 36% = 877.8888.
    r = tax.box3_tax(100_000)
    assert r.taxable_base == pytest.approx(40_643.0)
    assert r.deemed_return == pytest.approx(2_438.58)
    assert r.tax == pytest.approx(877.8888)


def test_box3_below_allowance_is_untaxed():
    r = tax.box3_tax(50_000)
    assert r.taxable_base == 0.0
    assert r.tax == 0.0


def test_box3_partners_double_the_allowance():
    solo = tax.box3_tax(150_000)
    partners = tax.box3_tax(150_000, partners=True)
    assert partners.allowance == pytest.approx(2 * solo.allowance)
    assert partners.tax < solo.tax  # larger tax-free band → less tax


def test_box3_is_independent_of_income_received():
    # The whole point of the wealth tax: two funds of equal value owe the same box-3 tax
    # regardless of whether they distribute or accumulate.
    reinvested_value = 120_000
    cashed_out_value = 120_000  # same market value, coupons taken as cash but re-counted
    assert tax.box3_tax(reinvested_value).tax == tax.box3_tax(cashed_out_value).tax


def test_effective_wealth_tax_rate():
    # 6.00% forfait * 36% rate = 2.16% of the taxable asset base per year.
    assert tax.effective_wealth_tax_rate() == pytest.approx(0.0216)


# --------------------------------------------------------------------------- Actual return
def test_actual_return_tax_hand_computed():
    # €5,000 actual return. Taxable = 5,000 - 1,800 = 3,200. Tax = 3,200 * 36% = 1,152.
    r = tax.actual_return_tax(5_000)
    assert r.taxable_return == pytest.approx(3_200.0)
    assert r.tax == pytest.approx(1_152.0)


def test_actual_return_below_allowance_is_untaxed():
    assert tax.actual_return_tax(1_000).tax == 0.0


def test_actual_return_negative_yields_no_tax():
    r = tax.actual_return_tax(-2_000)
    assert r.taxable_return == 0.0
    assert r.tax == 0.0


def test_actual_return_partners_double_allowance():
    solo = tax.actual_return_tax(5_000)
    partners = tax.actual_return_tax(5_000, partners=True)
    assert partners.allowance == pytest.approx(2 * solo.allowance)
    assert partners.tax < solo.tax


def test_regimes_differ_for_a_high_yield_small_holding():
    # A €30,000 holding below the box-3 allowance owes zero wealth tax, but a large actual
    # return above €1,800 is taxed under the actual-return regime — the regimes genuinely
    # diverge, which is the decision the UI surfaces.
    assert tax.box3_tax(30_000).tax == 0.0
    assert tax.actual_return_tax(4_000).tax > 0.0
