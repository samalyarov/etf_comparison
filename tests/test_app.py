"""Smoke tests for the Streamlit UI: every page renders without error, in both themes
and both currency modes. These guard the whole presentation layer against regressions.

They require a populated ``data/etf.db`` (the committed dev database). If it's absent the
tests skip rather than fail, so CI without data still passes.
"""

from __future__ import annotations

import os

import pytest

from etf.config import DB_PATH

pytestmark = pytest.mark.skipif(not DB_PATH.exists(),
                                reason="needs a populated data/etf.db")

PAGES = ["Recommended", "Compare", "Screener", "Portfolio", "Detail", "Strategy", "Data"]


def _run(page: str, theme: str | None = None, currency: str | None = None):
    from streamlit.testing.v1 import AppTest
    os.environ["ETF_FORCE_PAGE"] = page
    at = AppTest.from_file("src/etf/app.py", default_timeout=120).run()
    if theme:
        at.radio[1].set_value(theme).run()   # radio[1] = Theme
    if currency:
        at.radio[0].set_value(currency).run()  # radio[0] = Currency
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = _run(page)
    assert not at.exception, f"{page} raised: {at.exception}"


@pytest.mark.parametrize("page", ["Compare", "Detail", "Strategy", "Portfolio"])
def test_page_eur_mode(page):
    at = _run(page, currency="EUR")
    assert not at.exception, f"{page} (EUR) raised: {at.exception}"


@pytest.mark.parametrize("theme", ["Light", "Dark"])
def test_theme_toggle(theme):
    at = _run("Compare", theme=theme)
    assert not at.exception, f"Compare ({theme}) raised: {at.exception}"


def test_portfolio_bond_income_toggles():
    """The Bonds/income section survives flipping reinvest-vs-cashout and the tax regime."""
    at = _run("Portfolio")
    assert not at.exception, at.exception
    at.radio(key="bond_view").set_value("Cashed out").run()
    assert not at.exception, at.exception
    at.radio(key="bond_regime").set_value("Actual return (2028)").run()
    assert not at.exception, at.exception
    # Both currency modes exercise the FX-conversion path on close + distributions.
    at.radio[0].set_value("EUR").run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("theme", ["Light", "Dark"])
@pytest.mark.parametrize("currency", ["Native", "EUR"])
def test_portfolio_bond_section_both_themes_and_currencies(theme, currency):
    """The bond-income charts/tables render in both themes and both currency modes."""
    at = _run("Portfolio", theme=theme, currency=currency)
    assert not at.exception, f"Portfolio bonds ({theme}/{currency}) raised: {at.exception}"
    at.radio(key="bond_view").set_value("Cashed out").run()
    assert not at.exception, f"Portfolio cash-out ({theme}/{currency}) raised: {at.exception}"


@pytest.mark.parametrize("theme", ["Light", "Dark"])
@pytest.mark.parametrize("currency", ["Native", "EUR"])
def test_portfolio_factor_model_both_themes_and_currencies(theme, currency):
    """The Factor-model section (sleeves, loadings, scenario fan) renders in both themes
    and both currency modes."""
    at = _run("Portfolio", theme=theme, currency=currency)
    assert not at.exception, f"Portfolio factor model ({theme}/{currency}) raised: {at.exception}"


def test_portfolio_factor_model_sleeve_and_horizon_interaction():
    """Changing the sleeve selection and the horizon re-runs the factor section cleanly."""
    at = _run("Portfolio")
    assert not at.exception, at.exception
    ms = at.multiselect(key="fm_sleeves")
    if ms.value and len(ms.value) > 1:
        ms.set_value(ms.value[:1]).run()  # drop to a single sleeve
        assert not at.exception, at.exception
    sl = at.slider(key="fm_horizon")
    sl.set_value(30).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("theme", ["Light", "Dark"])
@pytest.mark.parametrize("currency", ["Native", "EUR"])
def test_detail_profile_section_both_themes_and_currencies(theme, currency):
    """The Detail-page strategy/exposure look-through renders in both themes/currencies."""
    at = _run("Detail", theme=theme, currency=currency)
    assert not at.exception, f"Detail profile ({theme}/{currency}) raised: {at.exception}"


@pytest.mark.parametrize("theme", ["Light", "Dark"])
@pytest.mark.parametrize("currency", ["Native", "EUR"])
def test_portfolio_risk_section_both_themes_and_currencies(theme, currency):
    """The Risk section (VaR table, stress-scenario chart, contribution-to-risk) renders in
    both themes and both currency modes."""
    at = _run("Portfolio", theme=theme, currency=currency)
    assert not at.exception, f"Portfolio risk ({theme}/{currency}) raised: {at.exception}"


def test_portfolio_risk_horizon_interaction():
    """Changing the VaR horizon re-runs the risk section cleanly (√t scaling path)."""
    at = _run("Portfolio")
    assert not at.exception, at.exception
    at.selectbox(key="risk_horizon").set_value("1 month (21d)").run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("theme", ["Light", "Dark"])
@pytest.mark.parametrize("currency", ["Native", "EUR"])
def test_portfolio_optimizer_both_themes_and_currencies(theme, currency):
    """The Optimiser section (weights, frontier, exposure) renders in both themes/currencies."""
    at = _run("Portfolio", theme=theme, currency=currency)
    assert not at.exception, f"Portfolio optimizer ({theme}/{currency}) raised: {at.exception}"


def test_portfolio_optimizer_objective_and_constraints_interaction():
    """Switching to min-vol and adding a per-fund cap + sector cap re-solves cleanly."""
    at = _run("Portfolio")
    assert not at.exception, at.exception
    at.selectbox(key="opt_obj").set_value("Min volatility").run()
    assert not at.exception, at.exception
    at.number_input(key="opt_maxw").set_value(40).run()
    assert not at.exception, at.exception
    at.number_input(key="opt_sector").set_value(30).run()
    assert not at.exception, at.exception


def test_portfolio_optimizer_infeasible_is_graceful():
    """An impossible per-fund cap (10% across few funds) shows a message, never crashes."""
    at = _run("Portfolio")
    assert not at.exception, at.exception
    at.number_input(key="opt_maxw").set_value(10).run()
    assert not at.exception, at.exception  # infeasible handled, no exception leaks


def test_portfolio_optimizer_shorting_toggle():
    """Enabling shorting (gross cap) re-solves without error."""
    at = _run("Portfolio")
    assert not at.exception, at.exception
    at.selectbox(key="opt_lev").set_value("Allow shorting (gross cap)").run()
    assert not at.exception, at.exception


def test_data_page_profile_coverage():
    """The Data-page profile-coverage indicator renders."""
    at = _run("Data")
    assert not at.exception, at.exception


def teardown_module(module):
    os.environ.pop("ETF_FORCE_PAGE", None)
