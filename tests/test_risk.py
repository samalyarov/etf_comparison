"""Tests for the portfolio risk engine: VaR/CVaR (three methods), component-risk
decomposition, √t horizon scaling, and historical crash-window stress tests."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from etf import risk

_N = NormalDist()


def _normal_returns(n=8000, mu=0.0004, sigma=0.011, seed=7) -> pd.Series:
    """A long synthetic daily-return series drawn from a Normal (for closed-form checks)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(rng.normal(mu, sigma, size=n), index=idx, name="r")


# --------------------------------------------------------------------------- VaR methods agree

def test_parametric_var_matches_closed_form_gaussian():
    r = _normal_returns()
    m, s = float(r.mean()), float(r.std(ddof=1))
    for c in (0.95, 0.99):
        z = _N.inv_cdf(c)
        res = risk.parametric_var(r, c)
        assert res.var == pytest.approx(s * z - m, rel=1e-9)
        # Closed-form normal Expected Shortfall.
        assert res.cvar == pytest.approx(s * _N.pdf(z) / (1 - c) - m, rel=1e-9)
        assert res.cvar > res.var  # CVaR strictly exceeds VaR


def test_historical_var_equals_empirical_quantile():
    r = _normal_returns()
    for c in (0.95, 0.99):
        res = risk.historical_var(r, c)
        assert res.var == pytest.approx(-float(np.quantile(r.to_numpy(), 1 - c)), rel=1e-12)
        # CVaR is minus the mean of the sub-quantile tail, and dominates VaR.
        q = np.quantile(r.to_numpy(), 1 - c)
        assert res.cvar == pytest.approx(-float(r.to_numpy()[r.to_numpy() <= q].mean()), rel=1e-9)
        assert res.cvar >= res.var


def test_three_methods_agree_on_normal_within_tolerance():
    r = _normal_returns(n=20000)
    for c in (0.95, 0.99):
        hist = risk.historical_var(r, c).var
        para = risk.parametric_var(r, c).var
        mc = risk.monte_carlo_var(r, c, method="normal", n_sims=100000, seed=1).var
        assert hist == pytest.approx(para, rel=0.06)
        assert mc == pytest.approx(para, rel=0.06)


def test_monte_carlo_is_deterministic_under_seed():
    r = _normal_returns()
    a = risk.monte_carlo_var(r, 0.95, method="bootstrap", n_sims=20000, seed=42)
    b = risk.monte_carlo_var(r, 0.95, method="bootstrap", n_sims=20000, seed=42)
    c = risk.monte_carlo_var(r, 0.95, method="bootstrap", n_sims=20000, seed=43)
    assert a.var == b.var and a.cvar == b.cvar
    assert a.var != c.var  # a different seed gives a different draw


# --------------------------------------------------------------------------- Cornish-Fisher

def test_cornish_fisher_quantile_reduces_to_gaussian_at_zero_moments():
    for c in (0.90, 0.95, 0.99):
        za = _N.inv_cdf(1 - c)
        assert risk._cornish_fisher_quantile(za, 0.0, 0.0) == pytest.approx(za, rel=1e-12)


def test_cornish_fisher_var_reduces_to_gaussian_at_zero_skew_and_kurtosis():
    # At zero skew and zero excess kurtosis the Cornish-Fisher VaR *and* CVaR must equal the
    # Gaussian closed forms exactly (quantile collapses to z; Boudt ES bracket to 1).
    for c in (0.95, 0.99):
        for h in (1, 5):
            g_var, g_cvar = risk._parametric_var_cvar(0.001, 0.02, c, h, 0.0, 0.0, False)
            cf_var, cf_cvar = risk._parametric_var_cvar(0.001, 0.02, c, h, 0.0, 0.0, True)
            assert cf_var == pytest.approx(g_var, rel=1e-12)
            assert cf_cvar == pytest.approx(g_cvar, rel=1e-12)


def test_cornish_fisher_returns_nan_outside_domain_of_validity():
    # Extreme excess kurtosis pushes the Cornish-Fisher expansion out of its monotone domain;
    # a money tool must return NaN there, never a fabricated (or negative) VaR/CVaR.
    assert risk._cornish_fisher_is_valid(0.0, 0.0)          # Gaussian is valid
    assert not risk._cornish_fisher_is_valid(0.0, 50.0)     # huge kurtosis → invalid
    var, cvar = risk._parametric_var_cvar(0.0, 0.02, 0.99, 1, -1.0, 50.0, cornish_fisher=True)
    assert np.isnan(var) and np.isnan(cvar)
    # Whenever a Cornish-Fisher CVaR *is* finite it must dominate the VaR (coherence).
    v, cv = risk._parametric_var_cvar(0.0, 0.02, 0.95, 1, -0.3, 1.0, cornish_fisher=True)
    if np.isfinite(cv):
        assert cv >= v


def test_cornish_fisher_widens_var_for_left_skewed_fat_tails():
    # Left-skewed, fat-tailed returns: CF VaR should exceed the Gaussian VaR (fatter left tail).
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 0.01, size=5000)
    base[::50] -= 0.08  # inject periodic crashes → left skew + excess kurtosis
    r = pd.Series(base, index=pd.bdate_range("2000-01-03", periods=len(base)))
    gauss = risk.parametric_var(r, 0.99)
    cf = risk.parametric_var(r, 0.99, cornish_fisher=True)
    assert cf.skew < 0 and cf.excess_kurtosis > 0
    assert cf.var > gauss.var


# --------------------------------------------------------------------------- horizon √t scaling

def test_sqrt_time_scaling_is_correct():
    # On a demeaned series (μ=0) the horizon VaR must equal √h × the per-period VaR for every
    # method (Basel square-root-of-time rule).
    r = _normal_returns()
    r = r - r.mean()  # force zero drift so scaling is pure √h
    for method_call in (
        lambda h: risk.parametric_var(r, 0.95, horizon=h),
        lambda h: risk.parametric_var(r, 0.95, horizon=h, cornish_fisher=True),
        lambda h: risk.historical_var(r, 0.95, horizon=h),
        lambda h: risk.monte_carlo_var(r, 0.95, horizon=h, n_sims=40000, seed=5),
    ):
        base = method_call(1)
        for h in (4, 9):
            scaled = method_call(h)
            assert scaled.var == pytest.approx(base.var * np.sqrt(h), rel=1e-9)
            assert scaled.cvar == pytest.approx(base.cvar * np.sqrt(h), rel=1e-9)


# --------------------------------------------------------------------------- component risk

def _two_asset_prices(seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    a = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, size=len(idx))))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.020, size=len(idx))))  # riskier
    return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)


def test_component_vol_sums_to_portfolio_vol():
    prices = _two_asset_prices()
    weights = {"AAA": 0.6, "BBB": 0.4}
    cr = risk.component_risk(prices, weights, confidence=0.95)
    assert cr is not None
    # Component vols sum to the portfolio vol, which equals the target-weighted series' stdev.
    r_p, _R, _w, _cols = risk.portfolio_returns(prices, weights)
    assert cr.component_vol.sum() == pytest.approx(cr.portfolio_vol, rel=1e-9)
    assert cr.portfolio_vol == pytest.approx(float(r_p.std(ddof=1)), rel=1e-9)
    assert cr.pct_contribution.sum() == pytest.approx(1.0, rel=1e-9)


def test_component_var_sums_to_parametric_var():
    prices = _two_asset_prices()
    weights = {"AAA": 0.5, "BBB": 0.5}
    cr = risk.component_risk(prices, weights, confidence=0.99)
    z = _N.inv_cdf(0.99)
    assert cr.component_var.sum() == pytest.approx(cr.portfolio_var, rel=1e-9)
    assert cr.portfolio_var == pytest.approx(z * cr.portfolio_vol, rel=1e-9)
    # The riskier, higher-vol fund carries the larger share of risk at equal weight.
    share = dict(zip(cr.assets, cr.pct_contribution))
    assert share["BBB"] > share["AAA"]


def test_single_fund_portfolio_is_graceful():
    prices = _two_asset_prices()
    cr = risk.component_risk(prices[["AAA"]], {"AAA": 1.0})
    assert cr is not None
    assert cr.pct_contribution.sum() == pytest.approx(1.0)
    # VaR methods survive a single-fund series too.
    r_p, _R, _w, _cols = risk.portfolio_returns(prices[["AAA"]], {"AAA": 1.0})
    assert np.isfinite(risk.parametric_var(r_p, 0.95).var)


# --------------------------------------------------------------------------- stress tests

def _crash_prices():
    """A crafted price path: calm, then a known -40% crash inside the COVID window, recovery."""
    dates = pd.bdate_range("2019-06-03", "2020-12-31")
    s = pd.Series(100.0, index=dates)
    peak = pd.Timestamp("2020-02-19")
    trough = pd.Timestamp("2020-03-23")
    # Flat at 100 to the peak, linear crash to 60 at the trough, linear recovery to 110.
    for d in dates:
        if d <= peak:
            s[d] = 100.0
        elif d <= trough:
            frac = (d - peak).days / (trough - peak).days
            s[d] = 100.0 * (1 - 0.40 * frac)
        else:
            frac = min(1.0, (d - trough).days / (pd.Timestamp("2020-08-01") - trough).days)
            s[d] = 60.0 + (110.0 - 60.0) * frac
    return pd.DataFrame({"XXX": s})


def test_stress_window_selects_dates_and_drawdown():
    prices = _crash_prices()
    res = risk.stress_test(prices, {"XXX": 1.0}, "COVID crash 2020",
                           risk.CRASH_WINDOWS["COVID crash 2020"])
    assert res.covered
    assert res.start.isoformat() == "2020-02-19"
    assert res.end.isoformat() == "2020-03-23"
    # A single-fund blend replays the fund's own -40% peak-to-trough drawdown.
    assert res.drawdown == pytest.approx(-0.40, abs=1e-6)
    assert res.trough_date.isoformat() == "2020-03-23"
    assert res.worst_day < 0
    # Price recovers to 110 (above the 100 peak) by August → a finite recovery time.
    assert res.recovery_days is not None and res.recovery_days > 0


def test_stress_skips_window_without_coverage():
    # This fund only starts in 2019 → it cannot cover the 2008 GFC window.
    prices = _crash_prices()
    res = risk.stress_test(prices, {"XXX": 1.0}, "GFC 2008", risk.CRASH_WINDOWS["GFC 2008"])
    assert not res.covered
    assert np.isnan(res.drawdown)


def test_stress_tests_batch_reports_all_windows():
    prices = _crash_prices()
    results = risk.stress_tests(prices, {"XXX": 1.0})
    labels = {r.label for r in results}
    assert labels == set(risk.CRASH_WINDOWS)
    covered = [r for r in results if r.covered]
    # Only the COVID window is fully inside the crafted 2019-2020 history.
    assert any(r.label == "COVID crash 2020" for r in covered)


def test_no_lookahead_in_recovery_not_yet_recovered():
    # A path that crashes and never regains its peak within the data → recovery_days is None.
    dates = pd.bdate_range("2019-06-03", "2020-04-30")
    s = pd.Series(100.0, index=dates)
    peak, trough = pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")
    for d in dates:
        if d <= peak:
            s[d] = 100.0
        elif d <= trough:
            s[d] = 100.0 * (1 - 0.40 * (d - peak).days / (trough - peak).days)
        else:
            s[d] = 62.0  # stays depressed — never recovers before the data ends
    res = risk.stress_test(pd.DataFrame({"XXX": s}), {"XXX": 1.0}, "COVID crash 2020",
                           risk.CRASH_WINDOWS["COVID crash 2020"])
    assert res.covered
    assert res.recovery_days is None


# --------------------------------------------------------------------------- summary table

def test_var_summary_shape_and_ordering():
    prices = _two_asset_prices()
    summ = risk.var_summary(prices, {"AAA": 0.6, "BBB": 0.4}, n_sims=20000, seed=9)
    assert list(summ.index) == [risk.METHOD_HISTORICAL, risk.METHOD_PARAMETRIC,
                                risk.METHOD_CORNISH_FISHER, risk.METHOD_MONTE_CARLO]
    assert list(summ.columns) == ["VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%"]
    # CVaR ≥ VaR and 99% ≥ 95% in every row.
    for _, row in summ.iterrows():
        assert row["CVaR 95%"] >= row["VaR 95%"]
        assert row["CVaR 99%"] >= row["VaR 99%"]
        assert row["VaR 99%"] >= row["VaR 95%"]
