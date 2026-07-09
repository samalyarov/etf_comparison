"""Tests for the constrained portfolio optimiser (:mod:`etf.optimizer`).

Coverage:
- a 2-asset tangency portfolio matches the closed-form ``Σ⁻¹(μ−rf) / 1ᵀΣ⁻¹(μ−rf)``;
- **every** constraint is provably respected in the solution (long-only fully-invested,
  per-asset bounds, gross leverage with shorting, turnover L1, exposure caps/floors);
- an infeasible constraint set returns a clean ``success=False`` status with no exception;
- exposure coverage is reported, and is < 1 when some funds lack look-through data;
- the efficient frontier is monotone and the solver is deterministic across runs.

The constraint-satisfaction tests build a synthetic price panel so they need no DB; the
profile-driven exposure tests use a tiny in-test profiles YAML so coverage gaps are explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf import optimizer as opt

TOL = 1e-4


# --------------------------------------------------------------------------- fixtures
def _panel(seed: int = 7, n: int = 900, mus=(0.09, 0.11, 0.05, 0.13),
           vols=(0.13, 0.19, 0.07, 0.23)) -> pd.DataFrame:
    """A deterministic geometric-BM price panel with known drift/vol per column."""
    rng = np.random.default_rng(seed)
    cols = [f"F{i}" for i in range(len(mus))]
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    mu = np.array(mus) / opt.TRADING_DAYS
    sig = np.array(vols) / np.sqrt(opt.TRADING_DAYS)
    rets = rng.normal(mu, sig, size=(n, len(mus)))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=cols)


def _w(res: opt.OptimizeResult, assets: list[str]) -> np.ndarray:
    return np.array([res.weights.get(a, 0.0) for a in assets])


# --------------------------------------------------------------------------- analytical
def test_two_asset_tangency_matches_closed_form():
    mu = np.array([0.10, 0.15])
    sig = np.array([0.15, 0.25])
    rho = 0.2
    rf = 0.03
    S = np.array([[sig[0] ** 2, rho * sig[0] * sig[1]],
                  [rho * sig[0] * sig[1], sig[1] ** 2]])
    e = mu - rf
    raw = np.linalg.solve(S, e)
    w_cf = raw / raw.sum()
    status, w = opt._solve(mu, S, ["X", "Y"], "max_sharpe", rf, opt.OptConstraints(),
                           None, None)
    assert status == "optimal"
    assert np.max(np.abs(w - w_cf)) < 1e-6


def test_max_sharpe_beats_min_vol_on_sharpe():
    p = _panel()
    ms = opt.optimize_portfolio(p, objective="max_sharpe", risk_free_rate=0.02)
    mv = opt.optimize_portfolio(p, objective="min_volatility", risk_free_rate=0.02)
    assert ms.success and mv.success
    # Tangency maximises Sharpe; min-vol minimises variance.
    assert ms.sharpe >= mv.sharpe - 1e-9
    assert mv.volatility <= ms.volatility + 1e-9


# --------------------------------------------------------------------------- constraints
def test_long_only_fully_invested():
    p = _panel()
    r = opt.optimize_portfolio(p, objective="max_sharpe")
    w = _w(r, list(p.columns))
    assert r.success
    assert (w >= -TOL).all()
    assert abs(w.sum() - 1.0) < TOL


def test_max_weight_bound_respected():
    p = _panel()
    r = opt.optimize_portfolio(p, constraints=opt.OptConstraints(max_weight=0.35))
    w = _w(r, list(p.columns))
    assert r.success
    assert w.max() <= 0.35 + TOL
    assert any("max weight" in b for b in r.binding)


def test_min_weight_floor_respected():
    p = _panel()
    r = opt.optimize_portfolio(p, constraints=opt.OptConstraints(min_weight=0.1))
    w = _w(r, list(p.columns))
    assert r.success
    assert (w >= 0.1 - TOL).all()


def test_gross_leverage_cap_with_shorting():
    p = _panel()
    c = opt.OptConstraints(long_only=False, gross_leverage=1.6, min_weight=-0.6)
    r = opt.optimize_portfolio(p, constraints=c)
    w = _w(r, list(p.columns))
    assert r.success
    assert np.abs(w).sum() <= 1.6 + TOL   # gross-exposure cap
    assert abs(w.sum() - 1.0) < TOL       # still net fully-invested


def test_turnover_limit_respected_and_binding():
    p = _panel()
    current = {c: 0.25 for c in p.columns}
    c = opt.OptConstraints(turnover_limit=0.30)
    r = opt.optimize_portfolio(p, constraints=c, current_weights=current)
    w = _w(r, list(p.columns))
    prev = np.array([0.25] * 4)
    assert r.success
    assert np.abs(w - prev).sum() <= 0.30 + TOL
    # The unconstrained tangency differs enough that turnover actually bites here.
    assert any("turnover" in b for b in r.binding)


def test_turnover_ignored_without_current_weights():
    p = _panel()
    c = opt.OptConstraints(turnover_limit=0.05)
    r = opt.optimize_portfolio(p, constraints=c, current_weights=None)
    # With no anchor the turnover limit is a no-op, not an error.
    assert r.success
    assert not any("turnover" in b for b in r.binding)


def test_l2_regularisation_spreads_weights():
    p = _panel()
    base = opt.optimize_portfolio(p, objective="min_volatility")
    reg = opt.optimize_portfolio(p, objective="min_volatility",
                                 constraints=opt.OptConstraints(l2_gamma=1.0))
    assert base.success and reg.success
    # Ridge on weights (Σ ← Σ+γI) pulls toward equal weight → lower concentration (HHI).
    def hhi(r):
        return sum(v ** 2 for v in r.weights.values())
    assert hhi(reg) <= hhi(base) + 1e-9


# --------------------------------------------------------------------------- exposure map
def _profiles_yaml(tmp_path) -> object:
    """Write a tiny profiles file: two equities with sector data, one with none, one bond."""
    text = """
indices:
  IDX_TECH:
    index_name: Tech Index
    strategy: equity
    asset_class: equity
    data_complete: true
    sector_weights: {Information Technology: 0.7, Financials: 0.3}
    region_weights: {"North America": 1.0}
  IDX_BROAD:
    index_name: Broad Index
    strategy: equity
    asset_class: equity
    data_complete: true
    sector_weights: {Information Technology: 0.2, Health Care: 0.5, Financials: 0.3}
    region_weights: {"North America": 0.6, Europe: 0.4}
  IDX_NOSECTOR:
    index_name: No-Sector Equity
    strategy: equity
    asset_class: equity
    data_complete: false
    region_weights: {Europe: 1.0}
  IDX_BOND:
    index_name: Govt Bond
    strategy: bond
    asset_class: bond
    data_complete: true
    credit_quality: {AAA: 1.0}
funds:
  AAA000000001: {ticker: TECH, index: IDX_TECH}
  AAA000000002: {ticker: BRD, index: IDX_BROAD}
  AAA000000003: {ticker: NOS, index: IDX_NOSECTOR}
  AAA000000004: {ticker: BND, index: IDX_BOND}
"""
    f = tmp_path / "profiles.yaml"
    f.write_text(text, encoding="utf-8")
    return f


def _panel_named(cols, seed=3, n=800):
    p = _panel(seed=seed, n=n)
    p.columns = cols
    return p


def test_exposure_matrix_columns_and_coverage(tmp_path):
    path = _profiles_yaml(tmp_path)
    assets = ["AAA000000001", "AAA000000002", "AAA000000003", "AAA000000004"]
    labels, A, covered = opt.exposure_matrix(assets, "sector", path=path)
    # The no-sector equity and the bond contribute zero columns → not counted.
    assert list(covered) == [1.0, 1.0, 0.0, 0.0]
    # Each covered fund's column sums to 1 (its own sector breakdown, normalised).
    assert A[:, 0].sum() == pytest.approx(1.0)
    assert A[:, 2].sum() == pytest.approx(0.0)
    assert "Information Technology" in labels


def test_sector_cap_respected_and_coverage_reported(tmp_path):
    path = _profiles_yaml(tmp_path)
    assets = ["AAA000000001", "AAA000000002", "AAA000000003", "AAA000000004"]
    p = _panel_named(assets)
    lim = opt.ExposureLimit("sector", "Information Technology", upper=0.25)
    r = opt.optimize_portfolio(p, constraints=opt.OptConstraints(exposure_limits=(lim,)),
                               path=path)
    assert r.success
    w = _w(r, assets)
    # Absolute IT exposure = 0.7*w0 + 0.2*w1 must respect the cap.
    it = 0.7 * w[0] + 0.2 * w[1]
    assert it <= 0.25 + TOL
    rep = r.exposures["sector"]
    # Coverage < 1 because the no-sector equity and the bond carry no sector data.
    assert 0.0 < rep.coverage < 1.0


def test_asset_class_floor_respected(tmp_path):
    path = _profiles_yaml(tmp_path)
    assets = ["AAA000000001", "AAA000000002", "AAA000000003", "AAA000000004"]
    p = _panel_named(assets)
    lim = opt.ExposureLimit("asset_class", "bond", lower=0.30)
    r = opt.optimize_portfolio(p, constraints=opt.OptConstraints(exposure_limits=(lim,)),
                               path=path)
    assert r.success
    w = _w(r, assets)
    assert w[3] >= 0.30 - TOL  # the bond fund is the only asset_class == bond
    assert r.exposures["asset_class"].coverage == pytest.approx(1.0)


# --------------------------------------------------------------------------- robustness
def test_infeasible_returns_clean_status():
    p = _panel()
    # 4 assets, each capped at 10% → max total 40% < 100%: infeasible, must not raise.
    r = opt.optimize_portfolio(p, constraints=opt.OptConstraints(max_weight=0.10))
    assert not r.success
    assert r.weights == {}
    assert "feasible" in r.message.lower() or r.status != "optimal"


def test_infeasible_floor_no_exception(tmp_path):
    path = _profiles_yaml(tmp_path)
    assets = ["AAA000000001", "AAA000000002", "AAA000000003", "AAA000000004"]
    p = _panel_named(assets)
    # Require 50% Health Care but only one fund has 50% HC and it can't dominate enough
    # while also... actually demand an impossible 99% Health Care floor.
    lim = opt.ExposureLimit("sector", "Health Care", lower=0.99)
    r = opt.optimize_portfolio(p, constraints=opt.OptConstraints(exposure_limits=(lim,)),
                               path=path)
    assert not r.success  # only fund #2 carries any Health Care (0.5 of its weight)
    assert isinstance(r.message, str)


def test_single_asset_is_insufficient():
    p = _panel()[["F0"]]
    r = opt.optimize_portfolio(p)
    assert not r.success
    assert r.status == "insufficient_assets"


def test_frontier_monotone_and_spans_min_vol():
    p = _panel()
    f = opt.efficient_frontier(p, points=12)
    assert not f.empty
    # Sorted by volatility; expected return should be non-decreasing along the frontier.
    assert f["ret"].is_monotonic_increasing or f["ret"].diff().dropna().min() > -1e-6
    mv = opt.optimize_portfolio(p, objective="min_volatility")
    assert f["volatility"].min() <= mv.volatility + 1e-4


def test_solver_is_deterministic():
    p = _panel()
    r1 = opt.optimize_portfolio(p, objective="max_sharpe")
    r2 = opt.optimize_portfolio(p, objective="max_sharpe")
    w1 = _w(r1, list(p.columns))
    w2 = _w(r2, list(p.columns))
    assert np.max(np.abs(w1 - w2)) < 1e-8


def test_return_methods_both_run():
    p = _panel()
    for method in opt.RETURN_METHODS:
        r = opt.optimize_portfolio(p, return_method=method)
        assert r.success, method
    with pytest.raises(ValueError):
        opt.optimize_portfolio(p, return_method="bogus")


def test_unknown_objective_raises():
    p = _panel()
    with pytest.raises(ValueError):
        opt.optimize_portfolio(p, objective="max_return")
