"""Tests for the factor-model module: parsing, regression, decomposition, scenarios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf import db, factors, portfolio
from etf.ingest import kenfrench

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- Ken French parsing

def test_parse_ff_csv_reads_monthly_block_as_decimals():
    text = (FIXTURES / "Europe_5_Factors_sample.csv").read_text(encoding="utf-8")
    df = kenfrench.parse_ff_csv(text, frequency="monthly")
    assert not df.empty
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    # First row: 199007, Mkt-RF 4.46% -> 0.0446 decimal, month-end date.
    assert df.index[0] == pd.Timestamp("1990-07-31")
    assert df["Mkt-RF"].iloc[0] == pytest.approx(0.0446)
    assert df["RF"].iloc[0] == pytest.approx(0.0068)
    # Values are decimals (|monthly factor| well under 1).
    assert df.abs().to_numpy().max() < 1.0


def test_parse_ff_csv_stops_before_annual_section():
    text = (FIXTURES / "Europe_5_Factors_sample.csv").read_text(encoding="utf-8")
    df = kenfrench.parse_ff_csv(text, frequency="monthly")
    # The fixture appends an "Annual Factors" section with YYYY (4-digit) rows; those must be
    # excluded — every parsed index is a month-end within the monthly range.
    assert (df.index.day >= 28).all()
    assert df.index.year.min() == 1990


def test_parse_ff_csv_handles_missing_sentinel():
    text = "\n\n,Mkt-RF,SMB,RF\n199001  , 1.23 , -99.99 , 0.10\n\n"
    df = kenfrench.parse_ff_csv(text)
    assert df["Mkt-RF"].iloc[0] == pytest.approx(0.0123)
    assert np.isnan(df["SMB"].iloc[0])  # -99.99 sentinel -> NaN, not -0.9999


def test_merge_appends_momentum_column():
    five = kenfrench.parse_ff_csv((FIXTURES / "Europe_5_Factors_sample.csv").read_text("utf-8"))
    mom = kenfrench.parse_ff_csv((FIXTURES / "Europe_MOM_Factor_sample.csv").read_text("utf-8"))
    merged = kenfrench.merge_factor_frames(five, mom)
    assert "WML" in merged.columns
    # Canonical order: RF last, WML before it.
    assert list(merged.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "WML", "RF"]


# --------------------------------------------------------------------------- store/load roundtrip

def test_factor_returns_roundtrip(tmp_path):
    from etf import data
    dbp = tmp_path / "t.db"
    db.init_db(dbp)
    five = kenfrench.parse_ff_csv((FIXTURES / "Europe_5_Factors_sample.csv").read_text("utf-8"))
    mom = kenfrench.parse_ff_csv((FIXTURES / "Europe_MOM_Factor_sample.csv").read_text("utf-8"))
    matrix = kenfrench.merge_factor_frames(five, mom)
    with db.connect(dbp) as conn:
        n = db.upsert_factor_returns(conn, "Europe", "monthly", matrix, source="test")
    assert n > 0
    loaded = data.load_factor_returns("Europe", "monthly", db_path=dbp)
    assert set(["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]) <= set(loaded.columns)
    # NaN cells (missing momentum before its start) are simply absent, never stored as values.
    assert loaded["Mkt-RF"].iloc[0] == pytest.approx(matrix["Mkt-RF"].iloc[0])


# --------------------------------------------------------------------------- regression (B)

def _synthetic_factors(n=180, seed=1):
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    data = {
        "Mkt-RF": rng.normal(0.006, 0.045, n),
        "SMB": rng.normal(0.001, 0.03, n),
        "HML": rng.normal(0.001, 0.03, n),
        "RMW": rng.normal(0.001, 0.02, n),
        "CMA": rng.normal(0.001, 0.02, n),
        "WML": rng.normal(0.004, 0.04, n),
        "RF": np.full(n, 0.002),
    }
    return pd.DataFrame(data, index=idx)


def test_regression_recovers_known_betas():
    fr = _synthetic_factors()
    true_betas = {"Mkt-RF": 1.0, "SMB": 0.3, "HML": -0.2, "RMW": 0.4, "CMA": 0.0, "WML": 0.5}
    true_alpha_m = 0.001
    rng = np.random.default_rng(7)
    excess = np.full(len(fr), true_alpha_m)
    for f, b in true_betas.items():
        excess = excess + b * fr[f].to_numpy()
    excess = excess + rng.normal(0, 0.002, len(fr))  # small idiosyncratic noise
    # Portfolio *raw* return = excess + RF; the regression must subtract RF itself.
    port = pd.Series(excess + fr["RF"].to_numpy(), index=fr.index)

    reg = factors.factor_exposures(port, fr)
    for f, b in true_betas.items():
        assert reg.betas[f] == pytest.approx(b, abs=0.05), f
    assert reg.alpha == pytest.approx(true_alpha_m, abs=0.002)
    assert reg.r_squared > 0.95
    assert reg.n_obs == len(fr)
    # A strong, correctly-signed loading is statistically significant.
    assert reg.t_stats["Mkt-RF"] > 5
    assert ("WML", pytest.approx(0.5, abs=0.06)) in [
        (f, b) for f, b in reg.dominant_tilts()]


def test_excess_return_convention_matters():
    """Regressing without subtracting RF changes the intercept by ~mean(RF)."""
    fr = _synthetic_factors(seed=3)
    excess = 0.0 + 0.9 * fr["Mkt-RF"].to_numpy()
    port = pd.Series(excess + fr["RF"].to_numpy(), index=fr.index)
    with_rf = factors.factor_exposures(port, fr, factors=["Mkt-RF"])
    # Drop RF from the frame so the function cannot form the excess return.
    without_rf = factors.factor_exposures(port, fr.drop(columns=["RF"]), factors=["Mkt-RF"])
    # Betas agree; the intercept differs by roughly the mean risk-free rate.
    assert with_rf.betas["Mkt-RF"] == pytest.approx(without_rf.betas["Mkt-RF"], abs=1e-6)
    assert without_rf.alpha - with_rf.alpha == pytest.approx(fr["RF"].mean(), abs=1e-3)


def test_regression_aligns_dates_on_overlap():
    fr = _synthetic_factors(n=120)
    port = pd.Series(0.8 * fr["Mkt-RF"].to_numpy() + fr["RF"].to_numpy(), index=fr.index)
    # Portfolio only overlaps the last 60 months of the factor history.
    reg = factors.factor_exposures(port.iloc[-60:], fr, factors=["Mkt-RF"])
    assert reg.n_obs == 60
    assert reg.betas["Mkt-RF"] == pytest.approx(0.8, abs=1e-6)


def test_regression_insufficient_overlap_raises():
    fr = _synthetic_factors(n=120)
    short = pd.Series(fr["Mkt-RF"].to_numpy()[:3], index=fr.index[:3])
    with pytest.raises(factors.InsufficientData):
        factors.factor_exposures(short, fr)


# --------------------------------------------------------------------------- decomposition (A)

def _price_matrix(seed=0, n=600):
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    a = 100 * np.cumprod(1 + rng.normal(0.0006, 0.011, n))
    b = 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, n))
    c = 100 * np.cumprod(1 + rng.normal(0.0004, 0.009, n))
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


def test_sleeve_contributions_are_additive():
    P = _price_matrix()
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    sc = factors.sleeve_contributions(P, weights, initial=100.0)
    # Buy-and-hold contributions sum exactly to the blend's total gain.
    final = P.iloc[-1] / P.iloc[0]
    expected_gain = sum(weights[k] * 100 * (final[k] - 1) for k in weights)
    assert sc["contribution"].sum() == pytest.approx(expected_gain, rel=1e-9)
    assert sc["contribution_share"].sum() == pytest.approx(1.0, rel=1e-9)
    # Contribution equals weight * standalone return * initial.
    for k in weights:
        assert sc.loc[k, "contribution"] == pytest.approx(
            weights[k] * 100 * sc.loc[k, "total_return"], rel=1e-9)


def test_sleeve_contributions_empty_on_no_overlap():
    P = _price_matrix()
    assert factors.sleeve_contributions(P, {}).empty


def test_sleeve_factor_label():
    assert factors.sleeve_factor_label(["momentum", "factor"]) == "Momentum"
    assert factors.sleeve_factor_label([], "iShares Edge MSCI World Min Vol") == "Min Volatility"
    assert factors.sleeve_factor_label(["large-cap"]) is None
    # A multifactor fund carries several factor tags but must resolve to Multifactor, not
    # be mislabelled as the first single factor found.
    assert factors.sleeve_factor_label(
        ["multifactor", "value", "momentum", "quality", "size"]) == "Multifactor"


# --------------------------------------------------------------------------- scenarios

def _crashy_series():
    """A price series that rises, crashes ~30%, then recovers and rises again."""
    up1 = np.linspace(100, 150, 40)
    crash = np.linspace(150, 105, 12)       # -30% peak-to-trough
    recover = np.linspace(105, 160, 30)     # regains and passes the prior peak
    up2 = np.linspace(160, 200, 30)
    vals = np.concatenate([up1, crash, recover, up2])
    idx = pd.date_range("2010-01-31", periods=len(vals), freq="ME")
    return pd.Series(vals, index=idx)


def test_worst_crash_window_finds_drawdown_and_recovery():
    s = _crashy_series()
    rets = factors.projection.monthly_returns(s)
    seq, dd, wlen = factors.worst_crash_window(rets)
    assert dd == pytest.approx(-0.30, abs=0.03)
    assert wlen > 0
    # Replaying the window reproduces the drawdown at its trough.
    cum = np.cumprod(1 + seq)
    assert cum.min() - 1.0 == pytest.approx(dd, abs=0.02)


def test_worst_crash_window_empty_on_monotonic():
    idx = pd.date_range("2010-01-31", periods=40, freq="ME")
    s = pd.Series(np.linspace(100, 200, 40), index=idx)
    seq, dd, wlen = factors.worst_crash_window(factors.projection.monthly_returns(s))
    assert wlen == 0 and dd == 0.0 and len(seq) == 0


def test_plan_scenarios_fan_is_monotonic():
    s = _crashy_series()
    ps = factors.plan_scenarios(s, start_value=10000, monthly=300, years=15,
                                method="bootstrap", n_sims=400, seed=2)
    assert ps.worst <= ps.base <= ps.best
    assert ps.invested == pytest.approx(10000 + 300 * 12 * 15)
    assert ps.crash is not None
    assert ps.crash.window_drawdown < 0
    assert len(ps.crash.timeline) == 15 * 12


def test_crash_scenario_dips_below_start_when_lumpsum_dominates():
    s = _crashy_series()
    # Large lump sum, tiny monthly: the crash must pull net worth below the start value.
    cs = factors.crash_scenario(s, start_value=100000, monthly=50, years=20)
    assert cs is not None
    assert cs.trough_value < 100000 * 0.95
    assert cs.trough_month >= 1
    assert cs.final_value > cs.trough_value  # recovers/grows after the crash tail


def test_plan_scenarios_no_crash_on_monotonic_series():
    idx = pd.date_range("2010-01-31", periods=60, freq="ME")
    s = pd.Series(100 * (1.01 ** np.arange(60)), index=idx)
    ps = factors.plan_scenarios(s, start_value=1000, monthly=100, years=10,
                                method="ols", include_crash=True)
    assert ps.crash is None  # nothing to replay
    assert ps.worst <= ps.base <= ps.best


def test_plan_scenarios_reuse_blend_index():
    """A blend built via portfolio.blend_index feeds straight into plan_scenarios."""
    P = _price_matrix(seed=5)
    blend = portfolio.blend_index(P, {"A": 0.4, "B": 0.3, "C": 0.3}, rebalance="Q")
    ps = factors.plan_scenarios(blend, start_value=5000, monthly=200, years=10,
                                method="normal", n_sims=300)
    assert ps.base > 0
    assert len(ps.fan) == 120
