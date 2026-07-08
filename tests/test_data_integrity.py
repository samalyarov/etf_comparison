"""Data-integrity tests — this is a financial tool, so invariants on the data matter.

Two layers:

* **Fixture-based invariants** (always run, incl. in CI): build a small in-memory database
  and assert the storage/quality/FX pipeline preserves the guarantees downstream analytics
  rely on — no unflagged corruption, currencies resolvable, FX conversion sane.
* **Real-DB checks** (skipped when ``data/etf.db`` is absent, e.g. in CI, since the DB is
  git-ignored): assert the *actual* dataset a user would analyse is internally consistent —
  every fund has a quote currency, no series slips past the health guard unflagged, FX and
  macro caches are populated, and price ranges are physically plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf import fx, metrics, quality
from etf.config import DB_PATH

# --------------------------------------------------------------------------- fixtures


def _price_frame(values):
    idx = pd.date_range("2015-01-01", periods=len(values), freq="B")
    s = pd.Series(values, index=idx, dtype=float)
    return pd.DataFrame({"open": s, "high": s, "low": s, "close": s,
                         "adj_close": s, "volume": 1000})


# --------------------------------------------------------------------------- always-on invariants


def test_cleaned_series_never_leaves_unflagged_corruption():
    # A GBX flip must end either repaired-and-clean or explicitly flagged suspect —
    # it must never pass through silently as "clean".
    out, rep = quality.clean_prices(_price_frame([24.0, 24.5, 2493.0, 24.7, 25.0, 24.8]))
    worst = quality.assess_series(out["adj_close"])
    if worst["status"] == "suspect":
        assert rep.status == "suspect"
    else:
        assert rep.status in ("clean", "repaired")
        assert metrics.has_clean_history(out["adj_close"])


def test_fx_conversion_is_monotonic_in_rate():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    s = pd.Series(100.0, index=idx)
    low = pd.DataFrame({"GBP": np.full(5, 1.0)}, index=idx)
    high = pd.DataFrame({"GBP": np.full(5, 1.3)}, index=idx)
    assert fx.convert_to_base(s, "GBP", high).iloc[0] > fx.convert_to_base(s, "GBP", low).iloc[0]


def test_pence_is_exactly_one_hundredth_of_pounds():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    rate = pd.DataFrame({"GBP": np.full(3, 1.15)}, index=idx)
    pounds = fx.convert_to_base(pd.Series(100.0, index=idx), "GBP", rate)
    pence = fx.convert_to_base(pd.Series(10000.0, index=idx), "GBp", rate)
    assert np.allclose(pounds.values, pence.values)


def test_metrics_never_raise_on_degenerate_series():
    # Price series always carry a DatetimeIndex; assert metrics survive degenerate lengths.
    for vals in ([100.0], [100.0, 100.0], []):
        idx = pd.date_range("2020-01-01", periods=len(vals), freq="D")
        s = pd.Series(vals, index=idx, dtype=float)
        # These must return a finite number or NaN, not raise, on too-short / empty input.
        metrics.cagr(s)
        metrics.max_drawdown(s)
        metrics.annualized_volatility(s)
        metrics.sharpe_ratio(s)


# --------------------------------------------------------------------------- real-DB checks

_HAS_DB = DB_PATH.exists()
db_test = pytest.mark.skipif(not _HAS_DB, reason="needs a populated data/etf.db")


@db_test
def test_every_instrument_has_a_currency():
    from etf import data
    etfs = data.list_etfs()
    missing = etfs[etfs["currency"].isna()]["ticker"].tolist()
    assert not missing, f"instruments without a quote currency: {missing}"


@db_test
def test_no_series_slips_past_the_health_guard_unflagged():
    from etf import data
    etfs = data.list_etfs()
    health = data.data_health().set_index("isin")
    offenders = []
    for isin, ticker in zip(etfs["isin"], etfs["ticker"]):
        s = data.load_prices(isin)["adj_close"].dropna()
        if s.empty:
            continue
        if not metrics.has_clean_history(s):
            status = health.loc[isin, "status"] if isin in health.index else "missing"
            if status != "suspect":
                offenders.append(f"{ticker}:{status}")
    assert not offenders, f"corrupt series not flagged suspect: {offenders}"


@db_test
def test_fx_and_macro_caches_populated():
    from etf import data
    fxdf = fx.load_fx()
    assert not fxdf.empty, "fx_rates cache is empty — run `python -m etf.ingest --fx`"
    assert {"USD", "GBP"} <= set(fxdf.columns)
    macro = data.macro_series()
    assert not macro.empty, "macro cache empty — run `python -m etf.ingest --facts`"


@db_test
def test_distributing_bond_funds_have_a_coupon_stream():
    # Bond-income modelling relies on the distributions table: every *distributing* bond ETF
    # with price history should carry at least one recorded coupon (else the cash-out
    # scenario is silently empty). Accumulating funds legitimately have none.
    from etf import bonds, data
    etfs = data.list_etfs()
    b = etfs[(etfs["asset_class"] == "bond") & (etfs["acc_dist"] == "DIST")]
    missing = []
    for isin, ticker in zip(b["isin"], b["ticker"]):
        if data.load_prices(isin).empty:
            continue
        d = data.load_distributions(isin)
        # A distributing bond fund with a multi-year history but no coupons is suspect.
        if d.empty or "amount" not in d.columns or (d["amount"] > 0).sum() == 0:
            missing.append(ticker)
        else:
            # Income modelling must produce a non-negative cash stream from real data.
            sc = bonds.income_scenarios(data.load_prices(isin)["close"].dropna(), d,
                                        initial=10_000.0)
            assert sc.total_income >= 0.0, ticker
    # IBCI (inflation-linked) is a known Yahoo gap — coupons not reported; tolerate only it.
    assert set(missing) <= {"IBCI.DE", "IBCI.L"}, f"distributing bonds with no coupons: {missing}"


@db_test
def test_bond_reinvested_path_reconciles_with_adj_close():
    # Total-return reconciliation (roadmap: "distribution / total-return reconciliation").
    # The reinvested path derived from the stored `adj_close` must match the path
    # reconstructed independently from `close` + the `distributions` table. A wide gap means
    # the adjusted series is inconsistent with the raw price + coupons — an adjustment/ingest
    # artifact worth catching before it reaches the bond-income view.
    #
    # NB: we deliberately do NOT assert "reinvesting beats cashing out" here — for a bond
    # whose price fell after coupons were paid (e.g. long-duration govvies whose coupons
    # were reinvested near the 2020-21 price peak, then crushed by the 2022 rate shock),
    # banking the coupons as cash legitimately wins. That conditional ordering is covered by
    # the synthetic monotonic-price unit test in tests/test_bonds.py.
    from etf import bonds, data
    etfs = data.list_etfs()
    b = etfs[(etfs["asset_class"] == "bond") & (etfs["acc_dist"] == "DIST")]
    checked = 0
    for isin, ticker in zip(b["isin"], b["ticker"]):
        px = data.load_prices(isin)
        if px.empty:
            continue
        dists = data.load_distributions(isin)
        close = px["close"].dropna()
        sc = bonds.income_scenarios(close, dists, adj_close=px["adj_close"].dropna(),
                                    initial=10_000.0)
        recon = bonds.reinvest_from_price(close, dists, initial=10_000.0)
        if sc.reinvested.empty or recon.empty:
            continue
        assert sc.final_reinvested == pytest.approx(float(recon.iloc[-1]), rel=0.01), (
            f"{ticker}: adj_close total return disagrees with close+coupon reconstruction")
        checked += 1
    assert checked > 0, "no distributing bond with price history was reconciled"


@db_test
def test_prices_are_physically_plausible():
    from etf import data
    etfs = data.list_etfs()
    bad = []
    for isin, ticker in zip(etfs["isin"], etfs["ticker"]):
        s = data.load_prices(isin)["adj_close"].dropna()
        if len(s) < 2:
            continue
        if (s <= 0).any():
            bad.append(f"{ticker}: non-positive price")
    assert not bad, bad
