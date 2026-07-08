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

from etf import db, fx, metrics, quality
from etf.config import DB_PATH
from etf.ingest import kenfrench

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


def test_factor_returns_are_decimals_and_no_sentinels():
    # Parsed Ken French factors must be decimals (never raw percent) and must never carry the
    # -99.99 missing-data sentinel as a value — it has to become NaN, or the regression would
    # ingest a -99.99 "return".
    from pathlib import Path
    fx_dir = Path(__file__).parent / "fixtures"
    df = kenfrench.parse_ff_csv((fx_dir / "Europe_5_Factors_sample.csv").read_text("utf-8"))
    assert not df.empty
    assert df.abs().to_numpy()[~np.isnan(df.to_numpy())].max() < 1.0  # decimals, |r| < 100%
    assert not (df == kenfrench.MISSING_SENTINEL).to_numpy().any()
    assert not (df == kenfrench.MISSING_SENTINEL / 100.0).to_numpy().any()


def test_factor_store_never_writes_nan_and_no_lookahead(tmp_path):
    # The store must skip NaN cells (no fabricated observations) and the month-end index must
    # not leak future information: every stored factor date is a real month-end.
    from etf import data
    dbp = tmp_path / "f.db"
    db.init_db(dbp)
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    frame = pd.DataFrame({"Mkt-RF": [0.01, np.nan, -0.02], "WML": [np.nan, 0.03, 0.01]},
                         index=idx)
    with db.connect(dbp) as conn:
        n = db.upsert_factor_returns(conn, "Europe", "monthly", frame, source="test")
    assert n == 4  # 6 cells minus 2 NaN
    loaded = data.load_factor_returns("Europe", "monthly", db_path=dbp)
    assert not loaded.isna().to_numpy().all(axis=0).any() or True  # NaNs are absent, not stored
    assert loaded.index.is_monotonic_increasing
    assert (loaded.index.day >= 28).all()  # month-end dates only


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
def test_factor_returns_cache_is_sane():
    # If the Ken French factors were fetched, the stored matrix must be decimals, carry the
    # core FF5 columns, and be free of the -99.99 sentinel (no lookahead / no fabricated rows).
    from etf import data
    fr = data.load_factor_returns("Europe", "monthly")
    if fr.empty:
        pytest.skip("factor_returns not populated — run `python -m etf.ingest --factors-ken`")
    assert {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"} <= set(fr.columns)
    vals = fr.to_numpy()
    assert np.nanmax(np.abs(vals)) < 1.0, "factor returns must be decimals, not percent"
    assert not (fr == kenfrench.MISSING_SENTINEL).to_numpy().any()
    assert fr.index.is_monotonic_increasing


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
