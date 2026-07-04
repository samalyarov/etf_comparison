"""Tests for currency normalisation to EUR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf import fx


def _fx_frame():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.DataFrame({"GBP": np.full(10, 1.15), "USD": np.full(10, 0.90)}, index=idx)


def test_infer_currency_prefers_stored():
    assert fx.infer_currency("EQQQ.L", stored="GBp") == "GBp"
    assert fx.infer_currency("VWCE.DE") == "EUR"
    assert fx.infer_currency("CSPX.L") == "GBP"  # suffix fallback
    assert fx.infer_currency("UNKNOWN") == "EUR"


def test_pence_detection_and_normalisation():
    assert fx.is_pence("GBp") and fx.is_pence("GBX")
    assert not fx.is_pence("GBP")
    assert fx.normalized_currency("GBp") == "GBP"
    assert fx.normalized_currency("USD") == "USD"


def test_eur_quote_is_noop():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(np.linspace(100, 110, 10), index=idx)
    out = fx.convert_to_base(s, "EUR", _fx_frame())
    assert np.allclose(out.values, s.values)


def test_gbp_converted():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(np.full(10, 100.0), index=idx)
    out = fx.convert_to_base(s, "GBP", _fx_frame())
    assert np.allclose(out.values, 115.0)  # 100 GBP * 1.15 EUR/GBP


def test_pence_divided_by_100_then_converted():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(np.full(10, 10000.0), index=idx)  # 10000 pence = 100 GBP
    out = fx.convert_to_base(s, "GBp", _fx_frame())
    assert np.allclose(out.values, 115.0)


def test_usd_converted():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(np.full(10, 100.0), index=idx)
    out = fx.convert_to_base(s, "USD", _fx_frame())
    assert np.allclose(out.values, 90.0)


def test_missing_currency_passthrough():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    s = pd.Series(np.full(5, 100.0), index=idx)
    out = fx.convert_to_base(s, "SEK", _fx_frame())  # no SEK column -> unchanged
    assert np.allclose(out.values, s.values)


def test_eur_per_unit_inversion():
    pair = pd.Series([1.25, 1.25])  # USD per EUR
    inv = fx.eur_per_unit_from_pair("USD", pair)
    assert np.allclose(inv.values, 0.8)  # EUR per USD
