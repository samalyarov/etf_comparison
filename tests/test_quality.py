"""Tests for price-series data-quality repair (GBX/GBP rescale + de-spike)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf import quality


def _series(values, start="2010-01-01"):
    idx = pd.date_range(start=start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def _frame(values):
    s = _series(values)
    return pd.DataFrame({"open": s, "high": s, "low": s, "close": s,
                         "adj_close": s, "volume": 1000})


def test_clean_series_is_untouched():
    s = _series([100, 101, 99, 102, 103, 104, 100, 98, 101])
    out, rep = quality.clean_prices(_frame(s.values))
    assert rep.status == "clean"
    assert rep.rescaled_days == 0 and rep.despiked_days == 0
    assert np.allclose(out["adj_close"].values, s.values)


def test_transient_gbx_flip_is_rescaled():
    # One day quoted in pence among pounds (24 -> 2493 -> 24) is snapped back.
    vals = [24.0, 24.5, 2493.0, 24.7, 25.0, 24.8]
    out, rep = quality.clean_prices(_frame(vals))
    assert rep.rescaled_days >= 1
    assert out["adj_close"].max() < 50  # the 2493 pence print is divided by 100
    assert quality.assess_series(out["adj_close"])["status"] == "clean"


def test_persistent_regime_shift_unified_to_recent_scale():
    # Pence for the first half, pounds for the second (like IBTS.L in 2009).
    vals = [10000, 10050, 9980, 10020] + [100, 101, 99, 102, 100]
    out, rep = quality.clean_prices(_frame(vals))
    assert rep.rescaled_days >= 1
    # Everything expressed in the recent (pounds) denomination: no ~100x values remain.
    assert out["adj_close"].max() < 200
    assert quality.assess_series(out["adj_close"])["status"] == "clean"


def test_isolated_non_100x_spike_is_despiked():
    # A lone bad print that is ~1.6x and reverts (SGLN/EQQQ pattern) is interpolated out.
    vals = [1800, 1810, 1820, 2953, 1842, 1850, 1860]
    out, rep = quality.clean_prices(_frame(vals))
    assert rep.despiked_days >= 1
    assert 1800 < out["adj_close"].iloc[3] < 1860  # replaced by neighbour midpoint
    assert quality.assess_series(out["adj_close"])["status"] == "clean"


def test_genuine_two_day_move_not_despiked():
    # A real +20% then flat move must survive (not treated as a spike).
    vals = [100, 100, 120, 121, 122]
    out, rep = quality.clean_prices(_frame(vals))
    assert rep.despiked_days == 0
    assert out["adj_close"].iloc[2] == 120


def test_reconstruct_scale_noop_on_clean():
    s = _series([50, 51, 52, 51, 53, 54])
    factor = quality.reconstruct_scale(s)
    assert (factor == 1.0).all()


def test_volume_untouched_by_rescale():
    vals = [24.0, 2493.0, 24.7, 25.0]
    df = _frame(vals)
    df["volume"] = [111, 222, 333, 444]
    out, _ = quality.clean_prices(df)
    assert list(out["volume"]) == [111, 222, 333, 444]
