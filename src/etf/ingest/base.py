"""Common contract and helpers for price/fundamentals sources.

Each source is a small adapter that normalises whatever the provider returns into a
canonical OHLCV DataFrame so the storage layer never has to care where data came from.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

CANONICAL_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class SourceError(Exception):
    """Raised when a source cannot return data for an instrument."""


@runtime_checkable
class PriceSource(Protocol):
    """A provider that can return canonical OHLCV history for an instrument."""

    name: str

    def get_prices(self, ticker: str, start: date | None, end: date | None) -> pd.DataFrame:
        """Return a DataFrame indexed by date with :data:`CANONICAL_COLUMNS`."""
        ...


def normalize_ohlcv(df: pd.DataFrame, rename: dict[str, str]) -> pd.DataFrame:
    """Rename provider columns to the canonical schema and coerce types.

    ``rename`` maps provider column names -> canonical names. Missing canonical
    columns are added as NaN; ``adj_close`` falls back to ``close`` when absent.
    The index is coerced to a tz-naive DatetimeIndex and sorted.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    out = df.rename(columns=rename).copy()
    out = out[[c for c in out.columns if c in CANONICAL_COLUMNS]]

    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    if out["adj_close"].isna().all():
        out["adj_close"] = out["close"]

    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out[CANONICAL_COLUMNS].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out
