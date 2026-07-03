"""Stooq adapter (via pandas-datareader).

No API key needed. Useful as a fallback / cross-check, with reasonable European
coverage. Stooq tickers differ from Yahoo's: e.g. Xetra ``VWCE.DE`` -> ``VWCE.DE``
mostly works, London ``.L`` -> ``.UK``. Stooq does not provide a separate adjusted
close, so ``adj_close`` falls back to ``close`` (price-return basis only).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from .base import SourceError, normalize_ohlcv


def _to_stooq_symbol(ticker: str) -> str:
    """Best-effort map of a Yahoo-style ticker to Stooq's convention."""
    t = ticker.upper()
    if t.endswith(".L"):
        return t[:-2] + ".UK"
    return t


class StooqSource:
    name = "stooq"

    def get_prices(self, ticker: str, start: date | None, end: date | None) -> pd.DataFrame:
        from pandas_datareader import data as pdr

        symbol = _to_stooq_symbol(ticker)
        try:
            df = pdr.DataReader(symbol, "stooq", start=start, end=end)
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"stooq failed for {symbol}: {exc}") from exc

        if df is None or df.empty:
            raise SourceError(f"stooq returned no rows for {symbol}")

        # Stooq returns newest-first with capitalised columns and no adjusted close.
        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        return normalize_ohlcv(df, rename)
