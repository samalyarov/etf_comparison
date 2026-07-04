"""Yahoo Finance adapter (via yfinance).

Primary source for UCITS ETFs: exchange-suffixed tickers (.DE/.L/.AS) are generally
well covered, and it returns dividends for total-return reconstruction. It is an
unofficial, rate-limited endpoint, so callers should batch and back off.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from .base import SourceError, normalize_ohlcv


class YahooSource:
    name = "yfinance"

    def get_prices(self, ticker: str, start: date | None, end: date | None) -> pd.DataFrame:
        import yfinance as yf

        kwargs = dict(auto_adjust=False, progress=False, threads=False, actions=False)
        if start is None:
            # No start => full history. yfinance ignores `end`-only and defaults to
            # period="1mo", so ask for the max period explicitly.
            kwargs["period"] = "max"
        else:
            kwargs["start"] = str(start)
            if end:
                kwargs["end"] = str(end)

        try:
            df = yf.download(ticker, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface any provider error uniformly
            raise SourceError(f"yfinance download failed for {ticker}: {exc}") from exc

        if df is None or df.empty:
            raise SourceError(f"yfinance returned no rows for {ticker}")

        # yfinance may return MultiIndex columns (field, ticker) for a single symbol.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        return normalize_ohlcv(df, rename)

    def get_currency(self, ticker: str) -> str | None:
        """Return the quote currency Yahoo reports for a ticker (e.g. 'GBp', 'USD'), or None.

        This is the reliable signal for the pence-vs-pounds ambiguity on London listings:
        Yahoo reports pence-quoted funds as 'GBp' and pounds-quoted as 'GBP'.
        """
        import yfinance as yf

        try:
            cur = yf.Ticker(ticker).fast_info.get("currency")
            return str(cur) if cur else None
        except Exception:  # noqa: BLE001
            return None

    def get_fundamentals(self, ticker: str) -> dict:
        """Best-effort fund metadata from Yahoo: AUM, inception, category. Empty on failure."""
        import datetime as _dt

        import yfinance as yf

        out: dict = {}
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:  # noqa: BLE001
            return out
        if info.get("totalAssets"):
            out["aum"] = float(info["totalAssets"])
        incep = info.get("fundInceptionDate")
        if incep:
            try:
                out["inception"] = _dt.date.fromtimestamp(int(incep)).isoformat()
            except (ValueError, OSError, OverflowError):
                pass
        if info.get("category"):
            out["category_yahoo"] = info["category"]
        return out

    def get_dividends(self, ticker: str) -> list[tuple[date, float]]:
        """Return (ex_date, amount) dividend history; empty list on any failure."""
        import yfinance as yf

        try:
            divs = yf.Ticker(ticker).dividends
        except Exception:  # noqa: BLE001
            return []
        if divs is None or len(divs) == 0:
            return []
        idx = divs.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        return [(ts.date(), float(amt)) for ts, amt in zip(idx, divs.values)]
