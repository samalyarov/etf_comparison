"""Tiingo adapter (REST API, requires TIINGO_API_KEY).

Clean, well-maintained data with a generous free tier. Coverage is strongest for
US-listed securities; European UCITS listings may be missing, in which case the
orchestrator falls back to Yahoo/Stooq. Tiingo's adjusted fields are dividend- and
split-adjusted, which is exactly the total-return basis we want.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from ..config import get_api_key
from .base import SourceError, normalize_ohlcv

BASE_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"


class TiingoSource:
    name = "tiingo"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or get_api_key("TIINGO_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def get_prices(self, ticker: str, start: date | None, end: date | None) -> pd.DataFrame:
        import requests

        if not self.api_key:
            raise SourceError("TIINGO_API_KEY not set")

        params = {"format": "json", "resampleFreq": "daily", "token": self.api_key}
        if start:
            params["startDate"] = str(start)
        if end:
            params["endDate"] = str(end)

        try:
            resp = requests.get(
                BASE_URL.format(ticker=ticker), params=params, timeout=30,
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"tiingo request failed for {ticker}: {exc}") from exc

        if resp.status_code == 404:
            raise SourceError(f"tiingo has no data for {ticker}")
        if resp.status_code != 200:
            raise SourceError(f"tiingo HTTP {resp.status_code} for {ticker}: {resp.text[:200]}")

        rows = resp.json()
        if not rows:
            raise SourceError(f"tiingo returned no rows for {ticker}")

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        rename = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adjClose": "adj_close",
            "volume": "volume",
        }
        return normalize_ohlcv(df, rename)
