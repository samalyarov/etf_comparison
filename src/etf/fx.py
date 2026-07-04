"""Currency normalisation to a base currency (EUR) for honest cross-fund comparison.

The universe mixes quote currencies: Xetra/Amsterdam listings are EUR, London (``.L``)
listings are a mix of GBP, **GBp (pence)** and USD, and Swiss listings are CHF. A GBP
fund and a USD fund are not comparable in their own currencies, and a EUR-based DCA plan
buys in EUR — so amounts must be converted to a single base currency.

Design (local-first, mirrors the price pipeline):

* Each instrument's quote currency is captured from Yahoo (``fast_info['currency']``) at
  ingest and stored on ``instruments.currency``. Yahoo reports pence as ``GBp`` — the one
  reliable signal for the pence-vs-pounds ambiguity that price levels alone can't resolve.
* Daily FX history is fetched once and cached in ``fx_rates`` (EUR value of 1 unit of each
  foreign currency). The app then converts offline, date-by-date, so FX return is included
  honestly rather than assuming a single spot rate.

``convert_to_base`` turns a native-currency price series into EUR (dividing GBp by 100
first), forward-filling FX across non-trading days.
"""

from __future__ import annotations

import re

import pandas as pd

BASE_CURRENCY = "EUR"

# Yahoo FX symbols giving EUR per 1 unit of the quote currency after inversion.
# EURUSD=X is USD per 1 EUR, so EUR-per-USD = 1 / EURUSD. Same pattern for the others.
FX_YAHOO_SYMBOLS = {"USD": "EURUSD=X", "GBP": "EURGBP=X", "CHF": "EURCHF=X"}

# Fallback quote currency by exchange suffix, used only when Yahoo currency wasn't stored.
# Note: this cannot distinguish GBP from GBp on ``.L`` — prefer the stored currency.
SUFFIX_CURRENCY = {"DE": "EUR", "AS": "EUR", "MI": "EUR", "PA": "EUR",
                   "L": "GBP", "SW": "CHF", "SG": "EUR"}


def infer_currency(ticker: str, stored: str | None = None) -> str:
    """Best-effort quote currency: the stored (Yahoo) value if present, else by suffix."""
    if stored:
        return stored
    m = re.search(r"\.(\w+)$", ticker or "")
    return SUFFIX_CURRENCY.get(m.group(1).upper(), "EUR") if m else "EUR"


def is_pence(currency: str | None) -> bool:
    """True for the GBp (pence) pseudo-currency Yahoo uses on some London listings."""
    return (currency or "").strip() in {"GBp", "GBX"}


def normalized_currency(currency: str | None) -> str:
    """Map GBp/GBX to GBP (the real currency); leave others unchanged."""
    return "GBP" if is_pence(currency) else (currency or BASE_CURRENCY).strip()


def eur_per_unit_from_pair(quote: str, pair_series: pd.Series) -> pd.Series:
    """Convert a Yahoo ``EUR<quote>=X`` series (quote per EUR) to EUR-per-unit-of-quote."""
    s = pd.to_numeric(pair_series, errors="coerce").dropna()
    s = s[s > 0]
    return 1.0 / s


def convert_to_base(prices: pd.Series, currency: str | None, fx: pd.DataFrame,
                    base: str = BASE_CURRENCY) -> pd.Series:
    """Convert a native-currency price series to ``base`` (EUR) using cached daily FX.

    ``fx`` is a date-indexed frame with a column per quote currency holding EUR-per-unit
    (see :func:`load_fx`). Pence (``GBp``) is divided by 100 to pounds first. Currencies
    already equal to ``base``, or with no FX data, pass through unchanged (a no-op that
    keeps the app working offline before FX is fetched).
    """
    s = pd.to_numeric(prices, errors="coerce")
    ccy = normalized_currency(currency)
    if is_pence(currency):
        s = s / 100.0
    if ccy == base or fx is None or fx.empty or ccy not in fx.columns:
        return s
    rate = fx[ccy].reindex(s.index).ffill().bfill()
    return s * rate


def load_fx(db_path=None) -> pd.DataFrame:
    """Load cached FX rates as a date x currency frame of EUR-per-unit values."""
    from . import db as _db
    from .config import DB_PATH
    with _db.connect(db_path or DB_PATH) as conn:
        try:
            raw = pd.read_sql_query(
                "SELECT date, quote, eur_per_unit FROM fx_rates", conn,
                parse_dates=["date"])
        except Exception:  # noqa: BLE001 - table absent on an old DB
            return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    return raw.pivot(index="date", columns="quote", values="eur_per_unit").sort_index()
