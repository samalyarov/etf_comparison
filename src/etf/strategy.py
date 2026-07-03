"""Investment-strategy backtests — dollar-cost averaging (DCA) over historical prices.

Pure functions over an adjusted-close price Series (total-return basis). No DB or
network access. The core question answered: "if I had invested X every month (plus an
optional lump sum) into this ETF over a period, what would it be worth now?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DCAResult:
    """Outcome of a DCA backtest."""

    timeline: pd.DataFrame  # index=date, cols: invested, value, units
    total_invested: float
    final_value: float
    profit: float
    money_multiple: float          # final_value / total_invested
    xirr: float                    # money-weighted annualised return
    n_contributions: int
    start: pd.Timestamp
    end: pd.Timestamp
    contributions: pd.Series = field(repr=False)  # date -> amount (cashflows out)


def _month_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First available trading day within each calendar month of the index."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).min().values)


def simulate_dca(
    prices: pd.Series,
    monthly: float,
    *,
    start=None,
    end=None,
    initial: float = 0.0,
    annual_step_up: float = 0.0,
) -> DCAResult:
    """Simulate investing ``monthly`` on the first trading day of each month.

    ``initial`` is an optional lump sum added with the first contribution.
    ``annual_step_up`` increases the monthly amount by that fraction every 12 months
    (e.g. 0.03 = +3%/yr to track raises/inflation). Buys are fractional; prices are
    the total-return (adjusted-close) series.
    """
    s = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    if len(s) < 2 or monthly <= 0:
        raise ValueError("Need at least two months of prices and a positive contribution.")

    buy_days = _month_starts(s.index)
    contributions: dict[pd.Timestamp, float] = {}
    for i, day in enumerate(buy_days):
        amount = monthly * ((1 + annual_step_up) ** (i // 12))
        if i == 0:
            amount += initial
        contributions[day] = amount

    # Build cumulative units held over the whole daily index.
    contrib_units = pd.Series(
        {day: contributions[day] / s.loc[day] for day in buy_days}
    ).sort_index()
    cum_units = contrib_units.cumsum().reindex(s.index, method="ffill").fillna(0.0)
    invested = pd.Series(contributions).sort_index().cumsum().reindex(
        s.index, method="ffill").fillna(0.0)
    value = cum_units * s

    timeline = pd.DataFrame({"invested": invested, "value": value, "units": cum_units})

    total_invested = float(sum(contributions.values()))
    final_value = float(value.iloc[-1])
    contrib_series = pd.Series(contributions).sort_index()
    rate = _xirr(contrib_series, final_value, s.index[-1])

    return DCAResult(
        timeline=timeline,
        total_invested=total_invested,
        final_value=final_value,
        profit=final_value - total_invested,
        money_multiple=final_value / total_invested if total_invested else float("nan"),
        xirr=rate,
        n_contributions=len(contributions),
        start=s.index[0],
        end=s.index[-1],
        contributions=contrib_series,
    )


def _xirr(contributions: pd.Series, final_value: float, final_date, guess: float = 0.1) -> float:
    """Money-weighted annualised return (XIRR) via Newton's method with bisection backup.

    Cashflows: each contribution is money *out* (negative), the final portfolio value is
    money *in* (positive) on ``final_date``.
    """
    dates = list(contributions.index) + [pd.Timestamp(final_date)]
    amounts = [-float(a) for a in contributions.values] + [float(final_value)]
    t0 = dates[0]
    years = np.array([(d - t0).days / 365.25 for d in dates])
    amounts = np.array(amounts)

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** years))

    # Newton's method
    rate = guess
    for _ in range(100):
        f = npv(rate)
        # numerical derivative
        df = (npv(rate + 1e-6) - f) / 1e-6
        if df == 0 or not np.isfinite(df):
            break
        step = f / df
        rate -= step
        if not np.isfinite(rate) or rate <= -0.999:
            rate = -0.5
            break
        if abs(step) < 1e-8:
            return float(rate)

    # Bisection fallback on a wide bracket
    lo, hi = -0.9999, 10.0
    flo, fhi = npv(lo), npv(hi)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        fmid = npv(mid)
        if abs(fmid) < 1e-6:
            return float(mid)
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return float((lo + hi) / 2)
