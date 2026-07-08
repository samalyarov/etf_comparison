"""Bond-income modelling: reinvest-vs-cash-out scenarios and distribution yield.

Fixed income earns its return largely as a **coupon stream**, so the choice between
*reinvesting* distributions and *taking them as cash* matters more than for growth equity —
and it interacts directly with Dutch box-3 tax (see :mod:`etf.tax`). This module turns a
stored price series plus the ``distributions`` table into the two side-by-side scenarios the
Portfolio page shows:

* **(a) Distributions reinvested** — the accumulating-equivalent, total-return path. Each
  coupon buys more units at that day's price; net worth compounds. This equals the stored
  ``adj_close`` (total-return) series when both are built consistently, so we can either take
  ``adj_close`` directly or reconstruct it from ``close`` + distributions (and cross-check).

* **(b) Distributions cashed out** — units are bought once at the start and held; the
  holding tracks the **price-return** path (``close``), and each coupon is paid out and
  accumulated as a *separate* cash pile (uninvested by default). Net worth = holding value +
  accumulated cash income. This is what a retiree drawing the income actually experiences.

All functions are **pure** (a price ``Series`` + a distributions ``DataFrame`` in, results
out) — no DB or network. Amounts are per share, in whatever currency the caller passes; the
app converts to EUR upstream when the EUR toggle is on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BondScenarios:
    """Reinvested vs cashed-out outcomes for a bond holding (per-share prices, EUR amounts)."""

    reinvested: pd.Series = field(repr=False)              # total-return net worth path
    cashout_value: pd.Series = field(repr=False)           # price-only holding value path
    cashout_income: pd.Series = field(repr=False)          # cumulative cash income (step)
    cashout_total: pd.Series = field(repr=False)           # holding value + cumulative income
    income_events: pd.Series = field(repr=False)           # cash paid per ex-date
    initial: float
    units: float                                           # units held under cash-out
    total_income: float                                    # sum of all cash coupons
    final_reinvested: float
    final_cashout_value: float
    final_cashout_total: float
    dist_yield_ttm: float                                  # trailing-12m income / current price


def _clean_price(price: pd.Series) -> pd.Series:
    """Sorted, positive, numeric price series indexed by date."""
    s = pd.to_numeric(price, errors="coerce").dropna().sort_index()
    return s[s > 0]


def _amounts(distributions: pd.DataFrame | pd.Series | None) -> pd.Series:
    """Extract an ex-date-indexed Series of per-share cash amounts from a distributions frame."""
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    if distributions is None or len(distributions) == 0:
        return empty
    if isinstance(distributions, pd.Series):
        s = distributions
    else:
        col = "amount" if "amount" in distributions.columns else distributions.columns[0]
        s = distributions[col]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return empty
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def distribution_yield_ttm(price: pd.Series,
                           distributions: pd.DataFrame | pd.Series | None,
                           asof: pd.Timestamp | None = None) -> float:
    """Trailing-12-month distribution yield: coupons paid in the last year / current price.

    ``asof`` defaults to the last price date. Returns 0.0 when there is no price or no
    distribution in the trailing window (e.g. an accumulating fund).
    """
    p = _clean_price(price)
    amt = _amounts(distributions)
    if p.empty or amt.empty:
        return 0.0
    asof = asof or p.index[-1]
    window_start = asof - pd.DateOffset(years=1)
    ttm = amt[(amt.index > window_start) & (amt.index <= asof)].sum()
    current = float(p.asof(asof)) if not p.empty else 0.0
    if current <= 0:
        return 0.0
    return float(ttm) / current


def reinvest_from_price(price: pd.Series,
                        distributions: pd.DataFrame | pd.Series | None,
                        initial: float = 10_000.0) -> pd.Series:
    """Reconstruct a total-return net-worth path from price + distributions (reinvested).

    Buys ``initial / price[0]`` units at the start; on each ex-date the cash coupon
    (``units * amount``) buys new units at that day's price. The resulting ``units * price``
    is the accumulating-equivalent path — used to cross-check against the stored
    ``adj_close`` total-return series.
    """
    p = _clean_price(price)
    if p.empty:
        return pd.Series(dtype=float)
    amt = _amounts(distributions)
    # Iterate the price index, reinvesting each coupon on the first trading day at/after its
    # ex-date; a coupon is reinvested exactly once (tracked via ``pending``).
    units = initial / float(p.iloc[0])
    out = pd.Series(index=p.index, dtype=float)
    pending = amt.copy()
    prev_d = None
    for d, px_t in p.items():
        if not pending.empty:
            if prev_d is None:
                mask = pending.index <= d
            else:
                mask = (pending.index > prev_d) & (pending.index <= d)
            for coupon in pending[mask]:
                units += (units * float(coupon)) / float(px_t)
            pending = pending[~mask]
        out.loc[d] = units * float(px_t)
        prev_d = d
    return out


def income_scenarios(price: pd.Series,
                     distributions: pd.DataFrame | pd.Series | None,
                     *,
                     adj_close: pd.Series | None = None,
                     initial: float = 10_000.0) -> BondScenarios:
    """Build the reinvested-vs-cashed-out scenarios for a bond holding.

    ``price`` is the price-return series (raw ``close``); ``distributions`` the ex-date cash
    amounts per share. If ``adj_close`` (the stored total-return series) is given it is used
    directly for the reinvested path; otherwise the path is reconstructed from price +
    distributions via :func:`reinvest_from_price`. ``initial`` is the amount invested at the
    start of the common window. Returns a :class:`BondScenarios`.
    """
    p = _clean_price(price)
    if p.empty:
        empty = pd.Series(dtype=float)
        return BondScenarios(empty, empty, empty, empty, empty, initial, 0.0, 0.0,
                             initial, initial, initial, 0.0)
    amt = _amounts(distributions)
    start, end = p.index[0], p.index[-1]
    # Only coupons paid during the holding window count.
    if not amt.empty:
        amt = amt[(amt.index >= start) & (amt.index <= end)]

    # --- (b) cash-out: buy once, hold; coupons paid out as cash and accumulated ---
    units = initial / float(p.iloc[0])
    cashout_value = units * p
    # Cash paid per ex-date, mapped onto the next available trading day.
    events: dict[pd.Timestamp, float] = {}
    for d, coupon in amt.items():
        pos = p.index.searchsorted(d)
        if pos >= len(p.index):
            continue
        day = p.index[pos]
        events[day] = events.get(day, 0.0) + units * float(coupon)
    income_events = pd.Series(events, dtype=float).sort_index()
    cum = income_events.reindex(p.index).fillna(0.0).cumsum()
    cashout_total = cashout_value + cum

    # --- (a) reinvested: total-return path ---
    if adj_close is not None:
        a = _clean_price(adj_close).reindex(p.index).ffill().dropna()
        if not a.empty:
            reinvested = initial * a / float(a.iloc[0])
        else:
            reinvested = reinvest_from_price(p, amt, initial)
    else:
        reinvested = reinvest_from_price(p, amt, initial)
    reinvested = reinvested.reindex(p.index).ffill()

    total_income = float(income_events.sum()) if not income_events.empty else 0.0
    return BondScenarios(
        reinvested=reinvested,
        cashout_value=cashout_value,
        cashout_income=cum,
        cashout_total=cashout_total,
        income_events=income_events,
        initial=float(initial),
        units=float(units),
        total_income=total_income,
        final_reinvested=float(reinvested.iloc[-1]) if not reinvested.empty else initial,
        final_cashout_value=float(cashout_value.iloc[-1]),
        final_cashout_total=float(cashout_total.iloc[-1]),
        dist_yield_ttm=distribution_yield_ttm(p, amt),
    )
