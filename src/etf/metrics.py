"""Analytics: returns, growth, and risk metrics computed over price series.

All functions are pure and operate on a pandas Series of prices indexed by a
DatetimeIndex (use ``adj_close`` for a total-return basis). They make no DB calls.

Convention: ``TRADING_DAYS = 252`` for annualisation of daily statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

TRADING_DAYS = 252

# Standard comparison horizons, as (label, relativedelta-or-None). None = since inception.
STANDARD_PERIODS: dict[str, relativedelta | None] = {
    "1M": relativedelta(months=1),
    "3M": relativedelta(months=3),
    "6M": relativedelta(months=6),
    "YTD": None,  # handled specially
    "1Y": relativedelta(years=1),
    "3Y": relativedelta(years=3),
    "5Y": relativedelta(years=5),
    "10Y": relativedelta(years=10),
    "Max": None,  # since inception
}


def _clean(prices: pd.Series) -> pd.Series:
    """Drop NaNs and ensure a sorted, numeric, date-indexed series."""
    s = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    return s[s > 0]


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily returns."""
    return _clean(prices).pct_change().dropna()


def has_clean_history(prices: pd.Series, max_daily_move: float = 0.5) -> bool:
    """Cheap sanity check that a price series isn't corrupted.

    A genuine ETF never moves more than ~15-20% in a single day; a swing beyond
    ``max_daily_move`` (default 50%) signals a bad split/adjustment or a mis-scaled
    (GBX vs GBP) series that would otherwise produce nonsense returns. Returns ``False``
    for such series so callers can exclude them from rankings.
    """
    r = daily_returns(prices)
    if r.empty:
        return False
    return bool(r.abs().max() <= max_daily_move)


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns."""
    s = _clean(prices)
    return np.log(s / s.shift(1)).dropna()


def total_return(prices: pd.Series) -> float:
    """Cumulative return over the full series (last / first - 1)."""
    s = _clean(prices)
    if len(s) < 2:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def cagr(prices: pd.Series) -> float:
    """Compound annual growth rate over the full series."""
    s = _clean(prices)
    if len(s) < 2:
        return float("nan")
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0)


def annualized_volatility(prices: pd.Series) -> float:
    """Annualised standard deviation of daily returns."""
    r = daily_returns(prices)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def annualized_return(prices: pd.Series) -> float:
    """Annualised mean daily return (arithmetic; used for Sharpe)."""
    r = daily_returns(prices)
    if r.empty:
        return float("nan")
    return float(r.mean() * TRADING_DAYS)


def sharpe_ratio(prices: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio. ``risk_free`` is an annual rate (e.g. 0.03)."""
    vol = annualized_volatility(prices)
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return float((annualized_return(prices) - risk_free) / vol)


def sortino_ratio(prices: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio (downside deviation in the denominator)."""
    r = daily_returns(prices)
    if len(r) < 2:
        return float("nan")
    downside = r[r < 0]
    if downside.empty:
        return float("nan")
    downside_dev = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if downside_dev == 0:
        return float("nan")
    return float((annualized_return(prices) - risk_free) / downside_dev)


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Return the drawdown curve (fraction below running peak, <= 0)."""
    s = _clean(prices)
    running_max = s.cummax()
    return s / running_max - 1.0


def max_drawdown(prices: pd.Series) -> float:
    """Worst peak-to-trough decline over the series (a negative number)."""
    dd = drawdown_series(prices)
    if dd.empty:
        return float("nan")
    return float(dd.min())


def normalize_to_100(prices: pd.Series) -> pd.Series:
    """Rebase a series to start at 100 (for growth-comparison charts)."""
    s = _clean(prices)
    if s.empty:
        return s
    return s / s.iloc[0] * 100.0


def period_return(prices: pd.Series, period: str, asof=None) -> float:
    """Total return over a named period ending at ``asof`` (default: last date).

    Uses the price on or before the window start (``Series.asof``) so weekends /
    holidays don't cause a KeyError. Returns NaN when there isn't enough history.
    """
    s = _clean(prices)
    if len(s) < 2:
        return float("nan")
    asof_ts = pd.Timestamp(asof) if asof is not None else s.index[-1]
    end_val = s.asof(asof_ts)
    if pd.isna(end_val):
        return float("nan")

    if period == "YTD":
        start_ts = pd.Timestamp(year=asof_ts.year, month=1, day=1)
    elif period == "Max":
        return float(end_val / s.iloc[0] - 1.0)
    else:
        delta = STANDARD_PERIODS.get(period)
        if delta is None:
            raise ValueError(f"Unknown period: {period}")
        start_ts = asof_ts - delta

    if start_ts < s.index[0]:
        return float("nan")  # not enough history for a fair comparison
    start_val = s.asof(start_ts)
    if pd.isna(start_val) or start_val <= 0:
        return float("nan")
    return float(end_val / start_val - 1.0)


def period_returns(prices: pd.Series, asof=None) -> dict[str, float]:
    """Return a dict of {period_label: total_return} over the standard horizons."""
    return {label: period_return(prices, label, asof=asof) for label in STANDARD_PERIODS}


def summary(prices: pd.Series, risk_free: float = 0.0, ter: float | None = None) -> dict:
    """One-stop summary of headline metrics for a single ETF."""
    out = {
        "start": _clean(prices).index[0].date() if len(_clean(prices)) else None,
        "end": _clean(prices).index[-1].date() if len(_clean(prices)) else None,
        "total_return": total_return(prices),
        "cagr": cagr(prices),
        "volatility": annualized_volatility(prices),
        "sharpe": sharpe_ratio(prices, risk_free),
        "sortino": sortino_ratio(prices, risk_free),
        "max_drawdown": max_drawdown(prices),
        "ter": ter,
    }
    if ter is not None and np.isfinite(out["cagr"]):
        out["cagr_after_ter"] = out["cagr"] - ter
    return out


def correlation_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Correlation of daily returns across a date x isin price matrix.

    Aligns on common dates (inner join of non-null returns) so pairs are comparable.
    """
    if matrix is None or matrix.empty:
        return pd.DataFrame()
    returns = matrix.sort_index().pct_change()
    return returns.corr()


def rolling_returns(prices: pd.Series, window_years: int = 1) -> pd.Series:
    """Rolling total return over a trailing window (approx. trading days)."""
    s = _clean(prices)
    window = int(window_years * TRADING_DAYS)
    if len(s) <= window:
        return pd.Series(dtype=float)
    return s / s.shift(window) - 1.0


def rolling_volatility(prices: pd.Series, window: int = 63) -> pd.Series:
    """Annualised rolling volatility (default window ~3 months of trading days)."""
    r = daily_returns(prices)
    if len(r) <= window:
        return pd.Series(dtype=float)
    return r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def calendar_year_returns(prices: pd.Series) -> pd.Series:
    """Total return within each calendar year, indexed by year (int).

    The first (typically partial) year is measured from the first available price.
    """
    s = _clean(prices)
    if len(s) < 2:
        return pd.Series(dtype=float)
    year_end = s.resample("YE").last()
    returns = year_end.pct_change()
    returns.iloc[0] = year_end.iloc[0] / s.iloc[0] - 1.0
    returns.index = returns.index.year
    return returns


# Notable market regimes for a stress lens (peak-to-trough / recovery windows, approx.).
STRESS_REGIMES: dict[str, tuple[str, str]] = {
    "2018 Q4 selloff": ("2018-09-20", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "COVID recovery": ("2020-03-23", "2020-08-18"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
    "2023–24 rally": ("2022-10-12", "2024-03-28"),
}


def _aligned_returns(prices: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Inner-joined daily returns of a fund and a benchmark on their common dates."""
    a = daily_returns(prices).rename("fund")
    b = daily_returns(benchmark).rename("bench")
    return pd.concat([a, b], axis=1, join="inner").dropna()


def beta(prices: pd.Series, benchmark: pd.Series) -> float:
    """Sensitivity of the fund's daily returns to the benchmark's (cov / var)."""
    df = _aligned_returns(prices, benchmark)
    if len(df) < 2:
        return float("nan")
    var_b = float(df["bench"].var(ddof=1))
    if var_b == 0:
        return float("nan")
    return float(df["fund"].cov(df["bench"]) / var_b)


def tracking_error(prices: pd.Series, benchmark: pd.Series) -> float:
    """Annualised standard deviation of the fund-minus-benchmark daily return."""
    df = _aligned_returns(prices, benchmark)
    if len(df) < 2:
        return float("nan")
    active = df["fund"] - df["bench"]
    return float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(prices: pd.Series, benchmark: pd.Series) -> float:
    """Annualised active return divided by tracking error."""
    df = _aligned_returns(prices, benchmark)
    if len(df) < 2:
        return float("nan")
    active = df["fund"] - df["bench"]
    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if te == 0:
        return float("nan")
    return float(active.mean() * TRADING_DAYS / te)


def up_down_capture(prices: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Up/down capture ratios: mean fund return vs mean benchmark return on up/down days.

    >1 up-capture means the fund gains more than the benchmark on its up days; <1 down-
    capture means it falls less on the benchmark's down days (both desirable).
    """
    df = _aligned_returns(prices, benchmark)
    if len(df) < 2:
        return float("nan"), float("nan")
    up = df[df["bench"] > 0]
    down = df[df["bench"] < 0]
    up_cap = float(up["fund"].mean() / up["bench"].mean()) if len(up) and up["bench"].mean() else float("nan")
    down_cap = float(down["fund"].mean() / down["bench"].mean()) if len(down) and down["bench"].mean() else float("nan")
    return up_cap, down_cap


def horizon_return_stats(prices: pd.Series, window_years: int = 10) -> dict:
    """Distribution of total returns over every rolling ``window_years`` window.

    More honest for buy-and-hold than a single trailing number: "over any N-year window
    this fund returned X–Y%". Returns min/p25/median/p75/max and the window count.
    """
    rr = rolling_returns(prices, window_years).dropna()
    if rr.empty:
        return {"n": 0}
    return {
        "n": int(len(rr)),
        "min": float(rr.min()), "p25": float(rr.quantile(0.25)),
        "median": float(rr.median()), "p75": float(rr.quantile(0.75)),
        "max": float(rr.max()),
    }


def regime_returns(prices: pd.Series, regimes: dict | None = None) -> pd.Series:
    """Total return of the fund within each named market regime window."""
    regimes = regimes or STRESS_REGIMES
    s = _clean(prices)
    out = {}
    for name, (start, end) in regimes.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        if s.empty or s.index[0] > start_ts or s.index[-1] < start_ts:
            out[name] = float("nan")
            continue
        start_val = s.asof(start_ts)
        end_val = s.asof(end_ts)
        out[name] = (float(end_val / start_val - 1.0)
                     if pd.notna(start_val) and pd.notna(end_val) and start_val > 0
                     else float("nan"))
    return pd.Series(out)


def monthly_returns_matrix(prices: pd.Series) -> pd.DataFrame:
    """Year x month (1-12) matrix of monthly total returns, for a heatmap."""
    s = _clean(prices)
    if len(s) < 2:
        return pd.DataFrame()
    month_end = s.resample("ME").last()
    mr = month_end.pct_change()
    mr.iloc[0] = month_end.iloc[0] / s.iloc[0] - 1.0
    frame = pd.DataFrame({"year": mr.index.year, "month": mr.index.month, "ret": mr.values})
    return frame.pivot(index="year", columns="month", values="ret").sort_index(ascending=False)
