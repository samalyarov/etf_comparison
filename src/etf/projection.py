"""Forward net-worth projections for a buy-and-hold / DCA plan.

Deliberately simple, per the brief: an **OLS log-trend** extrapolation for a central
estimate, plus a **Monte-Carlo / bootstrap** fan chart for a range of outcomes. No
seasonality, no Prophet — just an honest "if the historical trend continues" projection
with an uncertainty band, up to 40 years.

Two return engines drive the simulation:

* ``bootstrap`` — resample historical monthly returns with replacement (captures the real
  distribution's skew/fat tails; the honest default).
* ``normal`` — draw from a Normal fitted to historical monthly mean/std (smoother, thinner
  tails).

and one deterministic path:

* ``ols`` — extrapolate the OLS slope of log price vs time (the "trend continues" line).

All simulate a monthly contribution plan compounding forward from a starting value. Past
performance does not predict future returns — this is scenario arithmetic, not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
PERCENTILES = [5, 25, 50, 75, 95]


@dataclass
class ProjectionResult:
    """Forward projection of a contribution plan."""

    fan: pd.DataFrame          # index=month offset (1..N), cols: p5,p25,p50,p75,p95, invested
    horizon_years: int
    total_contributed: float
    method: str
    annual_return_used: float  # central annualised return the projection is built on
    dates: pd.DatetimeIndex    # projected calendar dates for the fan index


def monthly_returns(prices: pd.Series) -> pd.Series:
    """Historical monthly simple returns from a (total-return) price series."""
    s = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    if len(s) < 3:
        return pd.Series(dtype=float)
    m = s.resample("ME").last().dropna()
    return m.pct_change().dropna()


def ols_annual_log_drift(prices: pd.Series) -> float:
    """Annualised drift from an OLS fit of log price on time (years).

    This is the "general trend" the brief asks to extrapolate: the slope of a straight line
    through log price, expressed as an annual growth rate ``exp(slope) - 1``.
    """
    s = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    s = s[s > 0]
    if len(s) < 3:
        return float("nan")
    t_years = (s.index - s.index[0]).days.to_numpy() / 365.25
    y = np.log(s.to_numpy())
    slope = np.polyfit(t_years, y, 1)[0]  # log-price per year
    return float(np.exp(slope) - 1.0)


def _monthly_moments(prices: pd.Series) -> tuple[float, float]:
    r = monthly_returns(prices)
    if r.empty:
        return 0.0, 0.0
    return float(r.mean()), float(r.std(ddof=1))


def project_plan(prices: pd.Series, *, start_value: float, monthly: float, years: int,
                 method: str = "bootstrap", annual_step_up: float = 0.0,
                 n_sims: int = 2000, annual_return: float | None = None,
                 seed: int = 12345) -> ProjectionResult:
    """Project net worth forward for a monthly-contribution plan.

    Starting from ``start_value``, contribute ``monthly`` (optionally stepped up each year)
    for ``years`` years, compounding monthly. ``method`` selects the return engine
    (``bootstrap`` / ``normal`` / ``ols``). ``annual_return`` overrides the return
    assumption (e.g. to project on CAGR-after-cost). Returns a percentile fan over time.
    """
    years = int(max(1, min(years, 40)))
    n_months = years * MONTHS_PER_YEAR
    last_date = (prices.dropna().index[-1] if len(prices.dropna())
                 else pd.Timestamp.today().normalize())
    dates = pd.DatetimeIndex([last_date + pd.DateOffset(months=k) for k in range(1, n_months + 1)])

    # Contribution schedule (per month), with annual step-up.
    contribs = np.array([monthly * ((1 + annual_step_up) ** (k // 12)) for k in range(n_months)])
    invested = start_value + np.cumsum(contribs)

    mu_m, sigma_m = _monthly_moments(prices)
    hist = monthly_returns(prices).to_numpy()

    if method == "ols":
        ann = annual_return if annual_return is not None else ols_annual_log_drift(prices)
        r_m = (1.0 + (ann if np.isfinite(ann) else 0.0)) ** (1 / 12) - 1.0
        value = start_value
        path = []
        for k in range(n_months):
            value = value * (1 + r_m) + contribs[k]
            path.append(value)
        col = np.array(path)
        fan = pd.DataFrame({f"p{p}": col for p in PERCENTILES}, index=range(1, n_months + 1))
        used = ann if np.isfinite(ann) else 0.0
    else:
        rng = np.random.default_rng(seed)
        if method == "normal" or len(hist) < 12:
            ann_override = annual_return
            base_mu = ((1 + ann_override) ** (1 / 12) - 1.0) if ann_override is not None else mu_m
            draws = rng.normal(base_mu, sigma_m or 0.0, size=(n_sims, n_months))
        else:  # bootstrap historical monthly returns (default)
            shift = 0.0
            if annual_return is not None:  # recentre bootstrap on a target annual return
                target_mu = (1 + annual_return) ** (1 / 12) - 1.0
                shift = target_mu - mu_m
            idx = rng.integers(0, len(hist), size=(n_sims, n_months))
            draws = hist[idx] + shift

        values = np.full(n_sims, float(start_value))
        paths = np.empty((n_sims, n_months))
        for k in range(n_months):
            values = values * (1 + draws[:, k]) + contribs[k]
            paths[:, k] = values
        pct = np.percentile(paths, PERCENTILES, axis=0)  # shape (len(PERCENTILES), n_months)
        fan = pd.DataFrame({f"p{p}": pct[i] for i, p in enumerate(PERCENTILES)},
                           index=range(1, n_months + 1))
        used = (annual_return if annual_return is not None
                else (1 + mu_m) ** 12 - 1.0)

    fan["invested"] = invested
    return ProjectionResult(fan=fan, horizon_years=years,
                            total_contributed=float(invested[-1]),
                            method=method, annual_return_used=float(used), dates=dates)
