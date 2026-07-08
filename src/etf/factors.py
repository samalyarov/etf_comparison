"""Factor-model portfolio analysis (pure, no DB / no network).

Two complementary views of factor exposure, plus a forward purchasing-strategy scenario
modeller for a factor-ETF blend:

**(B) Regression-estimated exposures** — :func:`factor_exposures` regresses a portfolio's
*excess* returns on the Fama/French-Carhart factor returns to recover its factor **betas**
(loadings), the annualised **alpha**, R²/adjusted-R², and per-coefficient standard errors and
t-statistics. This tells you what a *built* portfolio actually loads on, whatever ETFs it
holds.

**(A) Factor-ETF building blocks** — :func:`sleeve_contributions` decomposes the realised
growth of a blend of dedicated factor ETFs (value / momentum / quality / size / min-vol) into
each sleeve's additive contribution, so you can see which factor sleeve drove the result.

**Purchasing-strategy scenarios** — :func:`plan_scenarios` projects a lump-sum + recurring
plan applied to the blend into a **best / base / worst** percentile fan (reusing
:mod:`etf.projection`'s bootstrap/Monte-Carlo engine) and an explicit **market-crash**
scenario that replays the blend's own worst historical drawdown-and-recovery path onto the
plan. This is the forward, plan-level scenario fan only — the formal risk engine (VaR +
historical stress replays) is a separate later task, so no VaR is computed here.

Method notes (researched; see ``ingest/kenfrench.py`` for the data + convention):
- The market factor is an **excess** return (Mkt-RF), so the regression subtracts the same
  risk-free rate ``RF`` from the portfolio return — the standard Fama-French convention.
- Monthly frequency (academic standard; matches the monthly factor series and projection).
- Plain **OLS** with classical (homoskedastic) standard errors. HAC/Newey-White SEs are a
  documented optional refinement; for buy-and-hold factor loadings plain OLS with reported
  SEs is the conventional, transparent choice and keeps the module dependency-free (numpy
  only — no statsmodels/scipy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import portfolio, projection

# Fama/French 5-factor set and the Carhart momentum extension. RF is the risk-free rate
# (not a regressor — it converts the portfolio to an excess return).
FF5_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
CARHART_FACTORS = [*FF5_FACTORS, "WML"]
RF_COLUMN = "RF"

# Plain-English gloss for each factor loading (used by the UI's "read of the tilt").
FACTOR_MEANING = {
    "Mkt-RF": "market (equity beta)",
    "SMB": "size — small minus big (small-cap tilt if positive)",
    "HML": "value — high minus low book/price (value tilt if positive)",
    "RMW": "profitability — robust minus weak (quality tilt if positive)",
    "CMA": "investment — conservative minus aggressive (if positive)",
    "WML": "momentum — winners minus losers (momentum tilt if positive)",
}

MONTHS_PER_YEAR = 12


class InsufficientData(ValueError):
    """Raised when portfolio/factor overlap is too small to fit the regression."""


@dataclass(frozen=True)
class FactorRegression:
    """Result of regressing a portfolio's excess returns on factor returns.

    ``betas`` are the factor loadings; ``alpha`` is the per-period intercept and
    ``alpha_annual`` its annualised equivalent. ``std_errors`` and ``t_stats`` carry one entry
    per factor plus ``"alpha"``. All returns are decimals.
    """

    betas: dict[str, float]
    alpha: float
    alpha_annual: float
    r_squared: float
    adj_r_squared: float
    std_errors: dict[str, float]
    t_stats: dict[str, float]
    n_obs: int
    factors: list[str]
    frequency: str = "monthly"

    def dominant_tilts(self, threshold: float = 0.1) -> list[tuple[str, float]]:
        """Factors whose |beta| exceeds ``threshold``, largest-magnitude first (for the UI)."""
        tilts = [(f, b) for f, b in self.betas.items()
                 if f != "Mkt-RF" and abs(b) >= threshold]
        return sorted(tilts, key=lambda kv: abs(kv[1]), reverse=True)


def to_monthly_returns(prices: pd.Series) -> pd.Series:
    """Month-end simple returns from a (total-return) price series — the regression input.

    Thin wrapper over :func:`projection.monthly_returns` so callers have one obvious entry
    point and the month-end index lines up with the Ken French monthly factor dates.
    """
    return projection.monthly_returns(prices)


def factor_exposures(portfolio_returns: pd.Series, factor_returns: pd.DataFrame, *,
                     factors: list[str] | None = None, rf_column: str = RF_COLUMN,
                     frequency: str = "monthly", min_obs: int | None = None) -> FactorRegression:
    """OLS-regress a portfolio's excess returns on factor returns; return loadings + stats.

    ``portfolio_returns`` and ``factor_returns`` are date-indexed period returns (decimals).
    The market factor is an excess return, so ``rf_column`` (if present) is subtracted from the
    portfolio return to form the dependent variable. Only factors present in
    ``factor_returns`` are used. Dates are inner-aligned and rows with any NaN dropped.

    Raises :class:`InsufficientData` when fewer than ``min_obs`` aligned observations remain
    (default ``len(factors) + 2``) so a constraint is never fit on too little data.
    """
    want = factors if factors is not None else CARHART_FACTORS
    use = [f for f in want if f in factor_returns.columns]
    if not use:
        raise InsufficientData("None of the requested factors are present in the data.")

    y = pd.to_numeric(portfolio_returns, errors="coerce")
    y.index = pd.to_datetime(y.index)
    fr = factor_returns.copy()
    fr.index = pd.to_datetime(fr.index)

    cols = use + ([rf_column] if rf_column in fr.columns else [])
    joined = pd.concat([y.rename("_port"), fr[cols]], axis=1, join="inner").dropna()
    need = min_obs if min_obs is not None else len(use) + 2
    if len(joined) < max(need, len(use) + 2):
        raise InsufficientData(
            f"Only {len(joined)} overlapping periods; need at least "
            f"{max(need, len(use) + 2)} to fit {len(use)} factors.")

    excess = joined["_port"].to_numpy()
    if rf_column in joined.columns:
        excess = excess - joined[rf_column].to_numpy()
    X = joined[use].to_numpy()

    n = len(excess)
    k = len(use) + 1  # + intercept
    design = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(design, excess, rcond=None)
    resid = excess - design @ beta
    rss = float(resid @ resid)
    tss = float(((excess - excess.mean()) ** 2).sum())
    dof = max(n - k, 1)
    sigma2 = rss / dof

    # Classical OLS covariance: sigma^2 * (X'X)^-1. Pseudo-inverse guards near-collinearity.
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0, beta / se, np.nan)

    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    adj_r2 = (1.0 - (1.0 - r2) * (n - 1) / dof) if tss > 0 and dof > 0 else float("nan")

    per = MONTHS_PER_YEAR if frequency == "monthly" else 252
    alpha = float(beta[0])
    betas = {f: float(b) for f, b in zip(use, beta[1:])}
    std_errors = {"alpha": float(se[0]), **{f: float(s) for f, s in zip(use, se[1:])}}
    t_stats = {"alpha": float(tvals[0]), **{f: float(t) for f, t in zip(use, tvals[1:])}}

    return FactorRegression(
        betas=betas, alpha=alpha,
        alpha_annual=float((1.0 + alpha) ** per - 1.0),
        r_squared=float(r2), adj_r_squared=float(adj_r2),
        std_errors=std_errors, t_stats=t_stats,
        n_obs=n, factors=use, frequency=frequency,
    )


def sleeve_contributions(prices: pd.DataFrame, weights: dict[str, float],
                         initial: float = 100.0) -> pd.DataFrame:
    """Additive per-sleeve contribution to a **buy-and-hold** blend's growth of ``initial``.

    For each sleeve the blend allocates ``initial * w_i`` at the common start date and lets it
    ride; its end value is that times its own growth factor. Contributions are therefore
    exactly additive: ``sum(contribution) == final_value - initial``. (Periodic rebalancing
    makes the split path-dependent; buy-and-hold gives the clean, honest decomposition, so it
    is reported here — the rebalanced *total* is still available via
    :func:`portfolio.blend_index`.)

    Returns a DataFrame indexed by ISIN with columns: ``weight``, ``total_return`` (the
    sleeve's own return), ``start_value``, ``end_value``, ``contribution`` (to the blend gain)
    and ``contribution_share`` (of the total gain).
    """
    P, w = portfolio._aligned(prices, weights)
    if P.empty:
        return pd.DataFrame(columns=["weight", "total_return", "start_value", "end_value",
                                     "contribution", "contribution_share"])
    growth = (P.iloc[-1] / P.iloc[0]).to_numpy()
    start_val = w * initial
    end_val = start_val * growth
    contribution = end_val - start_val
    total_gain = float(end_val.sum() - initial)
    share = contribution / total_gain if abs(total_gain) > 1e-12 else np.full(len(w), np.nan)
    out = pd.DataFrame({
        "weight": w,
        "total_return": growth - 1.0,
        "start_value": start_val,
        "end_value": end_val,
        "contribution": contribution,
        "contribution_share": share,
    }, index=P.columns)
    return out.sort_values("contribution", ascending=False)


# --------------------------------------------------------------------------- scenarios

@dataclass(frozen=True)
class CrashScenario:
    """A market-crash replay of the plan: the blend's worst real drawdown+recovery path."""

    label: str
    timeline: pd.Series        # net worth by month offset (1..N)
    trough_value: float        # lowest net worth during the crash window
    trough_month: int          # month offset of the trough
    final_value: float
    window_drawdown: float     # peak-to-trough of the replayed historical window (<= 0)
    window_months: int         # length of the drawdown+recovery window applied


@dataclass(frozen=True)
class PlanScenarios:
    """Forward outcomes of a purchasing strategy applied to a factor-ETF blend."""

    fan: pd.DataFrame              # projection percentile fan (p5..p95, invested)
    dates: pd.DatetimeIndex
    worst: float                   # p5 final net worth
    base: float                    # p50 final net worth
    best: float                    # p95 final net worth
    invested: float                # total contributed over the horizon
    method: str
    horizon_years: int
    crash: CrashScenario | None = None
    monthly_return_used: float = float("nan")  # base monthly drift used for the crash tail


def worst_crash_window(monthly_rets: pd.Series) -> tuple[np.ndarray, float, int]:
    """Return the monthly-return sequence of the series' worst drawdown-and-recovery episode.

    Finds the peak-to-trough maximum drawdown of the cumulative wealth index, then extends the
    window to the point where the index first regains the prior peak (or to the series end if
    it never fully recovers). The returned array is the *real* historical monthly returns over
    that window (peak+1 … recovery), so replaying it fabricates nothing. Also returns the
    peak-to-trough drawdown (<= 0) and the window length in months. Empty/edge input yields an
    empty array.
    """
    r = pd.to_numeric(monthly_rets, errors="coerce").dropna()
    if len(r) < 2:
        return np.array([]), 0.0, 0
    wealth = (1.0 + r).cumprod().to_numpy()
    running_peak = np.maximum.accumulate(wealth)
    drawdown = wealth / running_peak - 1.0
    trough = int(np.argmin(drawdown))
    dd = float(drawdown[trough])
    if dd >= -1e-9:  # monotonic / no meaningful drawdown
        return np.array([]), 0.0, 0
    # Peak that preceded the trough.
    peak = int(np.argmax(wealth[: trough + 1]))
    peak_level = wealth[peak]
    # Recovery: first index after the trough that regains the peak level.
    recovery = trough
    for j in range(trough + 1, len(wealth)):
        if wealth[j] >= peak_level:
            recovery = j
            break
    else:
        recovery = len(wealth) - 1
    seq = r.to_numpy()[peak + 1: recovery + 1]
    return seq, dd, len(seq)


def _base_monthly_drift(blend_prices: pd.Series, annual_return: float | None) -> float:
    """Monthly drift for the crash-scenario tail: explicit override, else OLS log-drift."""
    if annual_return is not None:
        ann = annual_return
    else:
        ann = projection.ols_annual_log_drift(blend_prices)
        if not np.isfinite(ann):
            mu, _ = projection._monthly_moments(blend_prices)
            return mu
    return (1.0 + ann) ** (1.0 / MONTHS_PER_YEAR) - 1.0


def crash_scenario(blend_prices: pd.Series, *, start_value: float, monthly: float, years: int,
                   annual_step_up: float = 0.0, annual_return: float | None = None,
                   label: str = "Historical crash replay") -> CrashScenario | None:
    """Apply the blend's worst historical drawdown+recovery to the plan, then base drift.

    The plan contributes monthly from ``start_value``. The first months replay the real
    monthly returns of the blend's worst drawdown-and-recovery window; any remaining months
    grow at the base monthly drift. Returns ``None`` if the series shows no drawdown to replay.
    """
    monthly_rets = projection.monthly_returns(blend_prices)
    seq, dd, wlen = worst_crash_window(monthly_rets)
    if wlen == 0:
        return None
    n_months = int(max(1, min(years, 40))) * MONTHS_PER_YEAR
    base_m = _base_monthly_drift(blend_prices, annual_return)
    contribs = [monthly * ((1 + annual_step_up) ** (k // 12)) for k in range(n_months)]

    value = float(start_value)
    path: list[float] = []
    for k in range(n_months):
        r = seq[k] if k < len(seq) else base_m
        value = value * (1.0 + r) + contribs[k]
        path.append(value)
    timeline = pd.Series(path, index=range(1, n_months + 1), name="crash")

    window_end = min(len(seq), n_months)
    trough_slice = timeline.iloc[:window_end] if window_end else timeline
    trough_month = int(trough_slice.idxmin())
    return CrashScenario(
        label=label, timeline=timeline,
        trough_value=float(trough_slice.min()), trough_month=trough_month,
        final_value=float(timeline.iloc[-1]),
        window_drawdown=dd, window_months=wlen,
    )


def plan_scenarios(blend_prices: pd.Series, *, start_value: float, monthly: float, years: int,
                   method: str = "bootstrap", annual_step_up: float = 0.0,
                   annual_return: float | None = None, n_sims: int = 2000,
                   seed: int = 12345, include_crash: bool = True) -> PlanScenarios:
    """Best/base/worst percentile fan + a market-crash replay for a plan on the blend.

    Reuses :func:`projection.project_plan` for the fan (``worst=p5``, ``base=p50``,
    ``best=p95`` by construction, so ``worst <= base <= best``) and
    :func:`crash_scenario` for the explicit crash path. Returns a :class:`PlanScenarios`.
    """
    proj = projection.project_plan(
        blend_prices, start_value=start_value, monthly=monthly, years=years, method=method,
        annual_step_up=annual_step_up, n_sims=n_sims, annual_return=annual_return, seed=seed,
    )
    last = proj.fan.iloc[-1]
    crash = None
    if include_crash:
        crash = crash_scenario(blend_prices, start_value=start_value, monthly=monthly,
                               years=proj.horizon_years, annual_step_up=annual_step_up,
                               annual_return=annual_return)
    return PlanScenarios(
        fan=proj.fan, dates=proj.dates,
        worst=float(last["p5"]), base=float(last["p50"]), best=float(last["p95"]),
        invested=proj.total_contributed, method=proj.method,
        horizon_years=proj.horizon_years, crash=crash,
        monthly_return_used=_base_monthly_drift(blend_prices, annual_return),
    )


# Universe helper: map the factor-ETF sleeves in the profiles dataset to a canonical factor
# label, so the UI can offer a "value / momentum / quality / size / min-volatility" picker.
SLEEVE_FACTOR_KEYWORDS = {
    # Multifactor is checked FIRST: a multifactor fund legitimately carries value/momentum/
    # quality/size tags, so it must not be mislabelled as a single-factor sleeve.
    "multifactor": "Multifactor",
    "multi-factor": "Multifactor",
    "min-volatility": "Min Volatility",
    "min-vol": "Min Volatility",
    "min vol": "Min Volatility",
    "minimum vol": "Min Volatility",
    "low-volatility": "Min Volatility",
    "low volatility": "Min Volatility",
    "momentum": "Momentum",
    "quality": "Quality",
    "value": "Value",
    "size": "Size",
}


def sleeve_factor_label(tilt: list[str], name: str = "") -> str | None:
    """Best-guess canonical factor label for a sleeve from its tilt tags / name (pure).

    Keyword precedence matters: ``multifactor`` and ``min-volatility`` are matched before the
    single classic factors so a multi-factor or defensive fund (which carries several factor
    tags) is not mislabelled as ``Value``/``Size``/etc.
    """
    hay = " ".join(str(t).lower() for t in (tilt or [])) + " " + name.lower()
    for key, label in SLEEVE_FACTOR_KEYWORDS.items():
        if key in hay:
            return label
    return None
