"""Portfolio risk engine (pure, no DB / no network): Value-at-Risk, Expected Shortfall
(CVaR), risk-contribution decomposition, and historical crash-window stress tests.

This is the **formal risk layer** that complements the forward scenario fan in
:mod:`etf.factors` (``plan_scenarios``). Where that module projects a *plan* into a
best/base/worst percentile fan, this one answers the risk-management questions on a *built*
portfolio's realised return distribution:

* **How bad is a bad period?** — Value-at-Risk (VaR) and Conditional VaR / Expected Shortfall
  (CVaR) at a chosen confidence, by three independent methods (they should broadly agree; when
  they don't, the disagreement is the signal — usually fat tails).
* **Which funds drive the risk?** — a marginal/component (Euler) decomposition of portfolio
  volatility and VaR, so a small weight in a wild fund is not mistaken for a small risk.
* **What did real crashes do to *these* weights?** — replay of named historical crash windows
  (GFC, euro crisis, COVID, 2022 rate shock, …) against the current target weights, reporting
  drawdown, worst single day, and time-to-recovery from real ``adj_close`` history.

Method notes (researched — see the module docstrings of each function):

* **Sign convention.** Returns are signed (a loss is negative). VaR and CVaR are reported as
  **positive loss magnitudes** (a 95% VaR of 0.02 means "a 1-in-20 period loses ≥ 2%").
* **Confidence ``c``.** The loss exceeded with probability ``1 − c``. ``c = 0.95`` → the 5%
  worst tail.
* **VaR is not a worst case.** It is the *threshold* of the tail, silent about how bad the tail
  gets beyond it — hence CVaR (the tail's *mean*), which is coherent (sub-additive) and the
  Basel-endorsed successor. Both are estimates of the past distribution, not a forecast; the UI
  carries these caveats (cf. Taleb's critique of VaR).
* **Horizon √t scaling.** Per-period figures scale to an ``horizon``-period figure with the
  Basel *square-root-of-time* rule: the volatility term scales by ``√horizon`` and the mean
  (drift) term linearly. This assumes **i.i.d.** returns (no autocorrelation, no vol
  clustering) — a documented approximation, reasonable for the short horizons risk desks use
  and increasingly optimistic as the horizon grows.

Dependencies: numpy / pandas only. The standard-normal quantile ``Φ⁻¹`` and density ``φ`` come
from the stdlib :class:`statistics.NormalDist` (no scipy dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import portfolio

_N = NormalDist()  # standard normal — Φ⁻¹ via .inv_cdf, φ via .pdf

DEFAULT_CONFIDENCES = (0.95, 0.99)

# Method labels used across the summary table and the UI.
METHOD_HISTORICAL = "Historical"
METHOD_PARAMETRIC = "Parametric (Gaussian)"
METHOD_CORNISH_FISHER = "Cornish-Fisher (fat-tail)"
METHOD_MONTE_CARLO = "Monte-Carlo"


# --------------------------------------------------------------------------- portfolio series

def portfolio_returns(prices: pd.DataFrame,
                      weights: dict[str, float]) -> tuple[pd.Series, pd.DataFrame, np.ndarray,
                                                          list[str]]:
    """Target-weighted daily return series of a blend, plus its aligned building blocks.

    Reuses :func:`portfolio._aligned` (no re-implementation of alignment / normalisation): the
    price matrix is inner-joined on dates every held fund has data and the weights renormalised
    to sum to 1. The portfolio return is then ``r_p = R · w`` — i.e. the blend held at the
    **fixed target weights** every day (daily-rebalanced). Fixed weights are what make the
    component-risk (Euler) decomposition exact and keep VaR and contribution-to-risk mutually
    consistent.

    Returns ``(r_p, R, w, cols)``: the portfolio daily returns, the aligned per-fund daily
    return matrix ``R``, the normalised weight vector ``w`` and the column ISINs. All are empty
    when there is no shared history.
    """
    P, w = portfolio._aligned(prices, weights)
    if P.empty:
        return pd.Series(dtype=float), pd.DataFrame(), np.array([]), []
    cols = list(P.columns)
    R = P.pct_change().dropna(how="any")
    if R.empty:
        return pd.Series(dtype=float), R, w, cols
    r_p = pd.Series(R.to_numpy() @ w, index=R.index, name="portfolio")
    return r_p, R, w, cols


def portfolio_wealth(prices: pd.DataFrame, weights: dict[str, float],
                     initial: float = 100.0) -> pd.Series:
    """Wealth index (growth of ``initial``) of the target-weighted, daily-rebalanced blend.

    Built from :func:`portfolio_returns` so it holds the *current target weights* through every
    period — the faithful object to slice a historical crash window against. Indexed on the
    aligned price dates, starting at ``initial`` on the first common date.
    """
    r_p, _R, _w, _cols = portfolio_returns(prices, weights)
    P, _ = portfolio._aligned(prices, weights)
    if P.empty or r_p.empty:
        return pd.Series(dtype=float)
    wealth = np.empty(len(P))
    wealth[0] = initial
    wealth[1:] = initial * (1.0 + r_p.to_numpy()).cumprod()
    return pd.Series(wealth, index=P.index, name="wealth")


# --------------------------------------------------------------------------- VaR / CVaR

@dataclass(frozen=True)
class VaRResult:
    """Value-at-Risk and Conditional VaR for one method / confidence / horizon.

    ``var`` and ``cvar`` are **positive loss magnitudes** in return units (fractions) over the
    ``horizon`` (in periods, typically trading days). ``cvar >= var`` always. ``mean`` / ``vol``
    are the per-period moments used; ``skew`` / ``excess_kurtosis`` are populated for the
    Cornish-Fisher method. NaNs signal too little data.
    """

    method: str
    confidence: float
    horizon: int
    var: float
    cvar: float
    mean: float
    vol: float
    n_obs: int
    skew: float = float("nan")
    excess_kurtosis: float = float("nan")


def _moments(r: np.ndarray) -> tuple[float, float, float, float]:
    """Sample mean, std (ddof=1), skewness and *excess* kurtosis of a return array."""
    n = len(r)
    m = float(r.mean())
    s = float(r.std(ddof=1)) if n > 1 else 0.0
    if n < 3 or s == 0:
        return m, s, 0.0, 0.0
    z = (r - m) / s
    # Population g1/g2 moment estimators (excess kurtosis: 0 for a normal).
    skew = float((z ** 3).mean())
    exkurt = float((z ** 4).mean() - 3.0)
    return m, s, skew, exkurt


def _cornish_fisher_quantile(z: float, skew: float, exkurt: float) -> float:
    """Cornish-Fisher expansion of a standard-normal quantile ``z`` for skew / excess kurtosis.

    ``z`` is the Gaussian quantile at the tail probability (e.g. ``Φ⁻¹(1−c) ≈ −1.645``). Returns
    the adjusted quantile; it reduces **exactly** to ``z`` when ``skew == exkurt == 0``.
    """
    return (z
            + (z ** 2 - 1.0) / 6.0 * skew
            + (z ** 3 - 3.0 * z) / 24.0 * exkurt
            - (2.0 * z ** 3 - 5.0 * z) / 36.0 * skew ** 2)


def _scale(mean_part: float, dev_part: float, horizon: int) -> float:
    """Square-root-of-time scaling: drift ``× horizon``, deviation ``× √horizon``.

    ``mean_part`` and ``dev_part`` are the per-period drift and (positive) deviation components
    of a loss; returns the horizon loss magnitude ``dev·√h − mean·h`` (drift reduces the loss).
    """
    h = max(1, int(horizon))
    return dev_part * np.sqrt(h) - mean_part * h


def parametric_var(returns: pd.Series, confidence: float = 0.95, *, horizon: int = 1,
                   cornish_fisher: bool = False) -> VaRResult:
    """Parametric variance-covariance VaR/CVaR (Gaussian, or Cornish-Fisher fat-tailed).

    Gaussian: assumes returns ``~ N(μ, σ)`` so ``VaR = σ·z_c − μ`` and the closed-form normal
    Expected Shortfall ``CVaR = σ·φ(z_c)/(1−c) − μ`` (``z_c = Φ⁻¹(c)``).

    Cornish-Fisher (``cornish_fisher=True``): replaces the Gaussian quantile with its
    Cornish-Fisher expansion for the sample **skewness** and **excess kurtosis**, and the CVaR
    with the Boudt-Peterson-Croux modified Expected Shortfall. This is the money-honest variant:
    real return distributions are left-skewed and fat-tailed, which Gaussian VaR understates. It
    **reduces to the Gaussian formulas at zero skew/kurtosis** by construction.
    """
    r = pd.to_numeric(returns, errors="coerce").dropna().to_numpy()
    n = len(r)
    if n < 2:
        return VaRResult(METHOD_CORNISH_FISHER if cornish_fisher else METHOD_PARAMETRIC,
                         confidence, horizon, float("nan"), float("nan"),
                         float("nan"), float("nan"), n)
    m, s, skew, exkurt = _moments(r)
    var, cvar = _parametric_var_cvar(m, s, confidence, horizon, skew, exkurt, cornish_fisher)
    method = METHOD_CORNISH_FISHER if cornish_fisher else METHOD_PARAMETRIC
    return VaRResult(method, confidence, horizon, var, cvar, m, s, n, skew, exkurt)


def _cf_derivative(z: float, skew: float, exkurt: float) -> float:
    """Derivative ``w'(z)`` of the Cornish-Fisher quantile map (a quadratic in ``z``).

    ``w'(z) = a·z² + b·z + c`` with ``a = K/8 − S²/6``, ``b = S/3`` and
    ``c = 1 − K/8 + 5S²/36`` (Maillard 2012). ``w`` is a valid — monotone increasing — quantile
    map exactly where this is positive.
    """
    a = exkurt / 8.0 - skew ** 2 / 6.0
    b = skew / 3.0
    c = 1.0 - exkurt / 8.0 + 5.0 * skew ** 2 / 36.0
    return a * z * z + b * z + c


def _cornish_fisher_is_valid(skew: float, exkurt: float, z_tail: float | None = None) -> bool:
    """Whether the Cornish-Fisher map is a valid quantile on the tail-to-median interval used.

    The expansion is a proper (monotone-increasing) quantile function only on a bounded
    (skew, kurtosis) domain; pushed past it — which very fat-tailed daily returns reach — it
    folds back and yields nonsensical VaR/ES (Maillard 2012, "A User's Guide to the
    Cornish-Fisher Expansion"). A **left-tail** VaR, however, only evaluates the map on the
    interval from the tail quantile ``z_tail`` (< 0) up to the median (``z = 0``); a fold in the
    irrelevant *right* tail must not reject an otherwise usable left-skewed, fat-tailed
    distribution (nor a sampled Normal whose tiny negative excess kurtosis makes the map fold
    far out in the right tail). So we require monotonicity — ``w'(z) > 0`` — only on
    ``[z_tail, 0]``, the region actually traversed.

    ``w'`` is a quadratic, so it is positive across the closed interval iff it is positive at
    both endpoints and at any interior vertex (its minimum, only when the parabola opens
    upward). ``z_tail`` defaults to the 99% quantile ``Φ⁻¹(0.01)`` — the most extreme standard
    confidence — making the bare predicate a conservative gate; the VaR path passes the actual
    per-confidence quantile so validity is confidence-specific.
    """
    zt = z_tail if z_tail is not None else _N.inv_cdf(0.01)
    if zt >= 0.0:  # defensively coerce to a genuine left tail
        zt = -abs(zt) or -_N.inv_cdf(0.99)
    pts = [zt, 0.0]
    a = exkurt / 8.0 - skew ** 2 / 6.0
    if a > 0.0:  # upward parabola — its minimum sits at the vertex; check it if it's inside
        vertex = -(skew / 3.0) / (2.0 * a)
        if zt < vertex < 0.0:
            pts.append(vertex)
    return all(_cf_derivative(zp, skew, exkurt) > 0.0 for zp in pts)


def _parametric_var_cvar(m: float, s: float, confidence: float, horizon: int,
                         skew: float, exkurt: float,
                         cornish_fisher: bool) -> tuple[float, float]:
    """VaR/CVaR from moments — the Gaussian or Cornish-Fisher formulas (pure, testable).

    Reduces exactly to the Gaussian result when ``skew == exkurt == 0`` (the Cornish-Fisher
    quantile collapses to ``z`` and the Boudt ES bracket to 1). When the Cornish-Fisher
    expansion exceeds its domain of validity (non-monotone at the tail, or a non-positive ES
    bracket), the affected figure is returned as ``NaN`` rather than a fabricated value — the UI
    then shows "—" and defers to the historical/Gaussian estimates.
    """
    tail = 1.0 - confidence
    za = _N.inv_cdf(tail)  # negative lower-tail quantile, e.g. -1.645
    if cornish_fisher:
        if not _cornish_fisher_is_valid(skew, exkurt, za):
            return float("nan"), float("nan")
        g = _cornish_fisher_quantile(za, skew, exkurt)
        var_dev = -s * g  # g < 0 → positive deviation
        bracket = (1.0
                   + g ** 3 * skew / 6.0
                   + (g ** 6 - 9.0 * g ** 4 + 9.0 * g ** 2 + 3.0) * skew ** 2 / 72.0
                   + (g ** 4 - 2.0 * g ** 2 - 1.0) * exkurt / 24.0)
        cvar_dev = s * (_N.pdf(g) / tail) * bracket  # Boudt-Peterson-Croux modified ES
        var = float(_scale(m, var_dev, horizon))
        cvar = float(_scale(m, cvar_dev, horizon))
        # ES must dominate VaR; a degenerate bracket signals the domain was still exceeded.
        if not (bracket > 0.0) or not np.isfinite(cvar) or cvar < var:
            cvar = float("nan")
        return var, cvar
    var_dev = -s * za  # = s · z_c
    cvar_dev = s * _N.pdf(za) / tail  # = s·φ(z_c)/(1−c), closed-form normal ES
    return float(_scale(m, var_dev, horizon)), float(_scale(m, cvar_dev, horizon))


def historical_var(returns: pd.Series, confidence: float = 0.95, *,
                   horizon: int = 1) -> VaRResult:
    """Historical (empirical) VaR/CVaR — no distributional assumption.

    VaR is ``−`` the empirical ``(1−c)`` quantile of the actual returns; CVaR is ``−`` the mean
    of the returns at or below that quantile (the average loss *in* the tail). Makes no
    assumption about the shape of the distribution — its only assumption is that the sampled
    history is representative — but is silent about losses larger than anything observed.
    """
    r = pd.to_numeric(returns, errors="coerce").dropna().to_numpy()
    n = len(r)
    if n < 2:
        return VaRResult(METHOD_HISTORICAL, confidence, horizon, float("nan"), float("nan"),
                         float("nan"), float("nan"), n)
    m = float(r.mean())
    s = float(r.std(ddof=1))
    q = float(np.quantile(r, 1.0 - confidence))  # empirical lower-tail return quantile
    tail = r[r <= q]
    tail_mean = float(tail.mean()) if len(tail) else q
    # Decompose into drift + deviation so √t scaling is drift-aware (§ module docstring).
    return VaRResult(METHOD_HISTORICAL, confidence, horizon,
                     float(_scale(m, m - q, horizon)),
                     float(_scale(m, m - tail_mean, horizon)),
                     m, s, n)


def monte_carlo_var(returns: pd.Series, confidence: float = 0.95, *, horizon: int = 1,
                    method: str = "normal", n_sims: int = 50_000,
                    seed: int = 12345) -> VaRResult:
    """Monte-Carlo VaR/CVaR on a portfolio return series (seedable → deterministic).

    ``method="normal"`` draws ``n_sims`` returns from ``N(μ, σ)`` fitted to the series;
    ``method="bootstrap"`` resamples the historical returns with replacement (preserving their
    real skew/fat tails). VaR/CVaR are the empirical tail statistics of the simulated draws.
    Deterministic for a fixed ``seed``. For a cross-asset (multivariate-normal) simulation that
    respects the fund covariance, see :func:`monte_carlo_var_mvn`.
    """
    r = pd.to_numeric(returns, errors="coerce").dropna().to_numpy()
    n = len(r)
    if n < 2:
        return VaRResult(METHOD_MONTE_CARLO, confidence, horizon, float("nan"), float("nan"),
                         float("nan"), float("nan"), n)
    m = float(r.mean())
    s = float(r.std(ddof=1))
    rng = np.random.default_rng(seed)
    if method == "bootstrap":
        sims = rng.choice(r, size=n_sims, replace=True)
    else:
        sims = rng.normal(m, s, size=n_sims)
    return _empirical_from_draws(sims, confidence, horizon, n, m, s)


def monte_carlo_var_mvn(asset_returns: pd.DataFrame, weights: np.ndarray,
                        confidence: float = 0.95, *, horizon: int = 1, n_sims: int = 50_000,
                        seed: int = 12345) -> VaRResult:
    """Monte-Carlo VaR/CVaR from a **multivariate-normal** fit of the fund returns.

    Fits a mean vector and covariance matrix to the aligned per-fund return matrix, draws
    ``n_sims`` joint scenarios (so the funds' correlation structure is preserved), forms the
    portfolio return ``draws · w`` and takes the empirical tail. More faithful than the
    univariate normal when funds are imperfectly correlated. Seedable → deterministic.
    """
    R = asset_returns.dropna(how="any")
    n = len(R)
    w = np.asarray(weights, dtype=float)
    if n < 2 or R.shape[1] == 0:
        return VaRResult(METHOD_MONTE_CARLO, confidence, horizon, float("nan"), float("nan"),
                         float("nan"), float("nan"), n)
    mu = R.mean().to_numpy()
    cov = R.cov().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mu, cov, size=n_sims)  # (n_sims, n_assets)
    sims = draws @ w
    port = R.to_numpy() @ w
    return _empirical_from_draws(sims, confidence, horizon, n,
                                 float(port.mean()), float(port.std(ddof=1)))


def _empirical_from_draws(sims: np.ndarray, confidence: float, horizon: int, n_obs: int,
                          mean: float, vol: float) -> VaRResult:
    """Build a Monte-Carlo :class:`VaRResult` from simulated portfolio returns.

    The horizon drift is anchored on the estimated per-period ``mean`` (not the simulated
    sample mean), so the √t scaling is consistent with the historical/parametric methods.
    """
    q = float(np.quantile(sims, 1.0 - confidence))
    tail = sims[sims <= q]
    tail_mean = float(tail.mean()) if len(tail) else q
    return VaRResult(METHOD_MONTE_CARLO, confidence, horizon,
                     float(_scale(mean, mean - q, horizon)),
                     float(_scale(mean, mean - tail_mean, horizon)),
                     mean, vol, n_obs)


def var_summary(prices: pd.DataFrame, weights: dict[str, float], *,
                confidences: tuple[float, ...] = DEFAULT_CONFIDENCES, horizon: int = 1,
                n_sims: int = 50_000, seed: int = 12345) -> pd.DataFrame:
    """Method × confidence table of VaR/CVaR for a blend (positive loss fractions).

    Rows: Historical, Parametric (Gaussian), Cornish-Fisher, Monte-Carlo. Columns: ``VaR c%`` /
    ``CVaR c%`` for each confidence. The Monte-Carlo row uses the multivariate-normal fit across
    funds when there are ≥ 2 funds, else a univariate normal of the single series. Empty frame
    when there is no shared history.
    """
    r_p, R, w, _cols = portfolio_returns(prices, weights)
    if r_p.empty:
        return pd.DataFrame()

    rows: dict[str, dict[str, float]] = {}
    for method_name, fn in (
        (METHOD_HISTORICAL, lambda c: historical_var(r_p, c, horizon=horizon)),
        (METHOD_PARAMETRIC, lambda c: parametric_var(r_p, c, horizon=horizon)),
        (METHOD_CORNISH_FISHER,
         lambda c: parametric_var(r_p, c, horizon=horizon, cornish_fisher=True)),
        (METHOD_MONTE_CARLO,
         lambda c: (monte_carlo_var_mvn(R, w, c, horizon=horizon, n_sims=n_sims, seed=seed)
                    if R.shape[1] > 1
                    else monte_carlo_var(r_p, c, horizon=horizon, n_sims=n_sims, seed=seed))),
    ):
        row: dict[str, float] = {}
        for c in confidences:
            res = fn(c)
            row[f"VaR {c:.0%}"] = res.var
            row[f"CVaR {c:.0%}"] = res.cvar
        rows[method_name] = row
    cols: list[str] = []
    for c in confidences:
        cols += [f"VaR {c:.0%}", f"CVaR {c:.0%}"]
    return pd.DataFrame.from_dict(rows, orient="index")[cols]


# --------------------------------------------------------------------------- component risk

@dataclass(frozen=True)
class ComponentRisk:
    """Euler decomposition of portfolio risk across holdings.

    ``component_vol`` sums to ``portfolio_vol`` and ``component_var`` to the (zero-mean)
    parametric ``portfolio_var`` — each fund's *coherent* share of total risk, not its
    standalone risk. ``pct_contribution`` sums to 1. All volatilities/VaRs are per-period return
    fractions; ``marginal_*`` are the sensitivities ``∂risk/∂wᵢ``.
    """

    assets: list[str]
    weights: np.ndarray
    portfolio_vol: float
    portfolio_var: float
    confidence: float
    marginal_vol: np.ndarray
    component_vol: np.ndarray
    marginal_var: np.ndarray
    component_var: np.ndarray
    pct_contribution: np.ndarray

    def to_frame(self, labels: dict[str, str] | None = None) -> pd.DataFrame:
        """Tidy per-holding table (optionally relabelling ISINs → tickers) for the UI."""
        names = [(labels or {}).get(a, a) for a in self.assets]
        return pd.DataFrame({
            "Fund": names,
            "Weight": self.weights,
            "Marginal VaR": self.marginal_var,
            "Component VaR": self.component_var,
            "Risk share": self.pct_contribution,
        })


def component_risk(prices: pd.DataFrame, weights: dict[str, float],
                   confidence: float = 0.95) -> ComponentRisk | None:
    """Marginal & component contribution-to-risk per holding (Euler / Basel decomposition).

    Portfolio volatility ``σ_p = √(wᵀΣw)`` is homogeneous of degree 1 in the weights, so by
    Euler's theorem it splits exactly into additive per-fund pieces:

    * **marginal** ``∂σ_p/∂wᵢ = (Σw)ᵢ / σ_p`` — the risk added by a marginal unit of fund *i*;
    * **component** ``wᵢ · ∂σ_p/∂wᵢ`` — fund *i*'s share, and ``Σ component = σ_p``.

    Parametric VaR inherits the same split (``z_c ×`` the volatility pieces), so component VaRs
    sum to the portfolio's Gaussian VaR. Returns ``None`` when there is no shared history.
    """
    _r_p, R, w, cols = portfolio_returns(prices, weights)
    if R.empty or len(cols) == 0:
        return None
    sigma = R.cov().to_numpy()
    port_var_q = float(w @ sigma @ w)
    port_vol = float(np.sqrt(max(port_var_q, 0.0)))
    z = _N.inv_cdf(confidence)
    if port_vol <= 0:
        zeros = np.zeros(len(w))
        return ComponentRisk(cols, w, 0.0, 0.0, confidence, zeros, zeros, zeros, zeros,
                             zeros if len(w) else zeros)
    marginal_vol = (sigma @ w) / port_vol
    component_vol = w * marginal_vol
    marginal_var = z * marginal_vol
    component_var = z * component_vol
    pct = component_vol / port_vol
    return ComponentRisk(cols, w, port_vol, z * port_vol, confidence,
                         marginal_vol, component_vol, marginal_var, component_var, pct)


# --------------------------------------------------------------------------- stress tests

# Named historical crash windows, as documented ``(start, end)`` constants (ISO dates). Windows
# bracket the equity peak-to-trough of each episode (approximate index dates; sources below).
# All dates are firmly in the past — no lookahead. A portfolio with no data covering a window
# (young funds) is reported ``covered=False`` and skipped, never fabricated.
CRASH_WINDOWS: dict[str, tuple[str, str]] = {
    # Global Financial Crisis: MSCI World / S&P 500 peak (Oct 2007) → trough (Mar 2009).
    "GFC 2008": ("2007-10-09", "2009-03-09"),
    # Euro sovereign-debt crisis: summer 2011 selloff → early-Oct 2011 bottom.
    "Euro crisis 2011": ("2011-07-07", "2011-10-04"),
    # China devaluation / oil crash: Aug-2015 shock → Feb-2016 bottom.
    "China / oil 2015-16": ("2015-08-10", "2016-02-11"),
    # Q4-2018 selloff (rate-hike / growth scare): late-Sep peak → Christmas-Eve trough.
    "2018 Q4 selloff": ("2018-09-20", "2018-12-24"),
    # COVID-19 crash: fastest bear market on record, Feb-19 peak → Mar-23 trough 2020.
    "COVID crash 2020": ("2020-02-19", "2020-03-23"),
    # 2022 rate shock / inflation: Jan-3 peak → Oct-12 trough.
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
    # SVB / banking stress: Feb-2023 peak → mid-March-2023 banking-crisis low.
    "SVB banking 2023": ("2023-02-02", "2023-03-13"),
}


@dataclass(frozen=True)
class StressResult:
    """Replay of one named crash window against the portfolio's current target weights.

    ``covered`` is False (and the loss fields NaN) when the blend has no price history reaching
    the window start — the crash cannot be honestly replayed on funds that did not yet exist.
    ``drawdown`` / ``worst_day`` are negative fractions; ``recovery_days`` is the calendar days
    from the in-window trough back to the pre-trough peak (searching all later history), or
    ``None`` if not yet recovered by the end of the data.
    """

    label: str
    start: date
    end: date
    covered: bool
    n_days: int = 0
    window_return: float = float("nan")
    drawdown: float = float("nan")
    worst_day: float = float("nan")
    trough_date: date | None = None
    recovery_days: int | None = None
    data_start: date | None = None


def stress_test(prices: pd.DataFrame, weights: dict[str, float], label: str,
                window: tuple[str, str], *, wealth: pd.Series | None = None,
                returns: pd.Series | None = None) -> StressResult:
    """Replay a single crash ``window`` against the blend's current target weights.

    Slices the target-weighted wealth index to ``[start, end]`` and reports the intra-window
    drawdown (peak-to-trough), the worst single-day return, the total window return and the
    time-to-recovery. ``wealth`` / ``returns`` may be passed to avoid rebuilding them across a
    batch (see :func:`stress_tests`). Coverage requires history on/before the window start.
    """
    start = pd.Timestamp(window[0])
    end = pd.Timestamp(window[1])
    if wealth is None:
        wealth = portfolio_wealth(prices, weights)
    if returns is None:
        returns, _R, _w, _cols = portfolio_returns(prices, weights)

    if wealth.empty or wealth.index[0] > start:
        # No data at the crash onset — young funds; report as uncovered, do not fabricate.
        return StressResult(label, start.date(), end.date(), covered=False,
                            data_start=wealth.index[0].date() if not wealth.empty else None)

    seg = wealth[(wealth.index >= start) & (wealth.index <= end)]
    if len(seg) < 2:
        return StressResult(label, start.date(), end.date(), covered=False,
                            data_start=wealth.index[0].date())

    running_peak = seg.cummax()
    dd = seg / running_peak - 1.0
    trough_date = dd.idxmin()
    drawdown = float(dd.min())
    window_return = float(seg.iloc[-1] / seg.iloc[0] - 1.0)

    seg_rets = returns[(returns.index > start) & (returns.index <= end)]
    worst_day = float(seg_rets.min()) if len(seg_rets) else float("nan")

    # Recovery: first date after the trough regaining the pre-trough running peak (search all
    # later history, which may extend past the window end).
    peak_level = float(running_peak.loc[trough_date])
    after = wealth[wealth.index > trough_date]
    regained = after[after >= peak_level]
    recovery_days = int((regained.index[0] - trough_date).days) if len(regained) else None

    return StressResult(label, start.date(), end.date(), covered=True,
                        n_days=len(seg), window_return=window_return, drawdown=drawdown,
                        worst_day=worst_day, trough_date=trough_date.date(),
                        recovery_days=recovery_days, data_start=wealth.index[0].date())


def stress_tests(prices: pd.DataFrame, weights: dict[str, float],
                 windows: dict[str, tuple[str, str]] | None = None) -> list[StressResult]:
    """Replay every named crash window against the blend (builds the wealth index once)."""
    windows = windows if windows is not None else CRASH_WINDOWS
    wealth = portfolio_wealth(prices, weights)
    returns, _R, _w, _cols = portfolio_returns(prices, weights)
    return [stress_test(prices, weights, label, win, wealth=wealth, returns=returns)
            for label, win in windows.items()]
