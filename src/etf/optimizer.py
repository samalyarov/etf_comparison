"""Constrained mean-variance portfolio optimiser (pure numerics; no DB / no network).

Max-Sharpe (tangency), min-volatility and the efficient frontier under **real-world
constraints**: leverage (long-only, or a gross-exposure cap ``Σ|w| ≤ L`` with shorting),
turnover vs a current portfolio (``‖w − w_prev‖₁ ≤ τ``), and sector / region / asset-class
exposure limits built from the ETF *look-through* profiles (:mod:`etf.profiles`). Solved with
**cvxpy** (a convex QP / SOCP); the covariance uses **Ledoit-Wolf shrinkage** and the expected
returns come from **PyPortfolioOpt** estimators.

CAVEAT — read before you trust a max-Sharpe weight vector
---------------------------------------------------------
Max-Sharpe is notoriously sensitive to the *expected-return* estimate — Michaud's
"error-maximisation" critique (R. Michaud, 1989, *Financial Analysts Journal*): the optimiser
piles into whatever assets have the highest **estimated** mean, and those estimates are mostly
noise, so tiny input changes swing the weights wildly and out-of-sample performance is often
poor. Historical means are a weak forecast of future returns. Treat the tangency portfolio as
*one input to judgement, not a prescription*. **Min-volatility and the frontier's low-risk end
are far more stable** because they lean only on the covariance, which is estimated much better
than the mean. This module surfaces the frontier and min-vol precisely so the user is not shown
a single fragile point in isolation.

Method note — max-Sharpe under constraints
------------------------------------------
The tangency problem ``max (μ−rf)ᵀw / √(wᵀΣw)`` is a linear-fractional program. We convert it
to a convex QP with the **Charnes-Cooper transformation** (``y = w/κ``, ``κ > 0``), scaling
**every** constraint's constants by ``κ`` — crucially the turnover anchor becomes
``‖y − κ·w_prev‖₁ ≤ τ·κ``. PyPortfolioOpt's ``max_sharpe`` leaves ``w_prev`` unscaled and so
mis-handles turnover (it returns *infeasible*); solving the transform directly in cvxpy keeps
every constraint exact. cvxpy *is* the solver here — this is the standard textbook method, not
a hand-rolled QP.

Exposure constraints and coverage
---------------------------------
A portfolio's exposure along a dimension is a **linear** map ``A·w`` (fund × category share),
so caps/floors are linear constraints. ``A`` is built from :mod:`etf.profiles`; a fund with no
look-through data for the dimension contributes a **zero** column, so it is simply *not counted*
— never silently assumed. Constraints act on the **absolute** covered exposure (denominator =
the whole portfolio), which makes an upper cap *conservative* under partial coverage. Every
constrained dimension reports its **coverage** (share of portfolio weight that carried data) so
a low-coverage constraint is visible, not silent (CLAUDE.md: data quality is sacred).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models

from . import profiles

# Objectives the optimiser understands.
OBJECTIVES = ("max_sharpe", "min_volatility")
# Expected-return estimators (both from PyPortfolioOpt; see the module caveat).
RETURN_METHODS = ("mean_historical", "ema_historical")
# Dimensions an exposure limit may reference (profiles look-through + asset_class).
LIMIT_DIMENSIONS = (*profiles.WEIGHT_DIMENSIONS, "asset_class")

TRADING_DAYS = 252
_SOLVERS = ("CLARABEL", "SCS")  # deterministic; CLARABEL preferred, SCS fallback
_TOL = 1e-4  # slack tolerance for "binding constraint" detection


# --------------------------------------------------------------------------- constraint spec
@dataclass(frozen=True)
class ExposureLimit:
    """A cap and/or floor on the portfolio's exposure to one category of one dimension.

    ``dimension`` is one of :data:`LIMIT_DIMENSIONS`; ``label`` is a category within it (e.g.
    ``sector`` / ``"Information Technology"``, ``asset_class`` / ``"bond"``). ``lower`` /
    ``upper`` are fractions of the whole portfolio (0..1); either may be ``None``. Bounds are on
    **absolute covered exposure** (see the module docstring).
    """

    dimension: str
    label: str
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class OptConstraints:
    """The real-world constraint set. Sensible defaults = long-only, fully-invested."""

    long_only: bool = True
    gross_leverage: float | None = None   # Σ|w| ≤ L; used only when long_only is False
    min_weight: float | None = None       # per-asset lower bound (e.g. -0.1 to allow shorting)
    max_weight: float | None = None       # per-asset upper bound (e.g. 0.35 concentration cap)
    turnover_limit: float | None = None   # ‖w − current_weights‖₁ ≤ τ (needs current_weights)
    l2_gamma: float = 0.0                 # L2 (ridge) diversification: Σ ← Σ + γ·I
    exposure_limits: tuple[ExposureLimit, ...] = ()


# --------------------------------------------------------------------------- result types
@dataclass(frozen=True)
class ExposureReport:
    """Absolute covered exposure the constraints act on, plus its coverage caveat."""

    dimension: str
    exposure: dict[str, float]  # {label: absolute portfolio weight in covered funds}
    coverage: float             # share of portfolio weight with data for this dimension


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome of one optimisation. ``success`` is False on an infeasible / failed solve."""

    objective: str
    status: str
    success: bool
    message: str
    weights: dict[str, float] = field(default_factory=dict)
    expected_return: float = float("nan")
    volatility: float = float("nan")
    sharpe: float = float("nan")
    risk_free_rate: float = 0.0
    return_method: str = "mean_historical"
    exposures: dict[str, ExposureReport] = field(default_factory=dict)
    binding: list[str] = field(default_factory=list)
    frontier: pd.DataFrame | None = None


# --------------------------------------------------------------------------- estimators
def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop empty columns, coerce numeric, keep dates where *every* asset has a price."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    p = prices.apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)
    p = p[p > 0].dropna(how="any").sort_index()
    return p


def expected_returns_vector(prices: pd.DataFrame,
                            method: str = "mean_historical") -> pd.Series:
    """Annualised expected returns via PyPortfolioOpt (see caveat: a weak forecast)."""
    if method not in RETURN_METHODS:
        raise ValueError(f"Unknown return method {method!r}; use one of {RETURN_METHODS}")
    if method == "ema_historical":
        return expected_returns.ema_historical_return(prices, frequency=TRADING_DAYS)
    return expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)


def covariance_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Annualised Ledoit-Wolf shrinkage covariance (well-conditioned, PSD)."""
    return risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()


# --------------------------------------------------------------------------- exposure map
def exposure_matrix(assets: list[str], dimension: str,
                    path=None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build the linear exposure map for a dimension.

    Returns ``(labels, A, covered)`` where ``A`` is ``len(labels) × len(assets)`` with
    ``A[k, f]`` = fund *f*'s normalised share of category *k* (columns sum to 1 for a covered
    fund, 0 for an uncovered one), and ``covered`` is a 0/1 vector flagging funds that carried
    data for the dimension. ``A·w`` is then the portfolio's absolute exposure per label.
    """
    profs = profiles.load_profiles(path)
    n = len(assets)
    covered = np.zeros(n)
    col_weights: list[dict[str, float]] = []
    labels: list[str] = []
    seen: set[str] = set()
    for j, isin in enumerate(assets):
        prof = profs.get(isin)
        if prof is None:
            col_weights.append({})
            continue
        if dimension == "asset_class":
            dim_w = {prof.asset_class: 1.0}
        else:
            dim_w = prof.weights(dimension)
        if not dim_w:
            col_weights.append({})
            continue
        total = sum(dim_w.values()) or 1.0
        norm = {k: v / total for k, v in dim_w.items()}
        col_weights.append(norm)
        covered[j] = 1.0
        for k in norm:
            if k not in seen:
                seen.add(k)
                labels.append(k)
    A = np.zeros((len(labels), n))
    idx = {k: i for i, k in enumerate(labels)}
    for j, norm in enumerate(col_weights):
        for k, v in norm.items():
            A[idx[k], j] = v
    return labels, A, covered


# --------------------------------------------------------------------------- core solve
def _align_current(current_weights: dict[str, float] | None,
                   assets: list[str]) -> np.ndarray | None:
    """Reindex current weights onto the candidate assets, normalised to sum 1 (or None)."""
    if not current_weights:
        return None
    raw = np.array([max(current_weights.get(a, 0.0), 0.0) for a in assets], dtype=float)
    tot = raw.sum()
    if tot <= 0:
        return None
    return raw / tot


def _linear_constraints(v, s, assets, con: OptConstraints,
                        w_prev: np.ndarray | None, path):
    """Constraints on real weights ``w = v/s`` expressed on ``(v, s)`` (s>0).

    With ``s = 1`` this is the direct QP; with ``s = κ`` it is the Charnes-Cooper transform for
    max-Sharpe. Returns ``(constraints, meta)`` where ``meta`` describes each named constraint
    so binding-ness can be checked post-solve in real ``w`` space.
    """
    n = len(assets)
    cons = [cp.sum(v) == s]  # fully invested: Σw = 1
    meta: list[tuple[str, str]] = []  # (name, kind) for binding detection

    if con.long_only:
        cons.append(v >= 0)
    else:
        lo = con.min_weight if con.min_weight is not None else -1.0
        cons.append(v >= lo * s)
        if con.gross_leverage is not None:
            cons.append(cp.norm(v, 1) <= con.gross_leverage * s)
            meta.append(("gross_leverage", "gross"))
    if con.max_weight is not None:
        cons.append(v <= con.max_weight * s)
    if con.min_weight is not None and con.long_only:
        cons.append(v >= con.min_weight * s)

    if con.turnover_limit is not None and w_prev is not None:
        cons.append(cp.norm(v - s * w_prev, 1) <= con.turnover_limit * s)
        meta.append(("turnover", "turnover"))

    for lim in con.exposure_limits:
        labels, A, _ = exposure_matrix(assets, lim.dimension, path)
        if lim.label not in labels:
            # No covered fund carries this category — an upper cap is trivially satisfied
            # (0 exposure); a positive floor is infeasible, which the solver will surface.
            row = np.zeros(n)
        else:
            row = A[labels.index(lim.label)]
        expo = row @ v  # absolute exposure × s
        tag = f"{lim.dimension}:{lim.label}"
        if lim.upper is not None:
            cons.append(expo <= lim.upper * s)
            meta.append((f"{tag}≤{lim.upper:.0%}", "expo_upper"))
        if lim.lower is not None:
            cons.append(expo >= lim.lower * s)
            meta.append((f"{tag}≥{lim.lower:.0%}", "expo_lower"))
    return cons, meta


def _solve(mu: np.ndarray, S: np.ndarray, assets: list[str], objective: str,
           rf: float, con: OptConstraints, w_prev: np.ndarray | None,
           path, target_return: float | None = None):
    """Solve one program; return ``(status, weights|None)``. Never raises on infeasibility."""
    n = len(assets)
    S_eff = S + con.l2_gamma * np.eye(n) if con.l2_gamma > 0 else S
    S_psd = cp.psd_wrap(S_eff)

    if objective == "max_sharpe":
        # Charnes-Cooper: minimise yᵀΣy s.t. (μ−rf)ᵀy = 1, Σy = κ, plus scaled constraints.
        y = cp.Variable(n)
        kappa = cp.Variable(nonneg=True)
        cons, _ = _linear_constraints(y, kappa, assets, con, w_prev, path)
        cons.append((mu - rf) @ y == 1)
        cons.append(kappa >= 1e-8)
        prob = cp.Problem(cp.Minimize(cp.quad_form(y, S_psd)), cons)
        status = _run(prob)
        if y.value is None or kappa.value is None or kappa.value <= 1e-9:
            return status, None
        return status, np.asarray(y.value / kappa.value).flatten()

    # min_volatility (optionally with a return floor → traces the efficient frontier).
    w = cp.Variable(n)
    cons, _ = _linear_constraints(w, 1.0, assets, con, w_prev, path)
    if target_return is not None:
        cons.append(mu @ w >= target_return)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, S_psd)), cons)
    status = _run(prob)
    if w.value is None:
        return status, None
    return status, np.asarray(w.value).flatten()


def _max_return(mu: np.ndarray, assets: list[str], con: OptConstraints,
                w_prev: np.ndarray | None, path) -> float | None:
    """LP for the greatest attainable expected return under the constraints (frontier top)."""
    n = len(assets)
    w = cp.Variable(n)
    cons, _ = _linear_constraints(w, 1.0, assets, con, w_prev, path)
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    _run(prob)
    if w.value is None:
        return None
    return float(mu @ np.asarray(w.value).flatten())


def _run(prob: cp.Problem) -> str:
    """Solve with the preferred solver, falling back; return the cvxpy status string."""
    for solver in _SOLVERS:
        try:
            prob.solve(solver=solver)
        except Exception:  # noqa: BLE001 — any solver failure just falls through to the next
            continue
        if prob.status in ("optimal", "optimal_inaccurate"):
            return prob.status
    return prob.status or "solver_error"


# --------------------------------------------------------------------------- public API
def _portfolio_stats(w: np.ndarray, mu: np.ndarray, S: np.ndarray,
                     rf: float) -> tuple[float, float, float]:
    ret = float(mu @ w)
    var = float(w @ S @ w)
    vol = float(np.sqrt(var)) if var > 0 else 0.0
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")
    return ret, vol, sharpe


def _binding(w: np.ndarray, assets: list[str], con: OptConstraints,
             w_prev: np.ndarray | None, path) -> list[str]:
    """Report which constraints the solution sits on (within tolerance), in real w space."""
    out: list[str] = []
    if not con.long_only and con.gross_leverage is not None:
        if abs(np.abs(w).sum() - con.gross_leverage) <= _TOL:
            out.append(f"gross leverage = {con.gross_leverage:.2f}")
    if con.max_weight is not None and np.any(w >= con.max_weight - _TOL):
        hit = [assets[i] for i in range(len(assets)) if w[i] >= con.max_weight - _TOL]
        out.append(f"max weight {con.max_weight:.0%} ({', '.join(hit)})")
    if con.turnover_limit is not None and w_prev is not None:
        if abs(np.abs(w - w_prev).sum() - con.turnover_limit) <= _TOL:
            out.append(f"turnover = {con.turnover_limit:.0%}")
    for lim in con.exposure_limits:
        labels, A, _ = exposure_matrix(assets, lim.dimension, path)
        if lim.label not in labels:
            continue
        expo = float(A[labels.index(lim.label)] @ w)
        if lim.upper is not None and abs(expo - lim.upper) <= _TOL:
            out.append(f"{lim.dimension}:{lim.label} ≤ {lim.upper:.0%} (binding)")
        if lim.lower is not None and abs(expo - lim.lower) <= _TOL:
            out.append(f"{lim.dimension}:{lim.label} ≥ {lim.lower:.0%} (binding)")
    return out


def _exposure_reports(w: np.ndarray, assets: list[str], con: OptConstraints,
                      path) -> dict[str, ExposureReport]:
    """Absolute covered exposure + coverage for every dimension referenced by a limit,
    plus asset_class (always), so the caller can display what the constraints acted on."""
    dims = {lim.dimension for lim in con.exposure_limits} | {"asset_class"}
    reports: dict[str, ExposureReport] = {}
    for dim in dims:
        labels, A, covered = exposure_matrix(assets, dim, path)
        expo = A @ w
        pos = w > 0
        coverage = float(np.abs(w[pos]).sum() and (covered[pos] @ np.abs(w[pos]))
                         / np.abs(w[pos]).sum()) if np.any(pos) else 0.0
        exposure = {labels[k]: float(expo[k]) for k in range(len(labels))
                    if abs(expo[k]) > 1e-9}
        exposure = dict(sorted(exposure.items(), key=lambda kv: kv[1], reverse=True))
        reports[dim] = ExposureReport(dim, exposure, coverage)
    return reports


def efficient_frontier(prices: pd.DataFrame, *, risk_free_rate: float = 0.02,
                       current_weights: dict[str, float] | None = None,
                       constraints: OptConstraints | None = None,
                       return_method: str = "mean_historical",
                       points: int = 20, path=None) -> pd.DataFrame:
    """Trace the constrained efficient frontier as a ``(volatility, ret, sharpe)`` frame.

    Solves min-variance for a grid of target returns between the min-vol portfolio and the
    max-return portfolio, honouring every constraint. Empty if the set is infeasible.
    """
    con = constraints or OptConstraints()
    p = _clean_prices(prices)
    if p.shape[1] < 2:
        return pd.DataFrame(columns=["volatility", "ret", "sharpe"])
    assets = list(p.columns)
    mu = expected_returns_vector(p, return_method).reindex(assets).to_numpy()
    S = covariance_matrix(p).reindex(index=assets, columns=assets).to_numpy()
    w_prev = _align_current(current_weights, assets)

    st_lo, w_lo = _solve(mu, S, assets, "min_volatility", risk_free_rate, con, w_prev, path)
    r_hi = _max_return(mu, assets, con, w_prev, path)
    if w_lo is None or r_hi is None:
        return pd.DataFrame(columns=["volatility", "ret", "sharpe"])
    r_lo = float(mu @ w_lo)
    if r_hi <= r_lo + 1e-9:
        ret, vol, sh = _portfolio_stats(w_lo, mu, S, risk_free_rate)
        return pd.DataFrame([{"volatility": vol, "ret": ret, "sharpe": sh}])
    rows = []
    for target in np.linspace(r_lo, r_hi, max(points, 2)):
        _, w = _solve(mu, S, assets, "min_volatility", risk_free_rate, con, w_prev, path,
                      target_return=float(target))
        if w is None:
            continue
        ret, vol, sh = _portfolio_stats(w, mu, S, risk_free_rate)
        rows.append({"volatility": vol, "ret": ret, "sharpe": sh})
    return pd.DataFrame(rows).sort_values("volatility").reset_index(drop=True)


def optimize_portfolio(prices: pd.DataFrame, *, objective: str = "max_sharpe",
                       risk_free_rate: float = 0.02,
                       current_weights: dict[str, float] | None = None,
                       constraints: OptConstraints | None = None,
                       return_method: str = "mean_historical",
                       with_frontier: bool = False, frontier_points: int = 20,
                       path=None) -> OptimizeResult:
    """Solve a constrained portfolio problem and report weights, stats, exposure and status.

    ``prices`` is a date × ISIN adjusted-close matrix (total-return basis). ``objective`` is
    ``"max_sharpe"`` (tangency, via Charnes-Cooper) or ``"min_volatility"``. Constraints come
    from :class:`OptConstraints`; ``current_weights`` (``{isin: weight}``) enables the turnover
    limit. Never raises on an infeasible set — it returns ``success=False`` with the solver
    status so the UI can message it cleanly.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}; use one of {OBJECTIVES}")
    con = constraints or OptConstraints()
    p = _clean_prices(prices)
    if p.shape[1] < 2:
        return OptimizeResult(objective, "insufficient_assets", False,
                              "Need at least two funds with overlapping price history.",
                              risk_free_rate=risk_free_rate, return_method=return_method)
    assets = list(p.columns)
    mu = expected_returns_vector(p, return_method).reindex(assets).to_numpy()
    S = covariance_matrix(p).reindex(index=assets, columns=assets).to_numpy()
    w_prev = _align_current(current_weights, assets)

    status, w = _solve(mu, S, assets, objective, risk_free_rate, con, w_prev, path)
    if w is None:
        return OptimizeResult(
            objective, status, False,
            f"No feasible portfolio for these constraints (solver: {status}).",
            risk_free_rate=risk_free_rate, return_method=return_method)

    w = np.where(np.abs(w) < 1e-6, 0.0, w)  # clean numerical dust
    ret, vol, sharpe = _portfolio_stats(w, mu, S, risk_free_rate)
    weights = {assets[i]: float(w[i]) for i in range(len(assets)) if abs(w[i]) > 1e-6}
    frontier = None
    if with_frontier:
        frontier = efficient_frontier(p, risk_free_rate=risk_free_rate,
                                      current_weights=current_weights, constraints=con,
                                      return_method=return_method, points=frontier_points,
                                      path=path)
    return OptimizeResult(
        objective, status, True, "Optimal portfolio found.",
        weights=weights, expected_return=ret, volatility=vol, sharpe=sharpe,
        risk_free_rate=risk_free_rate, return_method=return_method,
        exposures=_exposure_reports(w, assets, con, path),
        binding=_binding(w, assets, con, w_prev, path),
        frontier=frontier)
