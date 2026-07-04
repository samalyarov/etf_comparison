"""Multi-ETF portfolio blends: rebalanced backtest + correlation-aware suggestions.

The natural step beyond single-ETF Compare and DCA: pick weights across several funds and
see how the *blend* behaves. The key idea is :func:`blend_index` — it collapses a weighted,
periodically-rebalanced basket into a single synthetic total-return series (growth of 100).
Every existing tool (``metrics.*``, ``strategy.simulate_dca``, ``projection.project_plan``)
then works on the blend unchanged, so the portfolio page reuses all of them.

Also here: :func:`rebalance_comparison` (rebalanced vs let-it-drift) and
:func:`suggest_low_correlation` (a correlation-aware starter blend for a risk target).
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd

REBALANCE_FREQ = {"Monthly": "M", "Quarterly": "Q", "Annually": "A", "Never": None}


def parse_positions(text: str) -> dict[str, float]:
    """Parse pasted/uploaded holdings into ``{symbol: amount}``.

    Accepts CSV or whitespace-separated ``symbol, amount`` rows (a header is tolerated and
    skipped). ``amount`` is interpreted by the caller as either units or value. Blank lines
    and unparseable rows are ignored so a messy paste still works.
    """
    out: dict[str, float] = {}
    if not text or not text.strip():
        return out
    reader = csv.reader(io.StringIO(text.strip()), skipinitialspace=True)
    for parts in reader:
        if len(parts) == 1:
            parts = parts[0].split()
        if len(parts) < 2:
            continue
        sym = parts[0].strip().upper()
        try:
            amt = float(parts[1].replace(",", "").replace("€", "").replace("$", "").strip())
        except ValueError:
            continue  # header row or junk
        if sym:
            out[sym] = out.get(sym, 0.0) + amt
    return out


def contribution_rebalance(current_values: dict[str, float], target_weights: dict[str, float],
                           contribution: float) -> dict[str, float]:
    """Allocate a new contribution to move toward target weights **without selling**.

    Tax-aware, contribution-only rebalancing: buy the underweight funds. The contribution is
    split across funds in proportion to how far each is *below* its target value; if every
    fund is already at/above target the money is split by target weight. Returns per-fund
    buy amounts summing to ``contribution``.
    """
    if contribution <= 0 or not target_weights:
        return {k: 0.0 for k in target_weights}
    tot_w = sum(w for w in target_weights.values() if w > 0) or 1.0
    tw = {k: (w / tot_w) for k, w in target_weights.items()}
    total_after = sum(current_values.get(k, 0.0) for k in tw) + contribution
    deficits = {k: max(tw[k] * total_after - current_values.get(k, 0.0), 0.0) for k in tw}
    total_deficit = sum(deficits.values())
    if total_deficit <= 1e-9:  # already balanced — split by target weight
        return {k: contribution * tw[k] for k in tw}
    scale = min(1.0, contribution / total_deficit)
    buys = {k: deficits[k] * scale for k in tw}
    # If contribution exceeds what's needed to reach targets, spread the excess by weight.
    spent = sum(buys.values())
    excess = contribution - spent
    if excess > 1e-9:
        for k in tw:
            buys[k] += excess * tw[k]
    return buys


def _aligned(prices: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.DataFrame, np.ndarray]:
    """Inner-align the price matrix on dates where every held fund has data; normalise w."""
    cols = [c for c in weights if c in prices.columns and weights[c] > 0]
    if not cols:
        return pd.DataFrame(), np.array([])
    P = prices[cols].dropna(how="any").sort_index()
    w = np.array([weights[c] for c in cols], dtype=float)
    total = w.sum()
    if total <= 0 or P.empty:
        return pd.DataFrame(), np.array([])
    return P, w / total


def blend_index(prices: pd.DataFrame, weights: dict[str, float],
                rebalance: str | None = "Q", initial: float = 100.0) -> pd.Series:
    """Synthetic total-return series for a weighted, periodically-rebalanced blend.

    Starts at ``initial``. ``rebalance`` is a pandas period alias ('M'/'Q'/'A') or None for
    buy-and-hold (weights drift with performance). Between rebalance dates units are held
    fixed; on each period boundary holdings are reset to target weights of the current total.
    """
    P, w = _aligned(prices, weights)
    if P.empty:
        return pd.Series(dtype=float)

    px0 = P.iloc[0].to_numpy()
    units = w * initial / px0
    periods = P.index.to_period(rebalance) if rebalance else None

    values = np.empty(len(P))
    prev_period = periods[0] if periods is not None else None
    for t in range(len(P)):
        px = P.iloc[t].to_numpy()
        total = float((units * px).sum())
        if periods is not None and t > 0 and periods[t] != prev_period:
            units = w * total / px  # rebalance to target weights
            prev_period = periods[t]
        values[t] = total
    return pd.Series(values, index=P.index, name="blend")


def blend_weights_drift(prices: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Final actual weights of a never-rebalanced blend (shows how far it drifts)."""
    P, w = _aligned(prices, weights)
    if P.empty:
        return pd.Series(dtype=float)
    units = w / P.iloc[0].to_numpy()
    final_vals = units * P.iloc[-1].to_numpy()
    final_w = final_vals / final_vals.sum()
    return pd.Series(final_w, index=P.columns)


def rebalance_comparison(prices: pd.DataFrame, weights: dict[str, float],
                         rebalance: str | None = "Q", initial: float = 100.0) -> dict:
    """Compare a rebalanced blend against the same blend left to drift (buy-and-hold)."""
    reb = blend_index(prices, weights, rebalance=rebalance, initial=initial)
    drift = blend_index(prices, weights, rebalance=None, initial=initial)
    out = {"rebalanced": reb, "drift": drift}
    if not reb.empty and not drift.empty:
        out["rebalanced_final"] = float(reb.iloc[-1])
        out["drift_final"] = float(drift.iloc[-1])
    return out


def suggest_low_correlation(prices: pd.DataFrame, n: int = 4,
                            seed_isin: str | None = None) -> list[str]:
    """Greedily pick ``n`` funds that are mutually least-correlated (diversification starter).

    Starts from ``seed_isin`` (or the fund with the highest Sharpe-ish return/vol), then
    repeatedly adds whichever remaining fund has the lowest average correlation to those
    already chosen. A quick, correlation-aware blend suggestion — not a mean-variance optimum.
    """
    P = prices.dropna(how="all").sort_index()
    rets = P.pct_change().dropna(how="all")
    if rets.shape[1] < 2:
        return list(P.columns[:n])
    corr = rets.corr()
    cols = list(corr.columns)

    if seed_isin and seed_isin in cols:
        chosen = [seed_isin]
    else:
        # seed with the best return/volatility ratio among candidates
        mean = rets.mean()
        vol = rets.std(ddof=1).replace(0, np.nan)
        score = (mean / vol).dropna()
        chosen = [score.idxmax()] if not score.empty else [cols[0]]

    while len(chosen) < min(n, len(cols)):
        remaining = [c for c in cols if c not in chosen]
        avg_corr = {c: float(corr.loc[c, chosen].mean()) for c in remaining}
        chosen.append(min(avg_corr, key=avg_corr.get))
    return chosen
