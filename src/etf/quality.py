"""Price-series data-quality checks and repair (built into ingestion).

Real ETF price feeds — Yahoo's especially — carry two recurring corruptions that
otherwise produce nonsense CAGR/volatility downstream:

1. **GBX/GBP (pence vs pounds) mis-denomination** on London (``.L``) listings. Yahoo
   intermittently switches a day (or a multi-month run) between pence and pounds, so a
   single price is ~100x its neighbours (e.g. ``SMEA.L`` 2488 → 23 → 2488). Some series
   have a *persistent* regime shift (``IBTS.L`` 10342 pence → 71 pounds in 2009).
2. **Isolated bad prints** — a lone day at a wrong (non-100x) scale that reverts the next
   day (e.g. ``SGLN.L``/``EQQQ.L`` spiking ~1.6x for one print).

The repair here is deliberately general and conservative:

* :func:`reconstruct_scale` rescales each day by the power of 100 that maximises
  continuity, **anchored on the most recent price** so the present-day denomination (what
  you would actually trade at) is authoritative. This unifies every GBX/GBP artifact —
  single-day flips, multi-month wrong-scale runs, and permanent regime shifts — in one pass
  and is a no-op on clean series.
* :func:`despike` then repairs any remaining *isolated* single-day outlier that deviates
  from the geometric midpoint of its neighbours and reverts (the non-100x bad prints).

Nothing here fabricates data: rescaling is a unit change, and de-spiking only touches lone
points that both neighbours contradict. Anything still discontinuous afterwards is *flagged*
(``status="suspect"``) rather than silently trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Price columns that share a denomination and must be rescaled together. Volume is not.
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]

# Log-return magnitude above which a single day is treated as a candidate corruption.
# A genuine broad ETF virtually never moves >40% vs the midpoint of the two adjacent days.
SPIKE_LOG_THRESHOLD = 0.40  # ln(1.49) ≈ 0.40  → ~49% deviation
# A residual daily move above this after repair means the series is still not trustworthy.
SUSPECT_DAILY_MOVE = 0.50
# A power-of-100 rescale is only applied when it brings the day-to-day jump below this (in
# log space); otherwise the discontinuity is *not* a GBX/GBP artifact (e.g. an unadjusted
# split) and we must not fabricate — leave it and let the series be flagged suspect.
CONTINUITY_LOG_TOL = 0.70  # ln(2) ≈ 0.69 → residual jump must fall under ~2x


@dataclass
class QualityReport:
    """Outcome of cleaning one instrument's price history."""

    n_rows: int
    rescaled_days: int          # days shifted by a power of 100 (GBX/GBP fix)
    despiked_days: int          # isolated bad prints repaired
    max_move_before: float      # largest |daily return| before repair
    max_move_after: float       # largest |daily return| after repair
    status: str                 # 'clean' | 'repaired' | 'suspect'
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "rescaled_days": self.rescaled_days,
            "despiked_days": self.despiked_days,
            "max_move_before": self.max_move_before,
            "max_move_after": self.max_move_after,
            "status": self.status,
            "notes": self.notes,
        }


def _max_abs_daily_move(s: pd.Series) -> float:
    r = s.dropna().pct_change().dropna()
    return float(r.abs().max()) if len(r) else 0.0


def reconstruct_scale(prices: pd.Series, base: int = 100, max_pow: int = 2) -> pd.Series:
    """Per-date multiplicative factor (a power of ``base``) that maximises continuity.

    Walks the series **backwards from the most recent point** (anchored at factor 1), at
    each step choosing the power of ``base`` for the earlier day that puts it closest, in
    log space, to the already-reconstructed later day. This expresses the whole history in
    the *current* denomination and repairs GBX/GBP flips of any duration. On a clean series
    every factor is 1 (any rescale would create a huge artificial jump), so it is a no-op.
    """
    s = pd.to_numeric(prices, errors="coerce").dropna()
    s = s[s > 0].sort_index()
    if len(s) < 2:
        return pd.Series(1.0, index=prices.index).reindex(prices.index).fillna(1.0)

    logs = np.log(s.values)
    logb = np.log(base)
    nonzero = np.array([k for k in range(-max_pow, max_pow + 1) if k != 0])
    powers = np.zeros(len(s), dtype=int)
    for i in range(len(s) - 2, -1, -1):
        target = logs[i + 1] + powers[i + 1] * logb  # reconstructed later day (log)
        base_cost = abs(logs[i] - target)             # keep current scale (k = 0)
        alt = logs[i] + nonzero * logb
        j = int(np.argmin(np.abs(alt - target)))
        alt_cost = abs(alt[j] - target)
        # Only rescale when the current scale is genuinely discontinuous AND a power of
        # 100 restores continuity; otherwise the break is not a GBX/GBP artifact.
        if base_cost > CONTINUITY_LOG_TOL and alt_cost < CONTINUITY_LOG_TOL:
            powers[i] = nonzero[j]
    factor = pd.Series(np.power(float(base), powers), index=s.index)
    return factor.reindex(prices.index).fillna(1.0)


def despike(prices: pd.Series, threshold: float = SPIKE_LOG_THRESHOLD) -> tuple[pd.Series, pd.Series]:
    """Repair isolated single-day outliers by geometric interpolation of their neighbours.

    A point is a spike only if it deviates from the geometric midpoint of the two adjacent
    days by more than ``threshold`` in log space *and* differs from each neighbour by more
    than ``threshold`` (i.e. both neighbours contradict it and agree with each other). This
    catches lone bad prints without touching genuine two-day moves. Returns ``(repaired,
    mask)`` where ``mask`` marks the repaired dates.
    """
    s = pd.to_numeric(prices, errors="coerce")
    valid = s.dropna()
    valid = valid[valid > 0]
    mask = pd.Series(False, index=prices.index)
    if len(valid) < 3:
        return prices.copy(), mask

    logs = np.log(valid.values)
    out = logs.copy()
    for i in range(1, len(valid) - 1):
        mid = (logs[i - 1] + logs[i + 1]) / 2.0
        if (abs(logs[i] - mid) > threshold
                and abs(logs[i] - logs[i - 1]) > threshold
                and abs(logs[i] - logs[i + 1]) > threshold):
            out[i] = mid
            mask.loc[valid.index[i]] = True

    repaired = s.copy()
    repaired.loc[valid.index] = np.exp(out)
    return repaired, mask


def clean_prices(df: pd.DataFrame, ref: str = "adj_close") -> tuple[pd.DataFrame, QualityReport]:
    """Clean an OHLCV frame: GBX/GBP rescale + de-spike, applied consistently across cols.

    Detection is driven off the ``ref`` column (falls back to ``close``). The per-date scale
    factor and the de-spike repair are applied to every price column so OHLC stay coherent;
    ``volume`` is left untouched. Returns the cleaned frame and a :class:`QualityReport`.
    """
    if df is None or df.empty:
        return df, QualityReport(0, 0, 0, 0.0, 0.0, "clean", "empty")

    out = df.copy()
    ref_col = ref if ref in out.columns and out[ref].notna().any() else "close"
    if ref_col not in out.columns or out[ref_col].notna().sum() < 2:
        return out, QualityReport(len(out), 0, 0, 0.0, 0.0, "clean", "insufficient prices")

    reference = pd.to_numeric(out[ref_col], errors="coerce")
    before = _max_abs_daily_move(reference)

    # 1) GBX/GBP power-of-100 continuity rescale (all price columns share denomination).
    factor = reconstruct_scale(reference)
    rescaled_days = int((factor != 1.0).sum())
    price_cols = [c for c in PRICE_COLUMNS if c in out.columns]
    for c in price_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce") * factor

    # 2) De-spike isolated bad prints, driven off the (now rescaled) reference column.
    ref_scaled = pd.to_numeric(out[ref_col], errors="coerce")
    _, mask = despike(ref_scaled)
    despiked_days = int(mask.sum())
    if despiked_days:
        for c in price_cols:
            col = pd.to_numeric(out[c], errors="coerce")
            col = col.where(col > 0)
            col[mask] = np.nan  # blank the flagged prints so they are re-interpolated
            interp = np.exp(np.log(col).interpolate("linear"))
            out.loc[mask, c] = interp.loc[mask]

    after = _max_abs_daily_move(pd.to_numeric(out[ref_col], errors="coerce"))
    if rescaled_days or despiked_days:
        status = "suspect" if after > SUSPECT_DAILY_MOVE else "repaired"
    else:
        status = "suspect" if after > SUSPECT_DAILY_MOVE else "clean"

    notes = []
    if rescaled_days:
        notes.append(f"{rescaled_days} GBX/GBP day(s) rescaled")
    if despiked_days:
        notes.append(f"{despiked_days} bad print(s) repaired")
    if status == "suspect":
        notes.append(f"residual {after:.0%} daily move remains")
    report = QualityReport(len(out), rescaled_days, despiked_days, before, after,
                           status, "; ".join(notes))
    return out, report


def reconcile_total_return(close: pd.Series, adj_close: pd.Series,
                           distributions: pd.Series | None = None,
                           tolerance: float = 0.03) -> dict:
    """Cross-check the total-return series against price + distributions.

    ``adj_close`` should equal ``close`` compounded with reinvested distributions, so the
    gap between total-return and price-return ought to be explained (roughly) by the sum of
    distributions relative to price. A large unexplained gap signals an adjustment error.
    Returns a dict with the two returns, the distribution yield, and an ``ok`` flag.
    """
    c = pd.to_numeric(close, errors="coerce").dropna()
    a = pd.to_numeric(adj_close, errors="coerce").dropna()
    if len(c) < 2 or len(a) < 2:
        return {"ok": True, "price_return": float("nan"), "tr_return": float("nan"),
                "dist_yield": float("nan"), "note": "insufficient data"}
    price_return = float(c.iloc[-1] / c.iloc[0] - 1.0)
    tr_return = float(a.iloc[-1] / a.iloc[0] - 1.0)
    dist_yield = 0.0
    if distributions is not None and len(distributions):
        d = pd.to_numeric(distributions, errors="coerce").dropna()
        # crude reinvestment-free yield: distributions summed against mean price
        dist_yield = float(d.sum() / c.mean()) if c.mean() else 0.0
    # Total return should be at least price return (dividends never subtract), and the
    # excess should be within a few % of the accumulated distribution yield.
    excess = tr_return - price_return
    ok = excess >= -tolerance and excess <= dist_yield * 3 + tolerance + abs(price_return)
    note = "" if ok else "total-return vs price+distributions mismatch"
    return {"ok": bool(ok), "price_return": price_return, "tr_return": tr_return,
            "dist_yield": dist_yield, "excess": excess, "note": note}


def assess_series(prices: pd.Series) -> dict:
    """Lightweight read-side health check (no repair): status + worst daily move.

    Used by the app to label a stored series without re-running the full clean. A series is
    ``clean`` if no single day moves more than :data:`SUSPECT_DAILY_MOVE`, else ``suspect``.
    """
    move = _max_abs_daily_move(pd.to_numeric(prices, errors="coerce"))
    status = "suspect" if move > SUSPECT_DAILY_MOVE else "clean"
    return {"status": status, "max_move": move}
