"""ETF strategy / exposure profiles — the look-through layer (pure, no DB, no network).

Every fund in the universe is mapped to the *index* it tracks; the index carries the
researched exposure breakdown (region / country / GICS sector / top holdings, and for bonds a
credit-quality bucket). :func:`get_profile` merges a fund's row onto its index profile so
callers see a flat per-ISIN view.

This is the authoritative basis for the portfolio optimiser's **sector / exposure
constraints**. :func:`portfolio_exposure` aggregates a set of fund weights into a weighted
exposure along any dimension, **renormalising over the portion that has data** and reporting
*coverage* so a constraint is never silently computed on partial information.

Data lives in a committed seed (``scripts/etf_profiles.yaml``) that survives a DB rebuild; a
git-ignored working copy (``data/etf_profiles.yaml``) takes precedence if present — mirroring
how :mod:`etf.config` treats the watchlist and :mod:`etf.settings` the JSON store. No numbers
are invented: fields that were not verified are empty and the profile is flagged
``data_complete: false`` (see the YAML header and CLAUDE.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .config import DATA_DIR, PROJECT_ROOT

# Working copy first (regenerable, git-ignored), then the committed seed.
_DATA_PROFILES = DATA_DIR / "etf_profiles.yaml"
_SEED_PROFILES = PROJECT_ROOT / "scripts" / "etf_profiles.yaml"

# Dimensions along which a portfolio's exposure can be aggregated.
WEIGHT_DIMENSIONS = ("region", "country", "sector", "credit_quality")
_DIM_FIELD = {
    "region": "region_weights",
    "country": "country_weights",
    "sector": "sector_weights",
    "credit_quality": "credit_quality",
}

# Keys every resolved profile carries (used for schema validation in tests).
REQUIRED_KEYS = ("index_name", "strategy", "asset_class", "as_of", "source", "data_complete")


@dataclass(frozen=True)
class Profile:
    """A fund's resolved strategy/exposure profile (fund row merged onto its index)."""

    isin: str
    ticker: str
    index_id: str
    index_name: str
    strategy: str
    asset_class: str
    replication: str | None
    factor_tilt: list[str]
    as_of: str | None
    source: str | None
    data_complete: bool
    region_weights: dict[str, float] = field(default_factory=dict)
    country_weights: dict[str, float] = field(default_factory=dict)
    sector_weights: dict[str, float] = field(default_factory=dict)
    credit_quality: dict[str, float] = field(default_factory=dict)
    top_holdings: list[dict] = field(default_factory=list)
    asset_mix: dict[str, float] = field(default_factory=dict)
    hedged: bool | None = None

    def weights(self, dimension: str) -> dict[str, float]:
        """Return the weight mapping for a dimension (empty if this fund lacks it)."""
        return getattr(self, _DIM_FIELD[dimension], {}) or {}

    def has_dimension(self, dimension: str) -> bool:
        """True if this profile carries usable (non-empty) weights for the dimension."""
        return bool(self.weights(dimension))


def _profiles_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    return _DATA_PROFILES if _DATA_PROFILES.exists() else _SEED_PROFILES


def _coerce_weights(raw) -> dict[str, float]:
    """Coerce a {label: number} mapping to floats, dropping blanks. Never raises."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


@lru_cache(maxsize=8)
def _load_raw(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"ETF profiles file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Malformed profiles file (expected a mapping): {path}")
    return raw


def load_profiles(path: Path | None = None) -> dict[str, Profile]:
    """Load and resolve every fund into a flat ``{isin: Profile}`` mapping.

    Each fund row in ``funds:`` is merged onto its referenced ``indices:`` entry. Funds whose
    index reference is missing are skipped (they surface via :func:`exposure_gaps`).
    """
    raw = _load_raw(str(_profiles_path(path)))
    indices = raw.get("indices", {}) or {}
    funds = raw.get("funds", {}) or {}
    out: dict[str, Profile] = {}
    for isin, frow in funds.items():
        frow = frow or {}
        idx_id = frow.get("index")
        idx = indices.get(idx_id)
        if idx is None:
            continue
        out[isin] = Profile(
            isin=isin,
            ticker=frow.get("ticker", ""),
            index_id=idx_id,
            index_name=idx.get("index_name", idx_id),
            strategy=(idx.get("strategy") or "").strip(),
            asset_class=idx.get("asset_class", "unknown"),
            replication=frow.get("replication", idx.get("replication")),
            factor_tilt=list(idx.get("factor_tilt") or []),
            as_of=str(idx.get("as_of")) if idx.get("as_of") is not None else None,
            source=idx.get("source"),
            data_complete=bool(idx.get("data_complete", False)),
            region_weights=_coerce_weights(idx.get("region_weights")),
            country_weights=_coerce_weights(idx.get("country_weights")),
            sector_weights=_coerce_weights(idx.get("sector_weights")),
            credit_quality=_coerce_weights(idx.get("credit_quality")),
            top_holdings=list(idx.get("top_holdings") or []),
            asset_mix=_coerce_weights(idx.get("asset_mix")),
            hedged=frow.get("hedged"),
        )
    return out


def get_profile(isin: str, path: Path | None = None) -> Profile | None:
    """Return one fund's resolved profile, or ``None`` if it has no entry."""
    return load_profiles(path).get(isin)


@dataclass(frozen=True)
class ExposureResult:
    """Weighted portfolio exposure along one dimension, with a coverage caveat.

    ``exposure`` sums to ~1.0 (it is renormalised over the covered portion). ``coverage`` is
    the share of the *input* portfolio weight whose funds carried data for this dimension —
    the optimiser should treat a low-coverage result as unreliable.
    """

    dimension: str
    exposure: dict[str, float]
    coverage: float
    covered_weight: float
    total_weight: float
    missing: list[str]  # isins that contributed no data for this dimension


def portfolio_exposure(weights: dict[str, float], dimension: str,
                       path: Path | None = None) -> ExposureResult:
    """Aggregate a portfolio's weighted exposure along ``dimension``.

    ``weights`` is ``{isin: weight}`` (need not sum to 1 — it is normalised internally).
    Funds lacking data for the dimension are excluded and reported in ``missing``; the
    exposure is renormalised over the covered portion so it still sums to ~1.0. ``coverage``
    tells the caller how much of the portfolio actually informed the result.
    """
    if dimension not in _DIM_FIELD:
        raise ValueError(f"Unknown dimension {dimension!r}; use one of {WEIGHT_DIMENSIONS}")
    profiles = load_profiles(path)
    total = sum(w for w in weights.values() if w and w > 0)
    if total <= 0:
        return ExposureResult(dimension, {}, 0.0, 0.0, 0.0, list(weights.keys()))

    exposure: dict[str, float] = {}
    covered = 0.0
    missing: list[str] = []
    for isin, w in weights.items():
        if not w or w <= 0:
            continue
        prof = profiles.get(isin)
        dim_w = prof.weights(dimension) if prof else {}
        if not dim_w:
            missing.append(isin)
            continue
        covered += w
        # Normalise this fund's own breakdown defensively (it may sum slightly off 1.0),
        # then contribute it at the fund's portfolio weight.
        fund_total = sum(dim_w.values()) or 1.0
        for label, lw in dim_w.items():
            exposure[label] = exposure.get(label, 0.0) + w * (lw / fund_total)

    if covered <= 0:
        return ExposureResult(dimension, {}, 0.0, 0.0, total, missing)
    # Renormalise over the covered portion so the result sums to ~1.0.
    exposure = {k: v / covered for k, v in sorted(exposure.items(),
                                                  key=lambda kv: kv[1], reverse=True)}
    return ExposureResult(dimension, exposure, covered / total, covered, total, missing)


def asset_class_exposure(weights: dict[str, float],
                         path: Path | None = None) -> ExposureResult:
    """Aggregate a portfolio's weighted exposure by asset class (equity/bond/...).

    Asset class comes from each profile's ``asset_class`` field, so coverage here is simply
    "does the fund have a profile at all" — near-complete for the curated universe.
    """
    profiles = load_profiles(path)
    total = sum(w for w in weights.values() if w and w > 0)
    if total <= 0:
        return ExposureResult("asset_class", {}, 0.0, 0.0, 0.0, list(weights.keys()))
    exposure: dict[str, float] = {}
    covered = 0.0
    missing: list[str] = []
    for isin, w in weights.items():
        if not w or w <= 0:
            continue
        prof = profiles.get(isin)
        if prof is None:
            missing.append(isin)
            continue
        covered += w
        exposure[prof.asset_class] = exposure.get(prof.asset_class, 0.0) + w
    if covered <= 0:
        return ExposureResult("asset_class", {}, 0.0, 0.0, total, missing)
    exposure = {k: v / covered for k, v in sorted(exposure.items(),
                                                  key=lambda kv: kv[1], reverse=True)}
    return ExposureResult("asset_class", exposure, covered / total, covered, total, missing)


def exposure_gaps(isins: list[str] | None = None,
                  path: Path | None = None) -> dict[str, list[str]]:
    """Report profile-coverage gaps across a set of ISINs (defaults to all profiled funds).

    Returns ``{"no_profile": [...], "incomplete": [...], "no_sector": [...]}`` — the funds the
    optimiser can't fully constrain, so coverage gaps are explicit rather than silent.
    """
    profiles = load_profiles(path)
    targets = list(isins) if isins is not None else list(profiles.keys())
    no_profile, incomplete, no_sector = [], [], []
    for isin in targets:
        prof = profiles.get(isin)
        if prof is None:
            no_profile.append(isin)
            continue
        if not prof.data_complete:
            incomplete.append(isin)
        # A bond/commodity legitimately has no equity GICS sectors; only flag equities/reits.
        if prof.asset_class in ("equity", "reit") and not prof.has_dimension("sector"):
            no_sector.append(isin)
    return {"no_profile": no_profile, "incomplete": incomplete, "no_sector": no_sector}


def coverage_summary(path: Path | None = None) -> dict[str, int]:
    """Counts of full vs partial profiles across the whole file (for UI/reporting)."""
    profiles = load_profiles(path)
    full = sum(1 for p in profiles.values() if p.data_complete)
    return {"total": len(profiles), "full": full, "partial": len(profiles) - full}
