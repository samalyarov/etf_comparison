"""Tests for the ETF strategy/exposure profiles (etf.profiles) and the committed seed.

Two layers, mirroring test_data_integrity:
* pure-logic tests on load / aggregation / renormalisation (always run);
* seed-schema validation (always run — the seed is committed);
* a real-DB check that every stored instrument has a profile (skipped without data/etf.db).
"""

from __future__ import annotations

import pytest

from etf import profiles
from etf.config import DB_PATH


# --------------------------------------------------------------------------- loader / seed
def test_seed_loads_and_resolves():
    profs = profiles.load_profiles()
    assert len(profs) > 80  # whole curated universe
    iwda = profs["IE00B4L5Y983"]
    assert iwda.ticker == "IWDA.AS"
    assert iwda.index_name == "MSCI World"
    assert iwda.asset_class == "equity"
    assert iwda.data_complete is True
    # A verified holding is carried through verbatim.
    assert iwda.top_holdings[0]["name"] == "NVIDIA"


def test_every_profile_has_required_keys_and_valid_weights():
    profs = profiles.load_profiles()
    for isin, p in profs.items():
        for key in profiles.REQUIRED_KEYS:
            assert getattr(p, key) not in (None, ""), f"{isin} missing {key}"
        # Any present weight mapping must sum to ~1.0 (an explicit "Other" bucket is allowed).
        for dim in ("region_weights", "country_weights", "sector_weights", "credit_quality"):
            w = getattr(p, dim)
            if w:
                s = sum(w.values())
                assert abs(s - 1.0) <= 0.02, f"{isin}.{dim} sums to {s:.4f}"
                assert all(0.0 <= v <= 1.0 for v in w.values()), f"{isin}.{dim} weight range"
        # Holdings weights are fractions in [0, 1].
        for h in p.top_holdings:
            assert 0.0 <= float(h["weight"]) <= 1.0, f"{isin} holding weight out of range"


def test_full_profiles_carry_sector_country_and_holdings():
    # data_complete: true is a promise — such a fund must actually carry the look-through.
    profs = profiles.load_profiles()
    for isin, p in profs.items():
        if p.data_complete and p.asset_class in ("equity", "reit"):
            assert p.sector_weights, f"{isin} flagged full but has no sectors"
            assert p.country_weights, f"{isin} flagged full but has no countries"
            assert p.top_holdings, f"{isin} flagged full but has no holdings"


def test_full_bond_profiles_carry_country_and_credit():
    # For bonds, "full" promises a country breakdown and a credit-quality bucket
    # (GICS sectors don't apply). Enforce that so a bond can't be flagged full empty.
    profs = profiles.load_profiles()
    for isin, p in profs.items():
        if p.data_complete and p.asset_class == "bond":
            assert p.country_weights, f"{isin} bond flagged full but has no countries"
            assert p.credit_quality, f"{isin} bond flagged full but has no credit quality"


def test_lookthrough_coverage_floor():
    # The 2026-07 look-through pass raised full profiles to 82/92 (41 partials upgraded).
    # Coverage is monotonic — assert a floor, never a brittle exact count, so future
    # research can only push it up.
    cs = profiles.coverage_summary()
    assert cs["total"] >= 90
    assert cs["full"] >= 82, cs
    assert cs["partial"] == cs["total"] - cs["full"]


def test_upgraded_funds_carry_full_lookthrough():
    # Spot-check funds that moved partial -> full: they must resolve to their index and
    # carry a real, non-empty look-through (guards against a future accidental blank-out).
    profs = profiles.load_profiles()
    for isin, index_id in [
        ("IE00BKX55T58", "ftse_developed"),      # VHVG
        ("IE00BKM4GZ66", "msci_em_imi"),         # EIMI
        ("IE00B14X4Q57", "msci_emu"),            # CEU2
        ("IE00BP3QZ601", "msci_world_quality"),  # IWQU
        ("IE00B0M63284", "epra_nareit_europe"),  # IPRP (reit)
    ]:
        p = profs[isin]
        assert p.index_id == index_id, isin
        assert p.data_complete is True, isin
        assert p.country_weights and p.sector_weights and p.top_holdings, isin


# --------------------------------------------------------------------------- aggregation
def test_portfolio_exposure_hand_computed():
    # 50% MSCI World (IT 0.3027) + 50% S&P 500 (IT 0.3888) -> IT 0.34575.
    res = profiles.portfolio_exposure(
        {"IE00B4L5Y983": 0.5, "IE00B5BMR087": 0.5}, "sector")
    assert res.coverage == pytest.approx(1.0)
    assert res.exposure["Information Technology"] == pytest.approx(0.34575, abs=1e-4)
    assert sum(res.exposure.values()) == pytest.approx(1.0, abs=1e-6)


def test_portfolio_exposure_normalises_input_weights():
    # Unnormalised inputs (2:2) must give the same result as 0.5:0.5.
    res = profiles.portfolio_exposure(
        {"IE00B4L5Y983": 2.0, "IE00B5BMR087": 2.0}, "sector")
    assert res.exposure["Information Technology"] == pytest.approx(0.34575, abs=1e-4)


def test_missing_data_renormalises_and_reports_coverage():
    # Gold (commodity) has no sector breakdown: it drops out, coverage halves, exposure
    # renormalises over the covered (equity) half and still sums to ~1.0.
    res = profiles.portfolio_exposure(
        {"IE00B4L5Y983": 0.5, "IE00B4ND3602": 0.5}, "sector")
    assert res.coverage == pytest.approx(0.5)
    assert "IE00B4ND3602" in res.missing
    assert sum(res.exposure.values()) == pytest.approx(1.0, abs=1e-6)
    # The covered portion equals MSCI World's own breakdown.
    assert res.exposure["Information Technology"] == pytest.approx(0.3027, abs=1e-4)


def test_zero_and_empty_portfolios_are_safe():
    assert profiles.portfolio_exposure({}, "sector").coverage == 0.0
    assert profiles.portfolio_exposure({"IE00B4L5Y983": 0.0}, "region").coverage == 0.0


def test_unknown_dimension_raises():
    with pytest.raises(ValueError):
        profiles.portfolio_exposure({"IE00B4L5Y983": 1.0}, "nonsense")


def test_asset_class_exposure():
    res = profiles.asset_class_exposure({"IE00B4L5Y983": 0.6, "IE00B4ND3602": 0.4})
    assert res.exposure["equity"] == pytest.approx(0.6)
    assert res.exposure["commodity"] == pytest.approx(0.4)
    assert res.coverage == pytest.approx(1.0)


def test_exposure_gaps_structure():
    gaps = profiles.exposure_gaps()
    assert set(gaps) == {"no_profile", "incomplete", "no_sector"}
    # Bonds/commodities legitimately have no GICS sector, but must NOT appear as no_profile.
    assert "IE00B4ND3602" not in gaps["no_profile"]


# --------------------------------------------------------------------------- real-DB check
_HAS_DB = DB_PATH.exists()
db_test = pytest.mark.skipif(not _HAS_DB, reason="needs a populated data/etf.db")


@db_test
def test_every_db_instrument_has_a_profile_or_is_pending():
    # Data-integrity invariant: the optimiser must not hit an unprofiled fund silently.
    import sqlite3

    from etf.config import DB_PATH as _DB
    conn = sqlite3.connect(_DB)
    try:
        db_isins = [r[0] for r in conn.execute("SELECT isin FROM instruments")]
    finally:
        conn.close()
    profs = profiles.load_profiles()
    # PENDING: ISINs deliberately not yet profiled (empty — every fund is covered today).
    pending: set[str] = set()
    missing = [i for i in db_isins if i not in profs and i not in pending]
    assert not missing, f"DB instruments with no exposure profile: {missing}"
