"""Persisted user preferences — a tiny JSON store next to the database.

Single-user, local: theme, base currency, risk-free rate, dividend-tax assumption, and a
per-fund favourites/tags map ("core" / "satellite" / "considering") survive between
sessions. Kept deliberately simple (one JSON file, git-ignored, regenerable) rather than a
settings table, matching the project's zero-setup ethos.
"""

from __future__ import annotations

import json

from .config import DATA_DIR

SETTINGS_PATH = DATA_DIR / "settings.json"

DEFAULTS: dict = {
    "theme": "Dark",
    "currency": "Native",
    "risk_free": 2.0,
    "dividend_tax": 26.0,
    "favourites": {},   # isin -> tag ("core" | "satellite" | "considering")
}

TAGS = ["core", "satellite", "considering"]


def load() -> dict:
    """Load settings, filling any missing keys from :data:`DEFAULTS`."""
    out = dict(DEFAULTS)
    try:
        if SETTINGS_PATH.exists():
            stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                out.update({k: v for k, v in stored.items() if k in DEFAULTS})
    except (json.JSONDecodeError, OSError):
        pass
    if not isinstance(out.get("favourites"), dict):
        out["favourites"] = {}
    return out


def save(settings: dict) -> None:
    """Persist settings (best-effort; failures are swallowed to never break the UI)."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        clean = {k: settings.get(k, DEFAULTS[k]) for k in DEFAULTS}
        SETTINGS_PATH.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    except OSError:
        pass


def set_favourite(settings: dict, isin: str, tag: str | None) -> dict:
    """Set or clear a fund's tag; returns the mutated settings (also persists)."""
    favs = dict(settings.get("favourites") or {})
    if tag:
        favs[isin] = tag
    else:
        favs.pop(isin, None)
    settings["favourites"] = favs
    save(settings)
    return settings
