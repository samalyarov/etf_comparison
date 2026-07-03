"""Project paths, environment, and watchlist loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# src/etf/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "etf.db"
WATCHLIST_PATH = PROJECT_ROOT / "watchlist.yaml"

# Load .env (real keys live here; it is git-ignored). Safe if the file is absent.
load_dotenv(PROJECT_ROOT / ".env")


def get_api_key(name: str) -> str | None:
    """Return an API key from the environment, or None if unset/blank."""
    value = os.environ.get(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Instrument:
    """A single ETF as declared in the watchlist."""

    isin: str
    ticker: str
    name: str
    category: str | None = None
    asset_class: str | None = None
    region: str | None = None
    domicile: str | None = None
    replication: str | None = None
    acc_dist: str | None = None
    index_name: str | None = None
    ter: float | None = None
    currency: str | None = None
    exchange: str | None = None


def load_watchlist(path: Path | None = None) -> list[Instrument]:
    """Parse watchlist.yaml into a list of Instrument records."""
    path = path or WATCHLIST_PATH
    if not path.exists():
        raise FileNotFoundError(f"Watchlist not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("etfs", [])
    known = Instrument.__dataclass_fields__.keys()
    instruments: list[Instrument] = []
    for entry in entries:
        filtered = {k: v for k, v in entry.items() if k in known}
        instruments.append(Instrument(**filtered))
    return instruments
