"""SQLite storage: schema, connections, and idempotent upserts.

The database is a single local file (``data/etf.db``). It holds the *raw* canonical
data pulled from sources; derived metrics are computed on top in :mod:`etf.metrics`.
Everything here is safe to re-run — writes are upserts.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import date

import pandas as pd

from .config import DB_PATH, DATA_DIR, Instrument

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments (
    isin        TEXT PRIMARY KEY,
    ticker      TEXT,
    name        TEXT,
    category    TEXT,
    exchange    TEXT,
    currency    TEXT,
    asset_class TEXT,
    region      TEXT,
    domicile    TEXT,
    replication TEXT,
    acc_dist    TEXT,
    index_name  TEXT,
    inception   DATE,
    added_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prices (
    isin      TEXT NOT NULL REFERENCES instruments(isin),
    date      DATE NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    adj_close REAL,
    volume    INTEGER,
    source    TEXT,
    PRIMARY KEY (isin, date)
);

CREATE TABLE IF NOT EXISTS distributions (
    isin     TEXT NOT NULL REFERENCES instruments(isin),
    ex_date  DATE NOT NULL,
    amount   REAL NOT NULL,
    currency TEXT,
    source   TEXT,
    PRIMARY KEY (isin, ex_date)
);

CREATE TABLE IF NOT EXISTS fund_facts (
    isin          TEXT NOT NULL REFERENCES instruments(isin),
    snapshot_date DATE NOT NULL,
    ter           REAL,
    aum           REAL,
    index_name    TEXT,
    yield_ttm     REAL,
    source        TEXT,
    PRIMARY KEY (isin, snapshot_date)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    isin      TEXT,
    source    TEXT,
    kind      TEXT,
    from_date DATE,
    to_date   DATE,
    rows      INTEGER,
    status    TEXT,
    message   TEXT,
    run_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS macro_series (
    date   DATE NOT NULL,
    series TEXT NOT NULL,   -- e.g. US10Y, VIX
    value  REAL NOT NULL,
    PRIMARY KEY (date, series)
);

CREATE TABLE IF NOT EXISTS fx_rates (
    date          DATE NOT NULL,
    quote         TEXT NOT NULL,   -- ISO currency of the priced instrument (USD, GBP, CHF)
    eur_per_unit  REAL NOT NULL,   -- EUR value of 1 unit of `quote` on `date`
    PRIMARY KEY (date, quote)
);

CREATE TABLE IF NOT EXISTS data_health (
    isin            TEXT PRIMARY KEY REFERENCES instruments(isin),
    checked_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status          TEXT,      -- clean | repaired | suspect
    rescaled_days   INTEGER,   -- GBX/GBP power-of-100 days repaired
    despiked_days   INTEGER,   -- isolated bad prints repaired
    max_move_before REAL,      -- worst |daily return| as fetched
    max_move_after  REAL,      -- worst |daily return| after repair
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
"""


@contextmanager
def connect(db_path=DB_PATH):
    """Yield a SQLite connection with foreign keys enabled, committing on success."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Columns added after the initial schema shipped: {table: {column: definition}}.
# init_db applies any that are missing, so an existing DB migrates in place.
_MIGRATIONS = {
    "instruments": {"category": "TEXT"},
}


def init_db(db_path=DB_PATH) -> None:
    """Create tables/indexes if absent, then apply additive column migrations."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        for table, columns in _MIGRATIONS.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for col, ddl in columns.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


def upsert_instrument(conn: sqlite3.Connection, inst: Instrument) -> None:
    """Insert or update an instrument's static metadata (preserves added_at)."""
    conn.execute(
        """
        INSERT INTO instruments
            (isin, ticker, name, category, exchange, currency, asset_class, region,
             domicile, replication, acc_dist, index_name, inception)
        VALUES (:isin, :ticker, :name, :category, :exchange, :currency, :asset_class, :region,
                :domicile, :replication, :acc_dist, :index_name, :inception)
        ON CONFLICT(isin) DO UPDATE SET
            ticker=excluded.ticker, name=excluded.name, category=excluded.category,
            exchange=excluded.exchange,
            currency=COALESCE(excluded.currency, instruments.currency),
            asset_class=excluded.asset_class, region=excluded.region,
            domicile=excluded.domicile, replication=excluded.replication,
            acc_dist=excluded.acc_dist, index_name=excluded.index_name,
            inception=COALESCE(excluded.inception, instruments.inception)
        """,
        {
            "isin": inst.isin,
            "ticker": inst.ticker,
            "name": inst.name,
            "category": inst.category,
            "exchange": inst.exchange,
            "currency": inst.currency,
            "asset_class": inst.asset_class,
            "region": inst.region,
            "domicile": inst.domicile,
            "replication": inst.replication,
            "acc_dist": inst.acc_dist,
            "index_name": inst.index_name,
            "inception": None,
        },
    )


def upsert_prices(conn: sqlite3.Connection, isin: str, df: pd.DataFrame, source: str) -> int:
    """Upsert an OHLCV DataFrame (indexed by date) for one instrument.

    Expects columns: open, high, low, close, adj_close, volume (missing ones tolerated).
    Returns the number of rows written.
    """
    if df is None or df.empty:
        return 0
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    rows: list[tuple] = []
    for idx, row in df.iterrows():
        d = idx.date() if hasattr(idx, "date") else idx
        rows.append(
            (
                isin,
                d.isoformat(),
                *[_num(row.get(c)) for c in cols],
                source,
            )
        )
    conn.executemany(
        """
        INSERT INTO prices (isin, date, open, high, low, close, adj_close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(isin, date) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, adj_close=excluded.adj_close,
            volume=excluded.volume, source=excluded.source
        """,
        rows,
    )
    return len(rows)


def upsert_distributions(
    conn: sqlite3.Connection, isin: str, dists: Iterable[tuple[date, float]], source: str
) -> int:
    """Upsert (ex_date, amount) distribution rows for one instrument."""
    rows = [
        (isin, d.isoformat() if hasattr(d, "isoformat") else str(d), float(amt), None, source)
        for d, amt in dists
        if amt is not None
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO distributions (isin, ex_date, amount, currency, source)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(isin, ex_date) DO UPDATE SET
            amount=excluded.amount, source=excluded.source
        """,
        rows,
    )
    return len(rows)


def upsert_fact(
    conn: sqlite3.Connection,
    isin: str,
    snapshot_date: date,
    *,
    ter: float | None = None,
    aum: float | None = None,
    index_name: str | None = None,
    yield_ttm: float | None = None,
    source: str = "watchlist",
) -> None:
    """Upsert a fund-facts snapshot (TER, AUM, ...) for one instrument on a date."""
    conn.execute(
        """
        INSERT INTO fund_facts (isin, snapshot_date, ter, aum, index_name, yield_ttm, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(isin, snapshot_date) DO UPDATE SET
            ter=COALESCE(excluded.ter, fund_facts.ter),
            aum=COALESCE(excluded.aum, fund_facts.aum),
            index_name=COALESCE(excluded.index_name, fund_facts.index_name),
            yield_ttm=COALESCE(excluded.yield_ttm, fund_facts.yield_ttm),
            source=excluded.source
        """,
        (isin, snapshot_date.isoformat(), ter, aum, index_name, yield_ttm, source),
    )


def set_currency(conn: sqlite3.Connection, isin: str, currency: str) -> None:
    """Store an instrument's quote currency (as reported by the source, e.g. GBp/GBP/USD)."""
    conn.execute("UPDATE instruments SET currency = ? WHERE isin = ?", (currency, isin))


def set_inception(conn: sqlite3.Connection, isin: str, inception: str) -> None:
    """Store an instrument's inception date (fund age)."""
    conn.execute(
        "UPDATE instruments SET inception = COALESCE(inception, ?) WHERE isin = ?",
        (inception, isin))


def upsert_macro(conn: sqlite3.Connection, series: str, values: pd.Series) -> int:
    """Upsert a date-indexed macro series (e.g. US10Y yield, VIX). Returns rows."""
    rows = [(idx.date().isoformat() if hasattr(idx, "date") else str(idx), series, float(val))
            for idx, val in values.dropna().items()]
    if not rows:
        return 0
    conn.executemany(
        "INSERT INTO macro_series (date, series, value) VALUES (?, ?, ?) "
        "ON CONFLICT(date, series) DO UPDATE SET value=excluded.value", rows)
    return len(rows)


def upsert_fx(conn: sqlite3.Connection, quote: str, series: pd.Series) -> int:
    """Upsert a date-indexed EUR-per-unit series for one quote currency. Returns rows."""
    rows = [(idx.date().isoformat() if hasattr(idx, "date") else str(idx), quote, float(val))
            for idx, val in series.dropna().items()]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO fx_rates (date, quote, eur_per_unit) VALUES (?, ?, ?)
        ON CONFLICT(date, quote) DO UPDATE SET eur_per_unit=excluded.eur_per_unit
        """,
        rows,
    )
    return len(rows)


def upsert_health(conn: sqlite3.Connection, isin: str, report) -> None:
    """Record the data-quality outcome for one instrument (one row per isin, upserted)."""
    conn.execute(
        """
        INSERT INTO data_health
            (isin, checked_at, status, rescaled_days, despiked_days,
             max_move_before, max_move_after, notes)
        VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(isin) DO UPDATE SET
            checked_at=CURRENT_TIMESTAMP, status=excluded.status,
            rescaled_days=excluded.rescaled_days, despiked_days=excluded.despiked_days,
            max_move_before=excluded.max_move_before, max_move_after=excluded.max_move_after,
            notes=excluded.notes
        """,
        (isin, report.status, report.rescaled_days, report.despiked_days,
         report.max_move_before, report.max_move_after, report.notes),
    )


def log_ingest(
    conn: sqlite3.Connection,
    isin: str,
    source: str,
    kind: str,
    *,
    from_date=None,
    to_date=None,
    rows: int = 0,
    status: str = "ok",
    message: str = "",
) -> None:
    """Append a row to the ingest audit log."""
    conn.execute(
        """
        INSERT INTO ingest_log (isin, source, kind, from_date, to_date, rows, status, message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            isin,
            source,
            kind,
            _isodate(from_date),
            _isodate(to_date),
            rows,
            status,
            message[:500],
        ),
    )


def last_price_date(conn: sqlite3.Connection, isin: str) -> date | None:
    """Return the most recent stored price date for an instrument, or None."""
    row = conn.execute("SELECT MAX(date) FROM prices WHERE isin = ?", (isin,)).fetchone()
    if row and row[0]:
        return date.fromisoformat(row[0])
    return None


def _num(value):
    """Coerce to float, mapping NaN/None to SQL NULL."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _isodate(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
