"""Read helpers over the SQLite database (the query side of the app)."""

from __future__ import annotations

import pandas as pd

from . import db
from .config import DB_PATH


def list_etfs(db_path=DB_PATH) -> pd.DataFrame:
    """Return the instruments table joined with the latest TER, plus price coverage."""
    with db.connect(db_path) as conn:
        instruments = pd.read_sql_query("SELECT * FROM instruments ORDER BY name", conn)
        facts = pd.read_sql_query(
            """
            SELECT f.isin, f.ter, f.aum, f.yield_ttm
            FROM fund_facts f
            JOIN (SELECT isin, MAX(snapshot_date) AS d FROM fund_facts GROUP BY isin) latest
              ON f.isin = latest.isin AND f.snapshot_date = latest.d
            """,
            conn,
        )
        coverage = pd.read_sql_query(
            "SELECT isin, MIN(date) AS first_date, MAX(date) AS last_date, "
            "COUNT(*) AS n_prices FROM prices GROUP BY isin",
            conn,
        )
    out = instruments.merge(facts, on="isin", how="left").merge(coverage, on="isin", how="left")
    return out


def load_prices(isin: str, start=None, end=None, db_path=DB_PATH) -> pd.DataFrame:
    """Load a single ETF's price history as a DataFrame indexed by date."""
    query = "SELECT * FROM prices WHERE isin = ?"
    params: list = [isin]
    if start is not None:
        query += " AND date >= ?"
        params.append(str(start))
    if end is not None:
        query += " AND date <= ?"
        params.append(str(end))
    query += " ORDER BY date"
    with db.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    return df.set_index("date")


def price_matrix(isins: list[str], field: str = "adj_close", start=None, end=None,
                 db_path=DB_PATH) -> pd.DataFrame:
    """Return a date x isin matrix of one price field, aligned across instruments.

    Used for correlation and like-for-like comparison. Missing days are forward/back
    handling is left to the caller; this just pivots what's stored.
    """
    if not isins:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in isins)
    query = f"SELECT isin, date, {field} FROM prices WHERE isin IN ({placeholders})"
    params: list = list(isins)
    if start is not None:
        query += " AND date >= ?"
        params.append(str(start))
    if end is not None:
        query += " AND date <= ?"
        params.append(str(end))
    with db.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="isin", values=field).sort_index()


def load_distributions(isin: str, db_path=DB_PATH) -> pd.DataFrame:
    """Load an ETF's distribution history, indexed by ex-date."""
    with db.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT ex_date, amount, currency FROM distributions WHERE isin = ? ORDER BY ex_date",
            conn,
            params=[isin],
            parse_dates=["ex_date"],
        )
    return df.set_index("ex_date")


def macro_series(series: str | None = None, db_path=DB_PATH) -> pd.DataFrame:
    """Return cached macro-context rows (optionally one series), indexed by date."""
    with db.connect(db_path) as conn:
        try:
            q = "SELECT date, series, value FROM macro_series"
            params: list = []
            if series:
                q += " WHERE series = ?"
                params = [series]
            return pd.read_sql_query(q + " ORDER BY date", conn, params=params,
                                     parse_dates=["date"])
        except Exception:  # noqa: BLE001
            return pd.DataFrame()


def data_health(db_path=DB_PATH) -> pd.DataFrame:
    """Return the per-instrument data-quality report recorded at ingest, if any."""
    with db.connect(db_path) as conn:
        try:
            return pd.read_sql_query(
                "SELECT isin, status, rescaled_days, despiked_days, "
                "max_move_before, max_move_after, notes, checked_at FROM data_health",
                conn,
            )
        except Exception:  # noqa: BLE001 - table may not exist on an old DB
            return pd.DataFrame()


def ingest_log(limit: int = 100, db_path=DB_PATH) -> pd.DataFrame:
    """Return the most recent ingest-log rows."""
    with db.connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT run_at, isin, source, kind, from_date, to_date, rows, status, message "
            "FROM ingest_log ORDER BY id DESC LIMIT ?",
            conn,
            params=[limit],
        )
