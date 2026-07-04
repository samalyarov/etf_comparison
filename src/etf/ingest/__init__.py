"""Ingestion orchestration and CLI.

Drives the watchlist: upserts instrument metadata, seeds fund facts (TER) from the
watchlist, then fetches prices from the configured sources (in priority order, with
fallback) and dividends from Yahoo. Every attempt is written to ``ingest_log``.

By default it re-fetches full history so that ``adj_close`` (a retro-adjusted,
total-return series) stays globally consistent — data volumes are tiny. Use
``--incremental`` to fetch only the gap since the last stored date (faster, but the
adjusted series can drift after a distribution).

Run:
    python -m etf.ingest                 # all ETFs, full refresh
    python -m etf.ingest --incremental   # only new rows
    python -m etf.ingest --sources tiingo,yahoo,stooq
    python -m etf.ingest --only VWCE.DE  # one ticker or ISIN
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import date, timedelta

from .. import db, quality
from ..config import Instrument, load_watchlist
from .base import SourceError
from .stooq import StooqSource
from .tiingo import TiingoSource
from .yahoo import YahooSource

# Yahoo-first: best coverage for UCITS listings; others are fallbacks/cross-checks.
DEFAULT_SOURCE_ORDER = ["yahoo", "tiingo", "stooq"]

_SOURCE_FACTORIES = {
    "yahoo": YahooSource,
    "tiingo": TiingoSource,
    "stooq": StooqSource,
}


def build_sources(order: list[str]):
    """Instantiate sources in the requested order, skipping unavailable ones."""
    sources = []
    for key in order:
        factory = _SOURCE_FACTORIES.get(key)
        if factory is None:
            print(f"  ! unknown source '{key}', skipping")
            continue
        src = factory()
        # Tiingo is only usable with a key.
        if getattr(src, "available", True) is False:
            print(f"  ! source '{key}' unavailable (no API key), skipping")
            continue
        sources.append(src)
    return sources


def _fetch_with_retry(source, ticker, start, end, retries=3):
    """Call a source with simple exponential backoff (handles Yahoo rate limits)."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return source.get_prices(ticker, start, end)
        except SourceError as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg or "too many" in msg:
                time.sleep(2 ** attempt + random.random())
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _has_recent_distribution(sources, ticker: str, since: date) -> bool:
    """True if Yahoo reports a dividend with ex-date >= ``since`` (drives full-refetch)."""
    for source in sources:
        if isinstance(source, YahooSource):
            try:
                divs = source.get_dividends(ticker)
            except Exception:  # noqa: BLE001
                return False
            return any(d >= since for d, _ in divs)
    return False


def data_age_days(conn) -> int | None:
    """Days since the most recent stored price across the universe (None if empty)."""
    row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
    if not row or not row[0]:
        return None
    return (date.today() - date.fromisoformat(row[0])).days


def ingest_instrument(conn, inst: Instrument, sources, *, incremental: bool) -> bool:
    """Fetch and store prices + dividends for one instrument. Returns True on success."""
    db.upsert_instrument(conn, inst)

    # Seed fund facts (TER) from the watchlist so the UI has costs even without an API.
    if inst.ter is not None:
        db.upsert_fact(
            conn, inst.isin, date.today(), ter=inst.ter, index_name=inst.index_name,
            source="watchlist",
        )

    start: date | None = None
    if incremental:
        last = db.last_price_date(conn, inst.isin)
        if last is not None:
            start = last - timedelta(days=5)  # small overlap to catch restatements
            # Incremental correctness: a distribution inside the gap retro-adjusts the whole
            # adj_close series, so an incremental fetch would drift. If any dividend has an
            # ex-date at/after the incremental start, force a FULL refetch for this fund.
            if _has_recent_distribution(sources, inst.ticker, start):
                print(f"  ↻ {inst.ticker:<10} distribution in gap → forcing full refetch")
                start = None
    end = date.today() + timedelta(days=1)

    for source in sources:
        try:
            df = _fetch_with_retry(source, inst.ticker, start, end)
        except SourceError as exc:
            db.log_ingest(
                conn, inst.isin, source.name, "prices", status="miss", message=str(exc)
            )
            continue

        # Data-quality gate: repair GBX/GBP mis-denomination and isolated bad prints
        # *before* storing, so downstream metrics never see the corruption. The outcome
        # is recorded in data_health and noted on the ingest log.
        df, report = quality.clean_prices(df)
        db.upsert_health(conn, inst.isin, report)

        rows = db.upsert_prices(conn, inst.isin, df, source.name)
        health_note = "" if report.status == "clean" else f" [{report.status}: {report.notes}]"
        db.log_ingest(
            conn, inst.isin, source.name, "prices",
            from_date=df.index.min().date() if len(df) else None,
            to_date=df.index.max().date() if len(df) else None,
            rows=rows, status="ok", message=report.notes,
        )
        print(f"  ✓ {inst.ticker:<10} {rows:>5} price rows from {source.name}{health_note}")

        # Quote currency: capture from Yahoo (reliable GBp/GBP/USD signal) for FX
        # normalisation. Best-effort; non-fatal.
        if isinstance(source, YahooSource):
            try:
                cur = source.get_currency(inst.ticker)
                if cur:
                    db.set_currency(conn, inst.isin, cur)
            except Exception:  # noqa: BLE001
                pass

        # Dividends: only Yahoo exposes them here; best-effort, non-fatal.
        if isinstance(source, YahooSource):
            try:
                divs = source.get_dividends(inst.ticker)
                n = db.upsert_distributions(conn, inst.isin, divs, source.name)
                if n:
                    db.log_ingest(conn, inst.isin, source.name, "distributions", rows=n)
            except Exception as exc:  # noqa: BLE001
                db.log_ingest(
                    conn, inst.isin, source.name, "distributions",
                    status="error", message=str(exc),
                )
        return True

    print(f"  ✗ {inst.ticker:<10} no source returned data")
    return False


def fetch_fx(conn) -> dict:
    """Fetch daily FX history for the base currency (EUR) and cache it in ``fx_rates``.

    Pulls each ``EUR<ccy>=X`` pair from Yahoo (full history), inverts to EUR-per-unit, and
    upserts. Called once per full ingest and by the ``--fx`` backfill. Returns per-currency
    row counts. Non-fatal: a failed pair is skipped.
    """
    import yfinance as yf

    from .. import fx

    counts: dict[str, int] = {}
    for ccy, symbol in fx.FX_YAHOO_SYMBOLS.items():
        try:
            raw = yf.download(symbol, period="max", progress=False, threads=False,
                              auto_adjust=False)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, __import__("pandas").MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            eur_per_unit = fx.eur_per_unit_from_pair(ccy, raw["Close"])
            counts[ccy] = db.upsert_fx(conn, ccy, eur_per_unit)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! FX {ccy} ({symbol}) failed: {exc}")
    print(f"  ↳ FX cached: {counts}")
    return counts


def fetch_macro(conn) -> dict:
    """Fetch macro-context series (US 10Y yield, VIX) from Yahoo and cache them."""
    import yfinance as yf

    symbols = {"US10Y": "^TNX", "VIX": "^VIX"}
    scale = {"US10Y": 0.1, "VIX": 1.0}  # ^TNX is yield*10
    counts: dict[str, int] = {}
    for name, sym in symbols.items():
        try:
            raw = yf.download(sym, period="max", progress=False, threads=False,
                              auto_adjust=False)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, __import__("pandas").MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            counts[name] = db.upsert_macro(conn, name, raw["Close"] * scale[name])
        except Exception as exc:  # noqa: BLE001
            print(f"  ! macro {name} ({sym}) failed: {exc}")
    print(f"  ↳ macro cached: {counts}")
    return counts


def backfill_facts(only: str | None = None) -> dict:
    """Backfill fund fundamentals (AUM, inception) + macro series — network, no price fetch."""
    db.init_db()
    watchlist = load_watchlist()
    if only:
        needle = only.upper()
        watchlist = [i for i in watchlist if needle in (i.ticker.upper(), i.isin.upper())]
    yahoo = YahooSource()
    n = 0
    with db.connect() as conn:
        for inst in watchlist:
            f = yahoo.get_fundamentals(inst.ticker)
            if not f:
                continue
            if f.get("aum"):
                db.upsert_fact(conn, inst.isin, date.today(), aum=f["aum"], source="yahoo")
            if f.get("inception"):
                db.set_inception(conn, inst.isin, f["inception"])
            n += 1
            print(f"  {inst.ticker:<10} aum={f.get('aum')} inception={f.get('inception')}")
        fetch_macro(conn)
    print(f"\nBackfilled fundamentals for {n}/{len(watchlist)} instruments.")
    return {"funds": n}


def backfill_fx(only: str | None = None) -> dict:
    """Backfill quote currencies (per instrument) and FX rates — network, no price fetch."""
    db.init_db()
    watchlist = load_watchlist()
    if only:
        needle = only.upper()
        watchlist = [i for i in watchlist if needle in (i.ticker.upper(), i.isin.upper())]
    yahoo = YahooSource()
    n_ccy = 0
    with db.connect() as conn:
        for inst in watchlist:
            cur = yahoo.get_currency(inst.ticker)
            if cur:
                db.set_currency(conn, inst.isin, cur)
                n_ccy += 1
                print(f"  {inst.ticker:<10} {cur}")
        fetch_fx(conn)
    print(f"\nBackfilled currency for {n_ccy}/{len(watchlist)} instruments.")
    return {"currencies": n_ccy}


def repair_stored(only: str | None = None) -> dict:
    """Re-run the data-quality repair over *already stored* prices (no network).

    Reads each instrument's price history, applies :func:`quality.clean_prices`, writes the
    cleaned rows back, and records the outcome in ``data_health``. Use after upgrading the
    quality logic, or to fix a database ingested before the gate existed. Returns a summary.
    """
    from ..data import load_prices  # local import: keeps the network-free path lightweight

    db.init_db()
    watchlist = load_watchlist()
    if only:
        needle = only.upper()
        watchlist = [i for i in watchlist if needle in (i.ticker.upper(), i.isin.upper())]
    counts = {"clean": 0, "repaired": 0, "suspect": 0}
    with db.connect() as conn:
        for inst in watchlist:
            df = load_prices(inst.isin)
            if df.empty:
                continue
            cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"]
                    if c in df.columns]
            cleaned, report = quality.clean_prices(df[cols])
            src = df["source"].mode().iat[0] if "source" in df and df["source"].notna().any() \
                else "repair"
            db.upsert_prices(conn, inst.isin, cleaned, str(src))
            db.upsert_health(conn, inst.isin, report)
            counts[report.status] = counts.get(report.status, 0) + 1
            if report.status != "clean":
                print(f"  {report.status:<8} {inst.ticker:<10} {report.notes}")
    print(f"\nRepair done: {counts}")
    return counts


def run_if_stale(max_age_days: int = 7, **kwargs) -> dict:
    """Run a full ingest only if the stored data is older than ``max_age_days``.

    Designed for an unattended weekly job (Windows Task Scheduler / cron) so a machine that
    was off doesn't skip a refresh, but a fresh DB isn't re-fetched needlessly.
    """
    db.init_db()
    with db.connect() as conn:
        age = data_age_days(conn)
    if age is not None and age < max_age_days:
        print(f"Data is {age} day(s) old (< {max_age_days}) — skipping fetch.")
        return {"skipped": True, "age_days": age}
    print(f"Data is {age} day(s) old — refreshing.")
    return run(**kwargs)


def run(only: str | None = None, sources_order: list[str] | None = None,
        incremental: bool = False) -> dict:
    """Ingest the whole watchlist (or one instrument). Returns a small summary dict."""
    db.init_db()
    watchlist = load_watchlist()
    if only:
        needle = only.upper()
        watchlist = [i for i in watchlist if needle in (i.ticker.upper(), i.isin.upper())]
        if not watchlist:
            raise SystemExit(f"No watchlist entry matching '{only}'")

    sources = build_sources(sources_order or DEFAULT_SOURCE_ORDER)
    if not sources:
        raise SystemExit("No usable data sources configured.")
    print(f"Sources (in priority order): {', '.join(s.name for s in sources)}")
    print(f"Ingesting {len(watchlist)} instrument(s), "
          f"mode={'incremental' if incremental else 'full'}\n")

    ok = 0
    with db.connect() as conn:
        for inst in watchlist:
            if ingest_instrument(conn, inst, sources, incremental=incremental):
                ok += 1
        # FX rates for base-currency (EUR) normalisation — once per run.
        if any(isinstance(s, YahooSource) for s in sources):
            try:
                fetch_fx(conn)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! FX fetch failed: {exc}")

    print(f"\nDone: {ok}/{len(watchlist)} instruments ingested.")
    return {"ok": ok, "total": len(watchlist)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest ETF data into the local database.")
    parser.add_argument("--only", help="Limit to one ticker or ISIN.")
    parser.add_argument(
        "--sources",
        help=f"Comma-separated source priority (default: {','.join(DEFAULT_SOURCE_ORDER)}).",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Fetch only new rows since the last stored date (faster; may drift adj_close).",
    )
    parser.add_argument(
        "--repair", action="store_true",
        help="Re-run data-quality repair over already-stored prices (no network fetch).",
    )
    parser.add_argument(
        "--fx", action="store_true",
        help="Backfill quote currencies and EUR FX rates (network, no price re-fetch).",
    )
    parser.add_argument(
        "--facts", action="store_true",
        help="Backfill fund fundamentals (AUM/inception) + macro series (network).",
    )
    parser.add_argument(
        "--if-stale", type=int, metavar="DAYS", default=None,
        help="Only fetch if stored data is older than DAYS (for a scheduled weekly job).",
    )
    args = parser.parse_args(argv)
    if args.repair:
        repair_stored(only=args.only)
        return 0
    if args.if_stale is not None:
        order = [s.strip() for s in args.sources.split(",")] if args.sources else None
        run_if_stale(args.if_stale, only=args.only, sources_order=order,
                     incremental=args.incremental)
        return 0
    if args.fx:
        backfill_fx(only=args.only)
        return 0
    if args.facts:
        backfill_facts(only=args.only)
        return 0
    order = [s.strip() for s in args.sources.split(",")] if args.sources else None
    run(only=args.only, sources_order=order, incremental=args.incremental)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
