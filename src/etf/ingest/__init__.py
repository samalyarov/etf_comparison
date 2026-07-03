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

from .. import db
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
    end = date.today() + timedelta(days=1)

    for source in sources:
        try:
            df = _fetch_with_retry(source, inst.ticker, start, end)
        except SourceError as exc:
            db.log_ingest(
                conn, inst.isin, source.name, "prices", status="miss", message=str(exc)
            )
            continue

        rows = db.upsert_prices(conn, inst.isin, df, source.name)
        db.log_ingest(
            conn, inst.isin, source.name, "prices",
            from_date=df.index.min().date() if len(df) else None,
            to_date=df.index.max().date() if len(df) else None,
            rows=rows, status="ok",
        )
        print(f"  ✓ {inst.ticker:<10} {rows:>5} price rows from {source.name}")

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
    args = parser.parse_args(argv)
    order = [s.strip() for s in args.sources.split(",")] if args.sources else None
    run(only=args.only, sources_order=order, incremental=args.incremental)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
