"""Tests for config-driven curation and scheduled/incremental ingest helpers."""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

from etf import db, ingest

_SPEC = importlib.util.spec_from_file_location(
    "build_watchlist",
    Path(__file__).resolve().parents[1] / "scripts" / "build_watchlist.py")
build_watchlist = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_watchlist)


def test_update_failures_increments_and_clears():
    existing = {"A": 1, "B": 2}
    updated = build_watchlist.update_failures(existing, resolved_isins={"A"},
                                              unresolved_isins={"B", "C"})
    assert "A" not in updated          # resolved -> streak cleared
    assert updated["B"] == 3           # still failing -> incremented
    assert updated["C"] == 1           # new failure -> starts at 1


def test_update_failures_resolved_not_in_ledger():
    updated = build_watchlist.update_failures({}, resolved_isins={"X"}, unresolved_isins=set())
    assert updated == {}


def test_data_age_days(tmp_path):
    dbp = tmp_path / "t.db"
    db.init_db(dbp)
    with db.connect(dbp) as conn:
        assert ingest.data_age_days(conn) is None  # empty
        conn.execute("INSERT INTO instruments (isin, ticker) VALUES ('IE1', 'X')")
        recent = (date.today() - timedelta(days=3)).isoformat()
        conn.execute("INSERT INTO prices (isin, date, adj_close) VALUES ('IE1', ?, 100)",
                     (recent,))
    with db.connect(dbp) as conn:
        assert ingest.data_age_days(conn) == 3


def test_run_if_stale_skips_fresh(tmp_path, monkeypatch):
    dbp = tmp_path / "t.db"
    db.init_db(dbp)
    with db.connect(dbp) as conn:
        conn.execute("INSERT INTO instruments (isin, ticker) VALUES ('IE1', 'X')")
        conn.execute("INSERT INTO prices (isin, date, adj_close) VALUES ('IE1', ?, 100)",
                     (date.today().isoformat(),))
    orig_connect = db.connect
    monkeypatch.setattr(ingest.db, "connect", lambda *a, **k: orig_connect(dbp))
    # Should skip because data is fresh (0 days < 7).
    called = {"ran": False}
    monkeypatch.setattr(ingest, "run", lambda **kw: called.__setitem__("ran", True))
    out = ingest.run_if_stale(7)
    assert out.get("skipped") is True and called["ran"] is False
