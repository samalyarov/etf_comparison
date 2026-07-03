# ETF Comparison

A local, personal tool to gather, store, and compare ETF data (returns, cost, risk,
structure) to support buy-and-hold investing through Interactive Brokers.

See **[architecture.md](architecture.md)** for the full plan, data-source research, schema,
and roadmap.

## Status

Phase 0 — scaffolding done: Python 3.13 `.venv` + git repo + plan. Nothing to run yet.

## Setup

```powershell
# venv already created at .venv (Python 3.13)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Layout (planned)

- `architecture.md` — the plan and research.
- `src/etf/` — ingestion, storage, metrics, and the Streamlit UI (to be built).
- `data/etf.db` — local SQLite database (git-ignored, regenerable).
- `watchlist.yaml` — the ETFs to track (to be added).

## Stack

Python 3.13 · SQLite · pandas · yfinance (→ Tiingo/Stooq) · Streamlit + Plotly.
