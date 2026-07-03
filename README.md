# ETF Comparison

A local, personal tool to gather, store, and compare **UCITS ETFs** (returns, cost, risk,
structure) to support buy-and-hold investing through Interactive Brokers.

See **[architecture.md](architecture.md)** for the full plan, data-source research, schema,
and roadmap.

## What it does

- **Ingests** end-of-day price history (+ dividends) for a watchlist of ETFs from Yahoo
  Finance, with Tiingo and Stooq as fallbacks, into a local **SQLite** database.
- **Computes** returns (trailing & CAGR), volatility, Sharpe/Sortino, max drawdown, and
  cross-ETF correlation — all on a total-return (adjusted-close) basis.
- **Presents** it in a **Streamlit** UI (Compare / Screener / Detail / Data) and keeps the
  raw data in a plain `.db` you can query with any SQL client.

## Setup

```powershell
.\.venv\Scripts\Activate.ps1        # venv is Python 3.13
pip install -e ".[dev]"             # installs the package + deps (also pytest, ruff)
```

Optional: copy `.env.example` to `.env` and add a free `TIINGO_API_KEY` (Yahoo works without
any key; Tiingo is used as a fallback where it has coverage).

## Usage

```powershell
# 1. Fetch data for everything in watchlist.yaml (creates data/etf.db)
python -m etf.ingest                 # full history
python -m etf.ingest --incremental   # only new rows since last run (faster)
python -m etf.ingest --only CSPX.L   # a single ETF
python -m etf.ingest --sources tiingo,yahoo,stooq   # change source priority

# 2. Launch the UI
streamlit run src/etf/app.py
```

Add/remove ETFs by editing [watchlist.yaml](watchlist.yaml) (keyed by ISIN; `ticker` is the
Yahoo/exchange symbol, e.g. `.DE` Xetra, `.L` London, `.AS` Amsterdam).

## Raw data access

The database is a single file at `data/etf.db`. Query it directly:

```python
import pandas as pd, sqlite3
con = sqlite3.connect("data/etf.db")
pd.read_sql("SELECT * FROM prices WHERE isin = 'IE00B5BMR087'", con)
```

Or use the helpers in `etf.data` (`list_etfs`, `load_prices`, `price_matrix`, ...).

## Tests & lint

```powershell
pytest        # metrics unit tests
ruff check src tests
```

## Layout

- `src/etf/` — `config`, `db`, `data`, `metrics`, `ingest/` (source adapters), `app.py` (UI).
- `data/etf.db` — local SQLite database (git-ignored, regenerable via ingest).
- `watchlist.yaml` — the ETFs to track.
- `tests/` — unit tests for the analytics layer.

## Stack

Python 3.13 · SQLite · pandas · yfinance (→ Tiingo/Stooq) · Streamlit + Plotly.
