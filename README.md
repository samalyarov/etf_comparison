# ETF Comparison

A local, personal tool to gather, store, and compare **UCITS ETFs** (returns, cost, risk,
structure) to support buy-and-hold investing through Interactive Brokers. I made it for personal use (looking into investment options right now) and the thing is ~95% vibe-coded, so if you somehow happen upon this page - judge it as such.

See **[architecture.md](architecture.md)** for the full plan, data-source research, schema,
and roadmap.

## What it does

- **Ingests** end-of-day price history (+ dividends) for a watchlist of ~88 UCITS ETFs
  (across 15 categories) from Yahoo Finance, with Tiingo and Stooq as fallbacks, into a
  local **SQLite** database.
- **Computes** returns (trailing, CAGR, calendar-year, rolling), volatility (incl. rolling),
  Sharpe/Sortino, max drawdown, correlation, and DCA backtests — all total-return basis.
- **Presents** it in a themed **Streamlit** UI with a top nav and five pages:
  - **Compare** — growth of 100, calendar-year & drawdown, risk-vs-return scatter,
    correlation heatmap, risk/return + trailing-return tables.
  - **Screener** — a risk-return map of the whole universe + a sortable, filterable table.
  - **Detail** — single-ETF price, monthly-return heatmap, rolling volatility/return.
  - **Strategy** — a dollar-cost-averaging backtest ("invest X/month for Y years") with
    final value, profit, money-multiple and XIRR.
  - **Data** — coverage/freshness and a one-click fetch.
- Keeps the raw data in a plain `.db` you can query with any SQL client.

The ETF universe is generated + verified against Yahoo by `scripts/build_watchlist.py`
(re-run it to extend the list). Charts use a colourblind-validated palette.

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

- `src/etf/` — `config`, `db`, `data`, `metrics`, `strategy` (DCA), `theme`, `ingest/`
  (source adapters), `app.py` (UI).
- `scripts/build_watchlist.py` — regenerates/verifies `watchlist.yaml` from a candidate list.
- `data/etf.db` — local SQLite database (git-ignored, regenerable via ingest).
- `watchlist.yaml` — the ETFs to track (auto-generated).
- `tests/` — unit tests for the analytics and strategy layers.

## Stack

Python 3.13 · SQLite · pandas · yfinance (→ Tiingo/Stooq) · Streamlit + Plotly ·
streamlit-option-menu.
