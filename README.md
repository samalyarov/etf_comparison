# ETF Comparison

A local, personal tool to gather, store, and compare **UCITS ETFs** (returns, cost, risk,
structure) to support buy-and-hold investing through Interactive Brokers. I made it for personal use (looking into investment options right now) and the thing is ~95% vibe-coded, so if you somehow happen upon this page - judge it as such.

The architecture is summarised in the [diagram below](#architecture); deeper working notes
live in a local, git-ignored `brain/` knowledge base.

## What it does

- **Ingests** end-of-day price history (+ dividends) for a watchlist of ~88 UCITS ETFs
  (across 15 categories) from Yahoo Finance, with Tiingo and Stooq as fallbacks, into a
  local **SQLite** database. **Data-quality repair is built into ingestion**: GBX/GBP
  (pence↔pounds) mis-denomination and isolated bad prints are auto-detected and fixed, and
  anything still suspect (e.g. an unadjusted split) is flagged rather than trusted.
- **Normalises currency**: the universe mixes EUR/GBP/GBp/USD/CHF; a Native/**EUR** toggle
  FX-converts everything for honest comparison and DCA (rates cached locally).
- **Computes** returns (trailing, CAGR, calendar-year, rolling), volatility, Sharpe/Sortino,
  max drawdown, correlation, DCA backtests, **benchmark-relative** metrics (beta, tracking
  error, information ratio, up/down capture), **rolling-window return distributions**, a
  **regime/stress lens**, and an all-in **cost & tax** model (IBKR commissions, tracking
  difference, spread/FX, EU dividend-tax drag, domicile notes) — all total-return basis.
- **Presents** it in a themed **Streamlit** UI (switchable **Light / Dark** — a Tokyo Night
  palette) with a top nav and seven pages:
  - **Recommended** — selectable ranking (return / CAGR / CAGR-after-TER / Sharpe / Sortino
    / lowest drawdown), 5/10/15y lookback, per-category winners, and a market-context strip
    (US 10Y yield, VIX).
  - **Compare** — growth of 100, calendar-year & drawdown, risk-vs-return scatter,
    correlation heatmap, risk/return (incl. all-in cost & CAGR-after-cost) + trailing tables.
  - **Screener** — a risk-return map of the whole universe, a filterable/taggable table,
    favourites filter, and CSV export.
  - **Portfolio** — build a weighted blend, rebalanced backtest (vs drift), blended metrics,
    low-correlation suggestions, lump-sum-vs-DCA, and a contribution-only **rebalancing
    assistant** (paste your holdings).
  - **Detail** — price, monthly-return heatmap, rolling vol/return, cost & tax panel,
    benchmark-relative + regime + any-window-returns panels, an opt-in local **sentiment**
    read, fund fact-sheet download, and a persisted favourite/tag.
  - **Strategy** — a DCA backtest (net of commissions/FX) **plus a forward projection**
    (OLS trend or Monte-Carlo/bootstrap fan) of net worth up to 40 years.
  - **Data** — coverage/freshness, a data-health report, staleness flags, and a fetch button.
- Keeps the raw data in a plain `.db` you can query with any SQL client.

The ETF universe is generated + verified against Yahoo by `scripts/build_watchlist.py`
(re-run it to extend the list; it auto-drops tickers that fail verification repeatedly).
Charts use a colourblind-validated palette. Preferences and tags persist between sessions.

## Architecture

Local-first, single-user, and built as four clean layers — **ingest → store → analyse →
present**. Everything runs on your machine; the network is touched only during a deliberate
*fetch*. Prices land in one **SQLite** file you can open with any SQL client, and every
analytic (returns, risk, correlation, DCA) is computed on top of that raw layer — never
written back into it. A separate, self-correcting pipeline turns a hand-maintained candidate
list into a *verified* watchlist by probing which Yahoo listings actually return data.

```mermaid
flowchart TD
    CAND["scripts/candidates.yaml<br/>(hand-maintained universe)"] --> BW["build_watchlist.py<br/>(probe Yahoo listings)"]
    BW --> WL["watchlist.yaml<br/>(verified)"]

    subgraph SRC["Data sources — network"]
        Y["yfinance (primary)"]
        TI["Tiingo (fallback)"]
        ST["Stooq (fallback)"]
    end

    subgraph ING["Ingest — src/etf/ingest"]
        AD["Source adapters<br/>one common interface"]
        ORC["Orchestrator + CLI<br/>python -m etf.ingest"]
        QC["quality.py<br/>GBX/GBP repair · de-spike · flag"]
    end

    DB[("SQLite<br/>data/etf.db<br/>instruments · prices · distributions ·<br/>fund_facts · fx_rates · macro · data_health · log")]

    subgraph ANA["Analyse — src/etf"]
        MET["metrics.py<br/>returns · risk · correlation · benchmark"]
        STR["strategy.py · projection.py<br/>DCA · XIRR · Monte-Carlo"]
        POR["portfolio.py · costs.py · fx.py<br/>blends · cost/tax · currency"]
    end

    subgraph PRE["Present"]
        UI["Streamlit app · app.py<br/>Recommended · Compare · Screener · Portfolio ·<br/>Detail · Strategy · Data"]
        RAW["Raw SQL / pandas<br/>(any DB client)"]
    end

    THEME["theme.py<br/>Light / Dark (Tokyo Night)<br/>validated palette"]

    WL --> ORC
    Y --> AD
    TI --> AD
    ST --> AD
    AD --> QC --> ORC --> DB
    DB --> MET --> UI
    DB --> STR --> UI
    DB --> POR --> UI
    DB --> RAW
    THEME -.-> UI
```

Layer boundaries map to modules: `ingest/` (adapters + orchestration), `db.py` + `data.py`
(storage + queries), `metrics.py` + `strategy.py` (analytics), `app.py` + `theme.py` (UI).
Swapping a data source, adding a metric, or restyling the UI each touches exactly one layer.

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
python -m etf.ingest                 # full history (repairs + FX rates included)
python -m etf.ingest --incremental   # only new rows since last run (faster)
python -m etf.ingest --only CSPX.L   # a single ETF
python -m etf.ingest --sources tiingo,yahoo,stooq   # change source priority
python -m etf.ingest --if-stale 7    # only fetch if data >7 days old (scheduled job)
python -m etf.ingest --repair        # re-run data-quality repair, no network
python -m etf.ingest --fx            # backfill quote currencies + EUR FX rates
python -m etf.ingest --facts         # backfill AUM/inception + macro (10Y yield, VIX)

# 2. Launch the UI
streamlit run src/etf/app.py
```

For an unattended weekly refresh, point Windows Task Scheduler / cron at
`python -m etf.ingest --if-stale 7`.

Add/remove ETFs by editing [scripts/candidates.yaml](scripts/candidates.yaml) (each entry
lists candidate Yahoo tickers — `.DE` Xetra, `.L` London, `.AS` Amsterdam), then run
`python scripts/build_watchlist.py` to re-verify and regenerate `watchlist.yaml`.

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

- `src/etf/` — `config`, `db`, `data`, `quality` (data-quality repair), `fx` (currency
  normalisation), `costs` (cost & tax model), `metrics`, `strategy` (DCA), `projection`
  (forward projections), `portfolio` (blends + rebalancing), `sentiment` (local scorer),
  `settings` (persisted prefs), `theme`, `ingest/` (source adapters), `app.py` (UI).
- `scripts/build_watchlist.py` — regenerates/verifies `watchlist.yaml`; auto-drops tickers
  that fail verification repeatedly (`scripts/failures.yaml` ledger, git-ignored).
- `data/etf.db` — local SQLite database (git-ignored, regenerable via ingest). Also holds
  `data_health`, `fx_rates`, `macro_series`; `data/settings.json` stores UI preferences.
- `watchlist.yaml` — the ETFs to track (auto-generated).
- `tests/` — unit tests for analytics, quality, FX, costs, projections, portfolio,
  sentiment and robustness, plus UI/theme smoke tests.

## Stack

Python 3.13 · SQLite · pandas / numpy · yfinance (→ Tiingo/Stooq) · Streamlit + Plotly ·
streamlit-option-menu.
