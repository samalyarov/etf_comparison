# ETF Comparison — Architecture & Plan

> A local, personal tool to gather, store, and compare ETF data (historical returns,
> costs, risk, holdings) so I can make informed decisions about what to buy through
> Interactive Brokers.

Status: **draft / planning** · Last updated: 2026-07-03

---

## 1. Goal & scope

**What I want to be able to do:**

1. **Discover** which ETFs exist / are relevant to me (by asset class, region, index, issuer).
2. **Compare** them on the things that actually matter for a buy-and-hold ETF investor:
   - Historical **total return** over multiple horizons (1M … 10Y, since inception).
   - **Cost** — the Total Expense Ratio (TER) is the single biggest controllable drag.
   - **Risk** — volatility, max drawdown, correlation to other holdings.
   - **Structure** — accumulating vs distributing, physical vs synthetic replication,
     domicile, fund currency, AUM (size / liquidity).
3. Have **raw data available locally** for ad-hoc querying (a local DB I can hit with SQL).
4. Have a **simple UI** to compare a handful of ETFs side by side with charts.

**Explicit non-goals (keep it simple):**

- Not a trading bot, not live/real-time data, not intraday. **End-of-day (EOD) data is enough.**
- Not multi-user, not a hosted web service. Runs locally, for one person.
- Not tax advice — but the data model *is* tax-aware (acc/dist, domicile) because it matters.

---

## 2. Guiding principles

- **Local-first & offline-capable.** Once data is fetched it lives in a local DB; the UI
  reads from the DB, never hits the network on the hot path. Fetch is a deliberate step.
- **Raw layer + derived layer.** Store raw prices/facts as pulled; compute metrics
  (returns, Sharpe, drawdown) on top. Never overwrite raw data with derived numbers.
- **Source-agnostic ingestion.** Each data provider sits behind a common interface so I can
  swap/add sources without touching storage or UI. No single free source is fully reliable.
- **Idempotent, incremental ingestion.** Re-running a fetch upserts; it only pulls the gap
  since the last stored date. Safe to run daily/weekly from a scheduled task.
- **Reproducible.** Pinned dependencies, a fixed Python (3.13), everything scriptable.

---

## 3. High-level architecture

```
                +---------------------------------------------------------------+
                |                        SOURCES (network)                      |
                |  yfinance | Tiingo API | Stooq | FMP | IBKR (ib_async) | ...  |
                +-----------------------------+---------------------------------+
                                              |  (deliberate "fetch" step)
                                              v
   +----------------------------------------------------------------------------+
   |  INGESTION LAYER  (etf.ingest)                                              |
   |   - one adapter per source, common interface: get_prices(), get_facts()    |
   |   - normalises to a canonical schema, upserts, incremental by last date    |
   +-----------------------------+----------------------------------------------+
                                 |
                                 v
   +----------------------------------------------------------------------------+
   |  STORAGE  (SQLite: etf.db)                                                  |
   |   instruments · prices · distributions · fund_facts · ingest_log           |
   |   (raw, canonical, queryable directly with any SQL client)                 |
   +-----------------------------+----------------------------------------------+
                                 |
                                 v
   +----------------------------------------------------------------------------+
   |  ANALYTICS LAYER  (etf.metrics)                                            |
   |   returns · CAGR · volatility · Sharpe/Sortino · max drawdown ·            |
   |   rolling returns · correlation matrix · TER-adjusted comparisons          |
   +----------------+-------------------------------------+---------------------+
                    |                                     |
                    v                                     v
        +-----------------------+            +--------------------------------+
        |  UI  (Streamlit app)  |            |  Raw access (SQL / notebooks)  |
        |  compare, chart, rank |            |  DBeaver, sqlite3, pandas      |
        +-----------------------+            +--------------------------------+
```

Four clean layers: **ingest → store → analyse → present**. Each is a separate Python
module so they can be tested and evolved independently.

---

## 4. What data do we actually need?

| Category | Fields | Why it matters | Cadence |
|---|---|---|---|
| **Identity** | ticker, ISIN, name, exchange, currency | Key everything off ISIN (stable) not ticker | once / rarely |
| **Prices** | date, open, high, low, close, **adj_close**, volume | Adjusted close drives *total return* | daily EOD |
| **Distributions** | ex-date, amount, currency | Dividend/coupon yield; needed if only price (not adj) close is available | as they occur |
| **Fund facts** | **TER**, AUM, inception date, domicile, replication (physical/synthetic), **acc/dist**, index tracked, asset class, region | The heart of ETF *comparison* — costs & structure | monthly-ish |
| **Holdings** *(optional)* | top-N holdings, sector/country weights | Overlap analysis between ETFs | monthly-ish |

> **Total return vs price return:** the key correctness point. A distributing ETF's raw
> `close` ignores dividends and *understates* return. Always compare on **adjusted close**
> (dividends reinvested) or reconstruct total return from `close + distributions`.

---

## 5. Data sources — research (current as of mid-2026)

No single free source is complete *and* reliable, so the design blends a few. Findings:

### Price / EOD history

| Source | Access | Coverage | Free tier | Verdict for this project |
|---|---|---|---|---|
| **yfinance** (Yahoo) | `pip install yfinance` | Huge, global ETFs; adj close + dividends | Free, **unofficial** | **Primary, easy start.** Still actively maintained (release Jun 2026) but scrapes Yahoo, so expect 429 "rate limited" errors — cache aggressively, fetch in small batches, back off. Fine for personal EOD use. |
| **Tiingo** | REST API + key | 45k+ ETFs/funds, 30+ yrs, cleaned | Free key (~500/hr, 1000/day) | **Recommended reliability upgrade.** Proper API, clean multi-exchange data. Best "serious" free option; use as primary once I have a key. |
| **Stooq** | CSV download / `pandas-datareader` `"stooq"` | 20k+ global securities & ETFs, 20+ yrs | Free, **no real API** | **Backup / redundancy.** Good for bulk CSV backfill and cross-checking Yahoo. |
| **IBKR** via `ib_async` | TWS/IB Gateway must be running | Exactly what I can actually trade | Free with account | **Best "truth" source** — it's the venue I'll trade on. Clunky: needs the desktop Gateway running + strict pacing limits. Add in a later phase. |

### Fundamentals (TER, holdings, structure)

| Source | Notes |
|---|---|
| **yfinance** `.info` / `.funds_data` | Free; gives expense ratio, top holdings, sector weights for many ETFs. Patchy for European (UCITS) listings. First stop. |
| **Financial Modeling Prep (FMP)** | Free tier; dedicated ETF holdings + expense-ratio endpoints. Good for holdings/overlap. |
| **Alpha Vantage** | Free key but very low quota (~25 req/day in 2026) — use sparingly for one-off fundamentals, not bulk. |
| **justETF** (Europe) | The de-facto European ETF screener (TER, domicile, acc/dist, replication). No official free API — use for manual research / one-off seeding of the instrument list. |
| **Issuer sites / factsheets** | iShares, Vanguard, Xtrackers, etc. — authoritative for TER, AUM, methodology. Manual or light scraping for a curated watchlist. |

### Recommended blend

- **Phase 1:** `yfinance` for both prices *and* basic fundamentals — zero signup, fastest path.
- **Phase 2:** add **Tiingo** (free key) as the reliable primary for prices; keep yfinance/Stooq as fallback + cross-check.
- **Phase 3 (optional):** `ib_async` against IB Gateway for prices that match my execution venue; FMP for richer holdings.

> ⚠️ **Domicile matters a lot and is an open decision (see §13).** If I'm investing from
> Europe, I'll mostly buy **UCITS** ETFs (US-domiciled ones are largely un-buyable under
> PRIIPs and carry US estate-tax/withholding issues). US-based investors buy US-domiciled
> ETFs. This changes the *universe*, the best fundamentals source (justETF vs US screeners),
> and the acc/dist + tax framing. The schema stores `domicile` and `acc_dist` from day one.

---

## 6. Storage design

**Choice: SQLite** (`etf.db`, a single file). Rationale: zero setup, ships with Python,
readable by every SQL client (DBeaver, `sqlite3` CLI, pandas `read_sql`), trivially
backed up (copy the file), perfect for a single-user local tool. Meets the "easy access to
raw data / local DB" requirement directly.

> **Alternative considered — DuckDB:** columnar, extremely fast for analytical scans over
> long price series, reads/writes Parquet, still a single local file. If the analytics get
> heavy I can point the analytics layer at DuckDB (or even have it query the SQLite file /
> Parquet exports) without changing the ingestion or UI. Starting with SQLite for
> simplicity; DuckDB is the documented escape hatch.

### Schema (v1)

```sql
-- One row per ETF (keyed by ISIN; ticker can change / differ per exchange)
CREATE TABLE instruments (
    isin          TEXT PRIMARY KEY,
    ticker        TEXT,
    name          TEXT,
    exchange      TEXT,
    currency      TEXT,
    asset_class   TEXT,          -- equity / bond / commodity / ...
    region        TEXT,          -- world / US / EM / europe / ...
    domicile      TEXT,          -- IE, LU, US, ...  (tax-relevant)
    replication   TEXT,          -- physical / synthetic / sampled
    acc_dist      TEXT,          -- 'ACC' | 'DIST'
    inception     DATE,
    added_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily EOD bars. adj_close = dividends reinvested (total-return basis).
CREATE TABLE prices (
    isin      TEXT NOT NULL REFERENCES instruments(isin),
    date      DATE NOT NULL,
    open      REAL, high REAL, low REAL, close REAL,
    adj_close REAL,
    volume    INTEGER,
    source    TEXT,              -- 'yfinance' | 'tiingo' | 'stooq' | 'ibkr'
    PRIMARY KEY (isin, date)
);

-- Cash distributions (for total-return reconstruction & yield)
CREATE TABLE distributions (
    isin    TEXT NOT NULL REFERENCES instruments(isin),
    ex_date DATE NOT NULL,
    amount  REAL NOT NULL,
    currency TEXT,
    PRIMARY KEY (isin, ex_date)
);

-- Slowly-changing fund facts; keep history via snapshot_date
CREATE TABLE fund_facts (
    isin          TEXT NOT NULL REFERENCES instruments(isin),
    snapshot_date DATE NOT NULL,
    ter           REAL,          -- e.g. 0.0007 = 0.07%
    aum           REAL,
    index_name    TEXT,
    yield_ttm     REAL,
    source        TEXT,
    PRIMARY KEY (isin, snapshot_date)
);

-- Audit: what was fetched, when, how far
CREATE TABLE ingest_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    isin        TEXT, source TEXT, kind TEXT,   -- 'prices'|'facts'|...
    from_date   DATE, to_date DATE,
    rows        INTEGER, status TEXT, message TEXT,
    run_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_prices_date ON prices(date);
```

Optional later: `holdings` (isin, snapshot_date, holding_name, weight) for overlap analysis.

---

## 7. Ingestion / ETL pipeline

- **Adapter interface** — each source implements the same small contract:
  ```python
  class PriceSource(Protocol):
      def get_prices(self, isin_or_ticker, start, end) -> pd.DataFrame: ...
      def get_facts(self, isin_or_ticker) -> dict: ...
  ```
- **Watchlist-driven.** A `watchlist.yaml` (or an `instruments` seed) lists the ETFs I care
  about. Ingestion iterates the watchlist, so I only pull what I'll actually compare.
- **Incremental.** For each ETF, look up `MAX(date)` in `prices`, fetch only the gap → upsert.
- **Idempotent upserts** (`INSERT ... ON CONFLICT DO UPDATE`). Re-runs are safe.
- **Resilience.** Per-ETF try/except with backoff + jitter (yfinance 429s); log every run to
  `ingest_log`; one bad ticker never aborts the batch.
- **Entry point.** `python -m etf.ingest --all` (or `--watchlist path`), runnable by hand or
  from Windows Task Scheduler on a weekly cadence.

---

## 8. Analytics / metrics layer

Pure functions over a price DataFrame (from the DB). Computed on demand, cached in-process.

**Return & growth**
- Period total returns: 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y, 10Y, since inception (on `adj_close`).
- **CAGR** (annualised).
- Growth-of-10k / normalised-to-100 series for charting.
- Rolling N-year returns (distribution of outcomes, not just one window).

**Risk**
- Annualised **volatility** (stdev of daily log returns × √252).
- **Max drawdown** (peak-to-trough) and drawdown curve.
- **Sharpe** & **Sortino** (configurable risk-free rate).
- **Correlation matrix** across the compared set (diversification / overlap check).

**Cost & structure**
- TER and **TER-adjusted** return comparisons (what fees cost over the horizon).
- Distribution yield (TTM) from `distributions`.
- Currency note: report base-currency and (later) optional FX-adjusted returns.

All metrics take explicit date ranges and a common calendar so ETFs are compared like-for-like.

---

## 9. Access layer

Two deliberate ways in, matching the "raw data **and** UI" requirement:

1. **Programmatic / raw** — `etf.db` is plain SQLite. Query it directly from:
   - a SQL client (DBeaver / `sqlite3` CLI),
   - pandas: `pd.read_sql(...)`,
   - a Jupyter notebook for exploratory analysis.
   A thin `etf.data` module exposes helpers (`load_prices(isin, start, end)`, `list_etfs()`).
2. **UI** — the Streamlit app (below) for point-and-click comparison.

---

## 10. UI

**Choice: Streamlit.** Minimal code, great for a local personal data app, first-class charts,
runs with `streamlit run app.py` in the browser. (Alternatives weighed: Plotly Dash — more
control, more boilerplate; a Jupyter notebook — great for exploration, weak as a reusable
tool; a static HTML report — no interactivity. Streamlit is the sweet spot.)

**Planned screens**
- **Compare** — multiselect ETFs → normalised growth chart, returns table, risk table, TER,
  correlation heatmap, drawdown chart. The core view.
- **Screener / rank** — sort the watchlist by CAGR / TER / Sharpe / drawdown with filters
  (asset class, region, acc-vs-dist).
- **Detail** — single-ETF page: facts, price history, distributions, rolling returns.
- **Data admin** — trigger a fetch, show `ingest_log`, freshness per ETF.

Charts via Plotly (interactive) or Altair. Streamlit `@st.cache_data` for DB reads.

---

## 11. Tech stack

| Concern | Choice |
|---|---|
| Language / runtime | **Python 3.13** in a local **`.venv`** |
| Storage | **SQLite** (`etf.db`); DuckDB as documented escape hatch |
| Data access | `sqlite3` / SQLAlchemy Core + **pandas** |
| Sources | `yfinance` → `tiingo`/`pandas-datareader` → `ib_async` (later) |
| Analytics | pandas, numpy (`quantstats` optional for ready-made risk metrics) |
| UI | **Streamlit** + Plotly |
| Config | `watchlist.yaml`, `.env` for API keys (git-ignored) |
| Quality | `pytest`, `ruff` (lint+format), `mypy` (optional) |
| Scheduling | Windows Task Scheduler → `python -m etf.ingest --all` (optional) |

---

## 12. Proposed project layout

```
etf_comparison/
├─ .venv/                     # local virtualenv (git-ignored)
├─ architecture.md            # this document
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ .env.example               # template for API keys (real .env is git-ignored)
├─ watchlist.yaml             # the ETFs I care about
├─ data/
│  └─ etf.db                  # SQLite database (git-ignored)
├─ src/etf/
│  ├─ __init__.py
│  ├─ db.py                   # connection, schema init, upserts
│  ├─ data.py                 # read helpers (load_prices, list_etfs, ...)
│  ├─ ingest/
│  │  ├─ __init__.py          # orchestration, watchlist loop, incremental logic
│  │  ├─ base.py              # PriceSource protocol / common normalisation
│  │  ├─ yahoo.py             # yfinance adapter
│  │  ├─ tiingo.py            # Tiingo adapter
│  │  └─ stooq.py             # Stooq adapter
│  ├─ metrics.py              # returns, CAGR, vol, Sharpe, drawdown, corr
│  └─ app.py                  # Streamlit UI
└─ tests/
   ├─ test_metrics.py
   └─ test_ingest.py
```

---

## 13. Open decisions (need my input before/while building)

1. **Region / domicile — ✅ DECIDED: Europe / UCITS** (2026-07-03). The universe is
   **UCITS ETFs** (IE/LU domiciled); US-domiciled ETFs are largely un-buyable under PRIIPs.
   Consequences baked into the build: **justETF** is the primary fundamentals/screening
   source; the watchlist seeds with UCITS tickers/ISINs; **accumulating vs distributing** and
   fund currency are front-and-centre; expect patchier yfinance fundamentals for European
   listings (favour ISIN + `.L`/`.DE`/`.AS` tickers, cross-check with justETF/issuer factsheets).
2. **Base currency** for return reporting (EUR? USD?). Affects FX handling.
3. **Reliability vs zero-setup:** start pure-yfinance (Phase 1) or grab a free **Tiingo** key
   up front for cleaner data?
4. **Watchlist seed:** which ETFs to start with (e.g. a world-equity core like an
   MSCI ACWI / FTSE All-World tracker, plus a few candidates to compare)?

---

## 14. Phased roadmap

- **Phase 0 — scaffolding** ✅ `.venv` (Python 3.13) + git initialised; this plan written.
- **Phase 1 — walking skeleton.** SQLite schema + `db.py`; yfinance adapter; ingest a small
  hard-coded watchlist; a minimal Streamlit "compare" page with a normalised-growth chart.
- **Phase 2 — real comparison.** Full metrics layer (returns/CAGR/vol/Sharpe/drawdown/corr);
  returns + risk tables; screener/rank page; `watchlist.yaml`; `ingest_log` + freshness view.
- **Phase 3 — reliability & depth.** Add Tiingo (primary) + Stooq (fallback/cross-check);
  fund_facts (TER/AUM) ingestion; distributions & yield; correlation heatmap.
- **Phase 4 — optional power-ups.** `ib_async` prices from IB Gateway; holdings & overlap
  analysis; rolling-returns view; FX-adjusted returns; scheduled weekly fetch.

---

## 15. Sources (research)

- yfinance — [PyPI](https://pypi.org/project/yfinance/) · [GitHub](https://github.com/ranaroussi/yfinance) · [why it gets rate-limited](https://medium.com/@trading.dude/why-yfinance-keeps-getting-blocked-and-what-to-use-instead-92d84bb2cc01)
- Tiingo — [data coverage review (QuantStart)](https://www.quantstart.com/articles/evaluating-data-coverage-with-tiingo/)
- Stooq — [intro (QuantStart)](https://www.quantstart.com/articles/an-introduction-to-stooq-pricing-data/) · [free DB](https://stooq.com/db/)
- Interactive Brokers Python API — [ib_async (PyPI)](https://pypi.org/project/ib_async/) · [docs](https://ib-api-reloaded.github.io/ib_async/) · [IBKR historical data sources](https://www.interactivebrokers.com/campus/ibkr-quant-news/historical-market-data-sources/)
- Fundamentals APIs — [Financial Modeling Prep ETF holdings](https://site.financialmodelingprep.com/developer/docs/historical-etf-holdings-api) · [Alpha Vantage](https://www.alphavantage.co/) · [EODHD fundamentals](https://eodhd.com/financial-apis/stock-etfs-fundamental-data-feeds)
