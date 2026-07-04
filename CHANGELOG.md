# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every change should be logged under **Unreleased** as it is merged, then rolled into a
version heading on release. This is the authoritative human-readable history of the tool.

## [Unreleased]

## [1.0.0] — 2026-07-04

First production release. The tool graduates from a personal price/metrics viewer to a
full UCITS ETF research desk with built-in data quality, currency/cost/tax honesty, and
forward-looking decision support.

### Added
- **Data-quality repair at ingest** (`etf/quality.py`): backward power-of-100 continuity
  reconstruction fixes GBX/GBP (pence↔pounds) mis-denomination — single-day flips,
  multi-month wrong-scale runs and permanent regime shifts — and a neighbour-midpoint
  de-spike repairs isolated bad prints. Genuinely ambiguous cases (e.g. unadjusted splits)
  are flagged `suspect`, never fabricated. New `data_health` table, Data-tab health +
  staleness report, `--repair` CLI (no network), and distribution/total-return
  reconciliation.
- **FX normalisation to EUR** (`etf/fx.py`): quote currency captured per fund from Yahoo
  (reliable GBp signal), daily FX cached in `fx_rates`, and a Native/EUR toggle across
  Compare/Screener/Detail/Strategy/Portfolio. `--fx` backfill CLI.
- **Cost & tax model** (`etf/costs.py`): IBKR-style commissions + FX conversion cost,
  total cost of ownership (TER + tracking difference + amortised spread/FX), EU dividend
  tax drag (accumulating vs distributing) and IE/LU domicile notes. DCA backtests are
  netted for trading friction.
- **Forward projections** (`etf/projection.py`): OLS-trend and Monte-Carlo/bootstrap
  net-worth fan charts (percentile bands) up to 40 years, on the Strategy page.
- **Portfolio builder** (`etf/portfolio.py`) + new page: weighted, periodically-rebalanced
  blend backtest (vs let-it-drift), blended metrics, correlation-aware low-correlation
  suggestions, lump-sum-vs-DCA, and a contribution-only rebalancing assistant (paste your
  holdings; buy-only, tax-aware).
- **Analytics depth** (`etf/metrics.py`): beta, tracking error, information ratio, up/down
  capture vs a benchmark; rolling any-window return distributions; a regime/stress lens
  (2018 Q4, COVID, 2022 rate shock, …).
- **Recommended tab**: selectable ranking basis (return / CAGR / CAGR-after-TER / Sharpe /
  Sortino / lowest drawdown), 5/10/15y lookback, and per-category winners.
- **Data breadth**: AUM + inception fundamentals and macro context (US 10Y yield, VIX) via
  `--facts`; surfaced on Detail and Recommended.
- **Sentiment** (`etf/sentiment.py`): a local, no-key finance-tone scorer with an opt-in,
  contrarian-framed Detail panel (paste headlines → context, never a trigger).
- **UX**: persisted preferences + per-fund favourites/tags (`etf/settings.py`,
  `data/settings.json`), CSV export, downloadable fact sheet.
- **Robustness**: `--if-stale N` scheduled fetch, distribution-triggered full refetch for
  incremental-ingest correctness, `build_watchlist.py` auto-drop of repeatedly-failing
  tickers, UI/theme smoke tests, and data-integrity tests.
- **CI**: GitHub Actions runs ruff + the full pytest suite on every push and PR to `main`.

### Notes
- Deferred (blocked on external data/infra): holdings & overlap and factor tilts (need an
  issuer/justETF holdings feed); live IBKR via `ib_async` (needs TWS/Gateway — the manual
  CSV import is the shipped alternative).

## [0.1.0] — prior

- Storage + ingest + metrics + themed Streamlit UI (Recommended/Compare/Screener/Detail/
  Strategy/Data); ~88 UCITS ETFs across 15 categories; DCA backtest with XIRR; Light/Dark
  (Tokyo Night) themes; 10Y-return recommendations.

[Unreleased]: https://github.com/samalyarov/etf_comparison/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/samalyarov/etf_comparison/releases/tag/v1.0.0
