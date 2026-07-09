# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every change should be logged under **Unreleased** as it is merged, then rolled into a
version heading on release. This is the authoritative human-readable history of the tool.

## [Unreleased]

### Changed
- **Visual redesign — the *Meridian* design system** (`etf/theme.py`, `.streamlit/config.toml`,
  `app.py` chrome). Re-skins the app from the developer-IDE Tokyo Night palette to a calm,
  institutional finance direction (after the professional feel of firms like Lowden Financial):
  a deep **slate-navy** desk (Dark: plane `#0e1620`, card `#141f2b`, ink `#e7edf3`) and a warm
  **paper** desk (Light: plane `#e9e7e0`, card `#f6f5f1`, ink `#1b2733`), unified by a single
  muted **teal** accent (Light `#0c6b62` / Dark `#4bc5b5`) with reserved green/red up-down
  semantics. Both Light and Dark themes and the Native/EUR currency toggle keep working; all
  native-widget text fixes and themed HTML tables are preserved.
  - **Accessibility:** every text/UI token pair meets **WCAG AA** contrast in both themes —
    the Light `accent`, `good`, and `muted` tokens were tuned down until links, gain figures,
    and captions clear 4.5:1 on the paper surfaces; Dark already cleared AA throughout.
  - **Chrome:** a refined masthead (rising-line product mark + wordmark), a signature **meridian
    rule**, **tabular figures** on metrics/tables (a research ledger aligns its numbers), and
    accessible keyboard focus rings.
  - **Attribution:** the tool is credited to **Sam Maliarov** in the header byline and a
    persistent footer, each linking to his LinkedIn (`linkedin.com/in/semyon-malyarov`). Text +
    monogram only, no photograph.
  - **Charts:** the CVD-safe categorical series palettes are unchanged and stay in-band against
    the new surfaces (the slate card `#141f2b` and warm card `#f6f5f1` sit at effectively the
    same luminance as the previous surfaces they were validated on); any series-vs-series
    adjacency is mitigated as before by legends / direct labels / tables. The diverging ramps
    keep their red/blue poles with a neutral midpoint tuned to each surface.
  - Tests: added attribution/LinkedIn-present assertions and both-themes token-invariant checks
    to `tests/test_app.py`; existing UI/theme smoke tests re-run green in both themes and both
    currency modes.

### Added
- **Portfolio risk engine** (`etf/risk.py`, pure — numpy/pandas only): formal downside-risk
  layer on a built blend's realised return distribution, complementing the forward scenario fan
  in `etf/factors.py`.
  - **Value-at-Risk + Conditional VaR / Expected Shortfall** at 95% and 99% by **three
    independent methods**, reported as positive loss magnitudes with drift-aware **√t horizon
    scaling** (Basel square-root-of-time rule; documented i.i.d. assumption): **Historical**
    (empirical tail quantile, no distribution assumption); **Parametric (Gaussian)** with the
    closed-form normal ES `σ·φ(z)/(1−c) − μ`; a **Cornish-Fisher** fat-tail variant (modified VaR
    + Boudt-Peterson-Croux modified ES) that adjusts for sample skew and excess kurtosis and
    reduces **exactly** to the Gaussian forms at zero skew/kurtosis; and **Monte-Carlo**
    (seedable → deterministic) — univariate normal/bootstrap, or a **multivariate-normal** fit
    across funds preserving their covariance. Cornish-Fisher returns `NaN` (never a fabricated
    number) when the expansion leaves its domain of validity on the tail-to-median interval the
    VaR actually traverses (Maillard 2012), rather than gating on over-strict global
    monotonicity. `var_summary(...)` builds the method × confidence table.
  - **Contribution-to-risk** (`component_risk`) — marginal & component VaR/volatility per holding
    via the **Euler allocation**: `σ_p = √(wᵀΣw)` splits exactly into additive component vols
    that **sum to portfolio vol** (and, times `z_c`, to the parametric portfolio VaR); a small
    weight in a wild fund is correctly surfaced as a large risk share.
  - **Historical stress tests** (`stress_tests`) — replay of seven documented, dated crash
    windows against the portfolio's *current* target weights using real `adj_close` history:
    **GFC 2008** (2007-10-09→2009-03-09), **Euro crisis 2011** (2011-07-07→2011-10-04),
    **China/oil 2015-16** (2015-08-10→2016-02-11), **2018 Q4 selloff** (2018-09-20→2018-12-24),
    **COVID 2020** (2020-02-19→2020-03-23), **2022 rate shock** (2022-01-03→2022-10-12) and
    **SVB banking 2023** (2023-02-02→2023-03-13). Each reports portfolio drawdown, worst single
    day, window return and time-to-recovery; windows a portfolio has no data for (young funds)
    are reported **uncovered** with `NaN` losses — coverage is explicit, never fabricated.
  - **Portfolio page** gains a **Risk** section: the VaR/CVaR table (method × confidence) with a
    horizon selector and money-value translation, a contribution-to-risk chart + table, and a
    stress-scenario drawdown bar chart with recovery info. Honest caveats throughout (VaR is a
    threshold not a worst case; parametric assumes a shape; all are estimates of the *past*).
    Currency- and theme-aware; single-fund and thin-history blends handled gracefully.
  - Tests: `tests/test_risk.py` — the three methods agree on a synthetic Normal within
    tolerance and match the closed-form Gaussian VaR; historical VaR = empirical quantile;
    CVaR ≥ VaR; Cornish-Fisher reduces to Gaussian at zero moments, widens the tail for
    left-skewed fat tails, and returns NaN out of domain; √t scaling exact; component risk sums
    to the total; stress windows select the right dates/drawdown on a crafted crash and skip
    uncovered windows; Monte-Carlo deterministic under a seed. Plus UI smoke (both themes/
    currencies + horizon interaction) and data-integrity invariants (crash windows dated, ordered
    and lookahead-free; no fabricated losses on uncovered windows). Suite 197 → 221; ruff clean.
- **Constrained portfolio optimiser** (`etf/optimizer.py`, pure numerics): mean-variance
  optimisation with real-world constraints, solved with **cvxpy** using **PyPortfolioOpt**
  estimators (added `pyportfolioopt` to `pyproject.toml`; installs cleanly on Windows/py3.13
  and, via `pip install -e ".[dev]"`, on CI).
  - `optimize_portfolio(prices, *, objective, risk_free_rate, current_weights, constraints,
    return_method, with_frontier)` → `OptimizeResult` with optimal weights, expected
    return/vol/Sharpe, the realised **exposure breakdown + coverage**, which constraints are
    **binding**, and the solver status. Objectives: `max_sharpe` (tangency) and
    `min_volatility`; plus `efficient_frontier(...)` for context.
  - **Constraints** (`OptConstraints`): leverage — long-only (`w≥0, Σw=1`) *or* a gross-exposure
    cap `Σ|w| ≤ L` with shorting bounds (toggle); **turnover** `‖w − current_weights‖₁ ≤ τ`;
    per-asset min/max weight bounds; L2 (ridge) diversification; and **sector / region /
    asset-class exposure limits** (`ExposureLimit`, caps *and* floors).
  - **Method**: max-Sharpe is solved via the **Charnes-Cooper transformation** (`y = w/κ`) so
    every constraint's constants are scaled by `κ` — crucially the turnover anchor becomes
    `‖y − κ·w_prev‖₁ ≤ τ·κ`. This fixes a real PyPortfolioOpt limitation: its `max_sharpe`
    leaves `w_prev` unscaled and returns *infeasible* on a turnover constraint. Covariance is
    **Ledoit-Wolf shrinkage**; expected returns are mean-historical (default) or
    exponentially-weighted, with a prominent **estimation-sensitivity caveat** (Michaud's
    error-maximisation) in the module and UI.
  - **Exposure constraints use the profiles look-through** (`profiles.portfolio_exposure`
    basis): the exposure is a linear map `A·w` built from per-fund category shares; a fund with
    no look-through data contributes a **zero column** (not counted, never assumed), and every
    constrained dimension reports its **coverage** so partial-profile funds are surfaced, not
    silently constrained on incomplete data. Caps act on absolute covered exposure (conservative
    under partial coverage).
  - **Portfolio page** gains an **Optimiser** section: candidate set, objective, leverage /
    shorting, per-fund cap, per-sector/region caps, min-bonds floor, and an optional
    turnover limit anchored to pasted holdings. Shows optimal weights vs equal-weight, the
    **efficient frontier** with the tangency / min-vol / equal-weight / per-fund points, the
    resulting exposure breakdown + coverage, and binding constraints. Infeasible sets show a
    clear message (no crash); currency- and theme-aware; respects the risk-free-rate setting.
  - Tests: `tests/test_optimizer.py` (19) — 2-asset tangency vs closed form; every constraint
    provably respected (leverage, turnover, bounds, sector/asset-class limits); infeasible
    returns a clean status with no exception; coverage < 1 on partial profiles; frontier
    monotonicity; solver determinism. UI smoke (7) for the new section in both themes and both
    currencies, incl. the infeasible and shorting paths. Suite 171 → 197; ruff clean.
- **Factor-model portfolio builder** (`etf/factors.py`, pure): two complementary views plus a
  forward scenario modeller for a factor-ETF blend.
  - *(B) Regression-estimated exposures* — `factor_exposures(portfolio_returns, factor_returns)`
    OLS-regresses a built portfolio's **excess** returns on the Fama/French-Carhart factors
    and returns factor **betas** (loadings), annualised **alpha**, R²/adjusted-R², and
    per-coefficient standard errors and t-statistics. Follows the standard convention: the
    market factor is an excess return so `RF` is subtracted from the portfolio too; monthly
    frequency; plain OLS with classical SEs (numpy only — no statsmodels/scipy). Aligns dates,
    drops NaN, and raises `InsufficientData` on too little overlap.
  - *(A) Factor-ETF building blocks* — `sleeve_contributions(prices, weights)` decomposes a
    blend's growth-of-100 into each factor sleeve's **additive** contribution (buy-and-hold
    basis; sums exactly to the total gain), reusing `portfolio.blend_index` alignment.
  - *Purchasing-strategy scenarios* — `plan_scenarios(...)` projects a lump-sum + recurring
    plan into a **best / base / worst** percentile fan (reusing `projection.py`'s
    bootstrap/Monte-Carlo) and an explicit **market-crash** replay that applies the blend's own
    worst historical drawdown-and-recovery path (`worst_crash_window`, real data — nothing
    fabricated) to the plan. Scope: the forward plan-level fan only; VaR/formal stress replays
    are a separate later task, deliberately not built here.
- **Factor return data** (Ken French Data Library — European factors): new ingest adapter
  `etf/ingest/kenfrench.py` fetches the developed-**Europe** Fama/French 5 factors
  (`Mkt-RF, SMB, HML, RMW, CMA`, plus `RF`) and Carhart momentum (`WML`) from the freely
  published CSV zips, parses percent→decimal with `-99.99`→NaN, and stores the **monthly**
  matrix in a new additive `factor_returns` table (`db.upsert_factor_returns`,
  `data.load_factor_returns`). New CLI flag `python -m etf.ingest --factors-ken`; non-blocking
  (reports the fetch as pending and falls back to a committed fixture if the download
  rate-limits). Monthly frequency is the documented choice (academic standard for factor
  regressions; matches the projection's monthly basis). Note: the European factors are
  USD-denominated, so loadings read as relative *tilts* rather than currency-exact betas.
- **Portfolio page — "Factor model" section**: pick MSCI World factor sleeves
  (value / momentum / quality / size / min-volatility / multifactor — already in the universe),
  set a per-sleeve purchasing strategy, and see (a) the realised blend with per-factor
  contribution, (b) the built portfolio's regression factor loadings (bar chart + R²/alpha and
  a plain-English tilt read), and (c) the best/base/worst + market-crash scenario fan — reusing
  `render_table`, the dataviz palette and `theme`; currency- and theme-aware.
- Tests: `tests/test_factors.py` (Ken French parsing incl. annual-section boundary and the
  `-99.99` sentinel, store/load roundtrip, regression recovering KNOWN betas and the
  excess-return convention, date alignment and insufficient-overlap handling, additive sleeve
  decomposition, crash-window drawdown/recovery detection, and scenario-fan monotonicity +
  crash shape); `factor_returns` decimals/no-lookahead invariants in
  `tests/test_data_integrity.py`; and Portfolio factor-section UI smoke coverage in both themes
  and both currency modes. Suite 145 → 171.
- **ETF strategy / exposure profiles** (`etf/profiles.py`, pure; `scripts/etf_profiles.yaml`,
  committed seed): a look-through dataset for the whole universe — every fund maps to the
  index it tracks, and each index carries a researched breakdown of `strategy`,
  `region_weights`, `country_weights`, GICS `sector_weights`, `top_holdings`, `factor_tilt`,
  `replication`, plus `as_of`/`source` and a `data_complete` flag. Bonds use a
  `credit_quality` bucket instead of equity sectors; multi-asset funds carry an `asset_mix`.
  Data is verified from justETF / issuer / index-provider factsheets (dated May 2026); where a
  breakdown wasn't sourced the field is left empty and `data_complete: false` — **no weights
  are invented** (CLAUDE.md §4). All 91 database instruments are profiled (41 full
  look-through, 51 partial index/region-level). Stored DRY as an index library + fund→index
  map; the seed survives a DB rebuild (git-ignored `data/etf_profiles.yaml` overrides if
  present, mirroring the watchlist/settings pattern).
- **Exposure-aggregation API** for the upcoming portfolio optimiser:
  `profiles.portfolio_exposure(weights, dimension)` aggregates a portfolio's weighted
  region/country/sector/credit exposure, **renormalising over the covered portion and
  reporting `coverage`** so a constraint is never computed silently on partial data;
  `asset_class_exposure`, `exposure_gaps` and `coverage_summary` round it out.
- **Detail page — "Strategy & exposure · look-through"**: per-fund strategy, region/sector/
  credit bar charts, top-countries and top-holdings tables, tilt tags and a Full/Partial data
  indicator — theme- and currency-aware, reusing `render_table` and the dataviz palette. The
  **Data page** gains a universe-level profile-coverage panel (profiled / full / partial).
- Tests: `tests/test_profiles.py` (loader, hand-computed aggregation, missing-data
  renormalisation/coverage, seed schema validity with present weights summing ~1.0, and a
  real-DB check that every stored instrument has a profile or is explicitly pending); plus
  Detail/Data UI smoke tests in both themes and both currency modes. Suite 129 → 145.
- **Bond income modelling** (`etf/bonds.py`, pure): turns a stored `close` series + the
  `distributions` table into two side-by-side scenarios — **(a) distributions reinvested**
  (accumulating-equivalent total-return path, cross-checked against the stored `adj_close`)
  and **(b) distributions cashed out** (units held on the price-return path plus a separate
  accumulated cash-income pile). Also trailing-12-month distribution yield. Surfaced in a new
  **Bonds · income, reinvest vs cash out, Dutch tax** section on the Portfolio page with a
  reinvest/cash-out toggle, net-worth + cumulative-income charts, and metrics — currency- and
  theme-aware.
- **Dutch box-3 tax** (`etf/tax.py`, pure): both regimes as configurable, sourced functions.
  *Box 3 (2026)* — the current **fictitious-return wealth tax**: a forfaitair rendement on
  assets above the heffingsvrij vermogen, taxed at the box-3 rate (independent of coupons
  actually received). *Werkelijk rendement* — the postponed **actual-return** reform: tax on
  real return (coupons + value change) above a small tax-free result. Parameters verified
  July 2026: heffingsvrij vermogen **€59,357/person**, investment forfait **6.00%** (definitive),
  savings 1.28% / debt 2.70% (provisional), rate **36%**; actual-return tax-free result
  **€1,800/person** at **36%**, intended start **2028** (postponed from 2027; adopted by the
  Tweede Kamer 2026-02-12, pending Eerste Kamer). Simplifying assumption for the actual-return
  view: it taxes the window's *average annual* return; the law applies the allowance yearly.
- **Universe**: added three Eurozone/German government bond ETFs to broaden duration and
  add Dutch/Eurozone govvie exposure — iShares Euro Govt Bond 1-3yr Acc (`CBE3.L`), iShares
  Euro Govt Bond 15-30yr Dist (`IBGL.AS`), iShares eb.rexx Government Germany Dist (`EXHA.DE`).
  All ingested with full price history; the two distributing funds carry real coupon streams.
- **Tests**: `tests/test_bonds.py` (yield, cash-out income, reinvest/cash-out equivalence and
  ordering), `tests/test_tax.py` (hand-computed box-3 and actual-return arithmetic), a
  total-return reconciliation data-integrity check (adj_close vs close+coupons for every
  distributing bond), a distributing-bond coupon-stream invariant, and Portfolio bond-section
  UI smoke coverage in both themes and both currency modes.

### Notes
- Bond finding surfaced by the reconciliation work: for long-duration govvies (`IBGL.AS`,
  `EXHA.DE`) whose coupons were reinvested near the 2020-21 price peak and then crushed by the
  2022 rate shock, **cashing out beat reinvesting** — reinvesting is not universally superior,
  so the ordering is only asserted on synthetic monotonic-rising prices, not real data.

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
