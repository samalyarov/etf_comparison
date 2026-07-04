"""Streamlit UI for comparing UCITS ETFs.

Run from the project root:
    streamlit run src/etf/app.py

Reads exclusively from the local SQLite database (populated by ``python -m etf.ingest``);
only the "Fetch" button on the Data page touches the network.
"""

from __future__ import annotations

import calendar
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from etf import costs, data, fx, metrics, portfolio, projection, strategy, theme
from etf.config import DB_PATH

st.set_page_config(page_title="ETF Comparison", layout="wide")


# --------------------------------------------------------------------------- data access
@st.cache_data(ttl=300)
def get_etfs() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    return data.list_etfs()


@st.cache_data(ttl=300)
def get_prices(isin: str) -> pd.Series:
    df = data.load_prices(isin)
    if df.empty:
        return pd.Series(dtype=float)
    return df["adj_close"].dropna()


@st.cache_data(ttl=300)
def get_price_matrix(isins: tuple[str, ...]) -> pd.DataFrame:
    return data.price_matrix(list(isins))


@st.cache_data(ttl=300)
def get_fx() -> pd.DataFrame:
    return fx.load_fx()


@st.cache_data(ttl=300)
def get_prices_cur(isin: str, to_eur: bool, currency: str | None) -> pd.Series:
    """Adjusted-close series, optionally converted to EUR using cached FX rates."""
    s = get_prices(isin)
    if not to_eur or s.empty:
        return s
    return fx.convert_to_base(s, currency, get_fx()).dropna()


def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:,.2f}%"


def money(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:,.0f}"


PERIOD_OFFSETS = {"1Y": pd.DateOffset(years=1), "3Y": pd.DateOffset(years=3),
                  "5Y": pd.DateOffset(years=5), "10Y": pd.DateOffset(years=10)}


def trim(s: pd.Series, lookback: str) -> pd.Series:
    if s.empty:
        return s
    if lookback in PERIOD_OFFSETS:
        return s[s.index >= s.index[-1] - PERIOD_OFFSETS[lookback]]
    if lookback == "YTD":
        return s[s.index >= pd.Timestamp(year=s.index[-1].year, month=1, day=1)]
    return s


@st.cache_data(ttl=300)
def get_recommended(isins: tuple[str, ...], years: int = 10, n: int = 3):
    """Rank the universe by total return over the last ``years`` and return the top ``n``.

    Returns ``(top_isins, leaderboard_df)``. Only ETFs with a full ``years``-year history
    are eligible for the ranking (a fair like-for-like window); if fewer than ``n`` qualify
    it falls back to the best full-history CAGR to fill the remaining slots.
    """
    period = f"{years}Y"
    rows = []
    for i in isins:
        s = get_prices(i)
        if s.empty or not metrics.has_clean_history(s):
            continue  # skip funds with corrupt price series (bad splits/adjustments)
        w = trim(s, period)
        rows.append({"isin": i, "ret": metrics.period_return(s, period),
                     "cagr": metrics.cagr(w), "vol": metrics.annualized_volatility(w),
                     "maxdd": metrics.max_drawdown(w)})
    board = pd.DataFrame(rows)
    if board.empty:
        return [], board
    board = board.sort_values("ret", ascending=False, na_position="last").reset_index(drop=True)
    top = board.dropna(subset=["ret"]).head(n)["isin"].tolist()
    if len(top) < n:  # not enough full-history funds — fill by best CAGR
        extra = board[~board["isin"].isin(top)].sort_values("cagr", ascending=False)
        top += extra.head(n - len(top))["isin"].tolist()
    return top, board


# --------------------------------------------------------------------------- masthead + nav
etfs = get_etfs()
n_etfs = len(etfs)
n_cats = etfs["category"].nunique() if not etfs.empty else 0

if "theme" not in st.session_state:
    st.session_state.theme = theme.DEFAULT_THEME

left, mid, right = st.columns([3, 1, 1])
with left:
    st.markdown('<p class="app-title">ETF Comparison</p>'
                '<p class="app-sub">Local research desk for UCITS ETFs · '
                f'{n_etfs} funds across {n_cats} categories</p>', unsafe_allow_html=True)
with mid:
    cur_choice = st.radio("Currency", ["Native", "EUR"], horizontal=True,
                          label_visibility="collapsed",
                          help="Native = each fund's own quote currency (mixed). "
                               "EUR = FX-normalised for honest comparison.")
with right:
    choice = st.radio("Theme", list(theme.THEMES.keys()),
                      index=list(theme.THEMES.keys()).index(st.session_state.theme),
                      horizontal=True, label_visibility="collapsed")
    st.session_state.theme = choice or st.session_state.theme

EUR_MODE = cur_choice == "EUR"
CUR = "EUR" if EUR_MODE else None  # None => native/mixed

T = theme.THEMES[st.session_state.theme]
st.markdown(theme.css(T), unsafe_allow_html=True)


def sf(fig, **kw):
    """Style a Plotly figure with the active theme."""
    return theme.style_fig(fig, T, **kw)


def render_table(obj, *, hide_index: bool = False, fmt: dict | None = None,
                 max_height: int | None = None):
    """Render a DataFrame (or Styler) as a themed HTML table that follows the toggle.

    ``st.dataframe`` paints on a canvas tied to the static base theme, so it can't switch
    to Light at runtime. HTML tables can, and give exact colour control in both modes.
    """
    styler = obj if hasattr(obj, "set_table_styles") else obj.style
    if fmt:
        styler = styler.format(fmt, na_rep="—")
    styler = styler.set_table_styles(theme.table_styles(T))
    if hide_index:
        styler = styler.hide(axis="index")
    style = f' style="max-height:{max_height}px;overflow-y:auto"' if max_height else ""
    st.markdown(f'<div class="tbl-wrap"{style}>{styler.to_html()}</div>',
                unsafe_allow_html=True)


_PAGE_NAMES = ["Recommended", "Compare", "Screener", "Portfolio", "Detail", "Strategy", "Data"]
_forced = os.environ.get("ETF_FORCE_PAGE")  # test hook: exercise any page via AppTest
if _forced in _PAGE_NAMES:
    page = _forced
else:
    page = option_menu(
        None, _PAGE_NAMES,
        icons=["star", "bar-chart-line", "funnel", "pie-chart", "graph-up", "calculator",
               "database"],
        orientation="horizontal", default_index=0, styles=theme.nav_styles(T),
    ) or "Recommended"

if etfs.empty:
    st.warning("No data yet. Open the **Data** tab and fetch, or run `python -m etf.ingest`.")
    st.stop()

name_by_isin = dict(zip(etfs["isin"], etfs["name"]))
ticker_by_isin = dict(zip(etfs["isin"], etfs["ticker"]))
ter_by_isin = dict(zip(etfs["isin"], etfs["ter"]))
cat_by_isin = dict(zip(etfs["isin"], etfs["category"]))
currency_by_isin = dict(zip(etfs["isin"], etfs.get("currency", pd.Series(dtype=object))))


def px(isin: str) -> pd.Series:
    """Adjusted-close series in the currently selected currency (native or EUR)."""
    return get_prices_cur(isin, EUR_MODE, currency_by_isin.get(isin))
label_by_isin = {r["isin"]: f"{r['name']}  ·  {r['ticker']}" for _, r in etfs.iterrows()}
isin_by_label = {v: k for k, v in label_by_isin.items()}

# Top performers over the last 10 years — used as the app-wide default selection.
REC_ISINS, REC_BOARD = get_recommended(tuple(etfs["isin"]))
rec_labels = [label_by_isin[i] for i in REC_ISINS if i in label_by_isin]
DEFAULT_TICKERS = ["VWCE.DE", "CSPX.L", "IWDA.AS", "EIMI.L"]
_fallback_labels = [label_by_isin[i] for i, t in ticker_by_isin.items()
                    if t in DEFAULT_TICKERS and i in label_by_isin]
# Everything defaults to the recommended top-3 (falling back to broad-market staples).
default_labels = rec_labels or _fallback_labels


# --------------------------------------------------------------------------- Compare
def render_compare():
    c1, c2, c3 = st.columns([3, 1, 1])
    labels = c1.multiselect("ETFs to compare", list(isin_by_label.keys()),
                            default=default_labels[:3] or list(isin_by_label.keys())[:3])
    lookback = c2.selectbox("Lookback", ["Max", "10Y", "5Y", "3Y", "1Y", "YTD"], index=2)
    rf = c3.number_input("Risk-free %", value=2.0, step=0.25) / 100.0

    if not labels:
        st.info("Pick at least one ETF above.")
        return
    selected = [isin_by_label[lbl] for lbl in labels]
    full = {i: px(i) for i in selected}
    full = {i: s for i, s in full.items() if not s.empty}
    if not full:
        st.warning("No price history for the selected ETFs.")
        return
    win = {i: trim(s, lookback) for i, s in full.items()}

    # --- Growth of 100 ---
    st.markdown('<p class="section-label">Growth of 100 · total return</p>', unsafe_allow_html=True)
    fig = go.Figure()
    for idx, i in enumerate(selected):
        s = win.get(i)
        if s is None or s.empty:
            continue
        norm = metrics.normalize_to_100(s)
        fig.add_trace(go.Scatter(x=norm.index, y=norm.values, name=ticker_by_isin[i],
                                 line=dict(color=T.color(idx), width=2)))
    sf(fig, height=380)
    fig.update_yaxes(title="Indexed to 100")
    st.plotly_chart(fig, width="stretch")

    # --- Annual returns | Drawdown ---
    a, b = st.columns(2)
    with a:
        st.markdown('<p class="section-label">Calendar-year return</p>', unsafe_allow_html=True)
        yfig = go.Figure()
        for idx, i in enumerate(selected):
            yr = metrics.calendar_year_returns(win.get(i, full[i]))
            if yr.empty:
                continue
            yfig.add_trace(go.Bar(x=yr.index.astype(str), y=yr.values, name=ticker_by_isin[i],
                                  marker_color=T.color(idx)))
        sf(yfig, height=320, hovermode="x")
        yfig.update_layout(barmode="group", bargap=0.25)
        yfig.update_yaxes(tickformat=".0%")
        st.plotly_chart(yfig, width="stretch")
    with b:
        st.markdown('<p class="section-label">Drawdown from peak</p>', unsafe_allow_html=True)
        dfig = go.Figure()
        for idx, i in enumerate(selected):
            s = win.get(i)
            if s is None or s.empty:
                continue
            dd = metrics.drawdown_series(s)
            dfig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=ticker_by_isin[i],
                                      line=dict(color=T.color(idx), width=1.5)))
        sf(dfig, height=320, showlegend=False)
        dfig.update_yaxes(tickformat=".0%")
        st.plotly_chart(dfig, width="stretch")

    # --- Risk-return scatter | Correlation ---
    c, d = st.columns(2)
    with c:
        st.markdown('<p class="section-label">Risk vs return (lookback window)</p>',
                    unsafe_allow_html=True)
        sfig = go.Figure()
        for idx, i in enumerate(selected):
            s = win.get(i)
            if s is None or s.empty:
                continue
            sfig.add_trace(go.Scatter(
                x=[metrics.annualized_volatility(s)], y=[metrics.cagr(s)],
                mode="markers+text", text=[ticker_by_isin[i]], textposition="top center",
                textfont=dict(size=10, color=T.ink2), name=ticker_by_isin[i],
                marker=dict(size=13, color=T.color(idx),
                            line=dict(width=1.5, color=T.surface)),
                hovertemplate=f"{name_by_isin[i]}<br>vol %{{x:.1%}} · CAGR %{{y:.1%}}<extra></extra>"))
        sf(sfig, height=320, hovermode="closest", showlegend=False)
        sfig.update_xaxes(title="Volatility (ann.)", tickformat=".0%")
        sfig.update_yaxes(title="CAGR", tickformat=".0%")
        st.plotly_chart(sfig, width="stretch")
    with d:
        st.markdown('<p class="section-label">Correlation of daily returns</p>',
                    unsafe_allow_html=True)
        corr = metrics.correlation_matrix(get_price_matrix(tuple(selected)))
        if not corr.empty:
            corr = corr.rename(index=ticker_by_isin, columns=ticker_by_isin)
            heat = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, zmid=0,
                colorscale=T.diverging, text=corr.round(2).values,
                texttemplate="%{text}", textfont=dict(size=10),
                colorbar=dict(thickness=10, len=0.8)))
            sf(heat, height=320, showlegend=False, hovermode=False)
            st.plotly_chart(heat, width="stretch")

    # --- Tables ---
    st.markdown('<p class="section-label">Risk &amp; return</p>', unsafe_allow_html=True)
    rows = []
    for i in selected:
        s = win.get(i)
        if s is None or s.empty:
            continue
        summ = metrics.summary(s, risk_free=rf, ter=ter_by_isin.get(i))
        r = etfs[etfs["isin"] == i].iloc[0]
        needs_fx = fx.normalized_currency(currency_by_isin.get(i)) != "EUR"
        tco = costs.total_cost_of_ownership(
            ter_by_isin.get(i),
            spread=costs.estimate_spread(r.get("asset_class"), r.get("category")),
            fx_bps=costs.FX_CONVERSION_BPS if needs_fx else 0.0,
            replication=r.get("replication"))
        cagr_after_cost = (summ["cagr"] - tco["total_annual"]
                           if summ["cagr"] == summ["cagr"] else summ["cagr"])  # NaN-safe
        rows.append({"ETF": name_by_isin[i], "CAGR": summ["cagr"], "Total": summ["total_return"],
                     "Volatility": summ["volatility"], "Max DD": summ["max_drawdown"],
                     "Sharpe": summ["sharpe"], "Sortino": summ["sortino"],
                     "TER": summ.get("ter"), "All-in cost": tco["total_annual"],
                     "CAGR–cost": cagr_after_cost})
    mdf = pd.DataFrame(rows).set_index("ETF")
    pcols = ["CAGR", "Total", "Volatility", "Max DD", "TER", "All-in cost", "CAGR–cost"]
    render_table(mdf, fmt={**{c: "{:.2%}" for c in pcols},
                           "Sharpe": "{:.2f}", "Sortino": "{:.2f}"})

    st.markdown('<p class="section-label">Trailing total returns</p>', unsafe_allow_html=True)
    pr_rows = []
    for i in selected:
        pr = metrics.period_returns(full[i])
        pr = {"ETF": name_by_isin[i], **pr}
        pr_rows.append(pr)
    prdf = pd.DataFrame(pr_rows).set_index("ETF")
    render_table(prdf, fmt={c: "{:.2%}" for c in prdf.columns})


# --------------------------------------------------------------------------- Screener
def render_screener():
    f1, f2 = st.columns([2, 2])
    cats = sorted(etfs["category"].dropna().unique())
    pick_cat = f1.multiselect("Category", cats, default=cats)
    lookback = f2.selectbox("Metrics lookback", ["Max", "10Y", "5Y", "3Y", "1Y"], index=2)

    view = etfs[etfs["category"].isin(pick_cat)] if pick_cat else etfs
    rows = []
    for _, r in view.iterrows():
        s = trim(px(r["isin"]), lookback)
        summ = metrics.summary(s, ter=r.get("ter")) if not s.empty else {}
        rows.append({"Name": r["name"], "Ticker": r["ticker"], "Category": r["category"],
                     "Class": r.get("asset_class"), "TER": r.get("ter"),
                     "CAGR": summ.get("cagr"), "Vol": summ.get("volatility"),
                     "Max DD": summ.get("max_drawdown"), "Sharpe": summ.get("sharpe")})
    sdf = pd.DataFrame(rows)

    # Risk-return scatter of the whole (filtered) universe, coloured by asset class.
    st.markdown('<p class="section-label">Risk vs return — universe map</p>',
                unsafe_allow_html=True)
    classes = sorted(sdf["Class"].dropna().unique())
    cls_color = {c: T.color(i) for i, c in enumerate(classes)}
    sfig = go.Figure()
    for c in classes:
        sub = sdf[(sdf["Class"] == c) & sdf["CAGR"].notna() & sdf["Vol"].notna()]
        sfig.add_trace(go.Scatter(
            x=sub["Vol"], y=sub["CAGR"], mode="markers", name=c,
            marker=dict(size=10, color=cls_color[c], line=dict(width=1, color=T.surface)),
            customdata=sub[["Name", "Ticker"]].values,
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>"
                          "vol %{x:.1%} · CAGR %{y:.1%}<extra></extra>"))
    sf(sfig, height=360, hovermode="closest")
    sfig.update_xaxes(title="Volatility (ann.)", tickformat=".0%")
    sfig.update_yaxes(title="CAGR", tickformat=".0%")
    st.plotly_chart(sfig, width="stretch")

    st.markdown('<p class="section-label">All funds — ranked by CAGR</p>',
                unsafe_allow_html=True)
    sdf = sdf.sort_values("CAGR", ascending=False, na_position="last").reset_index(drop=True)
    render_table(sdf, hide_index=True, max_height=460,
                 fmt={"TER": "{:.2%}", "CAGR": "{:.2%}", "Vol": "{:.2%}",
                      "Max DD": "{:.2%}", "Sharpe": "{:.2f}"})


# --------------------------------------------------------------------------- Detail
def render_detail():
    label = st.selectbox("ETF", list(isin_by_label.keys()),
                         index=(list(isin_by_label.keys()).index(default_labels[0])
                                if default_labels else 0))
    isin = isin_by_label[label]
    row = etfs[etfs["isin"] == isin].iloc[0]

    m = st.columns(5)
    m[0].metric("TER", pct(row.get("ter")))
    m[1].metric("Category", str(row.get("category") or "—"))
    m[2].metric("Region", str(row.get("region") or "—"))
    m[3].metric("Acc/Dist", str(row.get("acc_dist") or "—"))
    m[4].metric("Domicile", str(row.get("domicile") or "—"))
    native_ccy = fx.normalized_currency(currency_by_isin.get(isin))
    st.caption(f"ISIN {row['isin']} · {row.get('index_name') or ''} · quoted in {native_ccy}"
               f"{'  → shown in EUR' if EUR_MODE else ''}")

    s = px(isin)
    if s.empty:
        st.warning("No price history stored yet.")
        return

    summ = metrics.summary(s, ter=row.get("ter"))
    k = st.columns(5)
    k[0].metric("CAGR", pct(summ["cagr"]))
    k[1].metric("Total return", pct(summ["total_return"]))
    k[2].metric("Volatility", pct(summ["volatility"]))
    k[3].metric("Max drawdown", pct(summ["max_drawdown"]))
    k[4].metric("Sharpe", "—" if pd.isna(summ["sharpe"]) else f"{summ['sharpe']:.2f}")

    st.markdown('<p class="section-label">Price history · adjusted close</p>',
                unsafe_allow_html=True)
    pfig = go.Figure(go.Scatter(x=s.index, y=s.values, line=dict(color=T.series[0], width=1.8),
                                fill="tozeroy", fillcolor="rgba(42,120,214,0.06)"))
    sf(pfig, height=300, showlegend=False)
    st.plotly_chart(pfig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown('<p class="section-label">Monthly return heatmap</p>', unsafe_allow_html=True)
        mm = metrics.monthly_returns_matrix(s)
        if not mm.empty:
            hfig = go.Figure(go.Heatmap(
                z=mm.values, x=[calendar.month_abbr[c] for c in mm.columns],
                y=mm.index.astype(str), colorscale=T.diverging, zmid=0,
                colorbar=dict(thickness=10, len=0.9, tickformat=".0%")))
            sf(hfig, height=340, showlegend=False, hovermode=False)
            st.plotly_chart(hfig, width="stretch")
    with b:
        st.markdown('<p class="section-label">Rolling 3-month volatility (ann.)</p>',
                    unsafe_allow_html=True)
        rv = metrics.rolling_volatility(s).dropna()
        if not rv.empty:
            vfig = go.Figure(go.Scatter(x=rv.index, y=rv.values,
                                        line=dict(color=T.series[5], width=1.5)))
            sf(vfig, height=340, showlegend=False)
            vfig.update_yaxes(tickformat=".0%")
            st.plotly_chart(vfig, width="stretch")

    c, d = st.columns(2)
    with c:
        st.markdown('<p class="section-label">Rolling 1-year return</p>', unsafe_allow_html=True)
        rr = metrics.rolling_returns(s, 1).dropna()
        if not rr.empty:
            rfig = go.Figure(go.Scatter(x=rr.index, y=rr.values,
                                        line=dict(color=T.series[1], width=1.5)))
            sf(rfig, height=300, showlegend=False)
            rfig.update_yaxes(tickformat=".0%")
            st.plotly_chart(rfig, width="stretch")
    with d:
        st.markdown('<p class="section-label">Distributions</p>', unsafe_allow_html=True)
        dists = data.load_distributions(isin)
        if dists.empty:
            st.caption("None recorded — accumulating ETF, or not yet fetched.")
        else:
            render_table(dists.tail(12), max_height=340)

    # --- Cost & tax (estimated) ---
    st.markdown('<p class="section-label">Cost &amp; tax · estimated true cost of ownership</p>',
                unsafe_allow_html=True)
    ter = row.get("ter")
    acc_dist = row.get("acc_dist")
    dist_yield = row.get("yield_ttm")
    div_tax = st.number_input("Your dividend tax rate %", value=26.0, step=1.0,
                              min_value=0.0, max_value=60.0) / 100.0
    needs_fx = native_ccy != "EUR"
    tco = costs.total_cost_of_ownership(
        ter, spread=costs.estimate_spread(row.get("asset_class"), row.get("category")),
        fx_bps=costs.FX_CONVERSION_BPS if needs_fx else 0.0, holding_years=10,
        replication=row.get("replication"))
    tdrag = costs.tax_drag(dist_yield, acc_dist, dividend_tax=div_tax)
    cg = st.columns(5)
    cg[0].metric("TER", pct(ter))
    cg[1].metric("Tracking diff (est.)", pct(tco["tracking_difference"]))
    cg[2].metric("Spread + FX (ann.)", pct(tco["spread_annual"] + tco["fx_annual"]))
    cg[3].metric("All-in cost/yr", pct(tco["total_annual"]),
                 help="TER + tracking difference + amortised spread/FX. Subtract from CAGR.")
    cg[4].metric("Tax drag/yr", pct(tdrag),
                 delta=("deferred (ACC)" if (acc_dist or "").upper().startswith("ACC")
                        else "taxed (DIST)"), delta_color="off")
    st.caption(f"{'Accumulating' if (acc_dist or '').upper().startswith('ACC') else 'Distributing'}"
               f" · {costs.domicile_note(row.get('domicile'))} "
               f"{'FX conversion applies (foreign currency).' if needs_fx else 'EUR-quoted — no FX cost.'}"
               " Estimates, not tax advice.")


# --------------------------------------------------------------------------- Strategy (DCA)
def render_strategy():
    st.markdown('<p class="section-label">Dollar-cost-averaging backtest</p>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    label = c1.selectbox("Invest into", list(isin_by_label.keys()),
                         index=(list(isin_by_label.keys()).index(default_labels[0])
                                if default_labels else 0))
    monthly = c2.number_input("Monthly", value=500, step=50, min_value=0)
    initial = c3.number_input("Initial lump", value=0, step=500, min_value=0)
    step_up = c4.number_input("Step-up %/yr", value=0.0, step=1.0) / 100.0

    o1, _ = st.columns([1, 3])
    include_costs = o1.checkbox("Include IBKR costs", value=True,
                                help="Deduct broker commission (min €1.25/order, 0.035%) and "
                                     "FX conversion cost from each contribution.")

    isin = isin_by_label[label]
    native_ccy = fx.normalized_currency(currency_by_isin.get(isin))
    s = px(isin)
    if s.empty:
        st.warning("No price history for this ETF.")
        return

    years_available = (s.index[-1] - s.index[0]).days / 365.25
    horizon = st.slider("Years to back-test (ending today)", min_value=1,
                        max_value=max(2, int(years_available)),
                        value=min(10, max(2, int(years_available))))
    start = s.index[-1] - pd.DateOffset(years=horizon)

    commission = costs.DEFAULT_COMMISSION if include_costs else None
    try:
        res = strategy.simulate_dca(s, monthly=float(monthly), initial=float(initial),
                                    start=start, annual_step_up=step_up,
                                    commission=commission, currency=native_ccy,
                                    account_currency="EUR")
    except ValueError as exc:
        st.warning(str(exc))
        return

    m = st.columns(5)
    m[0].metric("Invested", money(res.total_invested))
    m[1].metric("Final value", money(res.final_value))
    m[2].metric("Profit", money(res.profit),
                delta=f"{res.money_multiple:.2f}× money")
    m[3].metric("XIRR (ann.)", pct(res.xirr))
    m[4].metric("Costs paid", money(res.total_costs) if include_costs else "—",
                help="Total commissions + FX conversion cost deducted over the plan.")
    if EUR_MODE:
        ccy_note = "amounts in EUR (FX-normalised)"
    else:
        native_ccy = fx.normalized_currency(currency_by_isin.get(isin))
        ccy_note = (f"amounts in {native_ccy} ({ticker_by_isin[isin]} quote currency) — "
                    "switch to EUR (top-right) to normalise")
    st.caption(f"{res.start.date()} → {res.end.date()} · {ccy_note}. "
               "Total return basis (dividends reinvested).")

    st.markdown('<p class="section-label">Portfolio value vs money invested</p>',
                unsafe_allow_html=True)
    tl = res.timeline
    gfig = go.Figure()
    gfig.add_trace(go.Scatter(x=tl.index, y=tl["value"], name="Portfolio value",
                              line=dict(color=T.series[0], width=2),
                              fill="tozeroy", fillcolor="rgba(42,120,214,0.08)"))
    gfig.add_trace(go.Scatter(x=tl.index, y=tl["invested"], name="Money invested",
                              line=dict(color=T.ink2, width=1.5, dash="dot")))
    sf(gfig, height=380)
    st.plotly_chart(gfig, width="stretch")

    st.caption("DCA buys on the first trading day of each month. A negative gap between the "
               "lines is a period where you were underwater on cumulative contributions.")

    # --- Forward projection: "if the trend continues" ---
    st.markdown('<p class="section-label">Projection · if you keep going</p>',
                unsafe_allow_html=True)
    p1, p2, p3 = st.columns([1, 1, 2])
    proj_years = p1.slider("Project years", min_value=1, max_value=40, value=20)
    method = p2.selectbox("Method", ["Monte-Carlo (bootstrap)", "Monte-Carlo (normal)",
                                     "OLS trend"], index=0)
    method_key = {"Monte-Carlo (bootstrap)": "bootstrap",
                  "Monte-Carlo (normal)": "normal", "OLS trend": "ols"}[method]
    net_of_cost = p3.checkbox("Project on cost/tax-adjusted return", value=True,
                              help="Reduce the historical return by the fund's all-in cost "
                                   "(TER + tracking diff + spread/FX) before projecting.")

    ann_override = None
    if net_of_cost:
        drow = etfs[etfs["isin"] == isin].iloc[0]
        needs_fx = native_ccy != "EUR"
        tco = costs.total_cost_of_ownership(
            drow.get("ter"),
            spread=costs.estimate_spread(drow.get("asset_class"), drow.get("category")),
            fx_bps=costs.FX_CONVERSION_BPS if needs_fx else 0.0,
            replication=drow.get("replication"))
        hist_cagr = metrics.cagr(trim(s, f"{min(horizon, 15)}Y"))
        if hist_cagr == hist_cagr:  # not NaN
            ann_override = hist_cagr - tco["total_annual"]

    proj = projection.project_plan(
        trim(s, f"{min(horizon, 15)}Y") if horizon >= 3 else s,
        start_value=float(res.final_value), monthly=float(monthly), years=proj_years,
        method=method_key, annual_step_up=step_up, annual_return=ann_override)

    pfig = go.Figure()
    x = proj.dates
    fancol = "rgba(42,120,214,0.10)"
    if method_key != "ols":
        pfig.add_trace(go.Scatter(x=x, y=proj.fan["p95"], line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        pfig.add_trace(go.Scatter(x=x, y=proj.fan["p5"], fill="tonexty", fillcolor=fancol,
                                  line=dict(width=0), name="5–95% range", hoverinfo="skip"))
        pfig.add_trace(go.Scatter(x=x, y=proj.fan["p75"], line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        pfig.add_trace(go.Scatter(x=x, y=proj.fan["p25"], fill="tonexty",
                                  fillcolor="rgba(42,120,214,0.18)", line=dict(width=0),
                                  name="25–75% range", hoverinfo="skip"))
    pfig.add_trace(go.Scatter(x=x, y=proj.fan["p50"], name="Median",
                              line=dict(color=T.series[0], width=2.5)))
    pfig.add_trace(go.Scatter(x=x, y=proj.fan["invested"], name="Money invested",
                              line=dict(color=T.ink2, width=1.5, dash="dot")))
    sf(pfig, height=380)
    st.plotly_chart(pfig, width="stretch")

    end = proj.fan.iloc[-1]
    pm = st.columns(4)
    pm[0].metric(f"Contributed by yr {proj_years}", money(proj.total_contributed))
    pm[1].metric("Median outcome", money(end["p50"]))
    pm[2].metric("Pessimistic (5th pct)", money(end["p5"]) if method_key != "ols" else "—")
    pm[3].metric("Optimistic (95th pct)", money(end["p95"]) if method_key != "ols" else "—")
    st.caption(f"Projected on a {proj.annual_return_used * 100:.1f}%/yr "
               f"{'cost-adjusted ' if net_of_cost else ''}return "
               f"({'bootstrapped from history' if method_key == 'bootstrap' else method_key}). "
               "Scenario arithmetic, not a forecast — past performance does not predict "
               "future returns.")


# --------------------------------------------------------------------------- Data admin
def render_data():
    st.markdown(f'<p class="section-label">Database · {DB_PATH}</p>', unsafe_allow_html=True)
    if st.button("↻ Fetch / refresh all (network)", type="primary"):
        from etf import ingest
        with st.spinner("Fetching from sources… this can take a few minutes."):
            try:
                result = ingest.run()
                st.success(f"Ingested {result['ok']}/{result['total']} instruments.")
                st.cache_data.clear()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ingest failed: {exc}")

    # --- Data-quality health: GBX/GBP repairs, bad prints, and suspect series ---
    st.markdown('<p class="section-label">Data health</p>', unsafe_allow_html=True)
    health = data.data_health()
    if health.empty:
        st.caption("No health records yet. Run `python -m etf.ingest --repair` (no network) "
                   "or re-fetch to populate the data-quality checks.")
    else:
        n_susp = int((health["status"] == "suspect").sum())
        n_rep = int((health["status"] == "repaired").sum())
        n_clean = int((health["status"] == "clean").sum())
        hc = st.columns(3)
        hc[0].metric("Clean", n_clean)
        hc[1].metric("Repaired", n_rep, help="GBX/GBP rescales and bad prints fixed at ingest.")
        hc[2].metric("Suspect", n_susp,
                     help="Residual anomalies (e.g. unadjusted splits) — excluded from rankings.")
        hv = health.merge(etfs[["isin", "name", "ticker"]], on="isin", how="left")
        hv = hv[hv["status"] != "clean"].copy()
        if not hv.empty:
            hv = hv.sort_values("status", ascending=False)  # suspect first
            disp = hv[["name", "ticker", "status", "rescaled_days", "despiked_days",
                       "max_move_before", "max_move_after", "notes"]].rename(columns={
                "name": "ETF", "ticker": "Ticker", "status": "Status",
                "rescaled_days": "Rescaled", "despiked_days": "De-spiked",
                "max_move_before": "Worst move (raw)", "max_move_after": "Worst move (fixed)",
                "notes": "Notes"})
            render_table(disp, hide_index=True, max_height=300,
                         fmt={"Worst move (raw)": "{:.0%}", "Worst move (fixed)": "{:.0%}"})
        else:
            st.caption("All series clean — no repairs needed.")

    st.markdown('<p class="section-label">Coverage &amp; freshness</p>', unsafe_allow_html=True)
    cov = etfs[["name", "ticker", "category", "first_date", "last_date", "n_prices"]].copy()
    # Staleness: flag funds whose last price is well behind the freshest in the universe.
    last = pd.to_datetime(cov["last_date"], errors="coerce")
    universe_max = last.max()
    cov["days_behind"] = (universe_max - last).dt.days
    stale = cov[cov["days_behind"] > 7]
    if not stale.empty and pd.notna(universe_max):
        st.caption(f"⚠ {len(stale)} fund(s) are >7 days behind the freshest data "
                   f"({universe_max.date()}). They may have been delisted or renamed.")
    render_table(cov, hide_index=True, max_height=380,
                 fmt={"days_behind": "{:.0f}"})

    st.markdown('<p class="section-label">Recent ingest log</p>', unsafe_allow_html=True)
    render_table(data.ingest_log(60), hide_index=True, max_height=380)


# --------------------------------------------------------------------------- Recommended
def render_recommended():
    st.markdown('<p class="section-label">Top performers · total return over the last 10 years</p>',
                unsafe_allow_html=True)
    if not REC_ISINS:
        st.info("Not enough price history to rank yet. Fetch data on the **Data** tab.")
        return
    st.caption("The three UCITS ETFs with the highest total return over the last 10 years "
               "(dividends reinvested). They are the default selection across every tab. "
               "Past performance does not predict future returns.")

    board = REC_BOARD.set_index("isin")
    cols = st.columns(len(REC_ISINS))
    for col, i in zip(cols, REC_ISINS):
        r = board.loc[i]
        col.metric(ticker_by_isin[i], pct(r["ret"]),
                   delta=f"CAGR {pct(r['cagr'])}", delta_color="off")
        col.caption(name_by_isin[i])

    st.markdown('<p class="section-label">Growth of 100 · last 10 years</p>',
                unsafe_allow_html=True)
    fig = go.Figure()
    for idx, i in enumerate(REC_ISINS):
        s = trim(px(i), "10Y")
        if s.empty:
            continue
        norm = metrics.normalize_to_100(s)
        fig.add_trace(go.Scatter(x=norm.index, y=norm.values, name=ticker_by_isin[i],
                                 line=dict(color=T.color(idx), width=2)))
    sf(fig, height=380)
    fig.update_yaxes(title="Indexed to 100")
    st.plotly_chart(fig, width="stretch")

    st.markdown('<p class="section-label">10-year leaderboard</p>', unsafe_allow_html=True)
    lb = REC_BOARD.dropna(subset=["ret"]).head(15).copy()
    lb.insert(0, "Rank", range(1, len(lb) + 1))
    lb["ETF"] = lb["isin"].map(name_by_isin)
    lb["Ticker"] = lb["isin"].map(ticker_by_isin)
    lb["Category"] = lb["isin"].map(cat_by_isin)
    disp = lb[["Rank", "ETF", "Ticker", "Category", "ret", "cagr", "vol", "maxdd"]].rename(
        columns={"ret": "10Y return", "cagr": "CAGR", "vol": "Volatility", "maxdd": "Max DD"})
    render_table(disp, hide_index=True,
                 fmt={"10Y return": "{:.2%}", "CAGR": "{:.2%}",
                      "Volatility": "{:.2%}", "Max DD": "{:.2%}"})
    st.caption("Ranked by total return over the last 10 years; only funds with a full "
               "10-year history are eligible for the ranking.")


# --------------------------------------------------------------------------- Portfolio
def render_portfolio():
    st.markdown('<p class="section-label">Portfolio builder · weighted blend backtest</p>',
                unsafe_allow_html=True)
    labels = st.multiselect("Funds in the blend", list(isin_by_label.keys()),
                            default=default_labels[:3] or list(isin_by_label.keys())[:3])
    if not labels:
        st.info("Pick at least two funds to build a blend.")
        return
    selected = [isin_by_label[lbl] for lbl in labels]

    # Correlation-aware suggestion (writes a suggested set the user can adopt).
    sc1, sc2 = st.columns([1, 3])
    if sc1.button("Suggest low-correlation set"):
        mat_all = get_price_matrix(tuple(selected)) if len(selected) > 1 else pd.DataFrame()
        if not mat_all.empty:
            picks = portfolio.suggest_low_correlation(mat_all, n=min(4, len(selected)))
            st.session_state["port_suggested"] = picks
            sc2.caption("Suggested (least-correlated among your picks): "
                        + ", ".join(ticker_by_isin.get(i, i) for i in picks))

    # Weight inputs — equal-weight default, live-normalised.
    st.markdown('<p class="section-label">Weights</p>', unsafe_allow_html=True)
    wcols = st.columns(min(len(selected), 5))
    raw_w = {}
    for idx, i in enumerate(selected):
        col = wcols[idx % len(wcols)]
        raw_w[i] = col.number_input(ticker_by_isin.get(i, i), min_value=0.0, value=100.0,
                                    step=5.0, key=f"w_{i}")
    tot = sum(raw_w.values()) or 1.0
    weights = {i: w / tot for i, w in raw_w.items()}

    r1, r2 = st.columns([1, 3])
    rebal = r1.selectbox("Rebalance", list(portfolio.REBALANCE_FREQ.keys()), index=1)
    rebal_freq = portfolio.REBALANCE_FREQ[rebal]
    r2.caption("Weights shown are normalised to 100%. "
               + " · ".join(f"{ticker_by_isin.get(i, i)} {weights[i] * 100:.0f}%"
                            for i in selected))

    # Build a currency-aware price matrix for the selected funds.
    cols = {i: px(i) for i in selected}
    mat = pd.DataFrame({i: s for i, s in cols.items() if not s.empty})
    if mat.shape[1] < 1:
        st.warning("No overlapping price history for the selected funds.")
        return

    cmp = portfolio.rebalance_comparison(mat, weights, rebalance=rebal_freq)
    blend = cmp.get("rebalanced", pd.Series(dtype=float))
    if blend.empty:
        st.warning("Funds don't share enough common history to blend. Try a different set.")
        return

    # --- Blended metrics ---
    rf = 0.02
    summ = metrics.summary(blend, risk_free=rf)
    k = st.columns(5)
    k[0].metric("Blend CAGR", pct(summ["cagr"]))
    k[1].metric("Volatility", pct(summ["volatility"]))
    k[2].metric("Max drawdown", pct(summ["max_drawdown"]))
    k[3].metric("Sharpe", "—" if pd.isna(summ["sharpe"]) else f"{summ['sharpe']:.2f}")
    reb_gain = (cmp.get("rebalanced_final", 0) - cmp.get("drift_final", 0))
    k[4].metric("Rebalance effect", money(reb_gain),
                help="Final-value difference vs the same blend left to drift (per 100 start).")
    st.caption(f"Common history {blend.index[0].date()} → {blend.index[-1].date()} · "
               f"rebalanced {rebal.lower()} · "
               f"{'EUR (FX-normalised)' if EUR_MODE else 'native currencies (mixed) — switch to EUR'}.")

    # --- Growth of 100: rebalanced vs drift, plus components ---
    st.markdown('<p class="section-label">Growth of 100 · blend vs components</p>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=blend.index, y=blend.values, name="Blend (rebalanced)",
                             line=dict(color=T.series[0], width=2.5)))
    drift = cmp.get("drift")
    if drift is not None and not drift.empty:
        fig.add_trace(go.Scatter(x=drift.index, y=drift.values, name="Blend (drift)",
                                 line=dict(color=T.ink2, width=1.3, dash="dot")))
    for idx, i in enumerate(selected):
        comp = metrics.normalize_to_100(mat[i].reindex(blend.index).dropna())
        if not comp.empty:
            fig.add_trace(go.Scatter(x=comp.index, y=comp.values, name=ticker_by_isin.get(i, i),
                                     line=dict(color=T.color(idx + 2), width=1)))
    sf(fig, height=380)
    fig.update_yaxes(title="Indexed to 100")
    st.plotly_chart(fig, width="stretch")

    # --- Target vs drifted weights + component table ---
    a, b = st.columns(2)
    with a:
        st.markdown('<p class="section-label">Target vs drifted weights</p>',
                    unsafe_allow_html=True)
        drift_w = portfolio.blend_weights_drift(mat, weights)
        wf = go.Figure()
        wf.add_trace(go.Bar(x=[ticker_by_isin.get(i, i) for i in selected],
                            y=[weights[i] for i in selected], name="Target",
                            marker_color=T.series[0]))
        wf.add_trace(go.Bar(x=[ticker_by_isin.get(i, i) for i in drift_w.index],
                            y=drift_w.values, name="If never rebalanced",
                            marker_color=T.ink2))
        sf(wf, height=300)
        wf.update_layout(barmode="group")
        wf.update_yaxes(tickformat=".0%")
        st.plotly_chart(wf, width="stretch")
    with b:
        st.markdown('<p class="section-label">Lump sum vs DCA (same horizon)</p>',
                    unsafe_allow_html=True)
        yrs = (blend.index[-1] - blend.index[0]).days / 365.25
        h = min(10, max(2, int(yrs)))
        start = blend.index[-1] - pd.DateOffset(years=h)
        wnd = blend[blend.index >= start]
        try:
            dca = strategy.simulate_dca(blend, monthly=500.0, start=start)
            # Lump sum equal to total DCA contributions, invested at window start.
            lump_units = dca.total_invested / wnd.iloc[0]
            lump_final = lump_units * wnd.iloc[-1]
            comp = pd.DataFrame({
                "Strategy": ["DCA €500/mo", "Lump sum (same total)"],
                "Invested": [dca.total_invested, dca.total_invested],
                "Final value": [dca.final_value, lump_final],
            })
            render_table(comp, hide_index=True,
                         fmt={"Invested": "{:,.0f}", "Final value": "{:,.0f}"})
            st.caption(f"Over the last {h}y of the blend's common history. Lump sum usually "
                       "wins in rising markets; DCA cuts timing risk.")
        except ValueError as exc:
            st.caption(str(exc))


# --------------------------------------------------------------------------- router
PAGES = {"Recommended": render_recommended, "Compare": render_compare,
         "Screener": render_screener, "Portfolio": render_portfolio,
         "Detail": render_detail, "Strategy": render_strategy, "Data": render_data}
PAGES.get(page, render_recommended)()
