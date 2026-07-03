"""Streamlit UI for comparing UCITS ETFs.

Run from the project root:
    streamlit run src/etf/app.py

Reads exclusively from the local SQLite database (populated by ``python -m etf.ingest``);
never hits the network on the hot path except when you press "Fetch" on the Data page.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from etf import data, metrics
from etf.config import DB_PATH

# Colourblind-friendly qualitative palette (Okabe-Ito).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#666666"]

st.set_page_config(page_title="ETF Comparison", page_icon="📊", layout="wide")


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


def color_for(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:,.2f}%"


# --------------------------------------------------------------------------- sidebar
etfs = get_etfs()

st.sidebar.title("📊 ETF Comparison")
if etfs.empty:
    st.sidebar.warning("No data yet. Go to **Data** and fetch, or run `python -m etf.ingest`.")
page = st.sidebar.radio("View", ["Compare", "Screener", "Detail", "Data"], label_visibility="collapsed")

name_by_isin = dict(zip(etfs["isin"], etfs["name"])) if not etfs.empty else {}
isin_by_label = {}
if not etfs.empty:
    for _, row in etfs.iterrows():
        label = f"{row['name']}  ·  {row['ticker']}"
        isin_by_label[label] = row["isin"]


# --------------------------------------------------------------------------- Compare
def render_compare():
    st.header("Compare ETFs")
    if etfs.empty:
        st.info("No instruments in the database yet.")
        return

    labels = st.multiselect(
        "Select ETFs to compare", list(isin_by_label.keys()),
        default=list(isin_by_label.keys())[: min(3, len(isin_by_label))],
    )
    col_a, col_b = st.columns(2)
    lookback = col_a.selectbox(
        "Lookback", ["Max", "10Y", "5Y", "3Y", "1Y", "YTD"], index=2
    )
    rf = col_b.number_input("Risk-free rate (annual, %)", value=2.0, step=0.25) / 100.0

    if not labels:
        st.info("Pick at least one ETF above.")
        return

    selected = [(lbl, isin_by_label[lbl]) for lbl in labels]
    series_map = {isin: get_prices(isin) for _, isin in selected}
    series_map = {k: v for k, v in series_map.items() if not v.empty}
    if not series_map:
        st.warning("No price history stored for the selected ETFs yet. Fetch on the Data page.")
        return

    # Trim to lookback window based on the longest available end date.
    end = max(s.index[-1] for s in series_map.values())
    start = None
    period_map = {"1Y": pd.DateOffset(years=1), "3Y": pd.DateOffset(years=3),
                  "5Y": pd.DateOffset(years=5), "10Y": pd.DateOffset(years=10)}
    if lookback in period_map:
        start = end - period_map[lookback]
    elif lookback == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    trimmed = {isin: (s[s.index >= start] if start is not None else s)
               for isin, s in series_map.items()}

    # --- Growth chart (rebased to 100) ---
    st.subheader("Growth of 100 (total return)")
    fig = go.Figure()
    for i, (_, isin) in enumerate(selected):
        s = trimmed.get(isin)
        if s is None or s.empty:
            continue
        norm = metrics.normalize_to_100(s)
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values, name=name_by_isin.get(isin, isin),
            line=dict(color=color_for(i), width=2),
        ))
    fig.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Indexed to 100")
    st.plotly_chart(fig, width="stretch")

    # --- Metrics table ---
    st.subheader("Risk & return")
    ter_by_isin = dict(zip(etfs["isin"], etfs["ter"]))
    rows = []
    for _, isin in selected:
        s = trimmed.get(isin)
        if s is None or s.empty:
            continue
        summ = metrics.summary(s, risk_free=rf, ter=ter_by_isin.get(isin))
        rows.append({
            "ETF": name_by_isin.get(isin, isin),
            "CAGR": summ["cagr"], "Total": summ["total_return"],
            "Volatility": summ["volatility"], "Max DD": summ["max_drawdown"],
            "Sharpe": summ["sharpe"], "Sortino": summ["sortino"],
            "TER": summ.get("ter"), "CAGR–TER": summ.get("cagr_after_ter"),
        })
    mdf = pd.DataFrame(rows).set_index("ETF")
    pct_cols = ["CAGR", "Total", "Volatility", "Max DD", "TER", "CAGR–TER"]
    st.dataframe(
        mdf.style.format({**{c: "{:.2%}".format for c in pct_cols},
                          "Sharpe": "{:.2f}", "Sortino": "{:.2f}"}),
        width="stretch",
    )

    # --- Period returns table ---
    st.subheader("Trailing total returns")
    pr_rows = []
    for _, isin in selected:
        s = series_map.get(isin)  # use full history for trailing windows
        if s is None or s.empty:
            continue
        pr = metrics.period_returns(s)
        pr["ETF"] = name_by_isin.get(isin, isin)
        pr_rows.append(pr)
    prdf = pd.DataFrame(pr_rows).set_index("ETF")
    st.dataframe(
        prdf.style.format({c: (lambda v: "—" if pd.isna(v) else f"{v:.2%}")
                           for c in prdf.columns}),
        width="stretch",
    )

    # --- Correlation + drawdown ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation (daily returns)")
        matrix = get_price_matrix(tuple(isin for _, isin in selected))
        corr = metrics.correlation_matrix(matrix)
        if not corr.empty:
            corr = corr.rename(index=name_by_isin, columns=name_by_isin)
            heat = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                colorscale="RdBu_r", text=corr.round(2).values, texttemplate="%{text}",
            ))
            heat.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(heat, width="stretch")
    with c2:
        st.subheader("Drawdown")
        ddfig = go.Figure()
        for i, (_, isin) in enumerate(selected):
            s = trimmed.get(isin)
            if s is None or s.empty:
                continue
            dd = metrics.drawdown_series(s)
            ddfig.add_trace(go.Scatter(
                x=dd.index, y=dd.values, name=name_by_isin.get(isin, isin),
                line=dict(color=color_for(i), width=1.5),
            ))
        ddfig.update_layout(height=380, hovermode="x unified", showlegend=False,
                            yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(ddfig, width="stretch")


# --------------------------------------------------------------------------- Screener
def render_screener():
    st.header("Screener")
    if etfs.empty:
        st.info("No instruments in the database yet.")
        return
    lookback = st.selectbox("Metrics lookback", ["Max", "10Y", "5Y", "3Y", "1Y"], index=2)
    period_map = {"1Y": pd.DateOffset(years=1), "3Y": pd.DateOffset(years=3),
                  "5Y": pd.DateOffset(years=5), "10Y": pd.DateOffset(years=10)}

    rows = []
    for _, row in etfs.iterrows():
        s = get_prices(row["isin"])
        if not s.empty and lookback in period_map:
            s = s[s.index >= s.index[-1] - period_map[lookback]]
        summ = metrics.summary(s, ter=row.get("ter")) if not s.empty else {}
        rows.append({
            "Name": row["name"], "Ticker": row["ticker"],
            "Class": row.get("asset_class"), "Region": row.get("region"),
            "TER": row.get("ter"),
            "CAGR": summ.get("cagr"), "Vol": summ.get("volatility"),
            "Max DD": summ.get("max_drawdown"), "Sharpe": summ.get("sharpe"),
            "History": None if s.empty else f"{s.index[0].date()} → {s.index[-1].date()}",
        })
    sdf = pd.DataFrame(rows)

    classes = sorted(c for c in sdf["Class"].dropna().unique())
    pick = st.multiselect("Filter asset class", classes, default=classes)
    if pick:
        sdf = sdf[sdf["Class"].isin(pick)]

    st.dataframe(
        sdf.style.format({"TER": "{:.2%}", "CAGR": "{:.2%}", "Vol": "{:.2%}",
                          "Max DD": "{:.2%}", "Sharpe": "{:.2f}"}, na_rep="—"),
        width="stretch", hide_index=True,
    )
    st.caption("Tip: click a column header to sort.")


# --------------------------------------------------------------------------- Detail
def render_detail():
    st.header("ETF detail")
    if etfs.empty:
        st.info("No instruments in the database yet.")
        return
    label = st.selectbox("ETF", list(isin_by_label.keys()))
    isin = isin_by_label[label]
    row = etfs[etfs["isin"] == isin].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TER", pct(row.get("ter")))
    c2.metric("Asset class", str(row.get("asset_class") or "—"))
    c3.metric("Region", str(row.get("region") or "—"))
    c4.metric("Acc/Dist", str(row.get("acc_dist") or "—"))
    st.caption(f"ISIN {row['isin']} · {row.get('index_name') or ''} · domicile {row.get('domicile') or '—'}")

    s = get_prices(isin)
    if s.empty:
        st.warning("No price history stored yet.")
        return

    summ = metrics.summary(s, ter=row.get("ter"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CAGR", pct(summ["cagr"]))
    m2.metric("Volatility", pct(summ["volatility"]))
    m3.metric("Max drawdown", pct(summ["max_drawdown"]))
    m4.metric("Sharpe", "—" if pd.isna(summ["sharpe"]) else f"{summ['sharpe']:.2f}")

    fig = go.Figure(go.Scatter(x=s.index, y=s.values, line=dict(color=PALETTE[0], width=1.8)))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Adjusted close")
    st.plotly_chart(fig, width="stretch")

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Rolling 1Y return")
        rr = metrics.rolling_returns(s, 1).dropna()
        if not rr.empty:
            rfig = go.Figure(go.Scatter(x=rr.index, y=rr.values, line=dict(color=PALETTE[2])))
            rfig.update_layout(height=280, yaxis_tickformat=".0%",
                               margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(rfig, width="stretch")
    with c_right:
        st.subheader("Distributions")
        dists = data.load_distributions(isin)
        if dists.empty:
            st.caption("None recorded (accumulating ETF, or not yet fetched).")
        else:
            st.dataframe(dists.tail(12), width="stretch")


# --------------------------------------------------------------------------- Data admin
def render_data():
    st.header("Data")
    st.caption(f"Database: `{DB_PATH}`")

    if st.button("↻ Fetch / refresh all (network)", type="primary"):
        from etf import ingest
        with st.spinner("Fetching from sources… this can take a minute."):
            try:
                result = ingest.run()
                st.success(f"Ingested {result['ok']}/{result['total']} instruments.")
                st.cache_data.clear()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ingest failed: {exc}")

    if etfs.empty:
        st.info("Database is empty. Fetch above, or run `python -m etf.ingest` in a terminal.")
        return

    st.subheader("Coverage & freshness")
    cov = etfs[["name", "ticker", "first_date", "last_date", "n_prices"]].copy()
    st.dataframe(cov, width="stretch", hide_index=True)

    st.subheader("Recent ingest log")
    st.dataframe(data.ingest_log(50), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- router
if page == "Compare":
    render_compare()
elif page == "Screener":
    render_screener()
elif page == "Detail":
    render_detail()
else:
    render_data()
