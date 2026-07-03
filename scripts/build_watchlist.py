"""Generate a verified watchlist.yaml from a large candidate universe of UCITS ETFs.

For each candidate we list one or more Yahoo tickers (different exchange listings of the
same ISIN, e.g. Xetra ``.DE`` / London ``.L`` / Amsterdam ``.AS``). We batch-probe them on
Yahoo Finance and keep the first listing that actually returns price history. Candidates
whose tickers all fail are dropped and reported, so the resulting watchlist only contains
funds we can genuinely fetch.

Run:  python scripts/build_watchlist.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

# Each entry: isin, name, category, asset_class, region, acc_dist, ter, domicile, tickers[]
# `tickers` are tried in order; the first with data wins. Prefer Xetra (.DE, EUR) then LSE.
CANDIDATES: list[dict] = [
    # ---------------- Global Equity (all-world / ACWI / developed world) ----------------
    dict(isin="IE00BK5BQT80", name="Vanguard FTSE All-World (Acc)", category="Global Equity",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0022,
         tickers=["VWCE.DE", "VWRA.L"]),
    dict(isin="IE00B3RBWM25", name="Vanguard FTSE All-World (Dist)", category="Global Equity",
         asset_class="equity", region="world", acc_dist="DIST", ter=0.0022,
         tickers=["VWRL.L", "VGWL.DE", "VWRL.AS"]),
    dict(isin="IE000716YHJ7", name="Invesco FTSE All-World (Acc)", category="Global Equity",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0015,
         tickers=["FWIA.DE", "FWRA.L"]),
    dict(isin="IE0003XJA0J9", name="Amundi Prime All Country World (Acc)", category="Global Equity",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0007,
         tickers=["WEBN.DE", "WEBG.L"]),
    dict(isin="IE00B3YLTY66", name="SPDR MSCI ACWI IMI (Acc)", category="Global Equity",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0017,
         tickers=["SPYI.DE", "IMID.L"]),
    dict(isin="IE00B44Z5B48", name="SPDR MSCI ACWI (Acc)", category="Global Equity",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0012,
         tickers=["SPYY.DE", "ACWI.L"]),
    dict(isin="IE00B6R52259", name="iShares MSCI ACWI (Acc)", category="Global Equity",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0020,
         tickers=["IUSQ.DE", "SSAC.L", "ISAC.L"]),
    dict(isin="IE00B4L5Y983", name="iShares Core MSCI World (Acc)", category="Global Equity",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0020,
         tickers=["IWDA.AS", "SWDA.L", "EUNL.DE"]),
    dict(isin="IE00B0M62Q58", name="iShares MSCI World (Dist)", category="Global Equity",
         asset_class="equity", region="developed", acc_dist="DIST", ter=0.0050,
         tickers=["IWRD.L", "IQQW.DE"]),
    dict(isin="IE00BKX55T58", name="Vanguard FTSE Developed World (Acc)", category="Global Equity",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0012,
         tickers=["VHVG.L", "VGVF.DE"]),
    dict(isin="IE00BK5BQV03", name="Vanguard FTSE Developed World (Dist)", category="Global Equity",
         asset_class="equity", region="developed", acc_dist="DIST", ter=0.0012,
         tickers=["VEVE.L", "VGVE.DE"]),
    dict(isin="IE00BJ0KDQ92", name="Xtrackers MSCI World (Acc)", category="Global Equity",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0012,
         tickers=["XDWD.DE", "XDWL.L"]),
    dict(isin="IE00BFY0GT14", name="SPDR MSCI World (Acc)", category="Global Equity",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0012,
         tickers=["SWRD.L", "SPPW.DE"]),
    dict(isin="IE00B60SX394", name="Invesco MSCI World (Acc)", category="Global Equity",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0019,
         tickers=["MXWS.L", "SC0J.DE"]),

    # ---------------- US Equity ----------------
    dict(isin="IE00B5BMR087", name="iShares Core S&P 500 (Acc)", category="US Equity",
         asset_class="equity", region="us", acc_dist="ACC", ter=0.0007,
         tickers=["CSPX.L", "SXR8.DE"]),
    dict(isin="IE0031442068", name="iShares Core S&P 500 (Dist)", category="US Equity",
         asset_class="equity", region="us", acc_dist="DIST", ter=0.0007,
         tickers=["IUSA.L", "IUSA.DE"]),
    dict(isin="IE00BFMXXD54", name="Vanguard S&P 500 (Acc)", category="US Equity",
         asset_class="equity", region="us", acc_dist="ACC", ter=0.0007,
         tickers=["VUAA.L", "VUAA.DE", "VUAG.L"]),
    dict(isin="IE00B3XXRP09", name="Vanguard S&P 500 (Dist)", category="US Equity",
         asset_class="equity", region="us", acc_dist="DIST", ter=0.0007,
         tickers=["VUSA.L", "VUSA.AS", "VUSD.DE"]),
    dict(isin="IE00B6YX5C33", name="SPDR S&P 500 (Dist)", category="US Equity",
         asset_class="equity", region="us", acc_dist="DIST", ter=0.0003,
         tickers=["SPY5.L", "SPY5.DE"]),
    dict(isin="IE00B3YCGJ38", name="Invesco S&P 500 (Acc)", category="US Equity",
         asset_class="equity", region="us", acc_dist="ACC", ter=0.0005,
         tickers=["SPXP.L", "SPXS.DE"]),
    dict(isin="IE00B60SX170", name="iShares MSCI USA (Acc)", category="US Equity",
         asset_class="equity", region="us", acc_dist="ACC", ter=0.0007,
         tickers=["CSUS.L", "SXR4.DE"]),
    dict(isin="IE00BKM4H197", name="iShares MSCI USA SRI (Acc)", category="US Equity",
         asset_class="equity", region="us", acc_dist="ACC", ter=0.0020,
         tickers=["SUAS.L", "2B7F.DE"]),

    # ---------------- Nasdaq & Tech ----------------
    dict(isin="IE00B53SZB19", name="iShares Nasdaq 100 (Acc)", category="Nasdaq & Tech",
         asset_class="equity", region="us", acc_dist="ACC", ter=0.0030,
         tickers=["CNDX.L", "SXRV.DE"]),
    dict(isin="IE0032077012", name="Invesco EQQQ Nasdaq-100 (Dist)", category="Nasdaq & Tech",
         asset_class="equity", region="us", acc_dist="DIST", ter=0.0030,
         tickers=["EQQQ.L", "EQQQ.DE"]),
    dict(isin="IE00BMFKG444", name="Xtrackers Nasdaq 100 (Acc)", category="Nasdaq & Tech",
         asset_class="equity", region="us", acc_dist="ACC", ter=0.0020,
         tickers=["XNAQ.DE", "XNAS.L"]),
    dict(isin="IE00B3WJKG14", name="iShares S&P 500 Info Tech (Acc)", category="Nasdaq & Tech",
         asset_class="equity", region="us", acc_dist="ACC", ter=0.0015,
         tickers=["IITU.L", "QDVE.DE"]),
    dict(isin="IE00BM67HT60", name="Xtrackers MSCI World Info Tech (Acc)", category="Nasdaq & Tech",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0025,
         tickers=["XDWT.DE", "XWTS.L"]),

    # ---------------- Europe Equity ----------------
    dict(isin="IE00B4K48X80", name="iShares Core MSCI Europe (Acc)", category="Europe Equity",
         asset_class="equity", region="europe", acc_dist="ACC", ter=0.0012,
         tickers=["SMEA.L", "IMEA.DE"]),
    dict(isin="LU0908500753", name="Amundi Stoxx Europe 600 (Acc)", category="Europe Equity",
         asset_class="equity", region="europe", acc_dist="ACC", ter=0.0007,
         tickers=["MEUD.PA", "LYP6.DE"]),
    dict(isin="DE0002635307", name="iShares STOXX Europe 600 (Dist)", category="Europe Equity",
         asset_class="equity", region="europe", acc_dist="DIST", ter=0.0020,
         tickers=["EXSA.DE"]),
    dict(isin="IE00B945VV12", name="Vanguard FTSE Developed Europe (Dist)", category="Europe Equity",
         asset_class="equity", region="europe", acc_dist="DIST", ter=0.0010,
         tickers=["VEUR.L", "VGEU.DE"]),
    dict(isin="IE00B14X4Q57", name="iShares MSCI EMU (Acc)", category="Europe Equity",
         asset_class="equity", region="eurozone", acc_dist="ACC", ter=0.0018,
         tickers=["CEU2.L", "SXR7.DE"]),

    # ---------------- Emerging Markets ----------------
    dict(isin="IE00BKM4GZ66", name="iShares Core MSCI EM IMI (Acc)", category="Emerging Markets",
         asset_class="equity", region="emerging", acc_dist="ACC", ter=0.0018,
         tickers=["EIMI.L", "IS3N.DE"]),
    dict(isin="IE00B4L5YC18", name="iShares MSCI EM (Acc)", category="Emerging Markets",
         asset_class="equity", region="emerging", acc_dist="ACC", ter=0.0018,
         tickers=["SEMA.L", "IQQE.DE"]),
    dict(isin="IE00BTJRMP35", name="Xtrackers MSCI EM (Acc)", category="Emerging Markets",
         asset_class="equity", region="emerging", acc_dist="ACC", ter=0.0018,
         tickers=["XMME.DE", "XMEM.L"]),
    dict(isin="IE00B3VVMM84", name="Vanguard FTSE Emerging Markets (Dist)", category="Emerging Markets",
         asset_class="equity", region="emerging", acc_dist="DIST", ter=0.0022,
         tickers=["VFEM.L", "VDEM.DE"]),
    dict(isin="IE00BMG6Z448", name="iShares MSCI EM ex-China (Acc)", category="Emerging Markets",
         asset_class="equity", region="emerging", acc_dist="ACC", ter=0.0018,
         tickers=["EMXC.L"]),

    # ---------------- Small Cap ----------------
    dict(isin="IE00BF4RFH31", name="iShares MSCI World Small Cap (Acc)", category="Small Cap",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0035,
         tickers=["WSML.L", "IUSN.DE"]),
    dict(isin="IE00BCBJG560", name="SPDR MSCI World Small Cap (Acc)", category="Small Cap",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0045,
         tickers=["ZPRS.DE", "WOSC.L"]),
    dict(isin="IE00B48X4842", name="iShares MSCI EM Small Cap (Dist)", category="Small Cap",
         asset_class="equity", region="emerging", acc_dist="DIST", ter=0.0074,
         tickers=["IEMS.L", "SEMS.DE"]),

    # ---------------- Factor / Smart Beta ----------------
    dict(isin="IE00BP3QZ825", name="iShares Edge MSCI World Momentum (Acc)", category="Factor / Smart Beta",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0030,
         tickers=["IWMO.L", "IS3R.DE"]),
    dict(isin="IE00BP3QZ601", name="iShares Edge MSCI World Quality (Acc)", category="Factor / Smart Beta",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0030,
         tickers=["IWQU.L", "IS3Q.DE"]),
    dict(isin="IE00BP3QZB59", name="iShares Edge MSCI World Value (Acc)", category="Factor / Smart Beta",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0030,
         tickers=["IWVL.L", "IS3S.DE"]),
    dict(isin="IE00BP3QZD73", name="iShares Edge MSCI World Size (Acc)", category="Factor / Smart Beta",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0030,
         tickers=["IWSZ.L"]),
    dict(isin="IE00B8FHGS14", name="iShares Edge MSCI World Min Vol (Acc)", category="Factor / Smart Beta",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0030,
         tickers=["MVOL.L", "MINV.DE"]),
    dict(isin="IE00BZ0PKT83", name="iShares Edge MSCI World Multifactor (Acc)", category="Factor / Smart Beta",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0050,
         tickers=["FSWD.L", "IFSW.DE"]),
    dict(isin="IE00BSPLC298", name="SPDR MSCI USA Small Cap Value Weighted", category="Factor / Smart Beta",
         asset_class="equity", region="us", acc_dist="ACC", ter=0.0030,
         tickers=["ZPRV.DE", "USSC.L"]),
    dict(isin="IE00BSPLC413", name="SPDR MSCI Europe Small Cap Value Weighted", category="Factor / Smart Beta",
         asset_class="equity", region="europe", acc_dist="ACC", ter=0.0030,
         tickers=["ZPRX.DE", "ESCV.L"]),

    # ---------------- Dividend ----------------
    dict(isin="IE00B8GKDB10", name="Vanguard FTSE All-World High Div (Dist)", category="Dividend",
         asset_class="equity", region="world", acc_dist="DIST", ter=0.0029,
         tickers=["VHYL.L", "VGWD.DE"]),
    dict(isin="IE00B9CQXS71", name="SPDR S&P Global Dividend Aristocrats", category="Dividend",
         asset_class="equity", region="world", acc_dist="DIST", ter=0.0045,
         tickers=["ZPRG.DE", "GBDV.L"]),
    dict(isin="DE000A0F5UH1", name="iShares STOXX Global Select Dividend 100", category="Dividend",
         asset_class="equity", region="world", acc_dist="DIST", ter=0.0046,
         tickers=["ISPA.DE"]),
    dict(isin="IE00BYXVGZ48", name="Fidelity Global Quality Income (Acc)", category="Dividend",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0040,
         tickers=["FGQI.L", "FGEG.DE"]),

    # ---------------- Sector & Thematic ----------------
    dict(isin="IE00B1XNHC34", name="iShares Global Clean Energy", category="Sector & Thematic",
         asset_class="equity", region="world", acc_dist="DIST", ter=0.0065,
         tickers=["INRG.L", "IQQH.DE"]),
    dict(isin="IE00BGV5VN51", name="Xtrackers AI & Big Data (Acc)", category="Sector & Thematic",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0035,
         tickers=["XAIX.DE", "XAIX.L"]),
    dict(isin="IE00BDVPNG13", name="WisdomTree AI (Acc)", category="Sector & Thematic",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0040,
         tickers=["WTAI.L", "WTI2.DE"]),
    dict(isin="IE00BMC38736", name="VanEck Semiconductor (Acc)", category="Sector & Thematic",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0035,
         tickers=["SMH.L", "VVSM.DE"]),
    dict(isin="IE00BYZK4552", name="iShares Automation & Robotics (Acc)", category="Sector & Thematic",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0040,
         tickers=["RBOT.L", "2B76.DE"]),
    dict(isin="IE00BYZK4669", name="iShares Digitalisation (Acc)", category="Sector & Thematic",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0040,
         tickers=["DGTL.L", "2B77.DE"]),
    dict(isin="IE00BYZK4776", name="iShares Healthcare Innovation (Acc)", category="Sector & Thematic",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0040,
         tickers=["HEAL.L", "2B78.DE"]),
    dict(isin="IE00B1TXK627", name="iShares Global Water (Dist)", category="Sector & Thematic",
         asset_class="equity", region="world", acc_dist="DIST", ter=0.0065,
         tickers=["IH2O.L", "DXGW.DE"]),
    dict(isin="IE00BGL86Z12", name="iShares Electric Vehicles & Driving Tech (Acc)",
         category="Sector & Thematic",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0040,
         tickers=["ECAR.L", "IEVD.DE"]),

    # ---------------- Regional / Country ----------------
    dict(isin="IE00B4L5YX21", name="iShares Core MSCI Japan (Acc)", category="Regional / Country",
         asset_class="equity", region="japan", acc_dist="ACC", ter=0.0012,
         tickers=["SJPA.L", "IJPA.L", "EUNN.DE"]),
    dict(isin="IE00B52MJY50", name="iShares Core MSCI Pacific ex-Japan (Acc)", category="Regional / Country",
         asset_class="equity", region="pacific", acc_dist="ACC", ter=0.0020,
         tickers=["CPJ1.L", "SXR1.DE"]),
    dict(isin="IE00BZCQB185", name="iShares MSCI India (Acc)", category="Regional / Country",
         asset_class="equity", region="india", acc_dist="ACC", ter=0.0065,
         tickers=["QDV5.DE", "NDIA.L"]),
    dict(isin="IE00BHZRR147", name="Franklin FTSE India (Acc)", category="Regional / Country",
         asset_class="equity", region="india", acc_dist="ACC", ter=0.0019,
         tickers=["FLXI.L", "FLXI.DE"]),
    dict(isin="IE00BJ5JPG56", name="iShares MSCI China (Acc)", category="Regional / Country",
         asset_class="equity", region="china", acc_dist="ACC", ter=0.0028,
         tickers=["ICGA.L", "36BZ.DE"]),

    # ---------------- Bonds ----------------
    dict(isin="IE00BDBRDM35", name="iShares Core Global Aggregate Bond EUR-H (Acc)", category="Bonds",
         asset_class="bond", region="world", acc_dist="ACC", ter=0.0010,
         tickers=["EUNA.DE", "AGGH.L"]),
    dict(isin="IE00B3F81409", name="iShares Core Global Aggregate Bond (Dist)", category="Bonds",
         asset_class="bond", region="world", acc_dist="DIST", ter=0.0010,
         tickers=["AGGG.L", "SAGG.L"]),
    dict(isin="IE00BG47KH54", name="Vanguard Global Aggregate Bond EUR-H (Acc)", category="Bonds",
         asset_class="bond", region="world", acc_dist="ACC", ter=0.0010,
         tickers=["VAGF.L", "VGEA.DE"]),
    dict(isin="IE00B4WXJJ64", name="iShares Core Euro Govt Bond (Dist)", category="Bonds",
         asset_class="bond", region="eurozone", acc_dist="DIST", ter=0.0009,
         tickers=["SEGA.L", "EUNH.DE"]),
    dict(isin="IE00B3F81R35", name="iShares Core Euro Corp Bond (Dist)", category="Bonds",
         asset_class="bond", region="eurozone", acc_dist="DIST", ter=0.0020,
         tickers=["IEAC.L", "EUN5.DE"]),
    dict(isin="IE00B66F4759", name="iShares Euro High Yield Corp Bond (Dist)", category="Bonds",
         asset_class="bond", region="eurozone", acc_dist="DIST", ter=0.0050,
         tickers=["IHYG.L", "EUNW.DE"]),
    dict(isin="IE00B1FZS798", name="iShares $ Treasury 7-10y (Dist)", category="Bonds",
         asset_class="bond", region="us", acc_dist="DIST", ter=0.0007,
         tickers=["IBTM.L", "IDTM.DE"]),
    dict(isin="IE00B14X4S71", name="iShares $ Treasury 1-3y (Dist)", category="Bonds",
         asset_class="bond", region="us", acc_dist="DIST", ter=0.0007,
         tickers=["IBTS.L", "IBTA.DE"]),
    dict(isin="IE00B0M62X26", name="iShares Euro Inflation Linked Govt Bond", category="Bonds",
         asset_class="bond", region="eurozone", acc_dist="DIST", ter=0.0009,
         tickers=["IBCI.DE", "IBCI.L"]),

    # ---------------- Gold & Commodities ----------------
    dict(isin="IE00B4ND3602", name="iShares Physical Gold ETC", category="Gold & Commodities",
         asset_class="commodity", region="global", acc_dist="ACC", ter=0.0012,
         tickers=["SGLN.L", "IGLN.L", "EGLN.DE"]),
    dict(isin="IE00B579F325", name="Invesco Physical Gold ETC", category="Gold & Commodities",
         asset_class="commodity", region="global", acc_dist="ACC", ter=0.0012,
         tickers=["SGLD.L", "8PSG.DE"]),
    dict(isin="DE000A0S9GB0", name="Xetra-Gold ETC", category="Gold & Commodities",
         asset_class="commodity", region="global", acc_dist="ACC", ter=0.0000,
         tickers=["4GLD.DE"]),
    dict(isin="IE00B4NCWG09", name="iShares Physical Silver ETC", category="Gold & Commodities",
         asset_class="commodity", region="global", acc_dist="ACC", ter=0.0020,
         tickers=["SSLN.L", "ISLN.DE"]),
    dict(isin="IE00BDFL4P12", name="iShares Diversified Commodity Swap (Acc)", category="Gold & Commodities",
         asset_class="commodity", region="global", acc_dist="ACC", ter=0.0019,
         tickers=["ICOM.L", "SXRS.DE"]),

    # ---------------- Real Estate ----------------
    dict(isin="IE00B1FZS350", name="iShares Developed Markets Property Yield (Dist)", category="Real Estate",
         asset_class="reit", region="developed", acc_dist="DIST", ter=0.0059,
         tickers=["IWDP.L", "IQQ6.DE"]),
    dict(isin="IE00B0M63284", name="iShares European Property Yield (Dist)", category="Real Estate",
         asset_class="reit", region="europe", acc_dist="DIST", ter=0.0040,
         tickers=["IPRP.L", "IQQP.DE"]),

    # ---------------- Multi-Asset (Vanguard LifeStrategy) ----------------
    dict(isin="IE00BMVB5R75", name="Vanguard LifeStrategy 60% Equity (Acc)", category="Multi-Asset",
         asset_class="multi", region="world", acc_dist="ACC", ter=0.0025,
         tickers=["VNGA60.MI", "VNGA60.DE"]),
    dict(isin="IE00BMVB5P51", name="Vanguard LifeStrategy 80% Equity (Acc)", category="Multi-Asset",
         asset_class="multi", region="world", acc_dist="ACC", ter=0.0025,
         tickers=["VNGA80.MI", "VNGA80.DE"]),
    dict(isin="IE00BMVB5M21", name="Vanguard LifeStrategy 100% Equity (Acc)", category="Multi-Asset",
         asset_class="multi", region="world", acc_dist="ACC", ter=0.0025,
         tickers=["VNGA100.MI", "VNGA100.DE"]),
    dict(isin="IE00BMVB5K07", name="Vanguard LifeStrategy 40% Equity (Acc)", category="Multi-Asset",
         asset_class="multi", region="world", acc_dist="ACC", ter=0.0025,
         tickers=["VNGA40.MI", "VNGA40.DE"]),

    # ---------------- ESG / SRI ----------------
    dict(isin="IE00BYX2JD69", name="iShares MSCI World SRI (Acc)", category="ESG / SRI",
         asset_class="equity", region="developed", acc_dist="ACC", ter=0.0020,
         tickers=["SUSW.L", "2B7K.DE"]),
    dict(isin="IE00BNG8L385", name="Vanguard ESG Global All Cap (Acc)", category="ESG / SRI",
         asset_class="equity", region="world", acc_dist="ACC", ter=0.0024,
         tickers=["V3AB.L", "V3AA.L"]),
    dict(isin="IE00BFNM3P36", name="iShares MSCI EM SRI (Acc)", category="ESG / SRI",
         asset_class="equity", region="emerging", acc_dist="ACC", ter=0.0025,
         tickers=["SUSM.L", "2B7L.DE"]),
]


def _has_data(df) -> bool:
    try:
        return df is not None and "Close" in df and df["Close"].dropna().shape[0] > 15
    except Exception:
        return False


def probe(tickers: list[str]) -> str | None:
    """Return the first ticker (of a batch) that has data, else None."""
    import yfinance as yf

    if not tickers:
        return None
    data = yf.download(tickers, period="3mo", progress=False, threads=True,
                       auto_adjust=False, group_by="ticker")
    for t in tickers:
        try:
            sub = data[t] if len(tickers) > 1 else data
        except (KeyError, TypeError):
            continue
        if _has_data(sub):
            return t
    return None


def main() -> int:
    resolved: list[dict] = []
    unresolved: list[dict] = []

    # Round-robin over ticker preference index so each round is a single batched download.
    max_alts = max(len(c["tickers"]) for c in CANDIDATES)
    pending = list(CANDIDATES)
    for level in range(max_alts):
        batch = {c["isin"]: c["tickers"][level] for c in pending if level < len(c["tickers"])}
        if not batch:
            break
        print(f"Probing round {level + 1}: {len(batch)} tickers …")
        import yfinance as yf
        data = yf.download(list(batch.values()), period="3mo", progress=False,
                           threads=True, auto_adjust=False, group_by="ticker")
        still_pending = []
        for c in pending:
            if level >= len(c["tickers"]):
                still_pending.append(c)
                continue
            t = c["tickers"][level]
            try:
                sub = data[t] if len(batch) > 1 else data
            except (KeyError, TypeError):
                sub = None
            if _has_data(sub):
                chosen = dict(c)
                chosen["ticker"] = t
                del chosen["tickers"]
                chosen["domicile"] = c["isin"][:2]
                resolved.append(chosen)
            else:
                still_pending.append(c)
        pending = still_pending
        time.sleep(1.0)

    unresolved = pending
    # De-duplicate by ISIN (keep first resolved)
    seen = set()
    final = []
    for c in resolved:
        if c["isin"] in seen:
            continue
        seen.add(c["isin"])
        final.append(c)

    # Order by category then name for a tidy file.
    final.sort(key=lambda c: (c["category"], c["name"]))

    out = {"etfs": [
        {k: c[k] for k in ["isin", "ticker", "name", "category", "asset_class",
                           "region", "domicile", "acc_dist", "ter"] if k in c}
        for c in final
    ]}
    path = Path(__file__).resolve().parents[1] / "watchlist.yaml"
    header = (
        "# UCITS ETF watchlist — AUTO-GENERATED by scripts/build_watchlist.py.\n"
        "# Each `ticker` was verified to return data on Yahoo Finance. `isin` is the key;\n"
        "# `ter` is seeded from issuer factsheets. Re-run the script to refresh/extend.\n\n"
    )
    path.write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")

    print(f"\nResolved {len(final)}/{len(CANDIDATES)} ETFs -> {path}")
    if unresolved:
        print("Could not resolve (dropped):")
        for c in unresolved:
            print(f"  - {c['name']} ({c['isin']}) tried {c['tickers']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
