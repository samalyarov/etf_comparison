"""Visual themes: a light theme and a dark theme — the *Meridian* design system.

Meridian is a calm, institutional finance palette: a deep **slate-navy** desk (dark) and
a warm **paper** desk (light), each unified by a single muted **teal** accent and reserved
green/red semantics for up/down. The direction is deliberately restrained and trustworthy —
a research instrument, not a dashboard — after the calm professional feel of firms like
Lowden Financial, and away from the previous developer-IDE (Tokyo Night) palette.

Each theme is a token set (surfaces, ink, grid, accent) plus a categorical chart palette
validated with the data-viz method's script — the dark series against the slate surface
(#141f2b), the light series against the warm card (#f6f5f1). Series colours are CVD-safe so
identity never relies on colour alone (charts always ship legends / direct labels / tables);
the accent is a chrome colour (nav, buttons, links, focus) kept analogous to — but distinct
from — the blue-led chart order, so data viz stays coherent with itself. In the UI the
themes are surfaced plainly as "Light" and "Dark".
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Theme:
    name: str
    is_dark: bool
    plane: str        # page background
    surface: str      # cards / chart surface
    elevated: str     # hover / inputs
    ink: str          # primary text
    ink2: str         # secondary text
    muted: str        # axes / labels
    grid: str         # hairline gridlines
    baseline: str     # axis / baseline
    accent: str       # primary UI accent (nav-selected, buttons, links)
    good: str
    bad: str
    series: list[str] = field(default_factory=list)
    diverging: list = field(default_factory=list)

    def color(self, i: int) -> str:
        """Categorical colour for series index i (folded upstream past 8)."""
        return self.series[i % len(self.series)]


# Categorical series — CVD-safe, blue-led order (light validated vs the warm card #f6f5f1,
# dark vs the slate surface #141f2b). Unchanged from the prior validated set: both surfaces
# sit at effectively the same luminance as before, so the series remain in-band.
_LIGHT_SERIES = ["#2e7de9", "#1a8a5f", "#b15c00", "#007197",
                 "#7a4fd0", "#e0245e", "#c84b8c", "#d95926"]
_DARK_SERIES = ["#3987e5", "#199e70", "#c98500", "#008300",
                "#9085e9", "#e66767", "#d55181", "#d95926"]

# Light — Meridian "Paper": warm greige page, deep slate ink, teal accent.
LIGHT = Theme(
    name="Light", is_dark=False,
    plane="#e9e7e0", surface="#f6f5f1", elevated="#fdfcf9",
    ink="#1b2733", ink2="#48586a", muted="#586472",
    grid="#dcd8ce", baseline="#c4bfb2", accent="#0c6b62",
    good="#1f7a3e", bad="#c23b46",
    series=_LIGHT_SERIES,
    diverging=[[0.0, "#d5405a"], [0.5, "#e0ddd4"], [1.0, "#2e7de9"]],
)

# Dark — Meridian "Slate": deep slate-navy desk, cool near-white ink, luminous teal accent.
DARK = Theme(
    name="Dark", is_dark=True,
    plane="#0e1620", surface="#141f2b", elevated="#1e2c3a",
    ink="#e7edf3", ink2="#a6b4c4", muted="#8093a8",
    grid="#233343", baseline="#324457", accent="#4bc5b5",
    good="#5cc98a", bad="#e5726e",
    series=_DARK_SERIES,
    diverging=[[0.0, "#e5726e"], [0.5, "#2b3c4e"], [1.0, "#3987e5"]],
)

THEMES = {"Light": LIGHT, "Dark": DARK}
DEFAULT_THEME = "Dark"


def style_fig(fig, T: Theme, *, height: int = 420, hovermode="x unified",
              showlegend: bool = True, legend_bottom: bool = True):
    """Apply the house Plotly style for theme ``T``."""
    fig.update_layout(
        height=height,
        colorway=T.series,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  size=13, color=T.ink2),
        paper_bgcolor=T.surface,
        plot_bgcolor=T.surface,
        hovermode=hovermode,
        showlegend=showlegend,
        margin=dict(l=12, r=12, t=14, b=12),
        hoverlabel=dict(bgcolor=T.elevated, bordercolor=T.grid,
                        font=dict(color=T.ink, size=12)),
    )
    if legend_bottom and showlegend:
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=T.ink2)))
    axis = dict(gridcolor=T.grid, zerolinecolor=T.grid, linecolor=T.baseline,
                tickfont=dict(color=T.muted, size=11), title_font=dict(color=T.ink2, size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def nav_styles(T: Theme) -> dict:
    """Style dict for streamlit-option-menu (horizontal top nav)."""
    hover = T.elevated
    return {
        # Paint the component container so the iframe never shows the static base
        # (dark) background through in Light mode — this is the top "tab bar".
        "container": {"padding": "4px 0", "background-color": T.plane,
                      "border-bottom": f"1px solid {T.grid}", "margin-bottom": "6px"},
        "icon": {"color": T.muted, "font-size": "15px"},
        "nav-link": {
            "font-size": "14px", "font-weight": "500", "color": T.ink2,
            "padding": "8px 16px", "margin": "0 2px", "border-radius": "8px",
            "background-color": T.plane, "--hover-color": hover,
        },
        "nav-link-selected": {"background-color": T.accent,
                              "color": T.plane if T.is_dark else "#ffffff",
                              "font-weight": "600"},
    }


def table_styles(T: Theme) -> list[dict]:
    """`Styler.set_table_styles` rules so HTML tables follow theme ``T``.

    Streamlit's ``st.dataframe`` renders to a canvas that reads the *static* config
    theme, so it can't follow the runtime toggle (it stays dark in Light mode). We render
    tables as themed HTML instead, which respects both themes exactly.
    """
    mono = 'system-ui, -apple-system, "Segoe UI", sans-serif'
    return [
        {"selector": "", "props": [
            ("border-collapse", "collapse"), ("width", "100%"),
            ("font-family", mono), ("font-size", "0.83rem"), ("color", T.ink)]},
        {"selector": "thead th", "props": [
            ("background", T.elevated), ("color", T.ink2), ("text-align", "right"),
            ("padding", "8px 12px"), ("border-bottom", f"1px solid {T.baseline}"),
            ("font-weight", "600"), ("white-space", "nowrap"),
            ("position", "sticky"), ("top", "0"), ("z-index", "1")]},
        {"selector": "tbody td", "props": [
            ("color", T.ink), ("text-align", "right"), ("padding", "7px 12px"),
            ("border-bottom", f"1px solid {T.grid}"), ("white-space", "nowrap")]},
        {"selector": "tbody th", "props": [
            ("color", T.ink), ("text-align", "left"), ("padding", "7px 12px"),
            ("border-bottom", f"1px solid {T.grid}"), ("font-weight", "500"),
            ("white-space", "nowrap")]},
        {"selector": "thead th.blank", "props": [("background", T.elevated)]},
        {"selector": "tbody td:first-child, thead th:first-child",
         "props": [("text-align", "left")]},
        {"selector": "tbody tr:hover td, tbody tr:hover th",
         "props": [("background", T.elevated)]},
    ]


def css(T: Theme) -> str:
    """Full-page CSS for theme ``T`` — reskins Streamlit chrome, cards, widgets, text."""
    # Ink that reads on the teal accent: the deep plane on dark, white on light.
    btn_ink = T.plane if T.is_dark else "#ffffff"
    # A soft tint of the accent for the masthead rule / focus glow (theme-appropriate alpha).
    glow = "rgba(75,197,181,0.35)" if T.is_dark else "rgba(12,107,98,0.25)"
    return f"""
<style>
:root {{
  --plane: {T.plane}; --surface: {T.surface}; --elevated: {T.elevated};
  --ink: {T.ink}; --ink2: {T.ink2}; --muted: {T.muted};
  --grid: {T.grid}; --baseline: {T.baseline}; --accent: {T.accent};
}}

/* Trim chrome, tighten top */
#MainMenu, footer, header[data-testid="stHeader"] {{visibility: hidden;}}
.block-container {{padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1400px;}}

/* Page + text. Tabular figures everywhere — a research ledger aligns its numbers. */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background: var(--plane); color: var(--ink);
    font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1, "cv01" 1;
}}
.stApp p, .stApp span, .stApp label, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{color: var(--ink);}}
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {{
    color: var(--ink2) !important;
}}

/* Masthead — wordmark, a teal monogram, and a signature meridian rule. */
.masthead {{display: flex; align-items: center; gap: 12px; margin: 0 0 2px 0;}}
.masthead-mark {{
    flex: 0 0 auto; width: 34px; height: 34px; border-radius: 9px;
    background: var(--accent); color: {btn_ink}; font-weight: 800; font-size: 0.92rem;
    letter-spacing: -0.02em; display: flex; align-items: center; justify-content: center;
    box-shadow: 0 1px 3px {glow};
}}
.app-title {{font-size: 1.5rem; font-weight: 750; color: var(--ink);
             letter-spacing: -0.015em; margin: 0; line-height: 1.1;}}
.app-title .accent {{color: var(--accent);}}
.app-sub {{color: var(--ink2); font-size: 0.9rem; margin: 3px 0 0 0;}}
.app-sub a {{color: var(--accent); text-decoration: none; font-weight: 600;}}
.app-sub a:hover {{text-decoration: underline;}}
/* The meridian: a thin accent rule that fades into the grid — the page's signature line. */
.meridian-rule {{height: 2px; border: 0; margin: 8px 0 2px 0; border-radius: 2px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent) 90px, var(--grid) 90px,
                var(--grid) 100%);}}

/* Section labels */
.section-label {{font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
                 letter-spacing: 0.06em; color: var(--muted); margin: 0.6rem 0 0.3rem 0;}}

/* Metric cards */
div[data-testid="stMetric"] {{
    background: var(--surface); border: 1px solid var(--grid); border-radius: 12px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] label p {{color: var(--muted) !important;}}
div[data-testid="stMetricValue"] {{font-size: 1.5rem; color: var(--ink);
    font-variant-numeric: tabular-nums;}}
div[data-testid="stMetricDelta"] {{color: var(--ink2);}}

/* Chart cards */
div[data-testid="stPlotlyChart"] {{
    background: var(--surface); border: 1px solid var(--grid); border-radius: 12px;
    padding: 8px 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}}

/* Themed HTML tables (rendered via pandas Styler) */
.tbl-wrap {{
    background: var(--surface); border: 1px solid var(--grid); border-radius: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06); overflow-x: auto; max-width: 100%;
    margin: 2px 0 4px 0;
}}
.tbl-wrap table {{margin: 0;}}
.tbl-wrap td, .tbl-wrap th {{background: transparent;}}
.tbl-wrap td {{font-variant-numeric: tabular-nums;}}

/* --- Inputs: selects, multiselect, number/text — force readable text in both modes --- */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
div[data-baseweb="base-input"], [data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {{
    background: var(--surface) !important; border-color: var(--grid) !important;
    color: var(--ink) !important;
}}
div[data-baseweb="select"] *, div[data-baseweb="input"] input {{color: var(--ink) !important;}}
div[data-baseweb="select"] svg {{fill: var(--muted) !important; color: var(--muted) !important;}}
/* Multiselect chips (selected ETF names) */
span[data-baseweb="tag"] {{
    background: var(--elevated) !important; color: var(--ink) !important;
    border: 1px solid var(--grid) !important;
}}
span[data-baseweb="tag"] span, span[data-baseweb="tag"] svg {{color: var(--ink) !important; fill: var(--ink) !important;}}

/* Dropdown popovers render in a body-level portal (outside .stApp) — theme globally */
ul[role="listbox"], div[data-baseweb="popover"] ul, div[data-baseweb="menu"] {{
    background: var(--surface) !important;
}}
ul[role="listbox"] li, div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li, li[role="option"] {{color: var(--ink) !important;}}
li[role="option"]:hover, ul[role="listbox"] li:hover {{background: var(--elevated) !important;}}

/* Radio (theme toggle) + slider labels */
[data-testid="stRadio"] label, [data-testid="stSlider"] label {{color: var(--ink) !important;}}
[data-testid="stSlider"] [data-baseweb="slider"] div {{color: var(--ink);}}

/* Buttons — Streamlit's own testid selectors are specific, so use !important */
.stButton > button, button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-secondary"] {{
    border-radius: 10px !important; font-weight: 600 !important;
    background: var(--accent) !important; color: {btn_ink} !important; border: none !important;
}}
.stButton > button *, button[data-testid="stBaseButton-primary"] * {{color: {btn_ink} !important;}}

/* Captions */
.stApp [data-testid="stCaptionContainer"], .stApp small {{color: var(--muted) !important;}}

button[data-baseweb="tab"] {{font-weight: 600; color: var(--ink);}}

/* Accessible keyboard focus — a calm accent ring on interactive chrome */
.stApp a:focus-visible, .stButton > button:focus-visible,
div[data-baseweb="select"] > div:focus-within,
[data-testid="stNumberInput"] input:focus, [data-testid="stTextInput"] input:focus,
[data-baseweb="tab"]:focus-visible {{
    outline: 2px solid var(--accent) !important; outline-offset: 2px;
    box-shadow: 0 0 0 4px {glow} !important;
}}

/* Persistent attribution footer — built-by credit + LinkedIn, no photograph */
.app-footer {{
    margin: 2.2rem 0 0.4rem 0; padding: 14px 2px 4px 2px;
    border-top: 1px solid var(--grid);
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px;
    font-size: 0.82rem; color: var(--muted);
}}
.app-footer .foot-mark {{
    flex: 0 0 auto; width: 22px; height: 22px; border-radius: 6px;
    background: var(--accent); color: {btn_ink};
    font-weight: 800; font-size: 0.62rem; letter-spacing: -0.02em;
    display: inline-flex; align-items: center; justify-content: center;
}}
.app-footer .foot-by {{color: var(--ink2);}}
.app-footer a {{color: var(--accent); text-decoration: none; font-weight: 600;}}
.app-footer a:hover {{text-decoration: underline;}}
.app-footer .foot-note {{margin-left: auto; color: var(--muted); text-align: right;
    max-width: 46ch;}}
@media (max-width: 720px) {{
    .app-footer .foot-note {{margin-left: 0; text-align: left; max-width: 100%;}}
}}
</style>
"""
