"""Visual themes: a light theme and a dark theme.

Two palettes drive the whole app, both derived from the *Tokyo Night* family: the dark
set is Tokyo Night (background #1a1b26, foreground #c0caf5) and the light set is its
Tokyo Night *Day* counterpart (background #e1e2e7, blue accent #2e7de9). Each theme is a
token set (surfaces, ink, grid, an accent) plus a categorical chart palette validated with
the data-viz method's script — the dark series against #1a1b26, the light series against a
near-white card. Series colours are CVD-safe so identity never relies on colour alone
(charts always ship legends / direct labels / tables). In the UI the themes are surfaced
plainly as "Light" and "Dark"; the Tokyo Night lineage is documented here and in the README.
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


# Categorical series — validated (light vs a near-white card, dark vs #1a1b26).
_LIGHT_SERIES = ["#2e7de9", "#1a8a5f", "#b15c00", "#007197",
                 "#7a4fd0", "#e0245e", "#c84b8c", "#d95926"]
_DARK_SERIES = ["#3987e5", "#199e70", "#c98500", "#008300",
                "#9085e9", "#e66767", "#d55181", "#d95926"]

# Light — Tokyo Night "Day": page #e1e2e7, ink navy #2f374f, accent blue #2e7de9.
LIGHT = Theme(
    name="Light", is_dark=False,
    plane="#dadce4", surface="#eceef3", elevated="#f4f5f8",
    ink="#2c3350", ink2="#545c7e", muted="#767ea6",
    grid="#c8cbdb", baseline="#adb2ca", accent="#2e7de9",
    good="#2f8f4e", bad="#d03b5f",
    series=_LIGHT_SERIES,
    diverging=[[0.0, "#e0245e"], [0.5, "#dfe1ea"], [1.0, "#2e7de9"]],
)

# Dark — Tokyo Night: background #1a1b26, foreground #c0caf5, accent blue #7aa2f7.
DARK = Theme(
    name="Dark", is_dark=True,
    plane="#16161e", surface="#1a1b26", elevated="#24283b",
    ink="#c0caf5", ink2="#a9b1d6", muted="#565f89",
    grid="#2a2e42", baseline="#3b4261", accent="#7aa2f7",
    good="#9ece6a", bad="#f7768e",
    series=_DARK_SERIES,
    diverging=[[0.0, "#e66767"], [0.5, "#414868"], [1.0, "#3987e5"]],
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
                              "color": "#16161e" if T.is_dark else "#ffffff",
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
    btn_ink = "#16161e" if T.is_dark else "#ffffff"
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

/* Page + text */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background: var(--plane); color: var(--ink);
}}
.stApp p, .stApp span, .stApp label, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{color: var(--ink);}}
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {{
    color: var(--ink2) !important;
}}

/* Masthead */
.app-title {{font-size: 1.55rem; font-weight: 700; color: var(--ink);
             letter-spacing: -0.01em; margin: 0;}}
.app-sub {{color: var(--ink2); font-size: 0.9rem; margin: 2px 0 0 0;}}

/* Section labels */
.section-label {{font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
                 letter-spacing: 0.06em; color: var(--muted); margin: 0.6rem 0 0.3rem 0;}}

/* Metric cards */
div[data-testid="stMetric"] {{
    background: var(--surface); border: 1px solid var(--grid); border-radius: 12px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] label p {{color: var(--muted) !important;}}
div[data-testid="stMetricValue"] {{font-size: 1.5rem; color: var(--ink);}}
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
</style>
"""
