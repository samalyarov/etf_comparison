"""Visual themes: a light theme and a Tokyo Night dark theme.

Each theme is a token set (surfaces, ink, grid, an accent) plus a categorical chart
palette. The palettes are validated with the data-viz method's script — the dark set is
validated against the Tokyo Night surface (#1a1b26), the light set against white. The
Tokyo Night *accent* hues drive the app chrome; chart series use the CVD-safe steps so
identity never relies on colour alone (charts always ship legends / direct labels / tables).
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


# Categorical series — validated (light vs #fcfcfb, dark vs #1a1b26).
_LIGHT_SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                 "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
_DARK_SERIES = ["#3987e5", "#199e70", "#c98500", "#008300",
                "#9085e9", "#e66767", "#d55181", "#d95926"]

LIGHT = Theme(
    name="Light", is_dark=False,
    plane="#f7f8fa", surface="#ffffff", elevated="#eef1f6",
    ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e6e7ea", baseline="#c3c2b7", accent="#2a78d6",
    good="#0ca30c", bad="#d03b3b",
    series=_LIGHT_SERIES,
    diverging=[[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]],
)

# Tokyo Night — background #1a1b26, foreground #c0caf5, accent blue #7aa2f7.
TOKYO = Theme(
    name="Tokyo Night", is_dark=True,
    plane="#16161e", surface="#1a1b26", elevated="#24283b",
    ink="#c0caf5", ink2="#a9b1d6", muted="#565f89",
    grid="#2a2e42", baseline="#3b4261", accent="#7aa2f7",
    good="#9ece6a", bad="#f7768e",
    series=_DARK_SERIES,
    diverging=[[0.0, "#e66767"], [0.5, "#414868"], [1.0, "#3987e5"]],
)

THEMES = {"Tokyo Night": TOKYO, "Light": LIGHT}
DEFAULT_THEME = "Tokyo Night"


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
        "container": {"padding": "4px 0", "background-color": "transparent",
                      "border-bottom": f"1px solid {T.grid}", "margin-bottom": "6px"},
        "icon": {"color": T.muted, "font-size": "15px"},
        "nav-link": {
            "font-size": "14px", "font-weight": "500", "color": T.ink2,
            "padding": "8px 16px", "margin": "0 2px", "border-radius": "8px",
            "--hover-color": hover,
        },
        "nav-link-selected": {"background-color": T.accent,
                              "color": "#ffffff" if not T.is_dark else "#16161e",
                              "font-weight": "600"},
    }


def css(T: Theme) -> str:
    """Full-page CSS for theme ``T`` — reskins Streamlit chrome, cards, and text.

    (Canvas-rendered dataframes follow the static config.toml base theme, so they match
    the default Tokyo Night theme; in Light mode they stay dark — a known limitation.)
    """
    nav_selected_ink = "#16161e" if T.is_dark else "#ffffff"
    return f"""
<style>
:root {{
  --plane: {T.plane}; --surface: {T.surface}; --elevated: {T.elevated};
  --ink: {T.ink}; --ink2: {T.ink2}; --muted: {T.muted};
  --grid: {T.grid}; --accent: {T.accent};
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
[data-testid="stWidgetLabel"] label {{color: var(--ink2) !important;}}

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
div[data-testid="stMetric"] label {{color: var(--muted);}}
div[data-testid="stMetricValue"] {{font-size: 1.5rem; color: var(--ink);}}

/* Chart cards */
div[data-testid="stPlotlyChart"] {{
    background: var(--surface); border: 1px solid var(--grid); border-radius: 12px;
    padding: 8px 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}}

/* Inputs (best-effort dark/light) */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {{
    background: var(--surface) !important; border-color: var(--grid) !important;
}}

/* Buttons */
.stButton > button {{border-radius: 10px; font-weight: 600;
                     background: var(--accent); color: {nav_selected_ink}; border: none;}}

button[data-baseweb="tab"] {{font-weight: 600;}}
</style>
"""
