"""Visual theme: validated data-viz palette, Plotly styling, and app CSS.

Colours are the reference categorical palette from the data-viz method (validated:
worst adjacent CVD ΔE 24.2 in light mode). Identity is never carried by colour alone —
charts always ship a legend and/or direct labels plus a table view as relief.
"""

from __future__ import annotations

import plotly.graph_objects as go

# Categorical series hues (light surface), assigned in fixed order, never cycled.
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
          "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

# Chrome & ink tokens (light).
SURFACE = "#ffffff"       # chart/card surface
PLANE = "#f7f8fa"         # page plane
INK = "#0b0b0b"           # primary
INK_2 = "#52514e"         # secondary
MUTED = "#898781"         # axes / labels
GRID = "#e6e7ea"          # hairline gridlines
BASELINE = "#c3c2b7"      # axis / baseline
GOOD = "#0ca30c"
BAD = "#d03b3b"

# Diverging scale (blue positive ↔ gray ↔ red negative) for correlation & returns.
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
# Sequential blue ramp (light→dark) for single-hue magnitude.
SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]


def series_color(i: int) -> str:
    """Categorical colour for series index i (folds to 'Other' beyond 8 upstream)."""
    return SERIES[i % len(SERIES)]


def style_fig(fig: go.Figure, *, height: int = 420, hovermode="x unified",
              showlegend: bool = True, legend_bottom: bool = True) -> go.Figure:
    """Apply the house Plotly style: recessive grid, muted axes, tidy legend."""
    fig.update_layout(
        height=height,
        colorway=SERIES,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  size=13, color=INK_2),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        hovermode=hovermode,
        showlegend=showlegend,
        margin=dict(l=12, r=12, t=14, b=12),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID,
                        font=dict(color=INK, size=12)),
    )
    if legend_bottom and showlegend:
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=INK_2)))
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BASELINE,
                tickfont=dict(color=MUTED, size=11), title_font=dict(color=INK_2, size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


# ---------------------------------------------------------------- option-menu styles
def nav_styles() -> dict:
    """Style dict for streamlit-option-menu (horizontal top nav)."""
    return {
        "container": {"padding": "4px 0", "background-color": "transparent",
                      "border-bottom": f"1px solid {GRID}", "margin-bottom": "6px"},
        "icon": {"color": MUTED, "font-size": "15px"},
        "nav-link": {
            "font-size": "14px", "font-weight": "500", "color": INK_2,
            "padding": "8px 16px", "margin": "0 2px", "border-radius": "8px",
            "--hover-color": "#eef2f8",
        },
        "nav-link-selected": {
            "background-color": "#2a78d6", "color": "#ffffff", "font-weight": "600",
        },
    }


# ---------------------------------------------------------------- global CSS
CUSTOM_CSS = """
<style>
/* Trim Streamlit chrome and tighten the top of the page */
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1400px;}

/* App masthead */
.app-title {font-size: 1.55rem; font-weight: 700; color: #0b0b0b; letter-spacing: -0.01em;
            margin: 0;}
.app-sub {color: #52514e; font-size: 0.9rem; margin: 2px 0 0 0;}

/* Section headings */
h2, h3 {letter-spacing: -0.01em;}
.section-label {font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.06em; color: #898781; margin: 0.6rem 0 0.3rem 0;}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e6e7ea; border-radius: 12px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(11,11,11,0.04);
}
div[data-testid="stMetric"] label {color: #898781; font-weight: 600;}
div[data-testid="stMetricValue"] {font-size: 1.5rem; color: #0b0b0b;}

/* Chart containers: give plots a card feel */
div[data-testid="stPlotlyChart"] {
    background: #ffffff; border: 1px solid #e6e7ea; border-radius: 12px;
    padding: 8px 10px; box-shadow: 0 1px 2px rgba(11,11,11,0.04);
}

/* Dataframes */
div[data-testid="stDataFrame"] {border-radius: 12px;}

/* Buttons */
.stButton > button {border-radius: 10px; font-weight: 600;}

/* Tabs a touch cleaner */
button[data-baseweb="tab"] {font-weight: 600;}
</style>
"""
