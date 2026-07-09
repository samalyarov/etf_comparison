"""Palette invariants for the *Meridian* design system (pure, no DB required).

These guard the accessibility contract of the theme layer independently of the DB-gated
UI smoke tests in ``test_app.py``: every text/UI token that carries meaning must clear
**WCAG AA** (4.5:1) against the surface it is actually painted on, in *both* themes. If a
future palette tweak dims a link, a gain figure, or a caption below AA, this fails.
"""

from __future__ import annotations

import pytest

from etf import theme as theme_mod


def _lin(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(fg: str, bg: str) -> float:
    la, lb = _lum(fg), _lum(bg)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# (foreground token, background token) pairs, named by how each is actually used in the UI.
# Each foreground is normal-size text (or a link) rendered on the named surface.
_AA_PAIRS = [
    ("ink", "plane"),        # primary text on the page
    ("ink", "surface"),      # primary text on cards / tables
    ("ink2", "surface"),     # secondary text on cards
    ("muted", "plane"),      # captions on the page
    ("muted", "surface"),    # captions / axis labels on cards
    ("accent", "plane"),     # links (masthead byline, footer) on the page
    ("accent", "surface"),   # links inside cards
    ("good", "surface"),     # gain figures in tables
    ("bad", "surface"),      # loss figures in tables
]


@pytest.mark.parametrize("theme_name", ["Light", "Dark"])
@pytest.mark.parametrize("fg,bg", _AA_PAIRS)
def test_text_tokens_meet_wcag_aa(theme_name, fg, bg):
    T = theme_mod.THEMES[theme_name]
    ratio = contrast(getattr(T, fg), getattr(T, bg))
    assert ratio >= 4.5, f"{theme_name}: {fg} on {bg} = {ratio:.2f} (< 4.5, fails WCAG AA)"


@pytest.mark.parametrize("theme_name", ["Light", "Dark"])
def test_accent_chrome_legible(theme_name):
    """Buttons and the selected nav tab paint ink on the teal accent — that ink (deep plane
    on dark, white on light) must clear AA so button/nav labels stay readable."""
    T = theme_mod.THEMES[theme_name]
    btn_ink = T.plane if T.is_dark else "#ffffff"
    ratio = contrast(btn_ink, T.accent)
    assert ratio >= 4.5, f"{theme_name}: button ink on accent = {ratio:.2f} (< 4.5)"
    nav = theme_mod.nav_styles(T)
    assert nav["nav-link-selected"]["background-color"] == T.accent


def test_both_themes_present_and_distinct():
    assert set(theme_mod.THEMES) == {"Light", "Dark"}
    assert theme_mod.LIGHT.plane != theme_mod.DARK.plane
    assert theme_mod.DEFAULT_THEME in theme_mod.THEMES
