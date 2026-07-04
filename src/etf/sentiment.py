"""Lightweight, local, finance-aware sentiment scoring — deliberately *supplementary*.

Per the feasibility work in ``brain/sentiment_analysis.md``: for broad-index UCITS ETFs on
a buy-and-hold horizon, sentiment is weak, short-horizon, gameable colour — **never a
buy/sell trigger**. So this module is intentionally modest and honest:

* A tiny built-in finance polarity lexicon (Loughran-McDonald / VADER-lite) scores text in
  ``[-1, +1]`` with **no network, no API key, no heavy dependency**. Good enough for an
  on-demand "what does this chatter read like?" panel; swap in FinBERT/an LLM later.
* Aggregation turns a batch of headlines/threads into a mean score, a bull/bear split, and
  a mention count (volume is often a cleaner signal than polarity).
* The decision-support piece is a **contrarian flag**, not a score to chase: extreme
  euphoria → caution; capitulation → possible value; otherwise neutral. Explicitly framed
  as context, and non-predictive for a multi-year hold.

The intended UI is an on-demand Detail-page panel: paste the headlines/threads you're
reading, get an instant scored read + the contrarian framing. A live Reddit/Finnhub feed
can later populate the same functions (see the design doc); the scoring here stays local.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Compact finance-tone lexicons (word -> polarity). Not exhaustive — an MVP heuristic.
_POSITIVE = {
    "beat": 1.0, "beats": 1.0, "surge": 1.0, "surged": 1.0, "rally": 1.0, "rallies": 1.0,
    "gain": 0.8, "gains": 0.8, "up": 0.5, "rise": 0.7, "rising": 0.7, "record": 0.8,
    "strong": 0.8, "outperform": 1.0, "bullish": 1.0, "buy": 0.7, "growth": 0.7,
    "profit": 0.8, "profits": 0.8, "upgrade": 0.9, "upgraded": 0.9, "boom": 1.0,
    "recovery": 0.7, "rebound": 0.8, "soar": 1.0, "soared": 1.0, "optimistic": 0.8,
    "resilient": 0.7, "momentum": 0.6, "dividend": 0.4, "cheap": 0.5, "undervalued": 0.8,
}
_NEGATIVE = {
    "miss": -1.0, "misses": -1.0, "plunge": -1.0, "plunged": -1.0, "crash": -1.0,
    "crashes": -1.0, "fall": -0.7, "falls": -0.7, "drop": -0.7, "dropped": -0.7,
    "loss": -0.8, "losses": -0.8, "down": -0.5, "weak": -0.8, "bearish": -1.0,
    "sell": -0.7, "selloff": -1.0, "downgrade": -0.9, "downgraded": -0.9, "slump": -1.0,
    "recession": -1.0, "fear": -0.8, "fears": -0.8, "panic": -1.0, "bubble": -0.7,
    "overvalued": -0.8, "risk": -0.4, "warning": -0.7, "cut": -0.6, "collapse": -1.0,
    "tumble": -1.0, "tumbled": -1.0, "correction": -0.6, "volatile": -0.4, "dump": -0.9,
}
_NEGATORS = {"not", "no", "never", "isn't", "wasn't", "don't", "won't", "without"}

_TOKEN_RE = re.compile(r"[a-z']+")

# Category/theme -> query keywords + suggested communities (for a future live source).
THEME_QUERIES = {
    "US Large Cap": (["S&P 500", "US stocks", "Nasdaq"], ["r/Bogleheads", "r/ETFs"]),
    "World": (["world index", "VWCE", "IWDA", "global stocks"], ["r/eupersonalfinance"]),
    "Technology": (["tech stocks", "Nasdaq 100", "semiconductors"], ["r/stocks", "r/ETFs"]),
    "Emerging Markets": (["emerging markets", "EM stocks", "China stocks"], ["r/ETFs"]),
    "Clean Energy": (["clean energy", "solar stocks", "renewables"], ["r/stocks"]),
    "Gold": (["gold price", "gold ETF"], ["r/investing"]),
    "Bonds": (["bond yields", "treasury", "rate cuts"], ["r/bonds", "r/eupersonalfinance"]),
}


@dataclass
class SentimentSummary:
    """Aggregate read over a batch of texts."""

    n: int
    mean_score: float          # average polarity in [-1, 1]
    pos: int
    neg: int
    neutral: int
    label: str                 # bullish | bearish | mixed | neutral

    @property
    def pos_share(self) -> float:
        return self.pos / self.n if self.n else 0.0


def score_text(text: str) -> float:
    """Polarity of one piece of text in ``[-1, 1]`` using the finance lexicon.

    Handles simple negation (a negator within two tokens flips the next scored word). The
    score is the mean polarity of matched words, so texts with no known words score 0.
    """
    if not text:
        return 0.0
    tokens = _TOKEN_RE.findall(text.lower())
    hits: list[float] = []
    for i, tok in enumerate(tokens):
        pol = _POSITIVE.get(tok, 0.0) + _NEGATIVE.get(tok, 0.0)
        if pol == 0.0:
            continue
        window = tokens[max(0, i - 2):i]
        if any(w in _NEGATORS for w in window):
            pol = -pol
        hits.append(pol)
    if not hits:
        return 0.0
    return max(-1.0, min(1.0, sum(hits) / len(hits)))


def classify(score: float, threshold: float = 0.15) -> str:
    """Bucket a score into pos/neg/neutral."""
    if score > threshold:
        return "pos"
    if score < -threshold:
        return "neg"
    return "neutral"


def summarize(texts: list[str]) -> SentimentSummary:
    """Aggregate a batch of texts into a :class:`SentimentSummary`."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return SentimentSummary(0, 0.0, 0, 0, 0, "neutral")
    scores = [score_text(t) for t in texts]
    buckets = [classify(s) for s in scores]
    pos = buckets.count("pos")
    neg = buckets.count("neg")
    neutral = buckets.count("neutral")
    mean = sum(scores) / len(scores)
    if pos and neg and abs(pos - neg) <= max(1, len(texts) // 5):
        label = "mixed"
    elif mean > 0.1:
        label = "bullish"
    elif mean < -0.1:
        label = "bearish"
    else:
        label = "neutral"
    return SentimentSummary(len(texts), mean, pos, neg, neutral, label)


def contrarian_flag(summary: SentimentSummary) -> tuple[str, str]:
    """Turn a sentiment read into a *contrarian* context flag (not a trigger).

    The only defensible buy-and-hold use of retail sentiment is at extremes: euphoria is a
    caution flag, capitulation a possible-value flag. Returns ``(tag, explanation)``.
    """
    if summary.n < 3:
        return ("insufficient", "Too few items to read — treat as noise.")
    strong = summary.pos_share
    if summary.mean_score > 0.4 and strong > 0.7:
        return ("euphoria — caution",
                "Crowd is strongly bullish. Historically a weak *contrarian* caution sign, "
                "not a sell signal for a long hold.")
    if summary.mean_score < -0.4:
        return ("capitulation — possible value",
                "Crowd is strongly bearish. For a buy-and-hold investor this is where "
                "regular contributions do their best work — again, context, not a trigger.")
    return ("neutral", "No sentiment extreme — nothing actionable for a buy-and-hold plan.")
