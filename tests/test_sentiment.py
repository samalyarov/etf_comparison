"""Tests for the local sentiment scorer."""

from __future__ import annotations

from etf import sentiment


def test_positive_and_negative_text():
    assert sentiment.score_text("Stocks surge to record highs, strong rally") > 0.5
    assert sentiment.score_text("Market crash, huge losses, panic selloff") < -0.5


def test_neutral_and_unknown_text():
    assert sentiment.score_text("The fund holds several hundred companies") == 0.0
    assert sentiment.score_text("") == 0.0


def test_negation_flips_polarity():
    assert sentiment.score_text("this is not bullish") < 0
    assert sentiment.score_text("no crash coming") > 0


def test_summarize_labels():
    bull = sentiment.summarize(["surge rally record", "strong gains upgrade", "boom soar"])
    assert bull.label == "bullish" and bull.pos == 3
    bear = sentiment.summarize(["crash plunge", "selloff losses", "recession fear"])
    assert bear.label == "bearish" and bear.neg == 3


def test_summarize_empty():
    s = sentiment.summarize([])
    assert s.n == 0 and s.label == "neutral"


def test_contrarian_flag_extremes():
    euphoria = sentiment.summarize(["surge rally record boom", "strong gains upgrade soar",
                                    "bullish rally record", "boom surge strong"])
    tag, _ = sentiment.contrarian_flag(euphoria)
    assert "caution" in tag

    capit = sentiment.summarize(["crash plunge collapse", "selloff losses panic",
                                 "recession fear slump"])
    tag2, _ = sentiment.contrarian_flag(capit)
    assert "value" in tag2


def test_contrarian_flag_insufficient():
    tag, _ = sentiment.contrarian_flag(sentiment.summarize(["surge"]))
    assert tag == "insufficient"
