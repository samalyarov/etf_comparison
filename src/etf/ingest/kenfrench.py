"""Ken French Data Library adapter — European Fama/French factors + momentum.

Downloads and parses the freely-published **European** factor return series from the Ken
French Data Library at Dartmouth:

* ``Europe_5_Factors_CSV.zip`` — the Fama/French 5 factors for developed Europe:
  ``Mkt-RF`` (market excess return), ``SMB`` (size), ``HML`` (value), ``RMW``
  (profitability), ``CMA`` (investment), plus the risk-free rate ``RF``.
* ``Europe_Mom_Factor_CSV.zip`` — the Carhart momentum factor ``WML`` (winners-minus-losers)
  for the same region.

**Convention (documented choice).** We use the **monthly** series, not daily. Monthly is the
academic standard for factor-loading regressions (Fama–French 1993/2015, Carhart 1997), it
matches the monthly basis of :mod:`etf.projection`, and it avoids the non-synchronous-trading
noise that inflates daily-return regressions across a multi-currency UCITS blend. The raw
files quote returns in **percent** with ``-99.99`` marking missing data; the parser converts
to **decimals** and maps sentinels to ``NaN``. We store the European regional set because the
tool's universe is EU/UCITS (CLAUDE.md), and the Mkt-RF/RF are region-consistent.

Only :func:`fetch_europe_factors` touches the network; :func:`parse_ff_csv` is pure and is
what the tests exercise against a committed fixture, so the loader + regression are verifiable
offline even when the download rate-limits.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd

KEN_FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# region/frequency -> {dataset: zip filename}. Monthly is the shipped default (see docstring).
EUROPE_FILES = {
    "monthly": {
        "five_factor": "Europe_5_Factors_CSV.zip",
        "momentum": "Europe_Mom_Factor_CSV.zip",
    },
    "daily": {
        "five_factor": "Europe_5_Factors_Daily_CSV.zip",
        "momentum": "Europe_Mom_Factor_Daily_CSV.zip",
    },
}

# Canonical ordering of the merged factor columns (RF last; WML appended from the mom file).
FACTOR_ORDER = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "WML", "RF"]

MISSING_SENTINEL = -99.99  # Ken French marks unavailable observations with this value


def _index_to_date(token: str, frequency: str) -> pd.Timestamp | None:
    """Map a Ken French period token to a period-end Timestamp, or None if not this block.

    Monthly rows are ``YYYYMM`` and map to the calendar month-end (aligning with
    ``projection.monthly_returns`` which resamples to ``ME``). Daily rows are ``YYYYMMDD``.
    Annual rows (``YYYY``) and any non-numeric header return None so the parser stops at the
    end of the first data block (the file appends an annual section after the periodic one).
    """
    token = token.strip()
    if not token.isdigit():
        return None
    if frequency == "monthly" and len(token) == 6:
        return pd.Period(f"{token[:4]}-{token[4:]}", freq="M").to_timestamp(how="end").normalize()
    if frequency == "daily" and len(token) == 8:
        return pd.Timestamp(f"{token[:4]}-{token[4:6]}-{token[6:]}")
    return None


def parse_ff_csv(text: str, frequency: str = "monthly") -> pd.DataFrame:
    """Parse a raw Ken French factor CSV into a date x factor DataFrame of **decimals**.

    Pure and offline. Reads the first data block that follows the factor header row (the file
    later appends an annual-average section, which is deliberately ignored). Returns are
    converted from percent to decimals and ``-99.99`` sentinels become ``NaN``. An empty
    frame is returned for text with no recognisable block rather than raising.
    """
    lines = text.splitlines()
    header: list[str] | None = None
    records: dict[pd.Timestamp, dict[str, float]] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            if header is not None and records:
                break  # blank line ends the periodic block; stop before the annual section
            continue
        cells = [c.strip() for c in line.split(",")]
        # A header row starts with an empty first cell followed by factor names.
        if cells[0] == "" and any(c for c in cells[1:]):
            if header is None:
                header = [c for c in cells[1:] if c]
            continue
        if header is None:
            continue  # still inside the descriptive preamble
        dt = _index_to_date(cells[0], frequency)
        if dt is None:
            if records:
                break  # left the periodic block (annual header/rows) — done
            continue
        row: dict[str, float] = {}
        for name, cell in zip(header, cells[1:]):
            try:
                val = float(cell)
            except ValueError:
                continue
            row[name] = float("nan") if val == MISSING_SENTINEL else val / 100.0
        if row:
            records[dt] = row
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(records, orient="index").sort_index()
    df.index.name = "date"
    return df


def _download_zip_csv(filename: str, timeout: int = 30) -> str:
    """Download a Ken French CSV zip and return the (single) CSV member's text. Network."""
    req = urllib.request.Request(KEN_FRENCH_BASE + filename,
                                 headers={"User-Agent": "Mozilla/5.0 (etf_comparison)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted host)
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        member = zf.namelist()[0]
        return zf.read(member).decode("latin-1")


def fetch_europe_factors(frequency: str = "monthly", timeout: int = 30) -> pd.DataFrame:
    """Fetch + merge the European FF5 and momentum series into one factor matrix. Network.

    Returns a date x factor DataFrame (decimals) with columns ordered per
    :data:`FACTOR_ORDER` (only those present). The momentum series begins later than the
    5-factor set, so ``WML`` is left-joined and carries ``NaN`` before its first observation.
    Raises on a failed/blocked download so the caller can report the fetch as pending.
    """
    files = EUROPE_FILES[frequency]
    five = parse_ff_csv(_download_zip_csv(files["five_factor"], timeout), frequency)
    mom = parse_ff_csv(_download_zip_csv(files["momentum"], timeout), frequency)
    return merge_factor_frames(five, mom)


def merge_factor_frames(five: pd.DataFrame, momentum: pd.DataFrame) -> pd.DataFrame:
    """Join the 5-factor frame with the momentum frame; order columns canonically (pure)."""
    if five.empty:
        return five
    merged = five.join(momentum, how="left") if not momentum.empty else five.copy()
    cols = [c for c in FACTOR_ORDER if c in merged.columns]
    extra = [c for c in merged.columns if c not in cols]
    return merged[cols + extra]
