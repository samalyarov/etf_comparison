"""Dutch box-3 tax modelling for a private buy-and-hold investor.

The Netherlands taxes personal savings and investments in **box 3**. This module models
the two regimes a Dutch investor needs to compare when deciding how to hold bonds (or any
security), as pure, testable functions — *not* tax advice, and deliberately simplified:

* **Box 3 (current, 2026)** — a *fictitious-return wealth tax*. You are **not** taxed on
  the coupons/dividends/gains you actually receive; instead a **forfaitair rendement**
  (assumed return) is imputed on the *value* of your assets above a tax-free allowance
  (``heffingsvrij vermogen``), and that imputed return is taxed at the box-3 rate. So two
  funds with the same market value owe the same box-3 tax regardless of whether they
  distribute or accumulate — the reinvest-vs-cash-out choice does **not** change the tax
  bill, only the after-tax wealth path.

* **"Werkelijk rendement" (actual-return regime)** — the reform that taxes the **actual**
  return: interest/coupons + dividends + rent, plus the annual change in value
  (``vermogensaanwastbelasting`` / capital-growth tax) for liquid assets like bond ETFs.
  Realised *and* unrealised value changes are captured each year, above a small tax-free
  *result* allowance, at the same 36% rate. Originally slated for 2027, it has been
  postponed (adopted by the Tweede Kamer on 2026-02-12; intended start now 1 Jan 2028,
  pending Eerste Kamer approval and possible amendment). We model the defensible simple
  version — ``tax = max(0, actual_return - allowance) * rate`` — and expose every parameter.

All monetary inputs/outputs are in EUR. All rates are fractions (0.36 = 36%). Parameters
are module constants with sourced comments so they are easy to update when the definitive
figures are published.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- Box 3 (2026)
# Sources (fetched 2026-07): Belastingdienst box-3 2026 calculation page; SRA / Stolwijk
# "Box 3 percentages, vrijstellingen en bedragen 2026". The savings and debt forfaits are
# only *provisionally* set for 2026 and are fixed definitively in early 2027; the
# investment ("overige bezittingen") forfait and the rate are fixed.
HEFFINGSVRIJ_VERMOGEN_2026 = 59_357.0   # tax-free capital, per person (partners: 2x)
FORFAIT_SAVINGS_2026 = 0.0128           # deemed return on bank savings (provisional)
FORFAIT_INVESTMENTS_2026 = 0.0600       # deemed return on "overige bezittingen" (bonds/ETFs)
FORFAIT_DEBT_2026 = 0.0270              # deemed cost rate on qualifying debt (provisional)
BOX3_RATE_2026 = 0.36                   # box-3 tax rate

# --------------------------------------------------------------------------- Actual return
# Sources (fetched 2026-07): Rijksoverheid "plannen werkelijk rendement box 3"; Deloitte /
# Meijburg summaries of the Wet werkelijk rendement box 3. Tax-free *result* (not capital)
# of ~EUR 1,800 per person, taxed at 36% on actual income + annual value change. Start now
# intended 2028 (postponed from 2027); figures may still change before enactment.
HEFFINGVRIJ_RESULTAAT = 1_800.0         # tax-free actual-return result, per person
ACTUAL_RETURN_RATE = 0.36               # rate applied to taxable actual return


@dataclass(frozen=True)
class Box3Result:
    """Breakdown of a current box-3 (fictitious-return wealth tax) computation."""

    asset_value: float
    allowance: float
    taxable_base: float      # asset_value - allowance, floored at 0
    deemed_return: float     # forfaitair rendement on the taxable base
    tax: float               # deemed_return * rate


@dataclass(frozen=True)
class ActualReturnResult:
    """Breakdown of a "werkelijk rendement" (actual-return) computation."""

    actual_return: float     # coupons/dividends + value change over the year
    allowance: float         # tax-free result
    taxable_return: float    # actual_return - allowance, floored at 0
    tax: float               # taxable_return * rate


def box3_tax(
    asset_value: float,
    *,
    allowance: float = HEFFINGSVRIJ_VERMOGEN_2026,
    forfait: float = FORFAIT_INVESTMENTS_2026,
    rate: float = BOX3_RATE_2026,
    partners: bool = False,
) -> Box3Result:
    """Current box-3 wealth tax on an *investment* holding (bonds/ETFs = overige bezittingen).

    Models a single-asset-class holding: a forfaitair rendement (``forfait``, default the
    2026 investment rate of 6.00%) is imputed on the value above the tax-free allowance and
    taxed at ``rate``. ``partners`` doubles the allowance (fiscal partners share it). Note
    this is independent of the coupons actually received — that is the whole point of the
    regime. Returns a :class:`Box3Result` breakdown.
    """
    av = max(float(asset_value), 0.0)
    alw = float(allowance) * (2.0 if partners else 1.0)
    base = max(av - alw, 0.0)
    deemed = base * forfait
    return Box3Result(asset_value=av, allowance=alw, taxable_base=base,
                      deemed_return=deemed, tax=deemed * rate)


def actual_return_tax(
    actual_return: float,
    *,
    allowance: float = HEFFINGVRIJ_RESULTAAT,
    rate: float = ACTUAL_RETURN_RATE,
    partners: bool = False,
) -> ActualReturnResult:
    """Proposed "werkelijk rendement" tax on the *actual* return realised over a period.

    ``actual_return`` is the real return in EUR — for a bond ETF: coupons/interest received
    **plus** the change in market value over the year (capital-growth basis, so realised and
    unrealised alike). Tax is ``max(0, actual_return - allowance) * rate``. A negative
    actual return yields zero tax here (loss set-off across years/box is out of scope for
    this simple model). ``partners`` doubles the tax-free result. Returns an
    :class:`ActualReturnResult` breakdown.
    """
    ar = float(actual_return)
    alw = float(allowance) * (2.0 if partners else 1.0)
    taxable = max(ar - alw, 0.0)
    return ActualReturnResult(actual_return=ar, allowance=alw,
                              taxable_return=taxable, tax=taxable * rate)


def effective_wealth_tax_rate(
    *,
    forfait: float = FORFAIT_INVESTMENTS_2026,
    rate: float = BOX3_RATE_2026,
) -> float:
    """Box-3 tax as a fraction of *asset value* for wealth above the allowance.

    A convenience for intuition: ``forfait * rate`` (e.g. 6.00% * 36% = 2.16% of assets per
    year on the taxable base). Handy for comparing the wealth tax's drag against a bond's
    running yield.
    """
    return forfait * rate
