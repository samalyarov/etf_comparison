"""Cost & tax modelling for a European (UCITS / IBKR) buy-and-hold investor.

The headline TER is only part of what a EU investor actually pays. This module estimates
the pieces that matter over a long hold and expresses them as an annual drag so funds can
be compared honestly and DCA plans netted down:

* **Broker commission** — IBKR-style per-trade fee (a bps rate with a currency-specific
  minimum). Matters most for small, frequent DCA contributions.
* **FX conversion cost** — buying a USD/GBP-quoted ETF from a EUR account incurs IBKR's FX
  spread (a few bps, with a minimum). EUR-quoted funds avoid it entirely — a real reason to
  prefer the Xetra (.DE) line.
* **Tracking difference** — fund-vs-index slippage. Often close to the TER for physical
  replication, but it is the number that actually shows up in returns; estimated here.
* **Bid-ask spread** — a small round-trip cost, rougher for niche/thematic funds.
* **Tax drag** — for *distributing* funds, dividends are taxed as received each year; for
  *accumulating* funds tax is deferred (though some countries tax deemed distributions).
  Modelled from the distribution yield and a configurable investor tax rate.

Everything is an **estimate** with transparent, overridable assumptions — not tax advice.
All rates are fractions (0.0007 = 7 bps = 0.07%).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---- Broker commission defaults (IBKR "Tiered"-ish, EU retail; override in the UI) -------
# Minimum commission per order, keyed by trade currency.
COMMISSION_MIN = {"EUR": 1.25, "GBP": 1.00, "USD": 1.00, "CHF": 1.50}
COMMISSION_BPS = 0.00035          # 0.035% of trade value (IBKR Tiered, capped/floored)
FX_CONVERSION_BPS = 0.00002       # 0.002% IBKR FX spread ...
FX_CONVERSION_MIN = 2.0           # ... with a ~2 (base-ccy) minimum per conversion

# ---- Tax defaults (investor-level; highly country-specific — these are placeholders) -----
DEFAULT_DIVIDEND_TAX = 0.26       # e.g. many EU flat rates land ~26-30% on dividends


@dataclass(frozen=True)
class CommissionModel:
    """A simple broker commission model: a bps rate with a per-currency minimum + FX cost."""

    bps: float = COMMISSION_BPS
    fx_bps: float = FX_CONVERSION_BPS
    fx_min: float = FX_CONVERSION_MIN
    min_fee: dict[str, float] | None = None

    def _min_for(self, currency: str) -> float:
        table = self.min_fee or COMMISSION_MIN
        return table.get((currency or "EUR").upper(), 1.0)

    def trade_commission(self, notional: float, currency: str = "EUR") -> float:
        """Commission for a single order of ``notional`` (in trade currency)."""
        if notional <= 0:
            return 0.0
        return max(self.bps * notional, self._min_for(currency))

    def fx_cost(self, notional: float, needs_fx: bool) -> float:
        """FX conversion cost when the account (EUR) currency differs from the fund's."""
        if not needs_fx or notional <= 0:
            return 0.0
        return max(self.fx_bps * notional, self.fx_min)

    def buy_cost(self, notional: float, currency: str = "EUR",
                 account_currency: str = "EUR") -> float:
        """Total friction to buy ``notional``: commission + any FX conversion cost."""
        needs_fx = (currency or "EUR").upper() != (account_currency or "EUR").upper()
        return self.trade_commission(notional, currency) + self.fx_cost(notional, needs_fx)


DEFAULT_COMMISSION = CommissionModel()


def estimate_tracking_difference(ter: float | None, replication: str | None = None) -> float:
    """Rough annual fund-vs-index tracking difference (a drag, >= 0).

    Physical replication typically tracks close to (sometimes better than) TER thanks to
    securities lending; synthetic/thematic funds tend to slip a little more. Without a real
    index series we approximate: ~90% of TER for physical, ~120% for synthetic/unknown.
    """
    if ter is None:
        return 0.0
    rep = (replication or "").lower()
    factor = 0.9 if rep.startswith("phys") else 1.2
    return max(0.0, ter * factor)


def estimate_spread(asset_class: str | None = None, category: str | None = None) -> float:
    """Rough round-trip bid-ask spread cost (annualised assumption is left to the caller).

    Broad, liquid equity/bond trackers are tight (~5 bps); niche thematic/EM funds wider.
    """
    text = f"{asset_class or ''} {category or ''}".lower()
    if any(k in text for k in ("thematic", "clean", "robot", "water", "emerging", "small")):
        return 0.0025
    if "bond" in text:
        return 0.0007
    return 0.0010


def total_cost_of_ownership(ter: float | None, *, tracking_difference: float | None = None,
                            spread: float = 0.0, fx_bps: float = 0.0,
                            holding_years: float = 10.0,
                            replication: str | None = None) -> dict:
    """Estimate the annual all-in cost of holding a fund (TCO), beyond the headline TER.

    Spread and FX are one-off (round-trip / at purchase) costs amortised over the holding
    horizon. Returns a breakdown dict plus ``total_annual`` (the drag to subtract from CAGR).
    """
    td = tracking_difference if tracking_difference is not None \
        else estimate_tracking_difference(ter, replication)
    years = max(holding_years, 1.0)
    spread_annual = spread / years
    fx_annual = fx_bps / years
    total = (td or 0.0) + spread_annual + fx_annual
    return {
        "ter": ter or 0.0,
        "tracking_difference": td or 0.0,
        "spread_annual": spread_annual,
        "fx_annual": fx_annual,
        "total_annual": total,
    }


def tax_drag(dist_yield: float | None, acc_dist: str | None,
             dividend_tax: float = DEFAULT_DIVIDEND_TAX,
             deemed_distribution: bool = False) -> float:
    """Estimated annual tax drag from dividends.

    *Distributing* funds pay dividends that are taxed as received, so the drag is
    ``dist_yield * dividend_tax`` every year. *Accumulating* funds reinvest inside the
    wrapper and defer tax until sale — a drag of ~0 during the hold, unless the investor's
    country taxes *deemed* distributions (e.g. DE Vorabpauschale), which we approximate with
    the same formula when ``deemed_distribution`` is set.
    """
    y = dist_yield or 0.0
    if y <= 0:
        return 0.0
    is_acc = (acc_dist or "").upper().startswith("ACC")
    if is_acc and not deemed_distribution:
        return 0.0
    return y * dividend_tax


def domicile_note(domicile: str | None) -> str:
    """Short withholding-tax note for a fund domicile (buy-and-hold relevant)."""
    d = (domicile or "").upper()
    if d == "IE":
        return ("Ireland: US-Ireland treaty cuts US dividend withholding 30%→15%; ~60% of "
                "UCITS assets domicile here — generally preferred for US-equity exposure.")
    if d == "LU":
        return ("Luxembourg: 15% US withholding via treaty for most funds, but historically "
                "less favourable than IE for US-heavy exposure; fine for ex-US/EU funds.")
    return "Domicile withholding treatment not characterised."
