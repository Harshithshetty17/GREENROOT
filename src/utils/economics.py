"""What the advice costs, and what it has to earn back.

A recommendation a farmer cannot afford is not a recommendation. The system
prescribes fertiliser in kilograms; this module turns that into money, and then
into the question that actually decides whether to buy it: *how much extra
yield does this have to produce before it pays for itself?*

On not inventing prices
-----------------------
Fertiliser and crop prices vary by district, by season, by subsidy status and
by the week. This module ships **no default prices whatsoever**. Every figure
is computed from numbers the farmer enters for their own mandi and their own
dealer, and nothing is displayed until they do. A plausible-looking national
average would be worse than no number at all: it would carry the authority of
the rest of the system while being wrong for the person reading it.

All results are reported per **acre**, because that is the unit Indian
smallholdings are measured and sold in, converted from the model's per-hectare
prescription.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils.plain_language import ACRES_PER_HECTARE, per_acre

#: A standard fertiliser sack.
BAG_KG: float = 50.0

#: One quintal, the unit crops are priced in at an Indian mandi.
QUINTAL_KG: float = 100.0


@dataclass(frozen=True)
class CostLine:
    """One fertiliser product priced at the farmer's own rate.

    Attributes
    ----------
    product:
        Product name as prescribed.
    kg_per_hectare:
        Prescribed dose.
    price_per_kg:
        The farmer's local price, in rupees.
    """

    product: str
    kg_per_hectare: float
    price_per_kg: float

    @property
    def kg_per_acre(self) -> float:
        """Dose converted to the unit the farmer works in."""
        return per_acre(self.kg_per_hectare)

    @property
    def bags_per_acre(self) -> float:
        """Dose expressed in 50 kg sacks."""
        return self.kg_per_acre / BAG_KG

    @property
    def cost_per_acre(self) -> float:
        """Cost of this product for one acre, in rupees."""
        return self.kg_per_acre * self.price_per_kg


@dataclass(frozen=True)
class CostEstimate:
    """The full cost of following a fertiliser prescription.

    Attributes
    ----------
    lines:
        Per-product costings.
    priced:
        ``True`` when at least one product has a price, so the caller knows
        whether there is anything to show.
    """

    lines: List[CostLine] = field(default_factory=list)
    priced: bool = False

    @property
    def total_per_acre(self) -> float:
        """Total input cost for one acre, in rupees."""
        return float(sum(line.cost_per_acre for line in self.lines))

    @property
    def total_per_hectare(self) -> float:
        """Total input cost for one hectare, in rupees."""
        return self.total_per_acre * ACRES_PER_HECTARE

    def break_even_kg_per_acre(self, crop_price_per_quintal: float) -> Optional[float]:
        """Extra yield needed to pay for the fertiliser, in kg per acre.

        Parameters
        ----------
        crop_price_per_quintal:
            What the farmer expects to be paid per quintal at the mandi.

        Returns
        -------
        float or None
            Kilograms of additional produce per acre required to cover the
            input cost. ``None`` when no crop price has been supplied — the
            question is unanswerable without it, and a guess would be worse
            than silence.
        """
        if crop_price_per_quintal <= 0 or self.total_per_acre <= 0:
            return None
        price_per_kg = crop_price_per_quintal / QUINTAL_KG
        return self.total_per_acre / price_per_kg

    def break_even_quintals_per_acre(
        self, crop_price_per_quintal: float
    ) -> Optional[float]:
        """Break-even yield expressed in quintals per acre."""
        kilograms = self.break_even_kg_per_acre(crop_price_per_quintal)
        return None if kilograms is None else kilograms / QUINTAL_KG

    def summary(self, simple: bool = True) -> str:
        """Return a one-line reading of the cost."""
        if not self.priced or self.total_per_acre <= 0:
            return (
                "Enter what you pay for fertiliser to see what this advice costs."
                if simple
                else "No prices supplied; cost is not computed."
            )
        return (
            f"Following this advice costs about ₹{self.total_per_acre:,.0f} "
            f"per acre in fertiliser."
            if simple
            else f"Prescription cost: ₹{self.total_per_acre:,.2f}/acre "
            f"(₹{self.total_per_hectare:,.2f}/ha)."
        )

    def break_even_message(
        self, crop: str, crop_price_per_quintal: float, simple: bool = True
    ) -> Optional[str]:
        """Return the break-even sentence, or ``None`` without a crop price."""
        kilograms = self.break_even_kg_per_acre(crop_price_per_quintal)
        if kilograms is None:
            return None
        quintals = kilograms / QUINTAL_KG
        return (
            f"At ₹{crop_price_per_quintal:,.0f} per quintal for {crop}, this "
            f"fertiliser pays for itself if it gains you about "
            f"{kilograms:,.0f} kg ({quintals:.2f} quintal) more per acre. "
            f"Below that, you lose money on it."
            if simple
            else f"Break-even at ₹{crop_price_per_quintal:,.2f}/quintal: "
            f"{kilograms:,.1f} kg/acre ({quintals:.2f} q/acre) of additional "
            f"yield."
        )


def estimate_cost(
    plan: Dict[str, float], prices_per_kg: Dict[str, float]
) -> CostEstimate:
    """Cost a fertiliser prescription at the farmer's own prices.

    Parameters
    ----------
    plan:
        ``product -> kg/ha`` from the advisory.
    prices_per_kg:
        ``product -> rupees per kg``, entered by the farmer. Products with no
        price, or a non-positive one, are excluded rather than assumed.

    Returns
    -------
    CostEstimate
        Costed lines for the priced products only.
    """
    lines: List[CostLine] = []
    for product, kilograms in (plan or {}).items():
        price = float(prices_per_kg.get(product, 0.0) or 0.0)
        if price <= 0 or kilograms <= 0:
            continue
        lines.append(
            CostLine(
                product=product,
                kg_per_hectare=float(kilograms),
                price_per_kg=price,
            )
        )
    return CostEstimate(lines=lines, priced=bool(lines))


def price_per_kg_from_bag(bag_price: float, bag_kg: float = BAG_KG) -> float:
    """Convert a sack price to a per-kilogram price.

    Dealers quote per bag, so this is how a farmer's own number reaches the
    calculation without them having to do the arithmetic.
    """
    if bag_price <= 0 or bag_kg <= 0:
        return 0.0
    return bag_price / bag_kg


__all__ = [
    "BAG_KG",
    "QUINTAL_KG",
    "CostLine",
    "CostEstimate",
    "estimate_cost",
    "price_per_kg_from_bag",
]
