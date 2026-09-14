"""Simulating the advisory: what happens if the farmer actually does this?

The system tells a cultivator to add 40 kg/ha of urea. The obvious next
question — and the one the dashboard could not previously answer — is *what
that buys*. Does the recommendation strengthen? Does a different crop become
viable? Is the fertiliser worth the money at all?

This module closes that loop. It converts the advisory's fertiliser
prescription back into the nutrient additions it represents, applies them to
the farmer's readings, and re-runs the same inference path. The comparison is
honest in both directions: if following the advice barely moves the posterior,
the dashboard says so, which is useful information about whether to spend on
inputs at all.

What is deliberately *not* simulated
------------------------------------
pH amendments. The advisory recommends lime or gypsum for soils outside the
workable range, but the pH response to a given dose depends on the soil's
buffering capacity — clay content, organic matter, cation exchange capacity —
none of which this system measures. Modelling it from the seven inputs would
be inventing a number. The dashboard reports the amendment as advice and
leaves its effect out of the simulation, saying why.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.config import FEATURE_BOUNDS, FEATURE_NAMES
from src.models.inference import CropRecommender, PredictionResult
from src.utils.agronomy_advisory import AdvisoryReport

logger = logging.getLogger(__name__)

#: Product name -> (nutrient, mass fraction of that nutrient in the product).
#: Mirrors the grades used to build the prescription, inverted so a dose can be
#: converted back into the element it supplies.
_PRODUCT_NUTRIENTS: Dict[str, Tuple[str, float]] = {
    "Urea": ("N", 0.46),
    "Single Super Phosphate (SSP)": ("P", 0.16),
    "Muriate of Potash (MOP)": ("K", 0.60),
}

#: Confidence change below which following the advice is not worth the outlay.
NEGLIGIBLE_GAIN_PP: float = 1.0


@dataclass(frozen=True)
class InterventionResult:
    """Before-and-after comparison of following the fertiliser advice.

    Attributes
    ----------
    before, after:
        Predictions on the original and amended readings.
    nutrients_added:
        ``nutrient -> kg/ha of the element`` supplied by the prescription.
    products:
        ``product -> kg/ha`` as prescribed, for display.
    amended_features:
        The readings after the prescription is applied.
    unsimulated:
        Advisory categories present but deliberately excluded from the
        simulation, so the UI can say what is missing rather than imply the
        comparison is complete.
    """

    before: PredictionResult
    after: PredictionResult
    nutrients_added: Dict[str, float] = field(default_factory=dict)
    products: Dict[str, float] = field(default_factory=dict)
    amended_features: Dict[str, float] = field(default_factory=dict)
    unsimulated: Tuple[str, ...] = ()

    @property
    def crop_changed(self) -> bool:
        """``True`` when the prescription changes which crop is recommended."""
        return self.before.crop != self.after.crop

    @property
    def confidence_delta(self) -> float:
        """Change in confidence, in percentage points.

        Measured on the *original* crop when the recommendation is unchanged,
        and on the new crop when it changes — in both cases, the confidence
        attached to whatever the farmer would be told to plant.
        """
        return self.after.confidence - self.before.confidence

    @property
    def original_crop_delta(self) -> float:
        """Change in confidence for the originally recommended crop.

        Distinct from :attr:`confidence_delta` when the argmax moves: this
        tracks one fixed crop, so it answers "did my chosen crop get a better
        fit?" rather than "did the top answer get stronger?".
        """
        try:
            index = self.after.class_names.index(self.before.crop)
        except ValueError:  # pragma: no cover - class sets always match
            return 0.0
        return float(self.after.probabilities[index]) * 100.0 - self.before.confidence

    @property
    def is_worthwhile(self) -> bool:
        """``True`` when the prescription meaningfully improves the outcome."""
        return self.crop_changed or self.original_crop_delta >= NEGLIGIBLE_GAIN_PP

    @property
    def total_product_kg(self) -> float:
        """Total fertiliser prescribed, kg/ha across all products."""
        return float(sum(self.products.values()))

    def verdict(self, simple: bool = True) -> str:
        """Return a plain reading of whether the advice is worth following."""
        if not self.products:
            return (
                "Your soil already has enough of everything this crop needs. "
                "No fertiliser to add."
                if simple
                else "No nutrient deficit; the prescription is empty."
            )
        if self.crop_changed:
            return (
                f"After adding this fertiliser, {self.after.crop} becomes the "
                f"better choice instead of {self.before.crop} — the extra "
                f"plant food changes what your land suits best."
                if simple
                else f"The prescription shifts the argmax from "
                f"{self.before.crop} to {self.after.crop}."
            )
        delta = self.original_crop_delta
        if delta >= NEGLIGIBLE_GAIN_PP:
            return (
                f"Adding this fertiliser makes {self.before.crop} a clearly "
                f"better fit — the match rises by about {delta:.0f} points "
                f"out of 100."
                if simple
                else f"The prescription raises confidence in "
                f"{self.before.crop} by {delta:+.2f} pp."
            )
        return (
            f"Adding this fertiliser barely changes the match for "
            f"{self.before.crop} (about {delta:+.0f} points out of 100). Your "
            f"soil is already close to what this crop wants, so spend the "
            f"money only if you also want the yield benefit."
            if simple
            else f"Confidence moves {delta:+.2f} pp — below the "
            f"{NEGLIGIBLE_GAIN_PP:.0f} pp threshold at which the intervention "
            f"is materially informative."
        )

    def comparison_rows(self) -> List[Dict[str, object]]:
        """Per-nutrient before/after rows for a comparison table."""
        rows: List[Dict[str, object]] = []
        for nutrient in ("N", "P", "K"):
            added = self.nutrients_added.get(nutrient, 0.0)
            if added <= 0:
                continue
            rows.append(
                {
                    "nutrient": nutrient,
                    "before": round(self.before.raw_features[nutrient], 1),
                    "added": round(added, 1),
                    "after": round(self.amended_features[nutrient], 1),
                }
            )
        return rows


def nutrients_from_plan(plan: Dict[str, float]) -> Dict[str, float]:
    """Convert a fertiliser prescription into the nutrients it supplies.

    Parameters
    ----------
    plan:
        ``product -> kg/ha`` as produced by the advisory.

    Returns
    -------
    dict
        ``nutrient -> kg/ha of the element``. Unknown products are skipped
        rather than guessed at.
    """
    supplied: Dict[str, float] = {}
    for product, quantity in plan.items():
        entry = _PRODUCT_NUTRIENTS.get(product)
        if entry is None:
            logger.debug("No nutrient grade recorded for %r; skipping", product)
            continue
        nutrient, fraction = entry
        supplied[nutrient] = supplied.get(nutrient, 0.0) + float(quantity) * fraction
    return supplied


def simulate_advisory(
    recommender: CropRecommender,
    features: Dict[str, float],
    advisory: AdvisoryReport,
    before: Optional[PredictionResult] = None,
) -> Optional[InterventionResult]:
    """Re-run the recommendation as if the fertiliser advice had been followed.

    Parameters
    ----------
    recommender:
        The deployed ensemble wrapper.
    features:
        The farmer's current readings, keyed by feature name.
    advisory:
        The advisory whose fertiliser plan should be applied.
    before:
        The existing prediction for ``features``, to avoid recomputing it.

    Returns
    -------
    InterventionResult or None
        ``None`` when the advisory prescribes nothing to add, so callers can
        skip the comparison entirely rather than render an empty one.
    """
    plan = dict(advisory.fertiliser_plan or {})
    supplied = nutrients_from_plan(plan)
    if not supplied:
        return None

    amended = dict(features)
    for nutrient, added in supplied.items():
        low, high = FEATURE_BOUNDS[nutrient]
        # Clip so a large prescription cannot push the reading outside the
        # envelope the validator would reject.
        amended[nutrient] = float(min(max(amended[nutrient] + added, low), high))

    original = before or recommender.predict(
        [features[name] for name in FEATURE_NAMES]
    )
    after = recommender.predict([amended[name] for name in FEATURE_NAMES])

    # Name what the comparison leaves out, so it is not read as complete.
    unsimulated: List[str] = []
    if any(item.category == "Soil Reaction" and item.severity != "info"
           for item in advisory.items):
        unsimulated.append("Soil Reaction")

    return InterventionResult(
        before=original,
        after=after,
        nutrients_added={k: round(v, 2) for k, v in supplied.items()},
        products=plan,
        amended_features=amended,
        unsimulated=tuple(unsimulated),
    )


__all__ = [
    "InterventionResult",
    "NEGLIGIBLE_GAIN_PP",
    "nutrients_from_plan",
    "simulate_advisory",
]
