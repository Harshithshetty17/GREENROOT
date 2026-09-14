"""Agronomic heuristics translating a prediction into field-actionable advice.

A crop label alone is not a decision. A cultivator needs to know *what to
change*: how much urea to apply, whether the plot will waterlog, whether the
soil needs liming before sowing. This module closes that gap.

Reference envelopes are derived empirically from the benchmark corpus rather
than hard-coded: for each crop the per-feature mean and standard deviation over
its 100 training exemplars define the agronomic envelope the model itself
learned. Advice is therefore always consistent with the recommendation, and
deficits are expressed in the same units the farmer measured.

Nutrient gaps are converted to commercial fertiliser quantities using the
standard Indian straight-fertiliser grades — urea (46% N), single super
phosphate (16% P₂O₅) and muriate of potash (60% K₂O).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.core.config import BENCHMARK_DATASET, FEATURE_NAMES
from src.utils.plain_language import describe_quantity

logger = logging.getLogger(__name__)

# Severity ladder for advisory items.
CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

#: Nutrient content of the common straight fertilisers, as a mass fraction.
#: Applying ``deficit / fraction`` kg/ha of the product supplies ``deficit``
#: kg/ha of the nutrient element.
_FERTILISER_GRADES: Dict[str, tuple[str, float]] = {
    "N": ("Urea", 0.46),
    "P": ("Single Super Phosphate (SSP)", 0.16),
    "K": ("Muriate of Potash (MOP)", 0.60),
}

#: Crops cultivated under puddled, continuously-flooded conditions. Their water
#: requirement is a standing head, not merely adequate rainfall.
_PADDY_CROPS = frozenset({"rice", "jute"})

#: Perennial and plantation crops for which a single-season nutrient
#: prescription must be framed as a split annual dose.
_PERENNIAL_CROPS = frozenset(
    {"coconut", "coffee", "apple", "orange", "papaya", "mango", "grapes", "pomegranate"}
)

#: Nitrogen-fixing legumes. Blanket nitrogen on these suppresses nodulation, so
#: a nitrogen "deficit" is not a deficiency to correct.
_LEGUME_CROPS = frozenset(
    {"chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil"}
)

#: Rainfall (mm/month) above which drainage becomes the limiting constraint for
#: a non-paddy crop, and below which a paddy crop cannot be sustained by rain.
_WATERLOGGING_RAINFALL_MM = 220.0
_PADDY_MINIMUM_RAINFALL_MM = 150.0

#: |Z| within the crop's own envelope beyond which a parameter is off-target.
_ENVELOPE_SIGMA = 1.5

#: Everyday names for the macronutrients, used by the plain register.
_NUTRIENT_WORDS: Dict[str, str] = {
    "N": "nitrogen (the one that makes leaves green)",
    "P": "phosphorus (the one that builds roots)",
    "K": "potassium (the one that fills the grain)",
}

_PROFILE_LOCK = threading.Lock()
_PROFILES: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None


@dataclass(frozen=True)
class AdvisoryItem:
    """One actionable recommendation, in two registers.

    Attributes
    ----------
    category:
        Grouping key, e.g. ``'Hydrology'`` or ``'Nutrition'``.
    severity:
        :data:`CRITICAL`, :data:`WARNING`, or :data:`INFO`.
    message:
        The technical statement, for the report and the evaluator.
    plain:
        The same advice in everyday words, for the farmer. Falls back to
        :attr:`message` when no plain wording was supplied, so an item can
        never render blank.
    """

    category: str
    severity: str
    message: str
    plain: str = ""

    def __post_init__(self) -> None:
        if not self.plain:
            object.__setattr__(self, "plain", self.message)

    def say(self, simple: bool = True) -> str:
        """Return this item's text in the requested register."""
        return self.plain if simple else self.message

    @property
    def icon(self) -> str:
        """A glyph suitable for dashboard rendering."""
        return {CRITICAL: "🔴", WARNING: "🟠", INFO: "🟢"}.get(self.severity, "•")


@dataclass(frozen=True)
class AdvisoryReport:
    """The complete advisory package for one recommendation.

    Attributes
    ----------
    crop:
        Crop the advice pertains to.
    items:
        Every advisory item, ordered critical-first.
    nutrient_gaps:
        Signed ``observed - required`` gap per macronutrient, in kg/ha. A
        negative value is a deficit.
    fertiliser_plan:
        Product name -> kg/ha to apply, for nutrients in deficit.
    """

    crop: str
    items: List[AdvisoryItem] = field(default_factory=list)
    nutrient_gaps: Dict[str, float] = field(default_factory=dict)
    fertiliser_plan: Dict[str, float] = field(default_factory=dict)

    @property
    def critical_items(self) -> List[AdvisoryItem]:
        """Items that block sowing until resolved."""
        return [item for item in self.items if item.severity == CRITICAL]

    def by_category(self, category: str) -> List[AdvisoryItem]:
        """Return the items filed under ``category``."""
        return [item for item in self.items if item.category == category]

    def messages(self, simple: bool = False) -> List[str]:
        """Return every advisory message, in the requested register."""
        return [item.say(simple) for item in self.items]


def _load_profiles(
    dataset_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Derive per-crop agronomic envelopes from the benchmark corpus.

    Returns
    -------
    dict
        ``{crop: {feature: {'mean': float, 'std': float,
        'low': float, 'high': float}}}`` where the bounds are ±1σ. Empty if the
        dataset is unavailable, in which case advisory degrades to the
        distribution-free heuristics only.
    """
    path = Path(dataset_path) if dataset_path else BENCHMARK_DATASET
    if not path.exists():
        logger.warning("Benchmark dataset unavailable at %s; envelopes disabled.", path)
        return {}

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning("Could not parse benchmark dataset: %s", exc)
        return {}

    profiles: Dict[str, Dict[str, Dict[str, float]]] = {}
    for crop, block in frame.groupby("label"):
        stats: Dict[str, Dict[str, float]] = {}
        for feature in FEATURE_NAMES:
            series = pd.to_numeric(block[feature], errors="coerce").dropna()
            if series.empty:
                continue
            mean = float(series.mean())
            std = float(series.std(ddof=0)) or 1e-9
            stats[feature] = {
                "mean": mean,
                "std": std,
                "low": float(series.quantile(0.10)),
                "high": float(series.quantile(0.90)),
            }
        profiles[str(crop)] = stats
    return profiles


def get_crop_profiles() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return the memoised per-crop agronomic envelopes."""
    global _PROFILES
    with _PROFILE_LOCK:
        if _PROFILES is None:
            _PROFILES = _load_profiles()
        return _PROFILES


def get_crop_profile(crop: str) -> Dict[str, Dict[str, float]]:
    """Return the envelope for one crop, or an empty mapping if unknown."""
    return get_crop_profiles().get(str(crop), {})


def _hydrology_items(crop: str, rainfall: float, humidity: float) -> List[AdvisoryItem]:
    """Assess water supply, drainage and disease pressure."""
    items: List[AdvisoryItem] = []
    lowered = crop.lower()

    if lowered in _PADDY_CROPS and rainfall < _PADDY_MINIMUM_RAINFALL_MM:
        deficit = _PADDY_MINIMUM_RAINFALL_MM - rainfall
        items.append(
            AdvisoryItem(
                "Hydrology",
                CRITICAL,
                f"{crop.capitalize()} requires a standing water head that "
                f"{rainfall:.0f} mm cannot sustain — a shortfall of about "
                f"{deficit:.0f} mm. Secure canal or borewell irrigation before "
                f"transplanting, or switch to the runner-up crop.",
                plain=(
                    f"{crop.capitalize()} needs standing water in the field. "
                    f"The rain you get ({rainfall:.0f} mm) is not enough. "
                    f"Make sure you have canal or borewell water before you "
                    f"plant. If you cannot, choose the next crop on the list."
                ),
            )
        )
    elif lowered not in _PADDY_CROPS and rainfall > _WATERLOGGING_RAINFALL_MM:
        items.append(
            AdvisoryItem(
                "Hydrology",
                WARNING,
                f"At {rainfall:.0f} mm, rainfall exceeds what an unbunded "
                f"{crop} plot drains freely. Open 30–45 cm field drains along "
                f"the slope and raise beds to keep the root zone aerated; "
                f"waterlogging here presents as root rot, not drought stress.",
                plain=(
                    f"You get a lot of rain ({rainfall:.0f} mm). Water will "
                    f"stand in the field and rot the roots. Dig drainage "
                    f"channels about knee-deep along the slope, and plant on "
                    f"raised beds so the roots can breathe."
                ),
            )
        )
    else:
        items.append(
            AdvisoryItem(
                "Hydrology",
                INFO,
                f"Rainfall of {rainfall:.0f} mm sits within the workable range "
                f"for {crop}. Schedule irrigation by soil moisture rather than "
                f"by calendar.",
                plain=(
                    f"Your rain ({rainfall:.0f} mm) is right for {crop}. "
                    f"Water when the soil feels dry, not on a fixed day."
                ),
            )
        )

    if humidity >= 85.0:
        items.append(
            AdvisoryItem(
                "Hydrology",
                WARNING,
                f"Relative humidity of {humidity:.0f}% sustains fungal "
                f"pressure. Widen row spacing for canopy airflow and keep a "
                f"prophylactic fungicide schedule ready.",
                plain=(
                    f"The air is very damp ({humidity:.0f}%). Damp air brings "
                    f"fungus and leaf disease. Leave more space between rows "
                    f"so air can move, and keep fungicide ready."
                ),
            )
        )
    elif humidity <= 35.0:
        items.append(
            AdvisoryItem(
                "Hydrology",
                INFO,
                f"Low humidity ({humidity:.0f}%) raises evapotranspiration. "
                f"Mulch the inter-row to curb surface evaporation losses.",
                plain=(
                    f"The air is dry ({humidity:.0f}%), so water leaves the "
                    f"soil fast. Spread straw or dry leaves between the rows "
                    f"to hold the moisture in."
                ),
            )
        )
    return items


def _nutrition_items(
    crop: str, observed: Dict[str, float], profile: Dict[str, Dict[str, float]]
) -> tuple[List[AdvisoryItem], Dict[str, float], Dict[str, float]]:
    """Compare observed macronutrients against the crop envelope.

    Returns the advisory items, the signed gaps, and the fertiliser plan.
    """
    items: List[AdvisoryItem] = []
    gaps: Dict[str, float] = {}
    plan: Dict[str, float] = {}
    lowered = crop.lower()

    for nutrient in ("N", "P", "K"):
        stats = profile.get(nutrient)
        if not stats:
            continue
        required = stats["mean"]
        present = float(observed.get(nutrient, 0.0))
        gap = present - required
        gaps[nutrient] = round(gap, 2)

        # Only act once the gap clears the crop's own natural spread.
        if abs(gap) <= _ENVELOPE_SIGMA * stats["std"]:
            tolerance = _ENVELOPE_SIGMA * stats["std"]
            items.append(
                AdvisoryItem(
                    "Nutrition",
                    INFO,
                    f"{nutrient} at {present:.0f} kg/ha sits inside the "
                    f"{max(required - tolerance, 0):.0f}–{required + tolerance:.0f} "
                    f"kg/ha band for {crop} (target {required:.0f}). A "
                    f"maintenance dose is sufficient.",
                    plain=(
                        f"Your {_NUTRIENT_WORDS[nutrient]} level is right for "
                        f"{crop}. Just add your normal amount."
                    ),
                )
            )
            continue

        if gap < 0:
            deficit = abs(gap)
            product, fraction = _FERTILISER_GRADES[nutrient]
            quantity = round(deficit / fraction, 1)

            if nutrient == "N" and lowered in _LEGUME_CROPS:
                items.append(
                    AdvisoryItem(
                        "Nutrition",
                        INFO,
                        f"Nitrogen reads {deficit:.0f} kg/ha below the "
                        f"envelope, but {crop} fixes its own through root "
                        f"nodules. Apply only a 15–20 kg/ha starter dose and "
                        f"inoculate the seed with Rhizobium instead — a full "
                        f"correction would suppress nodulation.",
                        plain=(
                            f"Your nitrogen looks low, but do not add much. "
                            f"{crop.capitalize()} makes its own nitrogen in "
                            f"its roots. Add only a small starter dose, and "
                            f"treat the seed with Rhizobium culture. Too much "
                            f"urea will stop the roots doing their job."
                        ),
                    )
                )
                continue

            plan[product] = quantity
            split = (
                " Split it across three applications through the year, as this "
                "is a perennial stand."
                if lowered in _PERENNIAL_CROPS
                else " Apply half at sowing and top-dress the remainder at "
                "active vegetative growth."
            )
            plain_split = (
                " Do not put it all at once — split it into three doses "
                "across the year, as this is a long-standing tree crop."
                if lowered in _PERENNIAL_CROPS
                else " Put half at sowing time, and the other half when the "
                "plants are growing well."
            )
            items.append(
                AdvisoryItem(
                    "Nutrition",
                    WARNING,
                    f"{nutrient} is {deficit:.0f} kg/ha short of the {crop} "
                    f"envelope. Apply roughly {quantity:.0f} kg/ha of "
                    f"{product}.{split}",
                    plain=(
                        f"Your soil is low on "
                        f"{_NUTRIENT_WORDS[nutrient]}. Add {product} — "
                        f"{describe_quantity(quantity)}.{plain_split}"
                    ),
                )
            )
        else:
            items.append(
                AdvisoryItem(
                    "Nutrition",
                    WARNING,
                    f"{nutrient} exceeds the {crop} envelope by {gap:.0f} "
                    f"kg/ha. Withhold {nutrient}-bearing fertiliser this "
                    f"season; the surplus leaches to groundwater and, for "
                    f"nitrogen, drives vegetative growth at the cost of yield.",
                    plain=(
                        f"You already have plenty of "
                        f"{_NUTRIENT_WORDS[nutrient]}. Do not add any more "
                        f"this season. It would be money wasted, it washes "
                        f"into the groundwater, and too much makes the plant "
                        f"grow leaves instead of grain."
                    ),
                )
            )
    return items, gaps, plan


def _ph_items(crop: str, ph: float, profile: Dict[str, Dict[str, float]]) -> List[AdvisoryItem]:
    """Assess soil reaction and prescribe an amendment where needed."""
    stats = profile.get("ph")
    target = stats["mean"] if stats else 6.5

    if ph < 5.5:
        return [
            AdvisoryItem(
                "Soil Reaction",
                CRITICAL,
                f"Soil pH of {ph:.1f} is strongly acidic. Phosphorus locks "
                f"into iron and aluminium complexes below 5.5 and becomes "
                f"unavailable regardless of how much is applied. Incorporate "
                f"agricultural lime at 2–3 t/ha two to three weeks before "
                f"sowing to lift the reaction toward {target:.1f}.",
                plain=(
                    f"Your soil is very sour (pH {ph:.1f}). This is the first "
                    f"thing to fix. In sour soil the plant cannot take in "
                    f"phosphorus, however much fertiliser you add — so the "
                    f"money is wasted. Spread agricultural lime, about "
                    f"400–500 kg per acre, and mix it into the soil 2 to 3 "
                    f"weeks before you sow."
                ),
            )
        ]
    if ph > 8.2:
        return [
            AdvisoryItem(
                "Soil Reaction",
                CRITICAL,
                f"Soil pH of {ph:.1f} is strongly alkaline, which immobilises "
                f"iron and zinc and induces interveinal chlorosis. Apply "
                f"gypsum at 1.5–2 t/ha and incorporate well-decomposed organic "
                f"matter to buffer the reaction toward {target:.1f}.",
                plain=(
                    f"Your soil is very salty and alkaline (pH {ph:.1f}). Fix "
                    f"this first. The plant cannot take in iron and zinc, so "
                    f"the leaves turn yellow between the veins. Spread gypsum, "
                    f"about 300–400 kg per acre, and mix in well-rotted farm "
                    f"yard manure or compost."
                ),
            )
        ]
    if abs(ph - target) > 1.0:
        direction = "below" if ph < target else "above"
        return [
            AdvisoryItem(
                "Soil Reaction",
                WARNING,
                f"Soil pH of {ph:.1f} sits {abs(ph - target):.1f} units "
                f"{direction} the {target:.1f} optimum for {crop}. The crop "
                f"remains viable, but expect reduced nutrient-use efficiency; "
                f"organic matter will narrow the gap over successive seasons.",
                plain=(
                    f"Your soil pH is {ph:.1f}. {crop.capitalize()} likes it "
                    f"nearer {target:.1f}. The crop will still grow, but it "
                    f"will not use the fertiliser as well. Adding farm yard "
                    f"manure or compost every season will slowly correct this."
                ),
            )
        ]
    return [
        AdvisoryItem(
            "Soil Reaction",
            INFO,
            f"Soil pH of {ph:.1f} is well matched to {crop} "
            f"(optimum ≈ {target:.1f}); macronutrients stay plant-available.",
            plain=(
                f"Your soil pH ({ph:.1f}) suits {crop} well. The plant can "
                f"take in the fertiliser you give it."
            ),
        )
    ]


def _thermal_items(
    crop: str, temperature: float, profile: Dict[str, Dict[str, float]]
) -> List[AdvisoryItem]:
    """Assess the thermal regime against the crop envelope."""
    stats = profile.get("temperature")
    if not stats:
        return []
    low, high = stats["low"], stats["high"]

    if temperature < low:
        return [
            AdvisoryItem(
                "Thermal Regime",
                WARNING,
                f"At {temperature:.1f} °C the plot runs below the "
                f"{low:.1f}–{high:.1f} °C band {crop} was characterised over. "
                f"Expect slower germination; delay sowing until the soil warms "
                f"or use a raised-bed nursery.",
                plain=(
                    f"It is cool on your land ({temperature:.0f} °C). "
                    f"{crop.capitalize()} likes {low:.0f} to {high:.0f} °C. "
                    f"Seeds will come up slowly. Wait for warmer days to sow, "
                    f"or raise the seedlings on a raised bed first."
                ),
            )
        ]
    if temperature > high:
        return [
            AdvisoryItem(
                "Thermal Regime",
                WARNING,
                f"At {temperature:.1f} °C the plot runs above the "
                f"{low:.1f}–{high:.1f} °C band for {crop}. Heat stress at "
                f"flowering causes pollen sterility — shift sowing to the "
                f"cooler part of the season and mulch to moderate soil "
                f"temperature.",
                plain=(
                    f"It is hot on your land ({temperature:.0f} °C). "
                    f"{crop.capitalize()} likes {low:.0f} to {high:.0f} °C. "
                    f"Too much heat at flowering time means empty grain. Sow "
                    f"in the cooler part of the season, and spread straw on "
                    f"the soil to keep it cool."
                ),
            )
        ]
    return [
        AdvisoryItem(
            "Thermal Regime",
            INFO,
            f"Temperature of {temperature:.1f} °C falls inside the "
            f"{low:.1f}–{high:.1f} °C band for {crop}.",
            plain=(
                f"The temperature on your land ({temperature:.0f} °C) is "
                f"good for {crop}."
            ),
        )
    ]


_SEVERITY_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}


def generate_advisory(
    crop: str,
    features: Dict[str, float],
    *,
    confidence: Optional[float] = None,
    ood_features: Optional[Sequence[str]] = None,
) -> AdvisoryReport:
    """Build the complete advisory package for a recommendation.

    Parameters
    ----------
    crop:
        The recommended crop.
    features:
        Observed raw values keyed by :data:`~src.core.config.FEATURE_NAMES`.
    confidence:
        Posterior for the recommendation as a percentage; a low value adds a
        corroboration caveat.
    ood_features:
        Features flagged as out-of-distribution, which add an extrapolation
        caveat.

    Returns
    -------
    AdvisoryReport
        Items ordered critical-first, plus nutrient gaps and a fertiliser plan.
    """
    profile = get_crop_profile(crop)
    items: List[AdvisoryItem] = []

    items += _hydrology_items(
        crop,
        float(features.get("rainfall", 0.0)),
        float(features.get("humidity", 0.0)),
    )
    nutrition, gaps, plan = _nutrition_items(crop, features, profile)
    items += nutrition
    items += _ph_items(crop, float(features.get("ph", 7.0)), profile)
    items += _thermal_items(crop, float(features.get("temperature", 25.0)), profile)

    if confidence is not None and confidence < 50.0:
        items.append(
            AdvisoryItem(
                "Model Certainty",
                WARNING,
                f"The ensemble assigns {crop} only {confidence:.1f}% posterior "
                f"probability — the reading falls between crop envelopes. "
                f"Review the runner-up options and corroborate with a "
                f"laboratory soil test before committing the season.",
                plain=(
                    f"We are not very sure about {crop}. Your land sits "
                    f"between what several crops like. Please look at the "
                    f"other crops on the list, and get your soil tested at a "
                    f"government lab before you decide."
                ),
            )
        )
    if ood_features:
        names = ", ".join(ood_features)
        items.append(
            AdvisoryItem(
                "Model Certainty",
                WARNING,
                f"Input parameters {names} lie beyond three standard "
                f"deviations of the training distribution. The recommendation "
                f"is an extrapolation rather than an interpolation and carries "
                f"correspondingly wider uncertainty.",
                plain=(
                    f"Your readings for {names} are very unusual — far from "
                    f"any farm we learned from. The advice may not fit your "
                    f"land well. Please check the numbers, and get a soil "
                    f"test if you can."
                ),
            )
        )

    items.sort(key=lambda item: _SEVERITY_ORDER.get(item.severity, 3))
    return AdvisoryReport(
        crop=crop, items=items, nutrient_gaps=gaps, fertiliser_plan=plan
    )


__all__ = [
    "CRITICAL",
    "WARNING",
    "INFO",
    "AdvisoryItem",
    "AdvisoryReport",
    "generate_advisory",
    "get_crop_profile",
    "get_crop_profiles",
]
