"""Plain-language layer: the same advice, in words a farmer can act on.

The system has two audiences with opposite needs. A cultivator needs "your soil
is too sour — add lime before sowing". An examiner needs "pH 5.0 lies below the
5.5 threshold at which phosphorus is sequestered into Fe/Al complexes".
Writing only the first loses the rigour; writing only the second loses the user.

So the dashboard carries **both registers** and switches between them. Simple is
the default, because the farmer is the primary user; Technical is one toggle
away, so the underlying precision is never actually removed.

Writing rules for the simple register
-------------------------------------
* Short sentences. One idea each.
* Everyday words: "plant food" not "macronutrient", "sour soil" not "acidic
  reaction", "how sure we are" not "posterior probability".
* Say what to *do*, not what is *true*: "add 2 bags of lime per acre" beats
  "the soil requires calcium amendment".
* Quantities in units a farmer buys and measures — bags and acres alongside
  kg/ha, since Karnataka smallholdings are measured in acres.
* Never imply more certainty than the model has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

SIMPLE = "simple"
TECHNICAL = "technical"

#: 1 hectare = 2.4710538 acres. Fertiliser is sold and applied by the acre here.
ACRES_PER_HECTARE: float = 2.4710538

#: A standard fertiliser sack in India.
BAG_KG: float = 50.0


# --------------------------------------------------------------------------- #
# Unit helpers
# --------------------------------------------------------------------------- #
def per_acre(per_hectare: float) -> float:
    """Convert a per-hectare quantity to per-acre."""
    return float(per_hectare) / ACRES_PER_HECTARE


def describe_quantity(kg_per_hectare: float) -> str:
    """Describe a fertiliser dose in bags and acres, not just kg/ha.

    A farmer buys 50 kg sacks and works an acre, so "about 1 bag per acre" is
    actionable in a way that "82.6 kg/ha" is not.
    """
    acre = per_acre(kg_per_hectare)
    bags = acre / BAG_KG
    if bags < 0.25:
        measure = "less than a quarter bag"
    elif bags < 0.75:
        measure = "about half a bag"
    elif bags < 1.4:
        measure = "about 1 bag"
    else:
        measure = f"about {bags:.1f} bags"
    return f"{acre:.0f} kg per acre ({measure} of 50 kg)"


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConfidenceBand:
    """A plain-language reading of a model confidence percentage."""

    label: str
    detail: str
    severity: str


def confidence_band(confidence: float) -> ConfidenceBand:
    """Translate a posterior percentage into words.

    Parameters
    ----------
    confidence:
        Model confidence in [0, 100].
    """
    if confidence >= 85:
        return ConfidenceBand(
            "Very good match",
            "Your soil and weather suit this crop well.",
            "good",
        )
    if confidence >= 60:
        return ConfidenceBand(
            "Good match",
            "This crop should do well. Look at the other options too.",
            "good",
        )
    if confidence >= 40:
        return ConfidenceBand(
            "Fair match",
            "This crop may work, but the next ones are close. "
            "Compare them before you decide.",
            "warning",
        )
    return ConfidenceBand(
        "Weak match",
        "No crop fits your readings clearly. Please get your soil tested "
        "at a government lab before sowing.",
        "critical",
    )


# --------------------------------------------------------------------------- #
# Interface copy
# --------------------------------------------------------------------------- #
#: ``key -> (simple, technical)``.
COPY: Dict[str, Tuple[str, str]] = {
    # Tabs
    "tab_recommend": ("🌱 Best Crop for You", "🎯 Precision Recommendation"),
    "tab_why": ("💡 Why This Crop?", "🔍 Explainable AI Consensus"),
    "tab_whatif": ("🔄 What If I Change Something?", "🧭 What-If Sensitivity"),
    "tab_bulk": ("📋 Many Farms at Once", "📦 Bulk Advisory"),
    "tab_records": ("🗂️ Past Records", "📊 Audit Trail & Governance"),
    "tab_card": ("🧾 Your Soil Card", "🧾 Farmer Soil Health Card"),

    # Hero
    "app_tagline": (
        "Tells you which crop suits your land, and why",
        "Stacking ensemble meta-learning · multi-explainer consensus auditing",
    ),

    # Sidebar
    "sidebar_place": ("📍 Your Place", "📍 Location & Climate"),
    "sidebar_soil": ("🧪 Your Soil Test", "🧪 Soil Chemistry"),
    "sidebar_weather": ("🌤️ Weather on Your Land", "🌡️ Microclimate"),
    "sidebar_season": ("📅 Sowing Season", "📅 Cropping Season"),
    "season_ok": ("Right season for this crop", "Season compatible"),
    "season_clash": ("Wrong season for this crop", "Season mismatch"),
    "worth_heading": (
        "Will this fertiliser be worth it?",
        "Intervention simulation and input costing",
    ),
    "worth_intro": (
        "We re-check your land as if you had already added the fertiliser "
        "above, so you can see what it buys before you spend anything.",
        "The prescription is converted back into nutrient additions, applied "
        "to the readings, and re-scored through the same inference path.",
    ),
    "price_heading": ("What do you pay for fertiliser?", "Local input prices"),
    "price_hint": (
        "Enter the price of one 50 kg bag at your dealer. We do not guess "
        "prices — they change too much between places.",
        "Per-bag prices are farmer-supplied; no defaults ship with the system.",
    ),
    "district": ("Your district or taluk", "District / Taluk"),
    "get_weather": ("🌦️ Get today's weather", "🌦️ Sync live weather"),
    "load_baseline": (
        "📥 Use typical soil for my area",
        "📥 Load district baseline",
    ),
    "run": ("🌱 Show me the best crop", "🚀 Generate recommendation"),

    # Field labels
    "field_N": ("Nitrogen (N) — for green leaves", "Nitrogen (N)"),
    "field_P": ("Phosphorus (P) — for strong roots", "Phosphorus (P)"),
    "field_K": ("Potassium (K) — for full grain", "Potassium (K)"),
    "field_ph": ("Soil pH — sour or salty", "Soil pH"),
    "field_temperature": ("Temperature", "Temperature"),
    "field_humidity": ("Moisture in the air", "Humidity"),
    "field_rainfall": ("Rain in a month", "Rainfall"),

    # Recommendation tab
    "primary_label": ("Best crop for your land", "Recommended primary crop"),
    "ranked_heading": ("Other crops that could work", "Ranked suitability"),
    "advisory_heading": ("What to do in your field", "Agronomic advisory"),
    "fertiliser_heading": ("Fertiliser to add", "Fertiliser prescription (per hectare)"),
    "save_heading": ("Save this advice", "Persist to audit ledger"),
    "save_button": ("💾 Save this recommendation", "💾 Commit recommendation"),
    "compare_heading": (
        "How your soil compares with what this crop likes",
        "Input deviation from the benchmark training distribution",
    ),
    "compare_caption": (
        "A bar to the right means you have more than usual. "
        "A bar to the left means you have less.",
        "Standardised deviation z = (x − μ) ⁄ σ from the benchmark mean "
        "encoded in scaler.pkl.",
    ),

    # Why-this-crop tab
    "why_heading": ("Why was this crop chosen?", "Multi-explainer consensus audit"),
    "why_intro": (
        "We check the answer twice, in two different ways. "
        "If both checks point to the same reasons, you can trust the advice "
        "more.",
        "Two methodologically independent explainers — exact Shapley "
        "attribution and a local linear surrogate — are run over the same "
        "instance and their top-k driver sets compared.",
    ),
    "agreement_label": ("How much the two checks agree", "Jaccard Agreement Index"),
    "check_one": ("Check 1", "TreeSHAP"),
    "check_two": ("Check 2", "LIME"),
    "agree_yes": ("Both checks agree", "High Fidelity"),
    "agree_no": ("The two checks do not agree", "Local Divergence"),

    # What-if tab
    "whatif_heading": (
        "What happens if one thing changes?",
        "Single-factor perturbation analysis",
    ),
    "whatif_intro": (
        "Pick one thing about your land and see how the best crop changes "
        "as that one thing goes up or down. Everything else stays the same.",
        "One feature is swept across a ±100% band around its current value "
        "while the other six are held constant, tracing the decision surface "
        "along that axis.",
    ),
    "whatif_feature": ("What do you want to change?", "Feature to perturb"),
    "switch_heading": (
        "Where the best crop changes",
        "Recommendation switch points",
    ),

    # Records tab
    "records_heading": ("Advice you have saved", "Recommendation audit ledger"),

    # Card tab
    "card_heading": ("Your soil card — ready to print", "Printable soil health card"),

    # Bulk tab
    "bulk_heading": ("Advice for many farms at once", "Bulk advisory from a soil survey"),
    "bulk_intro": (
        "Have soil results for many farmers? Upload the file and get advice "
        "for every one of them together.",
        "Upload a laboratory export and receive a recommendation for every "
        "sample, with lenient column resolution and per-row validation.",
    ),
    "needs_review": ("Needs a closer look", "Needs review"),

    # Warnings
    "unusual_input": (
        "Your readings are very different from the farms we learned from. "
        "Treat this advice carefully and get a soil test.",
        "Input lies beyond three standard deviations of the training "
        "distribution; the recommendation is an extrapolation.",
    ),
    "low_confidence": (
        "We are not very sure about this one. Look at the other crops, "
        "and get your soil tested if you can.",
        "Posterior below the 50% advisory threshold; the reading falls "
        "between crop envelopes.",
    ),
    "disclaimer": (
        "This is advice to help you decide. It is not a promise. "
        "Please check with your local agriculture officer before sowing.",
        "Advisory output only. Corroborate with a certified laboratory soil "
        "test before committing the season.",
    ),
}

#: Advisory category names, ``technical -> simple``.
CATEGORY_NAMES: Dict[str, str] = {
    "Hydrology": "Water",
    "Nutrition": "Plant food",
    "Soil Reaction": "Sour or salty soil",
    "Thermal Regime": "Heat",
    "Model Certainty": "How sure we are",
}

#: Severity names, ``severity -> simple label``.
SEVERITY_NAMES: Dict[str, str] = {
    "critical": "Must fix first",
    "warning": "Worth doing",
    "info": "All good",
}


def text(key: str, simple: bool = True) -> str:
    """Return interface copy in the requested register.

    Parameters
    ----------
    key:
        A key of :data:`COPY`.
    simple:
        ``True`` for the farmer-facing wording, ``False`` for the technical
        wording.

    Returns
    -------
    str
        The copy, or ``key`` itself if no entry exists — so a missing string
        is visible during development rather than silently blank.
    """
    entry = COPY.get(key)
    if entry is None:
        return key
    return entry[0] if simple else entry[1]


def category_name(category: str, simple: bool = True) -> str:
    """Return an advisory category name in the requested register."""
    return CATEGORY_NAMES.get(category, category) if simple else category


def severity_name(severity: str, simple: bool = True) -> str:
    """Return an advisory severity label in the requested register."""
    if simple:
        return SEVERITY_NAMES.get(severity, severity)
    return {"critical": "Critical", "warning": "Advisory", "info": "Nominal"}.get(
        severity, severity
    )


__all__ = [
    "SIMPLE", "TECHNICAL", "ACRES_PER_HECTARE", "BAG_KG",
    "ConfidenceBand", "confidence_band",
    "per_acre", "describe_quantity",
    "COPY", "CATEGORY_NAMES", "SEVERITY_NAMES",
    "text", "category_name", "severity_name",
]
