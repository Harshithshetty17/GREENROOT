"""Localised agronomic and commercial intelligence for the 22 crop classes.

This module turns a model output into the three things a farmer actually acts
on: what the crop is called in their own language, what it is likely to earn,
and how many sacks of fertiliser to carry home from the dealer.

On the money figures
--------------------
Yield, mandi price and cost of cultivation are **indicative planning
benchmarks**, not live market data. They are order-of-magnitude figures for
Karnataka compiled from published extension-service ranges, and every one of
them is wrong for somebody: mandi rates move weekly, cost of cultivation
depends on whether labour is hired or family, and yield depends on the season
you actually get.

They are therefore:

* stamped with :data:`BENCHMARK_BASIS`, so a reader knows how stale they are;
* overridable per crop through :func:`with_overrides`, which is what the
  dashboard wires its price box to;
* always presented as an estimate, never as a quotation.

:mod:`src.utils.economics` remains the path for a farmer who knows their own
numbers -- it ships no defaults at all and computes purely from what they type.
This module is the *starting point* that gives them something to correct.

On the Kannada names
--------------------
Each crop carries the Kannada spelling and an ISO 15919 style romanisation, in
the form ``ಭತ್ತ (Bhatta)``. These are the common agricultural names used in
Karnataka rather than literal dictionary translations -- ``bengalgram`` for
chickpea, for instance, is what a Karnataka mandi board actually says. They
have been cross-checked against the crop-name strings in the NFSM soil survey
(``data/Cleaned_NFSM_Dataset.csv``), which carries bilingual labels of the form
``Bengalgram/ಕಡಲೆಕಾಳು``. Sixteen of the twenty-two are corroborated that way
and are listed in :data:`SURVEY_CORROBORATED`; the other six -- apple,
coconut, kidneybeans, lentil, maize and mothbeans -- are not attested in the
survey and still want a native speaker's eye before field use. The dashboard
marks the difference rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence

# --------------------------------------------------------------------------- #
# Commercial fertiliser grades
# --------------------------------------------------------------------------- #

#: A standard Indian fertiliser sack.
BAG_KG: float = 50.0

#: One quintal, the unit crops are priced in at a mandi.
QUINTAL_KG: float = 100.0

#: Urea: 46% N, the cheapest straight nitrogen source.
UREA_N: float = 0.46

#: DAP, 18:46:0 -- 18% N and 46% P2O5. Supplying phosphorus through DAP also
#: delivers nitrogen, which must be credited against the urea dose.
DAP_N: float = 0.18
DAP_P2O5: float = 0.46

#: Muriate of potash: 60% K2O.
MOP_K2O: float = 0.60

#: What the benchmark economics are anchored to. Shown wherever money is.
BENCHMARK_BASIS: str = "indicative Karnataka ranges, 2024-25 basis"

#: Crops whose Kannada name appears in the bilingual crop labels of the NFSM
#: soil survey shipped in ``data/``. Derived, not asserted:
#: ``tests/test_agronomy.py`` re-computes this set from the CSV and fails if
#: it drifts.
#:
#: Maize is deliberately absent. The survey labels it ``Maize/ಜೋಳ``, but it
#: also carries ``Jowar/ಜೋಳ`` -- ಜೋಳ is jowar, and the survey's own
#: ``Maize(fodder)/ಮುಸುಕಿನಜೋಳ`` shows it knows the difference. That entry is a
#: data-entry conflation, so it corroborates nothing.
SURVEY_CORROBORATED: FrozenSet[str] = frozenset(
    {
        "banana", "blackgram", "chickpea", "coffee", "cotton", "grapes",
        "jute", "mango", "mungbean", "muskmelon", "orange", "papaya",
        "pigeonpeas", "pomegranate", "rice", "watermelon",
    }
)


@dataclass(frozen=True)
class CropProfile:
    """Everything the farmer-facing views need about one crop class.

    Attributes
    ----------
    crop:
        The model's class label, lower case.
    english:
        Display name.
    kannada:
        Kannada spelling with a romanisation, e.g. ``ಭತ್ತ (Bhatta)``.
    yield_quintal_per_acre:
        Benchmark marketable yield.
    price_per_quintal:
        Benchmark APMC mandi rate in rupees.
    cost_per_acre:
        Benchmark cost of cultivation in rupees, excluding land rent.
    duration_days:
        Sowing to harvest. Perennials report a bearing-cycle year.
    """

    crop: str
    english: str
    kannada: str
    yield_quintal_per_acre: float
    price_per_quintal: float
    cost_per_acre: float
    duration_days: int

    @property
    def kannada_script(self) -> str:
        """Just the Kannada, without the romanisation in brackets."""
        return self.kannada.split(" (")[0]

    @property
    def bilingual(self) -> str:
        """``Rice · ಭತ್ತ (Bhatta)`` -- the form the crop card uses."""
        return f"{self.english} · {self.kannada}"

    @property
    def gross_per_acre(self) -> float:
        """Expected gross income per acre, in rupees."""
        return self.yield_quintal_per_acre * self.price_per_quintal

    @property
    def net_per_acre(self) -> float:
        """Expected net margin per acre, in rupees. May be negative."""
        return self.gross_per_acre - self.cost_per_acre

    @property
    def name_is_corroborated(self) -> bool:
        """Whether the Kannada name is backed by the shipped survey labels."""
        return self.crop in SURVEY_CORROBORATED

    def economics(self, acres: float) -> "CropEconomics":
        """Scale the benchmarks to a holding."""
        return CropEconomics(profile=self, acres=max(float(acres), 0.0))


@dataclass(frozen=True)
class CropEconomics:
    """Benchmark economics scaled to a given acreage."""

    profile: CropProfile
    acres: float

    @property
    def yield_quintals(self) -> float:
        return self.profile.yield_quintal_per_acre * self.acres

    @property
    def gross(self) -> float:
        return self.profile.gross_per_acre * self.acres

    @property
    def cost(self) -> float:
        return self.profile.cost_per_acre * self.acres

    @property
    def net(self) -> float:
        return self.gross - self.cost

    @property
    def is_loss(self) -> bool:
        """A benchmark that does not clear its own cost is worth saying aloud."""
        return self.net < 0


# --------------------------------------------------------------------------- #
# The 22 crop classes
# --------------------------------------------------------------------------- #

def _p(
    crop: str, english: str, kannada: str,
    yield_q: float, price: float, cost: float, days: int,
) -> CropProfile:
    return CropProfile(crop, english, kannada, yield_q, price, cost, days)


#: Keyed by the model's own class label so lookups cannot drift from the
#: classifier. Perennials report one bearing year as their duration.
CROPS: Dict[str, CropProfile] = {
    p.crop: p
    for p in (
        _p("rice",        "Rice",         "ಭತ್ತ (Bhatta)",              22, 2300,  28000, 135),
        _p("maize",       "Maize",        "ಮೆಕ್ಕೆಜೋಳ (Mekkejola)",      25, 2100,  22000, 110),
        _p("chickpea",    "Chickpea",     "ಕಡಲೆ (Kadale)",              7, 5400,  18000, 105),
        _p("kidneybeans", "Kidney Beans", "ರಾಜ್ಮಾ (Rajma)",             6, 7000,  24000, 120),
        _p("pigeonpeas",  "Pigeon Peas",  "ತೊಗರಿ (Togari)",             6, 7000,  20000, 165),
        _p("mothbeans",   "Moth Beans",   "ಮಠ (Matha)",                 4, 6000,  14000,  80),
        _p("mungbean",    "Mung Bean",    "ಹೆಸರು (Hesaru)",             4, 8000,  16000,  70),
        _p("blackgram",   "Black Gram",   "ಉದ್ದು (Uddu)",               4, 7000,  16000,  85),
        _p("lentil",      "Lentil",       "ಮಸೂರ (Masura)",              5, 6200,  17000, 110),
        _p("pomegranate", "Pomegranate",  "ದಾಳಿಂಬೆ (Dalimbe)",         45, 7000, 120000, 365),
        _p("banana",      "Banana",       "ಬಾಳೆ (Bale)",              220, 1600, 130000, 330),
        _p("mango",       "Mango",        "ಮಾವು (Mavu)",               35, 4500,  70000, 365),
        _p("grapes",      "Grapes",       "ದ್ರಾಕ್ಷಿ (Drakshi)",       100, 5500, 220000, 365),
        _p("watermelon",  "Watermelon",   "ಕಲ್ಲಂಗಡಿ (Kallangadi)",    120,  900,  45000,  85),
        _p("muskmelon",   "Muskmelon",    "ಕರ್ಬೂಜ (Karbuja)",          90, 1400,  42000,  80),
        _p("apple",       "Apple",        "ಸೇಬು (Sebu)",               60, 6000, 150000, 365),
        _p("orange",      "Orange",       "ಕಿತ್ತಳೆ (Kittale)",         70, 3000,  90000, 365),
        _p("papaya",      "Papaya",       "ಪಪ್ಪಾಯ (Pappaya)",         350, 1200,  90000, 300),
        _p("coconut",     "Coconut",      "ತೆಂಗು (Tengu)",             60, 3500,  55000, 365),
        _p("cotton",      "Cotton",       "ಹತ್ತಿ (Hatti)",              9, 7200,  38000, 165),
        _p("jute",        "Jute",         "ಸೆಣಬು (Senabu)",            12, 5000,  30000, 120),
        _p("coffee",      "Coffee",       "ಕಾಫಿ (Kaphi)",              10, 9000, 110000, 365),
    )
}


def profile(crop: str) -> Optional[CropProfile]:
    """Look up a crop profile by the model's class label. Case-insensitive."""
    return CROPS.get(str(crop).strip().lower())


def kannada_name(crop: str) -> str:
    """Kannada name, falling back to the English label for an unknown crop."""
    found = profile(crop)
    return found.kannada if found else str(crop).title()


def bilingual_name(crop: str) -> str:
    """``Rice · ಭತ್ತ (Bhatta)``, or a title-cased fallback."""
    found = profile(crop)
    return found.bilingual if found else str(crop).title()


def with_overrides(
    crop: str,
    *,
    price_per_quintal: Optional[float] = None,
    yield_quintal_per_acre: Optional[float] = None,
    cost_per_acre: Optional[float] = None,
) -> Optional[CropProfile]:
    """A profile with the farmer's own figures substituted in.

    Any argument left as ``None`` keeps the benchmark. This is the seam the
    dashboard uses: the benchmark gives them a number to react to, and what
    they type replaces it.
    """
    base = profile(crop)
    if base is None:
        return None
    changes = {
        k: float(v)
        for k, v in (
            ("price_per_quintal", price_per_quintal),
            ("yield_quintal_per_acre", yield_quintal_per_acre),
            ("cost_per_acre", cost_per_acre),
        )
        if v is not None
    }
    return replace(base, **changes) if changes else base


def coverage() -> int:
    """How many crop classes carry a profile. Pinned by the test suite."""
    return len(CROPS)


# --------------------------------------------------------------------------- #
# Commercial fertiliser bag plan
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BagLine:
    """One product, as kilograms and as sacks off the dealer's shelf."""

    product: str
    grade: str
    kg: float

    @property
    def bags(self) -> float:
        """Sacks, to one decimal -- what the dose actually works out to."""
        return round(self.kg / BAG_KG, 1)

    @property
    def whole_bags(self) -> int:
        """Sacks to carry home. Rounded up: a part-sack cannot be bought.

        A dose is rounded up rather than to nearest because under-applying a
        deficient nutrient wastes the whole intervention, while the surplus
        from rounding up is a few kilograms across an acre.
        """
        if self.kg <= 0:
            return 0
        whole = int(self.kg // BAG_KG)
        return whole + (1 if self.kg % BAG_KG > 1e-9 else 0)

    def say(self) -> str:
        """One line a farmer can read at the counter."""
        if self.whole_bags == 0:
            return f"{self.product} — none needed"
        sacks = "bag" if self.whole_bags == 1 else "bags"
        return (
            f"{self.product} ({self.grade}) — {self.whole_bags} {sacks} "
            f"of {BAG_KG:.0f} kg · {self.kg:.0f} kg"
        )


@dataclass(frozen=True)
class BagPlan:
    """The three-sack plan for a holding."""

    urea: BagLine
    dap: BagLine
    mop: BagLine
    acres: float
    #: Nitrogen delivered by the DAP and credited against the urea dose.
    nitrogen_from_dap_kg: float

    @property
    def lines(self) -> List[BagLine]:
        return [self.urea, self.dap, self.mop]

    @property
    def total_bags(self) -> int:
        return sum(line.whole_bags for line in self.lines)

    @property
    def is_empty(self) -> bool:
        """No product needed at all -- the soil already has enough."""
        return self.total_bags == 0

    def as_rows(self) -> List[Dict[str, object]]:
        """Table rows for the dashboard."""
        return [
            {
                "Fertiliser": line.product,
                "Grade": line.grade,
                "Bags (50 kg)": line.whole_bags,
                "Exact quantity": f"{line.kg:.0f} kg",
            }
            for line in self.lines
        ]


def bag_plan(
    *,
    n_kg_per_hectare: float,
    p_kg_per_hectare: float,
    k_kg_per_hectare: float,
    acres: float,
    acres_per_hectare: float = 2.4710538,
) -> BagPlan:
    """Convert an N-P-K deficit into sacks of Urea, DAP and MOP.

    The order matters and is the standard agronomic sequence:

    1. Meet the phosphorus deficit with DAP, at 46% P2O5.
    2. DAP is 18% nitrogen, so credit what it delivers against the N target.
    3. Top up whatever nitrogen remains with urea, at 46% N.
    4. Meet the potassium deficit with MOP, at 60% K2O.

    Sizing urea from the raw N target without crediting the DAP's nitrogen is
    the common mistake, and over-applies nitrogen on every plan that needs
    both -- which is most of them.

    Parameters
    ----------
    n_kg_per_hectare, p_kg_per_hectare, k_kg_per_hectare:
        The prescribed deficit. Phosphorus and potassium are taken as P2O5 and
        K2O, consistent with how the advisory prescribes them and with how
        Indian fertiliser grades are stated.
    acres:
        The holding. The prescription is per hectare; the farmer buys per acre.
    """
    factor = max(float(acres), 0.0) / acres_per_hectare

    p2o5 = max(float(p_kg_per_hectare), 0.0) * factor
    dap_kg = p2o5 / DAP_P2O5 if p2o5 > 0 else 0.0

    n_target = max(float(n_kg_per_hectare), 0.0) * factor
    n_from_dap = dap_kg * DAP_N
    urea_kg = max(n_target - n_from_dap, 0.0) / UREA_N

    k2o = max(float(k_kg_per_hectare), 0.0) * factor
    mop_kg = k2o / MOP_K2O if k2o > 0 else 0.0

    return BagPlan(
        urea=BagLine("Urea", "46-0-0", round(urea_kg, 1)),
        dap=BagLine("DAP", "18-46-0", round(dap_kg, 1)),
        mop=BagLine("Potash (MOP)", "0-0-60", round(mop_kg, 1)),
        acres=max(float(acres), 0.0),
        nitrogen_from_dap_kg=round(n_from_dap, 1),
    )


# --------------------------------------------------------------------------- #
# Spray advisory
# --------------------------------------------------------------------------- #

#: Above this, rain is likely to wash the spray off before it is absorbed.
RUNOFF_RAINFALL_MM: float = 12.0

#: Above this, foliage stays wet long enough for fungal infection to establish.
FUNGAL_HUMIDITY_PCT: float = 85.0

#: Spraying in this heat evaporates the carrier and can scorch the leaf.
SCORCH_TEMPERATURE_C: float = 36.0

CLEAR = "clear"
CAUTION = "caution"
HOLD = "hold"


@dataclass(frozen=True)
class SprayAdvice:
    """Whether today is a day to spray.

    Attributes
    ----------
    status:
        One of :data:`CLEAR`, :data:`CAUTION`, :data:`HOLD`.
    headline:
        Short verdict, e.g. ``Safe to spray``.
    detail:
        Why, in plain words, including what to do instead.
    is_live:
        Whether the weather behind this was a live reading. A verdict from
        fallback weather is a guess about the sky and must say so.
    """

    status: str
    headline: str
    detail: str
    is_live: bool = True

    @property
    def icon(self) -> str:
        return {CLEAR: "✅", CAUTION: "⚠️", HOLD: "🚫"}.get(self.status, "•")

    @property
    def should_spray(self) -> bool:
        return self.status == CLEAR


def spray_advice(
    *,
    rainfall_mm: float,
    humidity_pct: float,
    temperature_c: float,
    is_live: bool = True,
) -> SprayAdvice:
    """Clear, caution or hold, from the current weather.

    Ordered worst-first: runoff wastes the chemical outright, so it outranks
    the fungal warning, which outranks heat.
    """
    caveat = (
        ""
        if is_live
        else " This is based on typical weather, not a live reading — look at "
        "the sky before you decide."
    )

    if rainfall_mm >= RUNOFF_RAINFALL_MM:
        return SprayAdvice(
            HOLD,
            "Do not spray today — rain will wash it off",
            f"About {rainfall_mm:.0f} mm of rain is expected. Spray applied "
            f"now runs off before the plant takes it in, so the money is "
            f"wasted and it reaches the groundwater instead. Wait for a dry "
            f"spell of a few hours." + caveat,
            is_live,
        )

    if humidity_pct >= FUNGAL_HUMIDITY_PCT:
        return SprayAdvice(
            CAUTION,
            "Spray with care — the air is very damp",
            f"Humidity is around {humidity_pct:.0f}%. Leaves stay wet for a "
            f"long time in this air, which is how fungus takes hold. If you "
            f"are spraying for fungus this is the right day; if you are "
            f"spraying for insects, wait for drier air." + caveat,
            is_live,
        )

    if temperature_c >= SCORCH_TEMPERATURE_C:
        return SprayAdvice(
            CAUTION,
            "Spray early or late — it is too hot at midday",
            f"It is about {temperature_c:.0f}°C. Spray dries before the leaf "
            f"absorbs it and can burn the crop. Spray before 10 in the "
            f"morning or after 4 in the evening." + caveat,
            is_live,
        )

    return SprayAdvice(
        CLEAR,
        "Safe to spray today",
        f"No heavy rain expected, humidity around {humidity_pct:.0f}% and "
        f"about {temperature_c:.0f}°C. Still avoid the midday sun and do not "
        f"spray into the wind." + caveat,
        is_live,
    )


# --------------------------------------------------------------------------- #
# Crop roadmap
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RoadmapStage:
    """One stage of the season, with the day it falls on."""

    name: str
    day: int
    action: str

    @property
    def when(self) -> str:
        if self.day == 0:
            return "Before sowing"
        if self.day >= 365:
            return "Through the year"
        return f"Around day {self.day}"


def roadmap(crop: str) -> List[RoadmapStage]:
    """Four stages from land preparation to harvest, scaled to the crop.

    The fractions are the conventional split of a season: basal at sowing,
    first top dressing at the vegetative stage, second at flowering, then
    harvest. Perennials are described as an annual cycle instead, because
    "day 274 of a coconut palm" is not a useful thing to tell anyone.
    """
    found = profile(crop)
    days = found.duration_days if found else 120

    if days >= 365:
        return [
            RoadmapStage("Prepare", 0,
                         "Clear the basin, apply compost and the basal dose."),
            RoadmapStage("Growing season", 90,
                         "First top dressing after the rains establish."),
            RoadmapStage("Bearing", 210,
                         "Second dressing before the fruiting flush; watch "
                         "for pests weekly."),
            RoadmapStage("Harvest cycle", 365,
                         "Pick in rounds as the fruit matures, not all at once."),
        ]

    return [
        RoadmapStage("Prepare and sow", 0,
                     "Plough, apply the full DAP and potash with one third of "
                     "the urea as the basal dose, then sow."),
        RoadmapStage("Early growth", round(days * 0.25),
                     "First top dressing: one third of the urea. Weed now, "
                     "before the crop closes over."),
        RoadmapStage("Flowering", round(days * 0.55),
                     "Last third of the urea. This is when water matters most "
                     "— do not let the crop dry out."),
        RoadmapStage("Harvest", days,
                     f"Ready around day {days}. Stop spraying at least two "
                     f"weeks before you cut."),
    ]


__all__ = [
    "BAG_KG", "QUINTAL_KG", "BENCHMARK_BASIS", "SURVEY_CORROBORATED",
    "UREA_N", "DAP_N", "DAP_P2O5", "MOP_K2O",
    "CropProfile", "CropEconomics", "CROPS",
    "profile", "kannada_name", "bilingual_name", "with_overrides", "coverage",
    "BagLine", "BagPlan", "bag_plan",
    "SprayAdvice", "spray_advice", "CLEAR", "CAUTION", "HOLD",
    "RUNOFF_RAINFALL_MM", "FUNGAL_HUMIDITY_PCT", "SCORCH_TEMPERATURE_C",
    "RoadmapStage", "roadmap",
]
