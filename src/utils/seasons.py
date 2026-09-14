"""Cropping-season suitability for the Indian agricultural calendar.

The model scores a crop on soil chemistry and microclimate. It has no notion of
*when* the crop is sown, and that omission is the single most common way a
technically-correct recommendation becomes useless advice: chickpea suits a
plot perfectly and is still impossible if the farmer is standing in a field in
June, because chickpea is a rabi crop.

This module supplies that missing axis. It does not override the model — a
season mismatch is surfaced as a warning beside the recommendation, and the
ranked alternatives are annotated so the farmer can drop to the next crop that
*is* sowable now.

.. note::

   Season windows are indicative. They follow the standard Indian cropping
   calendar and are broadly right for peninsular India, but sowing dates shift
   with latitude, irrigation and local practice. They are advisory, and the UI
   says so. Treat a mismatch as "check with your extension officer", not as a
   prohibition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

KHARIF = "kharif"
RABI = "rabi"
SUMMER = "summer"
PERENNIAL = "perennial"

#: Display names and sowing windows, in the order a selector should show them.
SEASONS: Dict[str, Tuple[str, str]] = {
    KHARIF: ("Kharif (monsoon)", "Sown June–July, harvested September–October"),
    RABI: ("Rabi (winter)", "Sown October–December, harvested March–April"),
    SUMMER: ("Summer (zaid)", "Sown February–March, harvested May–June"),
    PERENNIAL: ("Perennial / plantation", "Planted once, harvested for years"),
}

#: Months (1–12) in which each season's sowing window falls, used to suggest a
#: sensible default from the current date.
_SOWING_MONTHS: Dict[str, FrozenSet[int]] = {
    KHARIF: frozenset({6, 7, 8}),
    RABI: frozenset({10, 11, 12}),
    SUMMER: frozenset({2, 3, 4}),
}

#: ``crop -> seasons in which it is normally sown``.
#:
#: Perennial and plantation crops are planted once and cropped for years, so
#: they carry :data:`PERENNIAL` and are never flagged as out of season.
CROP_SEASONS: Dict[str, FrozenSet[str]] = {
    # Kharif cereals, fibres and pulses
    "rice": frozenset({KHARIF, SUMMER}),
    "maize": frozenset({KHARIF, RABI, SUMMER}),
    "cotton": frozenset({KHARIF}),
    "jute": frozenset({KHARIF}),
    "pigeonpeas": frozenset({KHARIF}),
    "mothbeans": frozenset({KHARIF}),
    "mungbean": frozenset({KHARIF, SUMMER}),
    "blackgram": frozenset({KHARIF, RABI}),
    # Rabi pulses
    "chickpea": frozenset({RABI}),
    "lentil": frozenset({RABI}),
    "kidneybeans": frozenset({RABI}),
    # Summer cucurbits
    "watermelon": frozenset({SUMMER}),
    "muskmelon": frozenset({SUMMER}),
    # Perennial fruit and plantation
    "banana": frozenset({PERENNIAL}),
    "mango": frozenset({PERENNIAL}),
    "grapes": frozenset({PERENNIAL}),
    "apple": frozenset({PERENNIAL}),
    "orange": frozenset({PERENNIAL}),
    "papaya": frozenset({PERENNIAL}),
    "pomegranate": frozenset({PERENNIAL}),
    "coconut": frozenset({PERENNIAL}),
    "coffee": frozenset({PERENNIAL}),
}


@dataclass(frozen=True)
class SeasonFit:
    """Whether a crop can be sown in the selected season.

    Attributes
    ----------
    crop:
        The crop assessed.
    season:
        The season the farmer selected.
    suitable:
        ``True`` when the crop is normally sown then, or is perennial.
    is_perennial:
        Perennial crops are planted once, so the season question does not apply.
    sowable_in:
        The seasons this crop *is* normally sown in, for the advice message.
    """

    crop: str
    season: str
    suitable: bool
    is_perennial: bool
    sowable_in: Tuple[str, ...]

    @property
    def season_label(self) -> str:
        """Display name of the selected season."""
        return SEASONS.get(self.season, (self.season, ""))[0]

    def sowable_labels(self) -> str:
        """Human list of the seasons this crop suits."""
        names = [SEASONS.get(s, (s, ""))[0].split(" (")[0] for s in self.sowable_in]
        if not names:
            return "no recorded season"
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + " or " + names[-1]

    def sowable_labels_in(self, language: str = "en") -> str:
        """Seasons this crop suits, named in the given language."""
        from src.utils.i18n import season_name

        names = [
            season_name(s, SEASONS.get(s, (s, ""))[0].split(" (")[0], language)
            for s in self.sowable_in
        ]
        if not names:
            return "no recorded season"
        if len(names) == 1:
            return names[0]
        joiner = " ಅಥವಾ " if language == "kn" else " or "
        return ", ".join(names[:-1]) + joiner + names[-1]

    def message(self, simple: bool = True, language: str = "en") -> str:
        """Return the advice line for this fit.

        The Kannada versions are whole sentences rather than translated
        fragments assembled in English order -- Kannada puts the verb last,
        so building one from parts would read as nonsense.
        """
        if simple and language == "kn":
            from src.utils.agronomy import kannada_name
            from src.utils.i18n import season_message, season_name

            crop_kn = kannada_name(self.crop).split(" (")[0]
            kind = (
                "perennial" if self.is_perennial
                else "suitable" if self.suitable
                else "clash"
            )
            rendered = season_message(
                kind,
                language,
                crop=crop_kn,
                season=season_name(
                    self.season, self.season_label.split(" (")[0], language),
                seasons=self.sowable_labels_in(language),
            )
            if rendered:
                return rendered

        if self.is_perennial:
            return (
                f"{self.crop.capitalize()} is a long-term crop — you plant it "
                f"once and harvest for years, so it does not depend on the "
                f"season you sow."
                if simple
                else f"{self.crop.capitalize()} is a perennial; the sowing "
                f"season does not constrain establishment."
            )
        if self.suitable:
            return (
                f"{self.crop.capitalize()} is normally sown in "
                f"{self.sowable_labels()}, so this fits your season."
                if simple
                else f"{self.crop.capitalize()} is sown in "
                f"{self.sowable_labels()}; the selected season is compatible."
            )
        return (
            f"{self.crop.capitalize()} is not usually sown in "
            f"{self.season_label.split(' (')[0]} — it is a "
            f"{self.sowable_labels()} crop. Your soil suits it, but the timing "
            f"does not. Either wait for the right season, or pick the next "
            f"crop on the list that you can sow now."
            if simple
            else f"Season mismatch: {self.crop} is sown in "
            f"{self.sowable_labels()}, not {self.season_label}. The edaphic "
            f"match holds; the calendar does not."
        )


def seasons_for(crop: str) -> FrozenSet[str]:
    """Return the seasons a crop is normally sown in (empty if unknown)."""
    return CROP_SEASONS.get(str(crop).lower(), frozenset())


def is_perennial(crop: str) -> bool:
    """``True`` when the crop is planted once and cropped for years."""
    return PERENNIAL in seasons_for(crop)


def assess(crop: str, season: str) -> SeasonFit:
    """Assess whether ``crop`` can be sown in ``season``.

    An unknown crop is treated as suitable rather than flagged: the season
    table is advisory, and a missing entry is our gap, not the farmer's error.
    """
    windows = seasons_for(crop)
    perennial = PERENNIAL in windows
    if not windows:
        suitable = True
    elif perennial:
        suitable = True
    else:
        suitable = season in windows
    return SeasonFit(
        crop=str(crop),
        season=season,
        suitable=suitable,
        is_perennial=perennial,
        sowable_in=tuple(sorted(w for w in windows if w != PERENNIAL)) or (PERENNIAL,),
    )


def canonical(value: object, *, fallback: str = KHARIF) -> str:
    """Map anything that names a season back onto its key.

    Streamlit stores a ``selectbox``'s value as the string its ``format_func``
    produced, and restores it by looking that string back up among the
    formatted options. Change the interface language and the lookup misses --
    the stored label was formatted under the old language -- at which point
    Streamlit hands back the raw label instead of the option, and
    ``st.session_state["season"]`` holds ``"Rabi (winter)"`` or ``ಹಿಂಗಾರು``
    where every reader downstream expects ``"rabi"``.

    So a label is accepted here as well as a key: the English name with or
    without its parenthetical gloss, the Kannada name, or the key in any
    case. Anything unrecognised falls back rather than raising, because the
    caller is usually about to index :data:`SEASONS` with the result and a
    ``KeyError`` there takes the whole page down.
    """
    from src.utils.i18n import KANNADA, season_name

    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered in SEASONS:
            return lowered
        for key, (english, _) in SEASONS.items():
            if text in (english, english.split(" (")[0],
                        season_name(key, english, KANNADA)):
                return key
    return fallback


def default_season(month: int) -> str:
    """Suggest the season whose sowing window contains ``month``.

    Falls back to the nearest upcoming window so the selector always opens on
    something sensible rather than an arbitrary first entry.
    """
    for season, months in _SOWING_MONTHS.items():
        if month in months:
            return season
    # Between windows: pick the next one that opens.
    if month in (1,):
        return SUMMER
    if month in (5,):
        return KHARIF
    return RABI  # September, and anything unmatched


def filter_sowable(crops: List[str], season: str) -> List[str]:
    """Return only those crops that can be sown in ``season``."""
    return [crop for crop in crops if assess(crop, season).suitable]


def coverage() -> int:
    """Number of crops with a recorded season, for the tests."""
    return len(CROP_SEASONS)


__all__ = [
    "KHARIF", "RABI", "SUMMER", "PERENNIAL",
    "SEASONS", "CROP_SEASONS", "SeasonFit",
    "assess", "seasons_for", "is_perennial",
    "canonical", "default_season", "filter_sowable", "coverage",
]
