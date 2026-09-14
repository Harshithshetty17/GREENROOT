"""Kannada interface strings.

Until now only the crop names were localised: every label, heading and button
was English, in front of farmers for whom English is a second language at
best. This is the catalogue that fixes that.

Scope
-----
**Farmer-facing copy only.** Examiner / AI mode stays in English: its readers
are examiners and its vocabulary -- posterior probability, Jaccard index,
Z-score -- is English technical terminology that would be made worse, not
better, by translation. So a Kannada entry exists for the plain-language
register and nothing else, and :func:`translate` is a no-op in the technical
register by design.

Falling back rather than blanking
---------------------------------
Any key without a Kannada entry falls back to English. A partially translated
interface is useful; one with holes in it is not, and a missing string must
never render as an empty button.

**These translations have not been reviewed by a native speaker.**
:data:`REVIEW_STATUS` says so and the app repeats it in the language picker.
That is not boilerplate hedging: this is agronomic advice people act on, a
mistranslated fertiliser instruction has a cost in somebody's field, and the
author of this file does not speak Kannada. Treat every string here as a
draft that a Kannada-speaking agronomist should read before it reaches a real
farmer. The crop names in :mod:`src.utils.agronomy` are separately evidenced
-- sixteen of the twenty-two are corroborated against the shipped NFSM survey
-- and are the one part of the Kannada surface that rests on something firmer
than one author's judgement.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

#: Language codes.
ENGLISH: str = "en"
KANNADA: str = "kn"

#: What the picker shows. Each language names itself in its own script, which
#: is the only form somebody who cannot read the other one can recognise.
LANGUAGES: Dict[str, str] = {
    ENGLISH: "English",
    KANNADA: "ಕನ್ನಡ",
}

#: Shown wherever Kannada is offered. Deliberately unmissable.
REVIEW_STATUS: str = (
    "Kannada is a draft translation and has not been checked by a native "
    "speaker yet. If a word looks wrong, trust the English."
)

REVIEW_STATUS_KN: str = (
    "ಕನ್ನಡ ಅನುವಾದ ಇನ್ನೂ ಪರಿಶೀಲನೆ ಆಗಿಲ್ಲ. ಯಾವುದಾದರೂ ಪದ ತಪ್ಪು ಎನಿಸಿದರೆ "
    "ಇಂಗ್ಲಿಷ್ ಅನ್ನೇ ನಂಬಿ."
)

#: Kannada for the farmer-facing keys of
#: :data:`src.utils.plain_language.COPY`. Keys absent here fall back to
#: English; see the module docstring.
KANNADA_UI: Dict[str, str] = {
    # --- Navigation ------------------------------------------------------
    "tab_recommend": "🌱 ನಿಮ್ಮ ಬೆಳೆ",
    "tab_why": "💡 ಈ ಬೆಳೆ ಏಕೆ?",
    "tab_whatif": "🔄 ಏನಾದರೂ ಬದಲಾದರೆ?",
    "tab_bulk": "📋 ಹಲವು ಜಮೀನುಗಳು",
    "tab_records": "🗂️ ಹಳೆಯ ಸಲಹೆಗಳು",
    "tab_card": "🧾 ನಿಮ್ಮ ಮಣ್ಣಿನ ಕಾರ್ಡ್",

    # --- Hero and the main action ---------------------------------------
    "app_tagline": "ನಿಮ್ಮ ಜಮೀನಿಗೆ ಯಾವ ಬೆಳೆ ಸರಿ, ಮತ್ತು ಏಕೆ ಎಂದು ಹೇಳುತ್ತದೆ",
    "run": "🌱 ಸರಿಯಾದ ಬೆಳೆ ತೋರಿಸಿ",
    "district": "ನಿಮ್ಮ ಜಿಲ್ಲೆ ಅಥವಾ ತಾಲೂಕು",
    "get_weather": "🌦️ ಇಂದಿನ ಹವಾಮಾನ ತನ್ನಿ",
    "load_baseline": "📥 ನನ್ನ ಪ್ರದೇಶದ ಸಾಮಾನ್ಯ ಮಣ್ಣು ಬಳಸಿ",

    # --- Section headings ------------------------------------------------
    "sidebar_place": "📍 ನಿಮ್ಮ ಸ್ಥಳ",
    "sidebar_soil": "🧪 ನಿಮ್ಮ ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ",
    "sidebar_weather": "🌤️ ನಿಮ್ಮ ಜಮೀನಿನ ಹವಾಮಾನ",
    "sidebar_season": "📅 ಬಿತ್ತನೆ ಕಾಲ",

    # --- The soil readings ----------------------------------------------
    "field_N": "ಸಾರಜನಕ (N) — ಹಸಿರು ಎಲೆಗಳಿಗೆ",
    "field_P": "ರಂಜಕ (P) — ಬಲವಾದ ಬೇರುಗಳಿಗೆ",
    "field_K": "ಪೊಟ್ಯಾಶ್ (K) — ತುಂಬಿದ ಕಾಳಿಗೆ",
    "field_ph": "ಮಣ್ಣಿನ pH — ಹುಳಿ ಅಥವಾ ಉಪ್ಪು",
    "field_temperature": "ಉಷ್ಣಾಂಶ",
    "field_humidity": "ಗಾಳಿಯಲ್ಲಿನ ತೇವ",
    "field_rainfall": "ತಿಂಗಳ ಮಳೆ",

    # --- The answer -------------------------------------------------------
    "primary_label": "ನಿಮ್ಮ ಜಮೀನಿಗೆ ಸರಿಯಾದ ಬೆಳೆ",
    "ranked_heading": "ಬೇರೆ ಸಾಧ್ಯವಿರುವ ಬೆಳೆಗಳು",
    "advisory_heading": "ನಿಮ್ಮ ಜಮೀನಿನಲ್ಲಿ ಏನು ಮಾಡಬೇಕು",
    "fertiliser_heading": "ಹಾಕಬೇಕಾದ ಗೊಬ್ಬರ",
    "save_heading": "ಈ ಸಲಹೆ ಉಳಿಸಿ",
    "save_button": "💾 ಈ ಸಲಹೆಯನ್ನು ಉಳಿಸಿ",
    "records_heading": "ನೀವು ಉಳಿಸಿದ ಸಲಹೆಗಳು",
    "card_heading": "ನಿಮ್ಮ ಮಣ್ಣಿನ ಕಾರ್ಡ್ — ಮುದ್ರಿಸಲು ಸಿದ್ಧ",

    # --- Season -----------------------------------------------------------
    "season_ok": "ಈ ಬೆಳೆಗೆ ಸರಿಯಾದ ಕಾಲ",
    "season_clash": "ಈ ಬೆಳೆಗೆ ಸರಿಯಾದ ಕಾಲ ಅಲ್ಲ",

    # --- Warnings ---------------------------------------------------------
    "low_confidence": (
        "ಇದರ ಬಗ್ಗೆ ನಮಗೆ ಹೆಚ್ಚು ಖಚಿತವಿಲ್ಲ. ಬೇರೆ ಬೆಳೆಗಳನ್ನೂ ನೋಡಿ, "
        "ಸಾಧ್ಯವಾದರೆ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ."
    ),
    "unusual_input": (
        "ನಿಮ್ಮ ಅಳತೆಗಳು ನಾವು ಕಲಿತ ಜಮೀನುಗಳಿಗಿಂತ ತುಂಬಾ ಬೇರೆಯಾಗಿವೆ. "
        "ಈ ಸಲಹೆಯನ್ನು ಎಚ್ಚರಿಕೆಯಿಂದ ನೋಡಿ."
    ),
    "needs_review": "ಹತ್ತಿರದಿಂದ ನೋಡಬೇಕು",

    # --- Explanation ------------------------------------------------------
    "why_heading": "ಈ ಬೆಳೆಯನ್ನು ಏಕೆ ಆರಿಸಲಾಯಿತು?",
    "why_intro": (
        "ನಾವು ಉತ್ತರವನ್ನು ಎರಡು ಬೇರೆ ರೀತಿಯಲ್ಲಿ ಪರಿಶೀಲಿಸುತ್ತೇವೆ. ಎರಡೂ ಒಂದೇ "
        "ಕಾರಣ ತೋರಿಸಿದರೆ ಸಲಹೆಯನ್ನು ಹೆಚ್ಚು ನಂಬಬಹುದು."
    ),
    "agreement_label": "ಎರಡು ಪರಿಶೀಲನೆಗಳು ಎಷ್ಟು ಹೊಂದುತ್ತವೆ",
    "check_one": "ಪರಿಶೀಲನೆ ೧",
    "check_two": "ಪರಿಶೀಲನೆ ೨",
    "agree_yes": "ಎರಡೂ ಪರಿಶೀಲನೆಗಳು ಹೊಂದುತ್ತವೆ",
    "agree_no": "ಎರಡು ಪರಿಶೀಲನೆಗಳು ಹೊಂದುವುದಿಲ್ಲ",

    # --- Money -------------------------------------------------------------
    "worth_heading": "ಈ ಗೊಬ್ಬರ ಹಾಕುವುದು ಲಾಭವೇ?",
    "price_heading": "ಗೊಬ್ಬರಕ್ಕೆ ನೀವು ಎಷ್ಟು ಕೊಡುತ್ತೀರಿ?",

    # --- The line that matters most ---------------------------------------
    "disclaimer": (
        "ಇದು ನಿರ್ಧಾರ ತೆಗೆದುಕೊಳ್ಳಲು ಸಹಾಯ ಮಾಡುವ ಸಲಹೆ ಮಾತ್ರ. ಇದು ಭರವಸೆ ಅಲ್ಲ. "
        "ಬಿತ್ತನೆಗೆ ಮೊದಲು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಕೇಳಿ."
    ),
}

#: Strings the dashboard builds outside the COPY catalogue. Same fallback
#: rule: absent means English.
KANNADA_EXTRA: Dict[str, str] = {
    "your_district": "ನಿಮ್ಮ ಜಿಲ್ಲೆ",
    "how_many_acres": "ಎಷ್ಟು ಎಕರೆ?",
    "when_sow": "ಯಾವಾಗ ಬಿತ್ತುತ್ತೀರಿ?",
    "typical_weather": "ಸಾಮಾನ್ಯ ಹವಾಮಾನ",
    "live_weather": "ಈಗಿನ ಹವಾಮಾನ",
    "use_live_weather": "ಈಗಿನ ಹವಾಮಾನ ಬಳಸಿ",
    "change_readings": "ನನ್ನ ಮಣ್ಣಿನ ಅಳತೆ ಬದಲಿಸಿ",
    "your_advice": "🌱 ನಿಮ್ಮ ಸಲಹೆ",
    "saved_and_card": "🗂️ ಉಳಿಸಿದವು ಮತ್ತು ಕಾರ್ಡ್",
    "what_earn": "ಇದರಿಂದ ಎಷ್ಟು ಸಿಗಬಹುದು",
    "expected_harvest": "ನಿರೀಕ್ಷಿತ ಇಳುವರಿ",
    "mandi_rate": "ಮಂಡಿ ದರ",
    "money_left": "ಉಳಿಯುವ ಹಣ",
    "what_to_buy": "ಅಂಗಡಿಯಿಂದ ಏನು ತರಬೇಕು",
    "can_i_spray": "ಇಂದು ಸಿಂಪಡಿಸಬಹುದೇ?",
    "season_plan": "ಈ ಕಾಲದ ನಿಮ್ಮ ಯೋಜನೆ",
    "language": "ಭಾಷೆ",
    "account": "ನಿಮ್ಮ ಖಾತೆ",
    "sign_in": "ಒಳಗೆ ಬನ್ನಿ",
    "log_out": "ಹೊರಗೆ ಹೋಗಿ",
    "bags": "ಚೀಲ",
    "acre": "ಎಕರೆ",
    "not_clear": "ಸ್ಪಷ್ಟ ಉತ್ತರ ಅಲ್ಲ — ಎಚ್ಚರಿಕೆಯಿಂದ ಓದಿ",
    "changed_something": (
        "ಏನಾದರೂ ಬದಲಾಯಿತೇ? ಮೇಲೆ ನಿಮ್ಮ ಅಳತೆ ಸರಿಪಡಿಸಿ, ನಂತರ ಮತ್ತೆ ಒತ್ತಿ."
    ),
    "whats_due": "ಮುಂದೆ ಏನು ಮಾಡಬೇಕು",
    "share_heading": "ಈ ಸಲಹೆಯನ್ನು ಹಂಚಿಕೊಳ್ಳಿ",
    "share_whatsapp": "📤 ವಾಟ್ಸಾಪ್‌ನಲ್ಲಿ ಕಳುಹಿಸಿ",
    "share_copy": "ಅಥವಾ ಪಠ್ಯವನ್ನು ನಕಲಿಸಿ",
    "reminder_estimate": (
        "ಈ ದಿನಾಂಕಗಳು ನೀವು ಸಲಹೆ ಉಳಿಸಿದ ದಿನದಿಂದ ಲೆಕ್ಕ ಹಾಕಿದವು, ನೀವು ನಿಜವಾಗಿ "
        "ಬಿತ್ತಿದ ದಿನದಿಂದ ಅಲ್ಲ — ಹಾಗಾಗಿ ಇವು ಸರಿಸುಮಾರು, ನಿಖರವಲ್ಲ."
    ),
    "these_are_readings": (
        "ಇವು ನಿಮ್ಮ ಜಮೀನಿನ ಅಳತೆಗಳು. ತಪ್ಪಿದ್ದರೆ ಮೇಲೆ ಜಿಲ್ಲೆ ಬದಲಿಸಿ, "
        "ನಂತರ ಹಸಿರು ಗುಂಡಿ ಒತ್ತಿ."
    ),
}


#: Confidence bands. These carry the actual judgement -- "Fair match, 43 out
#: of 100" is the sentence a farmer decides on -- so leaving them in English
#: on an otherwise Kannada screen would strand the most important words on
#: the page.
KANNADA_BANDS: Dict[str, Tuple[str, str]] = {
    "Very good match": (
        "ತುಂಬಾ ಒಳ್ಳೆಯ ಹೊಂದಾಣಿಕೆ",
        "ನಿಮ್ಮ ಮಣ್ಣು ಮತ್ತು ಹವಾಮಾನ ಈ ಬೆಳೆಗೆ ಚೆನ್ನಾಗಿ ಹೊಂದುತ್ತವೆ.",
    ),
    "Good match": (
        "ಒಳ್ಳೆಯ ಹೊಂದಾಣಿಕೆ",
        "ಈ ಬೆಳೆ ಚೆನ್ನಾಗಿ ಬರಬೇಕು. ಬೇರೆ ಆಯ್ಕೆಗಳನ್ನೂ ನೋಡಿ.",
    ),
    "Fair match": (
        "ಸಾಧಾರಣ ಹೊಂದಾಣಿಕೆ",
        "ಈ ಬೆಳೆ ಆಗಬಹುದು, ಆದರೆ ಮುಂದಿನವೂ ಹತ್ತಿರವಿವೆ. "
        "ನಿರ್ಧರಿಸುವ ಮೊದಲು ಹೋಲಿಸಿ ನೋಡಿ.",
    ),
    "Weak match": (
        "ದುರ್ಬಲ ಹೊಂದಾಣಿಕೆ",
        "ಯಾವ ಬೆಳೆಯೂ ನಿಮ್ಮ ಅಳತೆಗಳಿಗೆ ಸ್ಪಷ್ಟವಾಗಿ ಹೊಂದುವುದಿಲ್ಲ. "
        "ಬಿತ್ತನೆಗೆ ಮೊದಲು ಸರ್ಕಾರಿ ಪ್ರಯೋಗಾಲಯದಲ್ಲಿ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ.",
    ),
}


def band_words(label: str, detail: str, language: str = ENGLISH) -> Tuple[str, str]:
    """Kannada for a confidence band, or the English pair unchanged."""
    if normalise(language) != KANNADA:
        return label, detail
    found = KANNADA_BANDS.get(label)
    return found if found else (label, detail)


#: Season names. The English carries a parenthetical gloss ("Kharif
#: (monsoon)") which Kannada does not need -- ಮುಂಗಾರು *is* the monsoon season.
KANNADA_SEASONS: Dict[str, str] = {
    "kharif": "ಮುಂಗಾರು",
    "rabi": "ಹಿಂಗಾರು",
    "summer": "ಬೇಸಿಗೆ",
    "perennial": "ಬಹುವಾರ್ಷಿಕ",
}

#: Season-fit sentences. Templates rather than fixed strings, because the
#: crop and the season are interpolated -- and Kannada word order is not
#: English word order, so these are written as whole sentences rather than
#: assembled from translated fragments.
KANNADA_SEASON_MESSAGES: Dict[str, str] = {
    "perennial": (
        "{crop} ಒಂದು ದೀರ್ಘಕಾಲದ ಬೆಳೆ — ಒಮ್ಮೆ ನೆಟ್ಟರೆ ವರ್ಷಗಟ್ಟಲೆ ಫಸಲು "
        "ಕೊಡುತ್ತದೆ, ಹಾಗಾಗಿ ಬಿತ್ತನೆ ಕಾಲ ಮುಖ್ಯವಲ್ಲ."
    ),
    "suitable": (
        "{crop} ಸಾಮಾನ್ಯವಾಗಿ {seasons} ಕಾಲದಲ್ಲಿ ಬಿತ್ತುತ್ತಾರೆ, ಹಾಗಾಗಿ ನಿಮ್ಮ "
        "ಕಾಲಕ್ಕೆ ಇದು ಸರಿಹೊಂದುತ್ತದೆ."
    ),
    "clash": (
        "{crop} ಅನ್ನು {season} ಕಾಲದಲ್ಲಿ ಸಾಮಾನ್ಯವಾಗಿ ಬಿತ್ತುವುದಿಲ್ಲ — ಇದು "
        "{seasons} ಬೆಳೆ. ನಿಮ್ಮ ಮಣ್ಣು ಸರಿಯಿದೆ, ಆದರೆ ಸಮಯ ಸರಿಯಿಲ್ಲ. ಸರಿಯಾದ "
        "ಕಾಲಕ್ಕೆ ಕಾಯಿರಿ, ಅಥವಾ ಈಗ ಬಿತ್ತಬಹುದಾದ ಮುಂದಿನ ಬೆಳೆಯನ್ನು ಆರಿಸಿ."
    ),
}


def season_name(key: str, english: str, language: str = ENGLISH) -> str:
    """A season's name, without the English parenthetical gloss."""
    if normalise(language) != KANNADA:
        return english
    return KANNADA_SEASONS.get(str(key).lower(), english)


def season_message(
    kind: str, language: str = ENGLISH, **parts: str
) -> Optional[str]:
    """A season-fit sentence in Kannada, or ``None`` to use the English.

    ``None`` rather than a fallback string, so the caller keeps whatever it
    would have said -- these sentences interpolate a crop name and a season
    list that the caller already has in the right form.
    """
    if normalise(language) != KANNADA:
        return None
    template = KANNADA_SEASON_MESSAGES.get(kind)
    return template.format(**parts) if template else None


#: Field-calendar stage names. These key off the English name returned by
#: :func:`src.utils.agronomy.roadmap`, which is stable and unique across both
#: the annual and the perennial set.
KANNADA_STAGES: Dict[str, str] = {
    "Prepare and sow": "ಸಿದ್ಧತೆ ಮತ್ತು ಬಿತ್ತನೆ",
    "Early growth": "ಆರಂಭಿಕ ಬೆಳವಣಿಗೆ",
    "Flowering": "ಹೂ ಬಿಡುವ ಹಂತ",
    "Harvest": "ಕೊಯ್ಲು",
    "Prepare": "ಸಿದ್ಧತೆ",
    "Growing season": "ಬೆಳವಣಿಗೆಯ ಕಾಲ",
    "Bearing": "ಫಸಲು ಬಿಡುವ ಹಂತ",
    "Harvest cycle": "ಕೊಯ್ಲಿನ ಸುತ್ತು",
}

#: What to do at each stage. Keyed by the English stage name for the same
#: reason. ``{days}`` is the crop's duration, interpolated by the caller --
#: only the harvest line uses it.
#:
#: These are field instructions, so they are the strings in this module a
#: reviewer should read first; they are grouped separately in the review
#: sheet for exactly that reason.
KANNADA_STAGE_ACTIONS: Dict[str, str] = {
    "Prepare and sow": (
        "ಉಳುಮೆ ಮಾಡಿ. ಪೂರ್ತಿ ಡಿಎಪಿ ಮತ್ತು ಪೊಟ್ಯಾಷ್ ಜೊತೆಗೆ ಯೂರಿಯಾದ ಮೂರರಲ್ಲಿ "
        "ಒಂದು ಭಾಗವನ್ನು ಬುಡಗೊಬ್ಬರವಾಗಿ ಹಾಕಿ, ನಂತರ ಬಿತ್ತಿ."
    ),
    "Early growth": (
        "ಮೊದಲ ಮೇಲುಗೊಬ್ಬರ: ಯೂರಿಯಾದ ಮೂರರಲ್ಲಿ ಒಂದು ಭಾಗ. ಬೆಳೆ ಮುಚ್ಚಿಕೊಳ್ಳುವ "
        "ಮೊದಲೇ ಕಳೆ ತೆಗೆಯಿರಿ."
    ),
    "Flowering": (
        "ಯೂರಿಯಾದ ಕೊನೆಯ ಮೂರನೇ ಒಂದು ಭಾಗ. ಈ ಹಂತದಲ್ಲಿ ನೀರು ಅತಿ ಮುಖ್ಯ — "
        "ಬೆಳೆ ಒಣಗಲು ಬಿಡಬೇಡಿ."
    ),
    "Harvest": (
        "ಸುಮಾರು {days} ದಿನಕ್ಕೆ ಕೊಯ್ಲಿಗೆ ಸಿದ್ಧ. ಕೊಯ್ಲಿಗೆ ಕನಿಷ್ಠ ಎರಡು ವಾರ "
        "ಮೊದಲು ಸಿಂಪಡಣೆ ನಿಲ್ಲಿಸಿ."
    ),
    "Prepare": "ಪಾತಿ ಸ್ವಚ್ಛ ಮಾಡಿ, ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ಮತ್ತು ಬುಡಗೊಬ್ಬರ ಹಾಕಿ.",
    "Growing season": "ಮಳೆ ಸ್ಥಿರವಾದ ನಂತರ ಮೊದಲ ಮೇಲುಗೊಬ್ಬರ ಹಾಕಿ.",
    "Bearing": (
        "ಫಸಲು ಬರುವ ಮೊದಲು ಎರಡನೇ ಗೊಬ್ಬರ ಹಾಕಿ; ವಾರಕ್ಕೊಮ್ಮೆ ಕೀಟ ಬಾಧೆ ನೋಡಿ."
    ),
    "Harvest cycle": (
        "ಹಣ್ಣು ಪಕ್ವವಾದಂತೆ ಸುತ್ತುಸುತ್ತಾಗಿ ಕೊಯ್ಯಿರಿ, ಎಲ್ಲವನ್ನೂ ಒಟ್ಟಿಗೆ ಅಲ್ಲ."
    ),
}

#: When a reminder falls. Singular and plural are separate entries because
#: Kannada inflects the noun ("ದಿನದಲ್ಲಿ" / "ದಿನಗಳಲ್ಲಿ"), so an English-style
#: trailing "s" would be wrong in both directions.
KANNADA_WHEN: Dict[str, str] = {
    "today": "ಇಂದು",
    "future_one": "ಸುಮಾರು {days} ದಿನದಲ್ಲಿ",
    "future_many": "ಸುಮಾರು {days} ದಿನಗಳಲ್ಲಿ",
    "past_one": "ಸುಮಾರು {days} ದಿನದ ಹಿಂದೆ",
    "past_many": "ಸುಮಾರು {days} ದಿನಗಳ ಹಿಂದೆ",
}


#: Sentences that interpolate a value -- a district, a sample count -- and so
#: cannot be assembled from translated fragments without getting Kannada word
#: order wrong. Each is a whole sentence with named placeholders.
KANNADA_PHRASES: Dict[str, str] = {
    "soil_from_survey": (
        "{district} ಜಿಲ್ಲೆಯ {count} ಮಣ್ಣು ಪರೀಕ್ಷೆಗಳ ಸರಾಸರಿ ಅಳತೆ ಇದು."
    ),
    "soil_not_survey": (
        "{district} ಜಿಲ್ಲೆಯ ಸಾಮಾನ್ಯ ಮಣ್ಣು — ಇದು ಸಮೀಕ್ಷೆಯಿಂದ ಬಂದದ್ದಲ್ಲ. "
        "ನಿಮ್ಮ ಬಳಿ ಮಣ್ಣು ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಇದ್ದರೆ ಕೆಳಗೆ ಸರಿಪಡಿಸಿ."
    ),
    "weather_normals": "{district} ಜಿಲ್ಲೆಯ ಸಾಮಾನ್ಯ ಹವಾಮಾನ",
}


def phrase(key: str, english: str, language: str = ENGLISH, **parts: object) -> str:
    """A Kannada sentence with values filled in, or ``english`` unchanged."""
    if normalise(language) != KANNADA:
        return english
    template = KANNADA_PHRASES.get(key)
    return template.format(**parts) if template else english


def stage_words(
    name: str, action: str, language: str = ENGLISH, *, day: int = 0
) -> Tuple[str, str]:
    """Kannada for one roadmap stage, or the English pair unchanged.

    ``day`` is the crop's duration in days, used only by the harvest line.
    Either half falls back independently, so a stage with a translated name
    and no translated action still shows the name in Kannada.
    """
    if normalise(language) != KANNADA:
        return name, action
    template = KANNADA_STAGE_ACTIONS.get(name)
    return (
        KANNADA_STAGES.get(name, name),
        template.format(days=int(day)) if template else action,
    )


def when_words(days_away: int, english: str, language: str = ENGLISH) -> str:
    """``in about 5 days`` in Kannada, or ``english`` unchanged.

    Takes the already-formatted English rather than looking it up, so this
    module never holds a second copy of :mod:`src.utils.reminders`' wording.
    """
    if normalise(language) != KANNADA:
        return english
    if days_away == 0:
        return KANNADA_WHEN["today"]
    count = abs(int(days_away))
    key = "past" if days_away < 0 else "future"
    key += "_one" if count == 1 else "_many"
    return KANNADA_WHEN[key].format(days=count)


def available_languages() -> List[Tuple[str, str]]:
    """``[("en", "English"), ("kn", "ಕನ್ನಡ")]`` for a picker."""
    return list(LANGUAGES.items())


def is_supported(language: Optional[str]) -> bool:
    return language in LANGUAGES


def normalise(language: Optional[str]) -> str:
    """Any unknown or missing code resolves to English."""
    return language if language in LANGUAGES else ENGLISH


def translate(
    key: str, english: str, language: str = ENGLISH, *, simple: bool = True
) -> str:
    """Kannada for ``key`` if there is one, otherwise ``english``.

    Parameters
    ----------
    key:
        A key of :data:`KANNADA_UI` or :data:`KANNADA_EXTRA`.
    english:
        What to show when there is no translation. Passed in rather than
        looked up so this module never has to hold a second copy of the
        English catalogue and drift from it.
    language:
        Target language code.
    simple:
        Technical register is English-only by design; see the module
        docstring. ``False`` returns ``english`` unchanged.
    """
    if not simple or normalise(language) != KANNADA:
        return english
    return KANNADA_UI.get(key) or KANNADA_EXTRA.get(key) or english


def coverage() -> Dict[str, int]:
    """How much of the interface is translated. Reported honestly in the UI."""
    from src.utils.plain_language import COPY

    translated = sum(1 for key in COPY if key in KANNADA_UI)
    return {
        "copy_keys": len(COPY),
        "copy_translated": translated,
        "extra_translated": len(KANNADA_EXTRA),
    }


def review_note(language: str = ENGLISH) -> str:
    """The unreviewed-translation warning, in the language being offered."""
    return REVIEW_STATUS_KN if normalise(language) == KANNADA else REVIEW_STATUS


__all__ = [
    "ENGLISH", "KANNADA", "LANGUAGES", "REVIEW_STATUS", "REVIEW_STATUS_KN",
    "KANNADA_UI", "KANNADA_EXTRA",
    "available_languages", "is_supported", "normalise", "translate",
    "coverage", "review_note", "KANNADA_BANDS", "band_words",
    "KANNADA_SEASONS", "KANNADA_SEASON_MESSAGES",
    "season_name", "season_message",
    "KANNADA_STAGES", "KANNADA_STAGE_ACTIONS", "KANNADA_WHEN",
    "stage_words", "when_words",
    "KANNADA_PHRASES", "phrase",
]
