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
    "out_of_100": "೧೦೦ ರಲ್ಲಿ",
    "not_clear": "ಸ್ಪಷ್ಟ ಉತ್ತರ ಅಲ್ಲ — ಎಚ್ಚರಿಕೆಯಿಂದ ಓದಿ",
    "changed_something": (
        "ಏನಾದರೂ ಬದಲಾಯಿತೇ? ಮೇಲೆ ನಿಮ್ಮ ಅಳತೆ ಸರಿಪಡಿಸಿ, ನಂತರ ಮತ್ತೆ ಒತ್ತಿ."
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
]
