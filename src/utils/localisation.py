"""Kannada localisation for farmer-facing output.

The system's stated users are cultivators in Karnataka. An advisory card
printed only in English is of limited use to the person meant to act on it, so
the Soil Health Card is rendered bilingually: every crop name and field label
carries its Kannada equivalent alongside the English.

Scope and honesty
-----------------
Only the *farmer-facing* card is localised. The analyst-facing tabs —
explainability consensus, sensitivity sweeps, the audit ledger — remain in
English, because their audience is the extension officer and the evaluator,
and machine-translating technical statistics would add risk without adding
value.

.. warning::

   These translations are a best-effort mapping prepared for an academic
   prototype and have **not been reviewed by a native Kannada speaker or an
   agricultural extension authority**. The card is deliberately *bilingual*
   rather than Kannada-only, so the English remains authoritative and a
   mistranslation cannot silently change the advice. Native review is required
   before any field deployment. Each entry carries an ISO 15919 transliteration
   to make that review straightforward.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: ISO 639-1 codes for the supported languages.
ENGLISH = "en"
KANNADA = "kn"

SUPPORTED_LANGUAGES: Dict[str, str] = {
    ENGLISH: "English",
    KANNADA: "ಕನ್ನಡ (Kannada)",
}

#: ``crop -> (Kannada, ISO 15919 transliteration)`` for all 22 target classes.
#: The transliteration column exists so a reviewer can verify the script
#: without reading Kannada.
CROP_NAMES: Dict[str, Tuple[str, str]] = {
    "apple": ("ಸೇಬು", "sēbu"),
    "banana": ("ಬಾಳೆಹಣ್ಣು", "bāḷehaṇṇu"),
    "blackgram": ("ಉದ್ದಿನಬೇಳೆ", "uddinabēḷe"),
    "chickpea": ("ಕಡಲೆ", "kaḍale"),
    "coconut": ("ತೆಂಗು", "teṅgu"),
    "coffee": ("ಕಾಫಿ", "kāphi"),
    "cotton": ("ಹತ್ತಿ", "hatti"),
    "grapes": ("ದ್ರಾಕ್ಷಿ", "drākṣi"),
    "jute": ("ಸೆಣಬು", "seṇabu"),
    "kidneybeans": ("ರಾಜ್ಮಾ", "rājmā"),
    "lentil": ("ಮಸೂರ", "masūra"),
    "maize": ("ಮೆಕ್ಕೆಜೋಳ", "mekkejōḷa"),
    "mango": ("ಮಾವು", "māvu"),
    "mothbeans": ("ಮಟಕಿ", "maṭaki"),
    "mungbean": ("ಹೆಸರುಕಾಳು", "hesarukāḷu"),
    "muskmelon": ("ಕರಬೂಜ", "karabūja"),
    "orange": ("ಕಿತ್ತಳೆ", "kittaḷe"),
    "papaya": ("ಪಪ್ಪಾಯಿ", "pappāyi"),
    "pigeonpeas": ("ತೊಗರಿ", "togari"),
    "pomegranate": ("ದಾಳಿಂಬೆ", "dāḷimbe"),
    "rice": ("ಭತ್ತ", "bhatta"),
    "watermelon": ("ಕಲ್ಲಂಗಡಿ", "kallaṅgaḍi"),
}

#: Fixed card labels, ``key -> {language: text}``.
LABELS: Dict[str, Dict[str, str]] = {
    "card_title": {
        ENGLISH: "Farmer Soil Health Card",
        KANNADA: "ರೈತ ಮಣ್ಣು ಆರೋಗ್ಯ ಕಾರ್ಡ್",
    },
    "subtitle": {
        ENGLISH: "Precision Agriculture Decision Support System",
        KANNADA: "ನಿಖರ ಕೃಷಿ ನಿರ್ಧಾರ ಬೆಂಬಲ ವ್ಯವಸ್ಥೆ",
    },
    "district": {ENGLISH: "District", KANNADA: "ಜಿಲ್ಲೆ"},
    "issued": {ENGLISH: "Issued", KANNADA: "ನೀಡಿದ ದಿನಾಂಕ"},
    "soil_chemistry": {
        ENGLISH: "Tested Soil Chemistry",
        KANNADA: "ಪರೀಕ್ಷಿಸಿದ ಮಣ್ಣಿನ ರಾಸಾಯನಿಕ ಗುಣ",
    },
    "microclimate": {
        ENGLISH: "Microclimate at Assessment",
        KANNADA: "ಹವಾಮಾನ ಸ್ಥಿತಿ",
    },
    "recommendation": {ENGLISH: "Recommendation", KANNADA: "ಶಿಫಾರಸು"},
    "primary_crop": {ENGLISH: "Recommended crop", KANNADA: "ಶಿಫಾರಸು ಮಾಡಿದ ಬೆಳೆ"},
    "confidence": {ENGLISH: "Model confidence", KANNADA: "ಮಾದರಿ ವಿಶ್ವಾಸ"},
    "alternatives": {ENGLISH: "Ranked Crop Suitability", KANNADA: "ಪರ್ಯಾಯ ಬೆಳೆಗಳು"},
    "rank": {ENGLISH: "Rank", KANNADA: "ಕ್ರಮ"},
    "crop": {ENGLISH: "Crop", KANNADA: "ಬೆಳೆ"},
    "advisory": {ENGLISH: "Agronomic Advisory", KANNADA: "ಕೃಷಿ ಸಲಹೆ"},
    "fertiliser": {
        ENGLISH: "Fertiliser Prescription (per hectare)",
        KANNADA: "ಗೊಬ್ಬರ ಶಿಫಾರಸು (ಪ್ರತಿ ಹೆಕ್ಟೇರ್)",
    },
    "product": {ENGLISH: "Product", KANNADA: "ಉತ್ಪನ್ನ"},
    "quantity": {ENGLISH: "Quantity", KANNADA: "ಪ್ರಮಾಣ"},
    "parameter": {ENGLISH: "Parameter", KANNADA: "ಅಂಶ"},
    "value": {ENGLISH: "Value", KANNADA: "ಮೌಲ್ಯ"},
    "unit": {ENGLISH: "Unit", KANNADA: "ಅಳತೆ"},
    "disclaimer": {
        ENGLISH: (
            "Advisory only. Corroborate with a certified laboratory soil test "
            "before committing the season."
        ),
        KANNADA: (
            "ಇದು ಸಲಹೆ ಮಾತ್ರ. ಬಿತ್ತನೆಗೆ ಮುನ್ನ ಪ್ರಮಾಣೀಕೃತ ಪ್ರಯೋಗಾಲಯದಲ್ಲಿ "
            "ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ."
        ),
    },
}

#: Feature labels for the card tables, ``feature -> {language: text}``.
FEATURE_LABELS_I18N: Dict[str, Dict[str, str]] = {
    "N": {ENGLISH: "Nitrogen (N)", KANNADA: "ಸಾರಜನಕ (N)"},
    "P": {ENGLISH: "Phosphorus (P)", KANNADA: "ರಂಜಕ (P)"},
    "K": {ENGLISH: "Potassium (K)", KANNADA: "ಪೊಟ್ಯಾಶಿಯಂ (K)"},
    "ph": {ENGLISH: "Soil pH", KANNADA: "ಮಣ್ಣಿನ ರಸಸಾರ"},
    "temperature": {ENGLISH: "Temperature", KANNADA: "ಉಷ್ಣಾಂಶ"},
    "humidity": {ENGLISH: "Humidity", KANNADA: "ಆರ್ದ್ರತೆ"},
    "rainfall": {ENGLISH: "Rainfall", KANNADA: "ಮಳೆ"},
}


def translate_crop(crop: str, language: str = KANNADA) -> str:
    """Return the crop name in ``language``, falling back to the English label.

    Parameters
    ----------
    crop:
        English class label as stored in ``class_names.pkl``.
    language:
        Target language code.

    Returns
    -------
    str
        The translated name, or the original capitalised label if no
        translation exists.
    """
    if language == ENGLISH:
        return crop.capitalize()
    entry = CROP_NAMES.get(str(crop).lower())
    return entry[0] if entry else crop.capitalize()


def crop_transliteration(crop: str) -> str:
    """Return the ISO 15919 transliteration, for verification and fallback."""
    entry = CROP_NAMES.get(str(crop).lower())
    return entry[1] if entry else str(crop).lower()


def bilingual_crop(crop: str) -> str:
    """Return ``"Kannada (English)"`` for a crop, or the English name alone."""
    entry = CROP_NAMES.get(str(crop).lower())
    if not entry:
        return crop.capitalize()
    return f"{entry[0]} ({crop.capitalize()})"


def label(key: str, language: str = ENGLISH) -> str:
    """Return a card label in ``language``, falling back to English then key."""
    entry = LABELS.get(key, {})
    return entry.get(language) or entry.get(ENGLISH) or key


def bilingual_label(key: str) -> str:
    """Return ``"English / Kannada"`` for a card label."""
    entry = LABELS.get(key, {})
    english = entry.get(ENGLISH, key)
    kannada = entry.get(KANNADA)
    return f"{english} / {kannada}" if kannada else english


def feature_label(feature: str, language: str = ENGLISH) -> str:
    """Return a feature label in ``language``, falling back to English."""
    entry = FEATURE_LABELS_I18N.get(feature, {})
    return entry.get(language) or entry.get(ENGLISH) or feature


def bilingual_feature(feature: str) -> str:
    """Return ``"English / Kannada"`` for a feature label."""
    entry = FEATURE_LABELS_I18N.get(feature, {})
    english = entry.get(ENGLISH, feature)
    kannada = entry.get(KANNADA)
    return f"{english} / {kannada}" if kannada else english


def coverage() -> Dict[str, float]:
    """Return translation coverage, for the documentation and tests."""
    return {
        "crops": len(CROP_NAMES),
        "labels": len(LABELS),
        "features": len(FEATURE_LABELS_I18N),
    }


__all__ = [
    "ENGLISH",
    "KANNADA",
    "SUPPORTED_LANGUAGES",
    "CROP_NAMES",
    "LABELS",
    "FEATURE_LABELS_I18N",
    "translate_crop",
    "crop_transliteration",
    "bilingual_crop",
    "label",
    "bilingual_label",
    "feature_label",
    "bilingual_feature",
    "coverage",
]
