"""District-level edaphic baseline retrieval from the NFSM laboratory survey.

Farmers consulting the system rarely arrive with a fresh soil test. This
service supplies a defensible prior: the robust central tendency of every
National Food Security Mission laboratory sample recorded in the requested
administrative unit.

Robust aggregation
------------------
The raw NFSM export is field-collected and carries transcription artefacts —
pH values up to 752 (a decimal-shift), available nitrogen up to 30,500 kg/ha.
Aggregating with a mean would let a single such record dominate a district.
This module therefore (1) clips every reading to the physiological envelope in
:data:`~src.core.config.FEATURE_BOUNDS`, discarding the inadmissible ones, and
(2) aggregates the survivors with the **median**, whose 50% breakdown point
makes it insensitive to the remaining contamination.

Resolution order
----------------
1. Exact (case- and whitespace-insensitive) match against a survey district.
2. Substring match in either direction, e.g. ``"Challakere"`` resolves the
   consolidated label ``"Chitradurga (Challakere)"``.
3. The curated Karnataka agro-climatic zone table below.
4. A neutral state-level composite, so a caller always receives a usable prior.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.core.config import FEATURE_BOUNDS, NFSM_DATASET

logger = logging.getLogger(__name__)

#: Raw NFSM column -> canonical feature name. ``avl_*`` are the plant-available
#: fractions reported by the laboratory; the bare ``p``/``k`` columns in the
#: export are placeholders and are ignored.
_NFSM_COLUMN_MAP: Dict[str, str] = {
    "avl_n": "N",
    "avl_p": "P",
    "avl_k": "K",
    "ph": "ph",
}

#: Administrative column in the NFSM export (taluk == sub-district).
_DISTRICT_COLUMN: str = "taluku"

#: Curated agro-climatic baselines for Karnataka zones absent from the survey.
#:
#: Values are representative departmental soil-health-card figures: laterite
#: coastal profiles (Udupi, Dakshina Kannada) are acidic and potassium-poor;
#: the southern and northern transitional zones (Mysuru, Dharwad) trend toward
#: neutral black-cotton chemistry.
KARNATAKA_FALLBACK: Dict[str, Dict[str, float]] = {
    "Udupi": {"N": 210.0, "P": 24.0, "K": 130.0, "ph": 5.8},
    "Dakshina Kannada": {"N": 205.0, "P": 22.0, "K": 125.0, "ph": 5.6},
    "Mysuru": {"N": 195.0, "P": 30.0, "K": 210.0, "ph": 6.8},
    "Dharwad": {"N": 185.0, "P": 28.0, "K": 250.0, "ph": 7.6},
    "Bengaluru Rural": {"N": 190.0, "P": 26.0, "K": 190.0, "ph": 6.4},
}

#: Neutral state-level composite, used when nothing else resolves.
_STATE_COMPOSITE: Dict[str, float] = {"N": 197.0, "P": 26.0, "K": 181.0, "ph": 6.4}

_CACHE_LOCK = threading.Lock()
_BASELINE_CACHE: Optional[Dict[str, Dict[str, float]]] = None


@dataclass(frozen=True)
class SoilBaseline:
    """A district-level macronutrient and pH prior.

    Attributes
    ----------
    district:
        Canonical name of the resolved administrative unit.
    N, P, K:
        Median plant-available macronutrients in kg/ha.
    ph:
        Median soil reaction.
    source:
        ``'nfsm'`` (survey median), ``'fallback'`` (curated zone table), or
        ``'default'`` (state composite).
    sample_count:
        Number of admissible laboratory samples behind the medians; ``0`` for
        non-survey sources.
    matched_on:
        How the query resolved — ``'exact'``, ``'substring'``, ``'fallback'``,
        or ``'default'``.
    """

    district: str
    N: float
    P: float
    K: float
    ph: float
    source: str
    sample_count: int = 0
    matched_on: str = "exact"

    @property
    def is_survey_backed(self) -> bool:
        """``True`` when the values are empirical NFSM medians."""
        return self.source == "nfsm"

    def as_dict(self) -> Dict[str, float]:
        """Return the four edaphic values keyed by canonical feature name."""
        return {"N": self.N, "P": self.P, "K": self.K, "ph": self.ph}


def _normalise(name: str) -> str:
    """Collapse a district string to a comparison key."""
    return " ".join(str(name).strip().lower().split())


def _titlecase(name: str) -> str:
    """Render a survey district in presentation case (``CHALLAKERE`` -> ``Challakere``)."""
    return " ".join(part.capitalize() for part in str(name).strip().split())


def _load_nfsm_baselines(path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """Compute clipped median baselines per district from the NFSM export.

    Returns an empty mapping when the dataset is missing or unparseable; the
    caller then relies on :data:`KARNATAKA_FALLBACK`.
    """
    dataset = Path(path) if path is not None else NFSM_DATASET
    if not dataset.exists():
        logger.warning("NFSM dataset not found at %s", dataset)
        return {}

    usecols = [_DISTRICT_COLUMN, *_NFSM_COLUMN_MAP]
    try:
        frame = pd.read_csv(dataset, usecols=usecols, low_memory=False)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        logger.warning("Could not parse NFSM dataset: %s", exc)
        return {}

    frame = frame.rename(columns=_NFSM_COLUMN_MAP)
    frame = frame.dropna(subset=[_DISTRICT_COLUMN])
    if frame.empty:
        return {}

    # Coerce laboratory strings ("N/A", blanks) to NaN, then reject readings
    # outside the physiological envelope so transcription artefacts cannot
    # influence the district median.
    for feature in _NFSM_COLUMN_MAP.values():
        series = pd.to_numeric(frame[feature], errors="coerce")
        low, high = FEATURE_BOUNDS[feature]
        frame[feature] = series.where(series.between(low, high))

    features = list(_NFSM_COLUMN_MAP.values())
    frame = frame.dropna(subset=features, how="all")
    if frame.empty:
        return {}

    # Fold the export's inconsistent casing ("CHALLAKERE" vs "Challakere")
    # into one administrative unit before aggregating.
    frame["_district"] = frame[_DISTRICT_COLUMN].map(_titlecase)

    grouped = frame.groupby("_district")
    medians = grouped[features].median()
    counts = grouped[features].apply(lambda block: int(block.notna().all(axis=1).sum()))

    baselines: Dict[str, Dict[str, float]] = {}
    for district, row in medians.iterrows():
        if row.isna().any():
            continue  # No admissible samples for at least one nutrient.
        baselines[str(district)] = {
            "N": round(float(row["N"]), 2),
            "P": round(float(row["P"]), 2),
            "K": round(float(row["K"]), 2),
            "ph": round(float(row["ph"]), 2),
            "_count": int(counts.get(district, 0)),
        }

    logger.info("Derived NFSM baselines for %d districts", len(baselines))
    return baselines


def _baselines(refresh: bool = False) -> Dict[str, Dict[str, float]]:
    """Return the memoised NFSM baseline table, loading it on first use."""
    global _BASELINE_CACHE
    with _CACHE_LOCK:
        if _BASELINE_CACHE is None or refresh:
            _BASELINE_CACHE = _load_nfsm_baselines()
        return _BASELINE_CACHE


def list_districts(include_fallback: bool = True) -> List[str]:
    """Return every selectable district, alphabetically sorted.

    Parameters
    ----------
    include_fallback:
        Append the curated Karnataka zones that the NFSM survey does not cover.
    """
    names = set(_baselines())
    if include_fallback:
        names.update(KARNATAKA_FALLBACK)
    return sorted(names)


def get_district_baseline(district_name: str) -> SoilBaseline:
    """Resolve a district to its edaphic baseline.

    Parameters
    ----------
    district_name:
        Free-text district or taluk name; matching is case-, whitespace- and
        substring-tolerant.

    Returns
    -------
    SoilBaseline
        Always a usable prior. Inspect
        :attr:`SoilBaseline.is_survey_backed` to tell empirical medians from
        the curated table, and :attr:`SoilBaseline.matched_on` for how the
        query resolved.
    """
    query = _normalise(district_name)
    survey = _baselines()

    if query:
        # 1. Exact match against the survey.
        for name, values in survey.items():
            if _normalise(name) == query:
                return _build(name, values, "nfsm", "exact")

        # 2. Bidirectional substring match, shortest candidate wins so that
        #    "Chitradurga" prefers the district over "Chitradurga (Hiriyuru)".
        candidates = [
            (name, values)
            for name, values in survey.items()
            if query in _normalise(name) or _normalise(name) in query
        ]
        if candidates:
            name, values = min(candidates, key=lambda item: len(item[0]))
            return _build(name, values, "nfsm", "substring")

        # 3. Curated Karnataka agro-climatic zones.
        for name, values in KARNATAKA_FALLBACK.items():
            key = _normalise(name)
            if key == query or query in key or key in query:
                return _build(name, values, "fallback", "fallback")

    # 4. Neutral state composite.
    logger.debug("No baseline matched %r; using state composite.", district_name)
    return _build(
        _titlecase(district_name) if query else "Karnataka (State Composite)",
        _STATE_COMPOSITE,
        "default",
        "default",
    )


def _build(
    district: str, values: Dict[str, float], source: str, matched_on: str
) -> SoilBaseline:
    """Assemble a :class:`SoilBaseline` from a raw baseline mapping."""
    return SoilBaseline(
        district=district,
        N=float(values["N"]),
        P=float(values["P"]),
        K=float(values["K"]),
        ph=float(values["ph"]),
        source=source,
        sample_count=int(values.get("_count", 0)),
        matched_on=matched_on,
    )


def refresh_cache() -> None:
    """Force the NFSM baseline table to be recomputed on next access."""
    _baselines(refresh=True)


__all__ = [
    "SoilBaseline",
    "KARNATAKA_FALLBACK",
    "get_district_baseline",
    "list_districts",
    "refresh_cache",
]
