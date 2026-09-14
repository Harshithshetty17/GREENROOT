"""Bulk advisory: recommendations for many soil samples in one pass.

An agricultural extension officer serves a village, not a single cultivator.
The interactive dashboard answers one reading at a time; this module answers a
whole survey — a laboratory's CSV export, a season's worth of soil health
cards, a district's sampling round.

Design constraints that follow from real field data
---------------------------------------------------
* **Column names vary.** Laboratory exports write ``N``, ``n``, ``Nitrogen``,
  ``avl_n`` or ``Available N`` for the same quantity. The resolver accepts all
  of them rather than forcing the officer to rewrite the file.
* **Rows fail individually.** One implausible pH must not abort a 500-row
  survey. Every row is validated on its own and failures are reported with a
  row number and a reason, so the officer can fix the source data.
* **The output is evidence, not just a label.** Each row carries its top-3
  ranking, its confidence, and an out-of-distribution flag, because a bulk run
  is exactly where a quietly extrapolated recommendation would go unnoticed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.config import FEATURE_BOUNDS, FEATURE_NAMES, OOD_ZSCORE_THRESHOLD
from src.models.inference import CropRecommender, ValidationError, get_recommender

logger = logging.getLogger(__name__)

#: Accepted spellings for each canonical feature, lower-cased and stripped of
#: punctuation. Drawn from the column names actually used by the NFSM export,
#: the benchmark corpus, and the consolidated master dataset.
_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "N": ("n", "nitrogen", "avln", "availablen", "navailable", "nkgha", "nitrogenn"),
    "P": ("p", "phosphorus", "avlp", "availablep", "pavailable", "pkgha", "phosphorusp"),
    "K": ("k", "potassium", "avlk", "availablek", "kavailable", "kkgha", "potassiumk"),
    "temperature": ("temperature", "temp", "tempc", "airtemperature", "temperaturec"),
    "humidity": ("humidity", "rh", "relativehumidity", "humiditypct", "humiditypercent"),
    "ph": ("ph", "soilph", "phvalue", "phlevel", "reaction"),
    "rainfall": ("rainfall", "rain", "precipitation", "rainfallmm", "rainmm"),
}

#: Optional passthrough columns preserved in the output for traceability.
_IDENTITY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "district": ("district", "taluku", "taluk", "block", "region", "mandal"),
    "farmer": ("farmer", "farmername", "farmersname", "cultivator", "name"),
    "sample_id": ("sampleid", "soilsampleid", "id", "sampleno", "labid"),
    "village": ("village", "villagename", "hobli"),
}

#: Maximum rows accepted in one upload, to keep the dashboard responsive.
MAX_BATCH_ROWS: int = 5000


def _normalise(column: str) -> str:
    """Reduce a column name to an alias-matching key."""
    return "".join(ch for ch in str(column).lower() if ch.isalnum())


@dataclass(frozen=True)
class BatchResult:
    """Outcome of a bulk advisory run.

    Attributes
    ----------
    recommendations:
        One row per successfully processed sample, carrying the resolved
        inputs, the recommended crop, confidence, top-3 alternatives, and the
        out-of-distribution flag.
    rejected:
        One row per sample that failed validation, with a human-readable
        reason and the original 1-based row number from the uploaded file.
    resolved_columns:
        ``canonical -> original`` mapping showing how the uploader's headers
        were interpreted, so an officer can confirm nothing was misread.
    missing_columns:
        Canonical features that could not be resolved at all.
    """

    recommendations: pd.DataFrame
    rejected: pd.DataFrame
    resolved_columns: Dict[str, str] = field(default_factory=dict)
    missing_columns: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Rows submitted."""
        return len(self.recommendations) + len(self.rejected)

    @property
    def success_rate(self) -> float:
        """Fraction of submitted rows that produced a recommendation."""
        return len(self.recommendations) / self.total if self.total else 0.0

    @property
    def is_empty(self) -> bool:
        """``True`` when nothing was successfully processed."""
        return self.recommendations.empty

    def summary(self) -> Dict[str, object]:
        """Aggregate statistics for the dashboard header."""
        if self.is_empty:
            return {
                "processed": 0,
                "rejected": len(self.rejected),
                "distinct_crops": 0,
                "mean_confidence": 0.0,
                "low_confidence": 0,
                "out_of_distribution": 0,
            }
        frame = self.recommendations
        return {
            "processed": len(frame),
            "rejected": len(self.rejected),
            "distinct_crops": int(frame["recommended_crop"].nunique()),
            "mean_confidence": float(frame["confidence"].mean()),
            "low_confidence": int((frame["confidence"] < 50.0).sum()),
            "out_of_distribution": int(frame["out_of_distribution"].sum()),
        }

    def crop_distribution(self) -> pd.DataFrame:
        """Recommended-crop counts with their mean confidence, commonest first."""
        if self.is_empty:
            return pd.DataFrame(columns=["crop", "count", "share", "mean_confidence"])
        grouped = (
            self.recommendations.groupby("recommended_crop")
            .agg(count=("recommended_crop", "size"), mean_confidence=("confidence", "mean"))
            .reset_index()
            .rename(columns={"recommended_crop": "crop"})
        )
        grouped["share"] = grouped["count"] / len(self.recommendations)
        return grouped.sort_values("count", ascending=False).reset_index(drop=True)

    def flagged(self) -> pd.DataFrame:
        """Rows needing human review: low confidence or out-of-distribution.

        This is the view an officer should read first — a bulk run is where an
        extrapolated recommendation would otherwise pass unnoticed.
        """
        if self.is_empty:
            return self.recommendations
        mask = (self.recommendations["confidence"] < 50.0) | (
            self.recommendations["out_of_distribution"]
        )
        return self.recommendations[mask]


def resolve_columns(frame: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    """Map the uploaded file's headers onto the canonical feature contract.

    Parameters
    ----------
    frame:
        The uploaded table.

    Returns
    -------
    tuple
        ``(resolved, missing)`` where ``resolved`` maps canonical feature name
        to the original column, and ``missing`` lists features not found.
    """
    lookup = {_normalise(column): column for column in frame.columns}
    resolved: Dict[str, str] = {}
    missing: List[str] = []

    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in (_normalise(canonical), *aliases):
            if alias in lookup:
                resolved[canonical] = lookup[alias]
                break
        else:
            missing.append(canonical)
    return resolved, missing


def resolve_identity_columns(frame: pd.DataFrame) -> Dict[str, str]:
    """Map optional traceability columns (district, farmer, sample id, village)."""
    lookup = {_normalise(column): column for column in frame.columns}
    found: Dict[str, str] = {}
    for canonical, aliases in _IDENTITY_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                found[canonical] = lookup[alias]
                break
    return found


def _describe_violation(values: Dict[str, float]) -> str:
    """Return a plain-language reason a row is inadmissible, or an empty string."""
    problems: List[str] = []
    for name in FEATURE_NAMES:
        value = values.get(name)
        if value is None or not np.isfinite(value):
            problems.append(f"{name} is missing or non-numeric")
            continue
        low, high = FEATURE_BOUNDS[name]
        if value < low or value > high:
            problems.append(f"{name}={value:g} outside admissible [{low:g}, {high:g}]")
    return "; ".join(problems)


class BatchProcessor:
    """Runs the recommendation pipeline over a table of soil samples.

    Parameters
    ----------
    recommender:
        Inference wrapper. Defaults to the process-wide singleton, so a bulk
        run does not reload the artefacts.
    top_k:
        Number of ranked alternatives recorded per row.
    """

    def __init__(
        self, recommender: Optional[CropRecommender] = None, top_k: int = 3
    ) -> None:
        self.recommender = recommender or get_recommender()
        self.top_k = max(1, int(top_k))

    def process(self, frame: pd.DataFrame) -> BatchResult:
        """Produce recommendations for every admissible row.

        Parameters
        ----------
        frame:
            Uploaded table. Column names are resolved leniently; rows are
            validated individually.

        Returns
        -------
        BatchResult
            Recommendations, rejections with reasons, and the header mapping.

        Raises
        ------
        ValueError
            Only if the table is empty, exceeds :data:`MAX_BATCH_ROWS`, or
            lacks one or more required features entirely — failures that no
            per-row handling can recover from.
        """
        if frame is None or frame.empty:
            raise ValueError("The uploaded file contains no rows.")
        if len(frame) > MAX_BATCH_ROWS:
            raise ValueError(
                f"{len(frame):,} rows exceeds the {MAX_BATCH_ROWS:,}-row limit. "
                f"Split the file and process it in parts."
            )

        resolved, missing = resolve_columns(frame)
        if missing:
            raise ValueError(
                "Could not find column(s) for: "
                + ", ".join(missing)
                + ". Expected headers like "
                + ", ".join(FEATURE_NAMES)
                + " (case-insensitive; common aliases such as 'Nitrogen' or "
                  "'avl_n' are also accepted)."
            )

        identity = resolve_identity_columns(frame)
        classes = self.recommender.class_names
        means = self.recommender.feature_means
        scales = self.recommender.feature_scales

        accepted_rows: List[Dict[str, object]] = []
        accepted_vectors: List[List[float]] = []
        rejected_rows: List[Dict[str, object]] = []

        for position, (_, row) in enumerate(frame.iterrows(), start=1):
            values: Dict[str, float] = {}
            for canonical in FEATURE_NAMES:
                raw = row[resolved[canonical]]
                try:
                    values[canonical] = float(raw)
                except (TypeError, ValueError):
                    values[canonical] = float("nan")

            reason = _describe_violation(values)
            record: Dict[str, object] = {"row": position}
            for label, column in identity.items():
                record[label] = row[column]

            if reason:
                rejected_rows.append({**record, "reason": reason})
                continue

            record.update({name: values[name] for name in FEATURE_NAMES})
            accepted_rows.append(record)
            accepted_vectors.append([values[name] for name in FEATURE_NAMES])

        recommendations = self._predict_accepted(
            accepted_rows, accepted_vectors, classes, means, scales
        )
        rejected = pd.DataFrame(rejected_rows) if rejected_rows else pd.DataFrame(
            columns=["row", "reason"]
        )

        logger.info(
            "Bulk advisory: %d processed, %d rejected", len(recommendations), len(rejected)
        )
        return BatchResult(
            recommendations=recommendations,
            rejected=rejected,
            resolved_columns=resolved,
            missing_columns=missing,
        )

    def _predict_accepted(
        self,
        records: List[Dict[str, object]],
        vectors: List[List[float]],
        classes: Sequence[str],
        means: np.ndarray,
        scales: np.ndarray,
    ) -> pd.DataFrame:
        """Score every admissible row in one vectorised pass."""
        if not vectors:
            return pd.DataFrame(
                columns=[
                    "row", *FEATURE_NAMES, "recommended_crop", "confidence",
                    "alternative_2", "alternative_3", "out_of_distribution",
                    "ood_features",
                ]
            )

        matrix = np.asarray(vectors, dtype=float)
        try:
            _, probabilities = self.recommender.predict_batch(matrix)
        except ValidationError as exc:  # pragma: no cover - rows pre-validated
            raise ValueError(f"Batch inference failed: {exc}") from exc

        # Z-scores are computed here rather than per row: the bulk view exists
        # precisely to surface silent extrapolation across many samples.
        z_scores = (matrix - means) / scales
        flags = np.abs(z_scores) > OOD_ZSCORE_THRESHOLD

        order = np.argsort(probabilities, axis=1)[:, ::-1]
        for index, record in enumerate(records):
            ranked = order[index][: self.top_k]
            record["recommended_crop"] = classes[int(ranked[0])]
            record["confidence"] = round(float(probabilities[index, ranked[0]]) * 100, 2)
            for rank, column in enumerate(ranked[1:], start=2):
                record[f"alternative_{rank}"] = (
                    f"{classes[int(column)]} "
                    f"({probabilities[index, int(column)] * 100:.1f}%)"
                )
            offending = [FEATURE_NAMES[j] for j in np.flatnonzero(flags[index])]
            record["out_of_distribution"] = bool(offending)
            record["ood_features"] = ", ".join(offending)

        return pd.DataFrame(records)


def build_template(rows: int = 5) -> pd.DataFrame:
    """Return a worked example file an officer can fill in and re-upload.

    Populated with plausible Karnataka readings rather than zeros, so the
    expected units and magnitudes are self-evident from the template itself.
    """
    samples = [
        ("Udupi", "Brahmavara", 90, 42, 43, 20.9, 82.0, 6.5, 202.9),
        ("Dakshina Kannada", "Puttur", 74, 35, 40, 26.5, 80.4, 5.8, 262.7),
        ("Mysuru", "Hunsur", 60, 55, 44, 23.0, 82.3, 7.0, 263.9),
        ("Dharwad", "Kalghatgi", 40, 72, 77, 25.6, 72.0, 7.6, 151.4),
        ("Bengaluru Rural", "Doddaballapura", 23, 68, 82, 24.0, 65.1, 6.4, 98.4),
    ]
    columns = [
        "sample_id", "district", "village", "N", "P", "K",
        "temperature", "humidity", "ph", "rainfall",
    ]
    data = [
        (f"GR-{index:03d}", district, village, n, p, k, temperature, humidity, ph, rainfall)
        for index, (district, village, n, p, k, temperature, humidity, ph, rainfall)
        in enumerate(samples[: max(1, rows)], start=1)
    ]
    return pd.DataFrame(data, columns=columns)


__all__ = [
    "BatchProcessor",
    "BatchResult",
    "MAX_BATCH_ROWS",
    "build_template",
    "resolve_columns",
    "resolve_identity_columns",
]
