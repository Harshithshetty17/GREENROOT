"""Inference wrapper around the pre-trained stacking ensemble.

The persisted artefacts form a two-stage pipeline:

.. code-block:: text

    x  ->  StandardScaler  ->  z  ->  {RF, AdaBoost, kNN}  ->  meta-features
                                   ->  LogisticRegression  ->  P(crop | z)

This module owns that pipeline end to end: it validates the raw agronomic
vector against the physiological envelope, applies the *exact* scaler the
ensemble was trained against, runs the meta-learner, and returns a ranked,
audit-ready :class:`PredictionResult`.

The artefacts in ``models/`` are treated as immutable. Nothing here refits,
mutates, or rewrites them.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from src.core.config import (
    CLASS_NAMES_PATH,
    FEATURE_BOUNDS,
    FEATURE_NAMES,
    LOW_CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    N_FEATURES,
    OOD_ZSCORE_THRESHOLD,
    SCALER_PATH,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CropCandidate:
    """One ranked crop hypothesis.

    Attributes
    ----------
    rank:
        1-indexed position in the descending-probability ordering.
    crop:
        Class label as stored in ``class_names.pkl``.
    probability:
        Meta-learner posterior in [0, 1].
    confidence_pct:
        The same posterior expressed as a percentage in [0, 100].
    """

    rank: int
    crop: str
    probability: float

    @property
    def confidence_pct(self) -> float:
        """Posterior probability as a percentage."""
        return self.probability * 100.0


@dataclass(frozen=True)
class PredictionResult:
    """A complete, auditable recommendation for one agronomic reading.

    Attributes
    ----------
    crop:
        Argmax class — the primary recommendation.
    confidence:
        Posterior for :attr:`crop`, as a percentage in [0, 100].
    top_k:
        Ranked alternatives, highest posterior first, including the primary.
    probabilities:
        Full posterior vector over all classes, index-aligned with
        :attr:`class_names`.
    class_names:
        Ordered class labels.
    raw_features:
        The validated raw input vector, keyed by feature name.
    z_scores:
        Standardised deviation of each input from the benchmark training mean,
        index-aligned with :data:`~src.core.config.FEATURE_NAMES`.
    scaled_vector:
        The standardised vector actually consumed by the ensemble.
    """

    crop: str
    confidence: float
    top_k: List[CropCandidate]
    probabilities: np.ndarray = field(repr=False)
    class_names: List[str] = field(repr=False)
    raw_features: Dict[str, float] = field(default_factory=dict)
    z_scores: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES), repr=False)
    scaled_vector: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES), repr=False)

    @property
    def is_low_confidence(self) -> bool:
        """``True`` when the posterior falls below the advisory threshold."""
        return (self.confidence / 100.0) < LOW_CONFIDENCE_THRESHOLD

    @property
    def ood_features(self) -> List[str]:
        """Features whose |Z| exceeds the covariate-shift threshold.

        A non-empty list means the reading sits outside the manifold the
        ensemble was trained on, so the posterior should be read as an
        extrapolation rather than an interpolation.
        """
        return [
            FEATURE_NAMES[i]
            for i, z in enumerate(self.z_scores)
            if abs(float(z)) > OOD_ZSCORE_THRESHOLD
        ]

    @property
    def is_out_of_distribution(self) -> bool:
        """``True`` when any feature breaches the covariate-shift threshold."""
        return bool(self.ood_features)

    def z_score_frame(self) -> pd.DataFrame:
        """Return the per-feature deviation profile as a tidy DataFrame."""
        return pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "value": [self.raw_features.get(name, np.nan) for name in FEATURE_NAMES],
                "z_score": [float(z) for z in self.z_scores],
            }
        )

    def top_k_frame(self) -> pd.DataFrame:
        """Return the ranked alternatives as a tidy DataFrame."""
        return pd.DataFrame(
            {
                "rank": [c.rank for c in self.top_k],
                "crop": [c.crop for c in self.top_k],
                "probability": [c.probability for c in self.top_k],
                "confidence_pct": [c.confidence_pct for c in self.top_k],
            }
        )


class ValidationError(ValueError):
    """Raised when a raw input vector violates the physiological envelope."""


class CropRecommender:
    """Loads the persisted stacking ensemble and serves ranked predictions.

    Artefact loading is lazy and guarded by a lock, so a single instance can be
    shared safely across Streamlit worker threads.

    Parameters
    ----------
    model_path, scaler_path, class_names_path:
        Overrides for the persisted artefacts. Default to the paths in
        :mod:`src.core.config`.

    Raises
    ------
    FileNotFoundError
        On first use, if any artefact is missing from ``models/``.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        class_names_path: Optional[Path] = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else MODEL_PATH
        self._scaler_path = Path(scaler_path) if scaler_path else SCALER_PATH
        self._class_names_path = (
            Path(class_names_path) if class_names_path else CLASS_NAMES_PATH
        )
        self._model: Any = None
        self._scaler: Any = None
        self._class_names: Optional[List[str]] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Artefact loading
    # ------------------------------------------------------------------ #
    def load(self) -> "CropRecommender":
        """Load the three artefacts if they are not already resident.

        Idempotent and thread-safe; returns ``self`` for chaining.
        """
        if self._model is not None:
            return self
        with self._lock:
            if self._model is not None:  # Another thread won the race.
                return self
            missing = [
                str(path)
                for path in (self._model_path, self._scaler_path, self._class_names_path)
                if not path.exists()
            ]
            if missing:
                raise FileNotFoundError(
                    "Missing model artefact(s): "
                    + ", ".join(missing)
                    + ". Run 'python train.py' to regenerate them."
                )
            self._model = joblib.load(self._model_path)
            self._scaler = joblib.load(self._scaler_path)
            self._class_names = [str(name) for name in joblib.load(self._class_names_path)]
            logger.info(
                "Loaded stacking ensemble over %d classes", len(self._class_names)
            )
        return self

    @property
    def model(self) -> Any:
        """The fitted :class:`~sklearn.ensemble.StackingClassifier`."""
        return self.load()._model

    @property
    def scaler(self) -> Any:
        """The fitted :class:`~sklearn.preprocessing.StandardScaler`."""
        return self.load()._scaler

    @property
    def class_names(self) -> List[str]:
        """Ordered crop class labels."""
        self.load()
        assert self._class_names is not None  # Guaranteed by load().
        return self._class_names

    @property
    def feature_means(self) -> np.ndarray:
        """Per-feature training means encoded in the scaler."""
        return np.asarray(self.scaler.mean_, dtype=float)

    @property
    def feature_scales(self) -> np.ndarray:
        """Per-feature training standard deviations encoded in the scaler."""
        return np.asarray(self.scaler.scale_, dtype=float)

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate(raw_input: Sequence[float] | np.ndarray) -> np.ndarray:
        """Coerce and range-check a raw agronomic vector.

        Parameters
        ----------
        raw_input:
            A length-7 sequence, or a ``(n, 7)`` array, ordered as
            :data:`~src.core.config.FEATURE_NAMES`.

        Returns
        -------
        numpy.ndarray
            A validated ``(n, 7)`` float array.

        Raises
        ------
        ValidationError
            If the input is non-numeric, has the wrong width, contains NaN or
            infinity, or breaches :data:`~src.core.config.FEATURE_BOUNDS`.
        """
        try:
            array = np.asarray(raw_input, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Input must be numeric: {exc}") from exc

        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValidationError(
                f"Expected a 1-D or 2-D input; got {array.ndim} dimensions."
            )
        if array.shape[1] != N_FEATURES:
            raise ValidationError(
                f"Expected {N_FEATURES} features {FEATURE_NAMES}; "
                f"got {array.shape[1]}."
            )
        if not np.isfinite(array).all():
            raise ValidationError("Input contains NaN or infinite values.")

        violations: List[str] = []
        for index, name in enumerate(FEATURE_NAMES):
            low, high = FEATURE_BOUNDS[name]
            column = array[:, index]
            if np.any(column < low) or np.any(column > high):
                offending = column[(column < low) | (column > high)]
                violations.append(
                    f"{name}={offending[0]:.2f} outside admissible [{low}, {high}]"
                )
        if violations:
            raise ValidationError(
                "Physiologically implausible input — " + "; ".join(violations)
            )
        return array

    def _to_frame(self, array: np.ndarray) -> pd.DataFrame:
        """Wrap a validated array in a named frame.

        The scaler was fitted on a named DataFrame, so feeding it a bare array
        emits a feature-name warning on every call. Preserving the names keeps
        the transform contract explicit and the logs clean.
        """
        return pd.DataFrame(array, columns=FEATURE_NAMES)

    def transform(self, raw_input: Sequence[float] | np.ndarray) -> np.ndarray:
        """Validate and standardise a raw vector into model space."""
        return np.asarray(self.scaler.transform(self._to_frame(self.validate(raw_input))))

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_proba(self, raw_input: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return the ``(n, 22)`` posterior matrix for raw agronomic inputs."""
        return np.asarray(self.model.predict_proba(self.transform(raw_input)), dtype=float)

    def predict(
        self, raw_input: Sequence[float] | np.ndarray, top_k: int = 3
    ) -> PredictionResult:
        """Produce a ranked recommendation for a single agronomic reading.

        Parameters
        ----------
        raw_input:
            Length-7 vector ordered as :data:`~src.core.config.FEATURE_NAMES`.
            A ``(1, 7)`` array is also accepted; wider batches are rejected —
            use :meth:`predict_batch` instead.
        top_k:
            Number of ranked alternatives to return, clamped to the class count.

        Returns
        -------
        PredictionResult
            Primary crop, confidence, ranked alternatives, the full posterior,
            and the Z-score deviation profile.

        Raises
        ------
        ValidationError
            If the input fails validation or carries more than one row.
        """
        array = self.validate(raw_input)
        if array.shape[0] != 1:
            raise ValidationError(
                f"predict() handles one reading; got {array.shape[0]} rows. "
                "Use predict_batch() for multiple readings."
            )

        scaled = np.asarray(self.scaler.transform(self._to_frame(array)), dtype=float)
        probabilities = np.asarray(self.model.predict_proba(scaled)[0], dtype=float)
        classes = self.class_names

        k = max(1, min(int(top_k), len(classes)))
        order = np.argsort(probabilities)[::-1][:k]
        candidates = [
            CropCandidate(
                rank=rank,
                crop=classes[int(index)],
                probability=float(probabilities[int(index)]),
            )
            for rank, index in enumerate(order, start=1)
        ]

        return PredictionResult(
            crop=candidates[0].crop,
            confidence=candidates[0].confidence_pct,
            top_k=candidates,
            probabilities=probabilities,
            class_names=classes,
            raw_features=dict(zip(FEATURE_NAMES, array[0].tolist())),
            z_scores=scaled[0],
            scaled_vector=scaled[0],
        )

    def predict_batch(
        self, raw_inputs: Sequence[Sequence[float]] | np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """Classify many readings at once.

        Returns
        -------
        tuple
            ``(labels, probabilities)`` where ``labels`` has length ``n`` and
            ``probabilities`` has shape ``(n, 22)``.
        """
        scaled = self.transform(raw_inputs)
        probabilities = np.asarray(self.model.predict_proba(scaled), dtype=float)
        classes = self.class_names
        labels = [classes[int(i)] for i in probabilities.argmax(axis=1)]
        return labels, probabilities

    def sweep(
        self,
        raw_input: Sequence[float],
        feature: str,
        values: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perturb one feature across ``values``, holding the other six fixed.

        This is the kernel of the what-if sensitivity analysis: it traces the
        decision surface along a single axis through the input point, which
        exposes the local decision boundaries the ensemble has learned.

        Parameters
        ----------
        raw_input:
            The anchor reading.
        feature:
            Name of the feature to perturb; must be in
            :data:`~src.core.config.FEATURE_NAMES`.
        values:
            Sweep grid for that feature. Values outside the physiological
            envelope are clipped rather than rejected, so a caller can request
            a naive ±100% sweep without pre-filtering it.

        Returns
        -------
        tuple
            ``(grid, probabilities)`` — the clipped grid actually evaluated,
            and the corresponding ``(len(grid), 22)`` posterior matrix.

        Raises
        ------
        ValidationError
            If ``feature`` is unknown or the anchor reading is invalid.
        """
        if feature not in FEATURE_NAMES:
            raise ValidationError(
                f"Unknown feature {feature!r}; expected one of {FEATURE_NAMES}."
            )
        anchor = self.validate(raw_input)[0]
        index = FEATURE_NAMES.index(feature)
        low, high = FEATURE_BOUNDS[feature]

        grid = np.clip(np.asarray(values, dtype=float), low, high)
        matrix = np.tile(anchor, (len(grid), 1))
        matrix[:, index] = grid

        scaled = np.asarray(self.scaler.transform(self._to_frame(matrix)), dtype=float)
        return grid, np.asarray(self.model.predict_proba(scaled), dtype=float)


#: Process-wide recommender, so Streamlit reruns do not re-read the artefacts.
_DEFAULT_RECOMMENDER: Optional[CropRecommender] = None
_DEFAULT_LOCK = threading.Lock()


def get_recommender() -> CropRecommender:
    """Return the lazily-instantiated process-wide :class:`CropRecommender`."""
    global _DEFAULT_RECOMMENDER
    if _DEFAULT_RECOMMENDER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_RECOMMENDER is None:
                _DEFAULT_RECOMMENDER = CropRecommender().load()
    return _DEFAULT_RECOMMENDER


__all__ = [
    "CropCandidate",
    "PredictionResult",
    "CropRecommender",
    "ValidationError",
    "get_recommender",
]
