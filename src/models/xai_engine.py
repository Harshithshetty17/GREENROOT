"""Multi-explainer consensus auditing via TreeSHAP and LIME.

Motivation
----------
A single post-hoc explainer is an unfalsifiable narrator: it always produces an
attribution, and nothing in its output indicates whether that attribution is
*stable*. This module therefore runs two methodologically independent
explainers over the same instance and quantifies their agreement.

* **TreeSHAP** — an exact, game-theoretic attribution. Shapley values are the
  unique credit assignment satisfying local accuracy, missingness and
  consistency, computed in polynomial time by exploiting tree structure.
* **LIME** — a model-agnostic surrogate. It samples a perturbation
  neighbourhood around the instance and fits a locally-weighted sparse linear
  model to the black box's responses there.

Because the two rest on different premises — cooperative game theory versus
local linear approximation — their concordance is evidence that the attribution
reflects genuine model structure rather than an artefact of one method.

Consensus metric
----------------
Let :math:`S` and :math:`L` be the top-:math:`k` driver sets from TreeSHAP and
LIME. Agreement is the Jaccard index

.. math:: J(S, L) = \\frac{|S \\cap L|}{|S \\cup L|} \\in [0, 1]

For :math:`k = 3` this takes exactly four values: 0 (0/6 shared), 0.2 (1/5),
0.5 (2/4), and 1.0 (3/3). A threshold of :math:`J \\ge 0.5` — a majority of the
drivers shared — is reported as **High Fidelity**; below it, **Local
Divergence**, indicating the instance sits near a decision boundary where the
attribution is not robust and the recommendation warrants human review.

Surrogate rationale
-------------------
TreeSHAP requires a tree ensemble. The deployed model is a *stacking* classifier
whose meta-learner is a logistic regression over base-learner posteriors, so it
is not directly tractable. The engine therefore fits an interpretable
RandomForest surrogate against the same standardised feature space and audits
that. :attr:`ExplainerConsensus.surrogate_fidelity` reports the surrogate's
label agreement with the deployed ensemble, so a reader can judge how far the
explanation transfers — an honest accounting the literature often omits.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.core.config import (
    BENCHMARK_DATASET,
    CONSENSUS_TOP_K,
    FEATURE_NAMES,
    JACCARD_FIDELITY_THRESHOLD,
    LIME_NUM_SAMPLES,
    RANDOM_STATE,
    SURROGATE_MAX_DEPTH,
    SURROGATE_N_ESTIMATORS,
    SURROGATE_PATH,
    TEST_SPLIT_SIZE,
)
from src.models.inference import CropRecommender, get_recommender

logger = logging.getLogger(__name__)

HIGH_FIDELITY = "High Fidelity"
LOCAL_DIVERGENCE = "Local Divergence"


def jaccard_index(left: Set[str], right: Set[str]) -> float:
    """Compute the Jaccard similarity :math:`|A \\cap B| / |A \\cup B|`.

    Parameters
    ----------
    left, right:
        Driver-name sets to compare.

    Returns
    -------
    float
        Similarity in [0, 1]. Two empty sets are defined as perfectly
        concordant (1.0), matching the convention that the empty union carries
        no disagreement.
    """
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


@dataclass(frozen=True)
class ConsensusReport:
    """The outcome of a two-explainer audit of a single instance.

    Attributes
    ----------
    predicted_crop:
        Class the attributions explain.
    shap_values, lime_values:
        Signed per-feature attributions, keyed by feature name. SHAP values are
        in log-odds-like model output units; LIME values are local linear
        coefficients. The two are *not* on a common scale — only their rankings
        are compared.
    shap_top_k, lime_top_k:
        Top-:math:`k` driver names by absolute attribution, highest first.
    jaccard:
        Agreement index :math:`J(S, L)`.
    intersection, union_size:
        Cardinalities behind the index, for transparent reporting.
    verdict:
        :data:`HIGH_FIDELITY` or :data:`LOCAL_DIVERGENCE`.
    top_k:
        The :math:`k` used.
    surrogate_agrees:
        Whether the surrogate's own label matches the deployed ensemble's for
        this instance.
    """

    predicted_crop: str
    shap_values: Dict[str, float]
    lime_values: Dict[str, float]
    shap_top_k: List[str]
    lime_top_k: List[str]
    jaccard: float
    intersection: int
    union_size: int
    verdict: str
    top_k: int = CONSENSUS_TOP_K
    surrogate_agrees: bool = True

    @property
    def is_high_fidelity(self) -> bool:
        """``True`` when the explainers reach the consensus threshold."""
        return self.verdict == HIGH_FIDELITY

    @property
    def primary_driver(self) -> str:
        """Highest-magnitude TreeSHAP feature, for the audit ledger."""
        return self.shap_top_k[0] if self.shap_top_k else ""

    @property
    def consensus_drivers(self) -> List[str]:
        """Drivers both explainers placed in their top-:math:`k`."""
        shared = set(self.shap_top_k) & set(self.lime_top_k)
        return [name for name in self.shap_top_k if name in shared]

    def interpretation(self) -> str:
        """Return a plain-language reading of the consensus outcome."""
        if self.is_high_fidelity:
            shared = ", ".join(self.consensus_drivers) or "the leading drivers"
            return (
                f"TreeSHAP and LIME agree on {self.intersection} of "
                f"{self.union_size} drivers (J = {self.jaccard:.2f}). The "
                f"attribution to {shared} is stable across both explainers, so "
                f"the {self.predicted_crop} recommendation rests on a "
                f"reproducible basis."
            )
        return (
            f"TreeSHAP and LIME share only {self.intersection} of "
            f"{self.union_size} drivers (J = {self.jaccard:.2f}). The instance "
            f"likely sits near a decision boundary where the local attribution "
            f"is unstable; treat the {self.predicted_crop} recommendation as "
            f"provisional and corroborate it with a field soil test."
        )

    def to_frame(self) -> pd.DataFrame:
        """Return a side-by-side attribution table for both explainers."""
        return pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "shap": [self.shap_values.get(name, 0.0) for name in FEATURE_NAMES],
                "lime": [self.lime_values.get(name, 0.0) for name in FEATURE_NAMES],
                "in_shap_top_k": [name in self.shap_top_k for name in FEATURE_NAMES],
                "in_lime_top_k": [name in self.lime_top_k for name in FEATURE_NAMES],
            }
        )


class ExplainerConsensus:
    """Runs TreeSHAP and LIME over one instance and scores their agreement.

    Construction is cheap; the surrogate, SHAP explainer and LIME explainer are
    built on first use and then reused, guarded by a lock so Streamlit worker
    threads can share one instance.

    Parameters
    ----------
    recommender:
        Deployed ensemble wrapper. Defaults to the process-wide singleton.
    dataset_path:
        Benchmark CSV the surrogate and LIME's perturbation prior are fitted
        on. Defaults to ``data/Crop_recommendation.csv``.
    top_k:
        Cardinality of the compared driver sets.
    cache_path:
        Where the fitted surrogate is memoised. Set to ``None`` to disable
        on-disk caching. This is a *derived* artefact, distinct from the three
        protected files in ``models/``.
    """

    def __init__(
        self,
        recommender: Optional[CropRecommender] = None,
        dataset_path: Optional[Path] = None,
        top_k: int = CONSENSUS_TOP_K,
        cache_path: Optional[Path] = SURROGATE_PATH,
    ) -> None:
        self.recommender = recommender or get_recommender()
        self.dataset_path = Path(dataset_path) if dataset_path else BENCHMARK_DATASET
        self.top_k = max(1, min(int(top_k), len(FEATURE_NAMES)))
        self.cache_path = Path(cache_path) if cache_path else None

        self._surrogate: Optional[RandomForestClassifier] = None
        self._shap_explainer = None
        self._lime_explainer = None
        self._training_scaled: Optional[np.ndarray] = None
        self._surrogate_fidelity: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def _load_training_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the standardised benchmark design matrix and its labels."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Benchmark dataset required for XAI is missing: {self.dataset_path}"
            )
        frame = pd.read_csv(self.dataset_path)
        features = frame[FEATURE_NAMES]
        labels = frame["label"].astype(str).to_numpy()
        scaled = np.asarray(self.recommender.scaler.transform(features), dtype=float)
        return scaled, labels

    def fit(self) -> "ExplainerConsensus":
        """Build the surrogate and both explainers if not already resident.

        Idempotent and thread-safe; returns ``self`` for chaining.
        """
        if self._shap_explainer is not None and self._lime_explainer is not None:
            return self
        with self._lock:
            if self._shap_explainer is not None and self._lime_explainer is not None:
                return self

            # Imported lazily: shap and lime are heavyweight, and the
            # recommendation path must stay usable without them.
            import shap
            from lime.lime_tabular import LimeTabularExplainer

            scaled, labels = self._load_training_matrix()
            self._training_scaled = scaled

            self._surrogate = self._build_surrogate(scaled, labels)
            self._shap_explainer = shap.TreeExplainer(self._surrogate)
            self._lime_explainer = LimeTabularExplainer(
                training_data=scaled,
                feature_names=list(FEATURE_NAMES),
                class_names=list(self.recommender.class_names),
                mode="classification",
                discretize_continuous=True,
                random_state=RANDOM_STATE,
            )
            logger.info(
                "XAI engine ready (surrogate fidelity %.4f)",
                self._surrogate_fidelity if self._surrogate_fidelity is not None else -1,
            )
        return self

    def _build_surrogate(
        self, scaled: np.ndarray, labels: np.ndarray
    ) -> RandomForestClassifier:
        """Load or fit the interpretable RandomForest that TreeSHAP audits.

        The surrogate is trained on a held-out split of the benchmark so that
        its fidelity to the deployed ensemble is measured on data neither model
        memorised at that split.
        """
        if self.cache_path and self.cache_path.exists():
            try:
                payload = joblib.load(self.cache_path)
                surrogate = payload["surrogate"]
                # A stale cache from a different feature contract would silently
                # misattribute, so verify the shape before trusting it.
                if getattr(surrogate, "n_features_in_", None) == len(FEATURE_NAMES):
                    self._surrogate_fidelity = float(payload.get("fidelity", float("nan")))
                    logger.debug("Reusing cached XAI surrogate from %s", self.cache_path)
                    return surrogate
                logger.warning("Cached surrogate has a stale feature contract; refitting.")
            except (OSError, KeyError, EOFError, ValueError) as exc:
                logger.warning("Could not reuse cached surrogate (%s); refitting.", exc)

        from sklearn.model_selection import train_test_split

        train_x, holdout_x, train_y, _ = train_test_split(
            scaled,
            labels,
            test_size=TEST_SPLIT_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
        surrogate = RandomForestClassifier(
            n_estimators=SURROGATE_N_ESTIMATORS,
            max_depth=SURROGATE_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ).fit(train_x, train_y)

        # Fidelity = agreement with the *deployed ensemble*, not with ground
        # truth. An explanation is only as transferable as this number.
        deployed = np.asarray(self.recommender.model.predict(holdout_x))
        self._surrogate_fidelity = float(np.mean(surrogate.predict(holdout_x) == deployed))

        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(
                    {"surrogate": surrogate, "fidelity": self._surrogate_fidelity},
                    self.cache_path,
                )
            except OSError as exc:  # pragma: no cover - read-only volume
                logger.warning("Could not cache surrogate: %s", exc)
        return surrogate

    @property
    def surrogate(self) -> RandomForestClassifier:
        """The interpretable RandomForest that TreeSHAP is computed against."""
        self.fit()
        assert self._surrogate is not None  # Guaranteed by fit().
        return self._surrogate

    @property
    def surrogate_fidelity(self) -> float:
        """Held-out label agreement between surrogate and deployed ensemble."""
        self.fit()
        return float(self._surrogate_fidelity if self._surrogate_fidelity is not None else 0.0)

    # ------------------------------------------------------------------ #
    # Attribution
    # ------------------------------------------------------------------ #
    def _class_index(self, crop: str) -> int:
        """Map a crop label to its column in the surrogate's output space."""
        classes = [str(c) for c in self.surrogate.classes_]
        return classes.index(crop) if crop in classes else 0

    def _shap_attributions(self, scaled_instance: np.ndarray, class_index: int) -> np.ndarray:
        """Return signed TreeSHAP values for one instance and one class.

        SHAP's multiclass output layout has changed across releases — older
        versions return a per-class list, newer ones a ``(n, features, classes)``
        tensor. Both are normalised here so the engine is version-tolerant.
        """
        raw = self._shap_explainer.shap_values(scaled_instance.reshape(1, -1))

        if isinstance(raw, list):  # legacy: list of (n, n_features)
            return np.asarray(raw[class_index][0], dtype=float)

        array = np.asarray(raw, dtype=float)
        if array.ndim == 3:  # modern: (n, n_features, n_classes)
            return array[0, :, class_index]
        if array.ndim == 2:  # binary or already reduced: (n, n_features)
            return array[0]
        return array.reshape(-1)[: len(FEATURE_NAMES)]

    def _lime_attributions(
        self, scaled_instance: np.ndarray, class_index: int
    ) -> Dict[str, float]:
        """Return signed LIME coefficients for one instance and one class."""
        explanation = self._lime_explainer.explain_instance(
            data_row=scaled_instance,
            predict_fn=self.surrogate.predict_proba,
            num_features=len(FEATURE_NAMES),
            num_samples=LIME_NUM_SAMPLES,
            labels=(class_index,),
        )
        return {
            FEATURE_NAMES[int(index)]: float(weight)
            for index, weight in explanation.as_map()[class_index]
        }

    @staticmethod
    def _rank_top_k(attributions: Dict[str, float], k: int) -> List[str]:
        """Return the ``k`` feature names with the largest |attribution|."""
        ordered = sorted(attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return [name for name, _ in ordered[:k]]

    def explain(
        self,
        raw_input: Sequence[float] | np.ndarray,
        predicted_crop: Optional[str] = None,
    ) -> ConsensusReport:
        """Audit one agronomic reading with both explainers.

        Parameters
        ----------
        raw_input:
            Length-7 raw vector ordered as
            :data:`~src.core.config.FEATURE_NAMES`. Standardisation is applied
            internally, so callers pass the same units they gave
            :meth:`~src.models.inference.CropRecommender.predict`.
        predicted_crop:
            Class to explain. Defaults to the deployed ensemble's own argmax,
            which is the interesting case for an audit.

        Returns
        -------
        ConsensusReport
            Both attribution vectors, both top-:math:`k` sets, the Jaccard
            index, and the fidelity verdict.
        """
        self.fit()
        scaled = self.recommender.transform(raw_input)[0]

        if predicted_crop is None:
            predicted_crop = str(self.recommender.model.predict(scaled.reshape(1, -1))[0])

        class_index = self._class_index(predicted_crop)
        surrogate_label = str(self.surrogate.predict(scaled.reshape(1, -1))[0])

        shap_vector = self._shap_attributions(scaled, class_index)
        shap_values = {
            name: float(shap_vector[i]) for i, name in enumerate(FEATURE_NAMES)
        }
        lime_values = self._lime_attributions(scaled, class_index)

        shap_top = self._rank_top_k(shap_values, self.top_k)
        lime_top = self._rank_top_k(lime_values, self.top_k)

        shap_set, lime_set = set(shap_top), set(lime_top)
        index = jaccard_index(shap_set, lime_set)

        return ConsensusReport(
            predicted_crop=predicted_crop,
            shap_values=shap_values,
            lime_values=lime_values,
            shap_top_k=shap_top,
            lime_top_k=lime_top,
            jaccard=index,
            intersection=len(shap_set & lime_set),
            union_size=len(shap_set | lime_set),
            verdict=(
                HIGH_FIDELITY if index >= JACCARD_FIDELITY_THRESHOLD else LOCAL_DIVERGENCE
            ),
            top_k=self.top_k,
            surrogate_agrees=surrogate_label == predicted_crop,
        )

    def audit_batch(
        self, raw_inputs: Sequence[Sequence[float]]
    ) -> Tuple[List[ConsensusReport], float]:
        """Audit several readings and return the reports plus mean agreement.

        The mean Jaccard index over a representative sample is the system-level
        explainability-stability statistic reported in the evaluation.
        """
        reports = [self.explain(row) for row in raw_inputs]
        mean_j = float(np.mean([r.jaccard for r in reports])) if reports else 0.0
        return reports, mean_j


#: Process-wide engine, so Streamlit reruns do not refit the surrogate.
_DEFAULT_ENGINE: Optional[ExplainerConsensus] = None
_DEFAULT_LOCK = threading.Lock()


def get_explainer() -> ExplainerConsensus:
    """Return the lazily-instantiated process-wide :class:`ExplainerConsensus`."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_ENGINE is None:
                _DEFAULT_ENGINE = ExplainerConsensus()
    return _DEFAULT_ENGINE


__all__ = [
    "ConsensusReport",
    "ExplainerConsensus",
    "jaccard_index",
    "get_explainer",
    "HIGH_FIDELITY",
    "LOCAL_DIVERGENCE",
]
