"""Verification of the inference pipeline and the XAI consensus engine.

Covers the model contract (shapes, probability normalisation, class space),
input validation at the physiological boundary, determinism, the what-if sweep,
and the exact arithmetic of the Jaccard consensus index.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from src.core.config import (
    CONSENSUS_TOP_K,
    FEATURE_BOUNDS,
    FEATURE_NAMES,
    JACCARD_FIDELITY_THRESHOLD,
    N_CLASSES,
    N_FEATURES,
)
from src.models.inference import (
    CropCandidate,
    CropRecommender,
    PredictionResult,
    ValidationError,
)
from src.models.xai_engine import (
    HIGH_FIDELITY,
    LOCAL_DIVERGENCE,
    ExplainerConsensus,
    jaccard_index,
)
from tests.conftest import artefacts_required

pytestmark = artefacts_required


# --------------------------------------------------------------------------- #
# Artefact contract
# --------------------------------------------------------------------------- #
class TestArtefactContract:
    """The persisted artefacts must match the declared feature contract."""

    def test_class_space_is_complete(self, recommender: CropRecommender) -> None:
        assert len(recommender.class_names) == N_CLASSES
        assert len(set(recommender.class_names)) == N_CLASSES

    def test_classes_are_alphabetically_ordered(self, recommender: CropRecommender) -> None:
        assert recommender.class_names == sorted(recommender.class_names)

    def test_scaler_matches_feature_contract(self, recommender: CropRecommender) -> None:
        assert recommender.feature_means.shape == (N_FEATURES,)
        assert recommender.feature_scales.shape == (N_FEATURES,)
        assert np.all(recommender.feature_scales > 0), "A zero scale would divide by zero"

    def test_model_and_scaler_agree_on_class_order(
        self, recommender: CropRecommender
    ) -> None:
        assert [str(c) for c in recommender.model.classes_] == recommender.class_names

    def test_load_is_idempotent(self, recommender: CropRecommender) -> None:
        first = recommender.load().model
        assert recommender.load().model is first

    def test_missing_artefacts_raise(self, tmp_path) -> None:
        absent = CropRecommender(
            model_path=tmp_path / "no_model.pkl",
            scaler_path=tmp_path / "no_scaler.pkl",
            class_names_path=tmp_path / "no_classes.pkl",
        )
        with pytest.raises(FileNotFoundError, match="Missing model artefact"):
            absent.load()


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
class TestInputValidation:
    """Physiologically implausible readings must never reach the scaler."""

    @pytest.mark.parametrize("nutrient_index", [0, 1, 2])
    def test_negative_nutrients_are_rejected(
        self, recommender: CropRecommender, valid_sample: List[float], nutrient_index: int
    ) -> None:
        valid_sample[nutrient_index] = -1.0
        with pytest.raises(ValidationError, match="outside admissible"):
            recommender.predict(valid_sample)

    @pytest.mark.parametrize("ph", [2.99, 10.01, -1.0, 752.0])
    def test_out_of_range_ph_is_rejected(
        self, recommender: CropRecommender, valid_sample: List[float], ph: float
    ) -> None:
        valid_sample[FEATURE_NAMES.index("ph")] = ph
        with pytest.raises(ValidationError):
            recommender.predict(valid_sample)

    @pytest.mark.parametrize("temperature", [-0.1, 55.1, 200.0])
    def test_out_of_range_temperature_is_rejected(
        self, recommender: CropRecommender, valid_sample: List[float], temperature: float
    ) -> None:
        valid_sample[FEATURE_NAMES.index("temperature")] = temperature
        with pytest.raises(ValidationError):
            recommender.predict(valid_sample)

    @pytest.mark.parametrize("width", [0, 1, 6, 8, 22])
    def test_wrong_feature_count_is_rejected(
        self, recommender: CropRecommender, width: int
    ) -> None:
        with pytest.raises(ValidationError, match="Expected 7 features"):
            recommender.predict([1.0] * width)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_are_rejected(
        self, recommender: CropRecommender, valid_sample: List[float], value: float
    ) -> None:
        valid_sample[0] = value
        with pytest.raises(ValidationError, match="NaN or infinite"):
            recommender.predict(valid_sample)

    def test_non_numeric_input_is_rejected(self, recommender: CropRecommender) -> None:
        with pytest.raises(ValidationError, match="numeric"):
            recommender.predict(["a", "b", "c", "d", "e", "f", "g"])

    def test_boundary_values_are_accepted(self, recommender: CropRecommender) -> None:
        """The envelope is inclusive: exactly-at-bound readings must pass."""
        lower = [FEATURE_BOUNDS[name][0] for name in FEATURE_NAMES]
        upper = [FEATURE_BOUNDS[name][1] for name in FEATURE_NAMES]
        for vector in (lower, upper):
            result = recommender.predict(vector)
            assert result.crop in recommender.class_names

    def test_batch_input_rejected_by_single_predict(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        with pytest.raises(ValidationError, match="predict_batch"):
            recommender.predict([valid_sample, valid_sample])


# --------------------------------------------------------------------------- #
# Prediction contract
# --------------------------------------------------------------------------- #
class TestPredictionContract:
    """Output dimensionality, calibration, and ranking invariants."""

    def test_returns_prediction_result(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        assert isinstance(recommender.predict(valid_sample), PredictionResult)

    def test_probability_vector_has_one_entry_per_class(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        assert recommender.predict(valid_sample).probabilities.shape == (N_CLASSES,)

    def test_probabilities_sum_to_unity(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        total = float(recommender.predict(valid_sample).probabilities.sum())
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_probabilities_are_in_the_unit_interval(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        probabilities = recommender.predict(valid_sample).probabilities
        assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)

    def test_probabilities_sum_to_unity_across_random_inputs(
        self, recommender: CropRecommender
    ) -> None:
        rng = np.random.default_rng(7)
        bounds = np.array([FEATURE_BOUNDS[name] for name in FEATURE_NAMES])
        matrix = rng.uniform(bounds[:, 0], bounds[:, 1], size=(40, N_FEATURES))
        _, probabilities = recommender.predict_batch(matrix)
        assert probabilities.shape == (40, N_CLASSES)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_primary_crop_is_the_argmax(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        result = recommender.predict(valid_sample)
        expected = result.class_names[int(np.argmax(result.probabilities))]
        assert result.crop == expected

    def test_confidence_matches_the_argmax_posterior(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        result = recommender.predict(valid_sample)
        assert math.isclose(
            result.confidence, float(result.probabilities.max()) * 100.0, abs_tol=1e-9
        )

    @pytest.mark.parametrize("k", [1, 3, 5, 22])
    def test_top_k_length_and_ordering(
        self, recommender: CropRecommender, valid_sample: List[float], k: int
    ) -> None:
        candidates = recommender.predict(valid_sample, top_k=k).top_k
        assert len(candidates) == k
        probabilities = [c.probability for c in candidates]
        assert probabilities == sorted(probabilities, reverse=True)
        assert [c.rank for c in candidates] == list(range(1, k + 1))
        assert all(isinstance(c, CropCandidate) for c in candidates)

    def test_top_k_is_clamped_to_the_class_count(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        assert len(recommender.predict(valid_sample, top_k=500).top_k) == N_CLASSES

    def test_z_scores_match_the_scaler_transform(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        result = recommender.predict(valid_sample)
        expected = (
            np.asarray(valid_sample) - recommender.feature_means
        ) / recommender.feature_scales
        assert np.allclose(result.z_scores, expected, atol=1e-9)

    def test_raw_features_round_trip(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        result = recommender.predict(valid_sample)
        assert list(result.raw_features) == FEATURE_NAMES
        assert np.allclose(
            [result.raw_features[name] for name in FEATURE_NAMES], valid_sample
        )

    def test_prediction_is_deterministic(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        first = recommender.predict(valid_sample)
        second = recommender.predict(valid_sample)
        assert first.crop == second.crop
        assert np.allclose(first.probabilities, second.probabilities)

    def test_benchmark_sample_classifies_as_rice(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        """Guards against artefact drift: this row is rice in the corpus."""
        result = recommender.predict(valid_sample)
        assert result.crop == "rice"
        assert result.confidence > 80.0

    def test_frames_have_the_expected_shape(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        result = recommender.predict(valid_sample)
        assert list(result.z_score_frame().columns) == ["feature", "value", "z_score"]
        assert len(result.z_score_frame()) == N_FEATURES
        assert len(result.top_k_frame()) == len(result.top_k)


# --------------------------------------------------------------------------- #
# Distribution monitoring
# --------------------------------------------------------------------------- #
class TestDistributionMonitoring:
    """The Z-score monitor must flag covariate shift and stay quiet otherwise."""

    def test_in_distribution_sample_is_not_flagged(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        assert not recommender.predict(valid_sample).is_out_of_distribution

    def test_extreme_rainfall_is_flagged(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        valid_sample[FEATURE_NAMES.index("rainfall")] = 1000.0
        result = recommender.predict(valid_sample)
        assert result.is_out_of_distribution
        assert "rainfall" in result.ood_features

    def test_low_confidence_flag_tracks_the_threshold(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        result = recommender.predict(valid_sample)
        assert result.is_low_confidence == (result.confidence < 50.0)


# --------------------------------------------------------------------------- #
# What-if sweep
# --------------------------------------------------------------------------- #
class TestSensitivitySweep:
    """The single-factor sweep underpinning the what-if tab."""

    def test_sweep_shape_matches_the_grid(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        grid, probabilities = recommender.sweep(
            valid_sample, "ph", np.linspace(4.0, 9.0, 25)
        )
        assert grid.shape == (25,)
        assert probabilities.shape == (25, N_CLASSES)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_sweep_clips_to_the_physiological_envelope(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        low, high = FEATURE_BOUNDS["ph"]
        grid, _ = recommender.sweep(valid_sample, "ph", np.linspace(-50, 50, 20))
        assert grid.min() >= low and grid.max() <= high

    def test_sweep_holds_other_features_constant(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        """At the anchor's own value the sweep must reproduce predict()."""
        anchor_ph = valid_sample[FEATURE_NAMES.index("ph")]
        _, probabilities = recommender.sweep(valid_sample, "ph", [anchor_ph])
        assert np.allclose(
            probabilities[0], recommender.predict(valid_sample).probabilities, atol=1e-9
        )

    def test_unknown_feature_is_rejected(
        self, recommender: CropRecommender, valid_sample: List[float]
    ) -> None:
        with pytest.raises(ValidationError, match="Unknown feature"):
            recommender.sweep(valid_sample, "salinity", [1.0, 2.0])


# --------------------------------------------------------------------------- #
# Jaccard consensus arithmetic
# --------------------------------------------------------------------------- #
class TestJaccardIndex:
    """The consensus metric is exact set arithmetic, not an approximation."""

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ({"N", "P", "K"}, {"N", "P", "K"}, 1.0),
            ({"N", "P", "K"}, {"N", "P", "ph"}, 0.5),
            ({"N", "P", "K"}, {"N", "ph", "rainfall"}, 0.2),
            ({"N", "P", "K"}, {"ph", "rainfall", "humidity"}, 0.0),
            (set(), set(), 1.0),
            ({"N"}, set(), 0.0),
        ],
    )
    def test_known_values(self, left: set, right: set, expected: float) -> None:
        assert math.isclose(jaccard_index(left, right), expected, abs_tol=1e-12)

    def test_is_symmetric(self) -> None:
        left, right = {"N", "P", "K"}, {"N", "ph", "humidity"}
        assert jaccard_index(left, right) == jaccard_index(right, left)

    def test_is_bounded(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(50):
            left = set(rng.choice(FEATURE_NAMES, size=3, replace=False))
            right = set(rng.choice(FEATURE_NAMES, size=3, replace=False))
            assert 0.0 <= jaccard_index(left, right) <= 1.0

    def test_top_3_takes_only_four_distinct_values(self) -> None:
        """|S| = |L| = 3 admits exactly {0, 0.2, 0.5, 1.0}."""
        observed = {
            jaccard_index(set(FEATURE_NAMES[:3]), set(FEATURE_NAMES[i : i + 3]))
            for i in range(5)
        }
        assert observed <= {0.0, 0.2, 0.5, 1.0}


# --------------------------------------------------------------------------- #
# Explainer consensus engine
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def engine() -> ExplainerConsensus:
    """A fitted consensus engine, shared across this module."""
    shap = pytest.importorskip("shap", reason="shap not installed")
    pytest.importorskip("lime", reason="lime not installed")
    del shap
    return ExplainerConsensus().fit()


class TestExplainerConsensus:
    """TreeSHAP and LIME must both attribute, and agreement must be scored."""

    def test_surrogate_is_faithful_to_the_deployed_ensemble(
        self, engine: ExplainerConsensus
    ) -> None:
        assert engine.surrogate_fidelity > 0.90, (
            "A low-fidelity surrogate would explain a different model than the "
            "one that produced the recommendation."
        )

    def test_report_covers_every_feature(
        self, engine: ExplainerConsensus, valid_sample: List[float]
    ) -> None:
        report = engine.explain(valid_sample)
        assert set(report.shap_values) == set(FEATURE_NAMES)
        assert set(report.lime_values) == set(FEATURE_NAMES)

    def test_top_k_cardinality(
        self, engine: ExplainerConsensus, valid_sample: List[float]
    ) -> None:
        report = engine.explain(valid_sample)
        assert len(report.shap_top_k) == CONSENSUS_TOP_K
        assert len(report.lime_top_k) == CONSENSUS_TOP_K
        assert len(set(report.shap_top_k)) == CONSENSUS_TOP_K

    def test_jaccard_is_consistent_with_the_reported_sets(
        self, engine: ExplainerConsensus, valid_sample: List[float]
    ) -> None:
        report = engine.explain(valid_sample)
        shap_set, lime_set = set(report.shap_top_k), set(report.lime_top_k)
        assert report.intersection == len(shap_set & lime_set)
        assert report.union_size == len(shap_set | lime_set)
        assert math.isclose(
            report.jaccard, report.intersection / report.union_size, abs_tol=1e-12
        )

    def test_verdict_tracks_the_threshold(
        self, engine: ExplainerConsensus, valid_sample: List[float]
    ) -> None:
        report = engine.explain(valid_sample)
        expected = (
            HIGH_FIDELITY
            if report.jaccard >= JACCARD_FIDELITY_THRESHOLD
            else LOCAL_DIVERGENCE
        )
        assert report.verdict == expected
        assert report.is_high_fidelity == (report.verdict == HIGH_FIDELITY)

    def test_explains_the_deployed_prediction_by_default(
        self,
        engine: ExplainerConsensus,
        recommender: CropRecommender,
        valid_sample: List[float],
    ) -> None:
        assert engine.explain(valid_sample).predicted_crop == recommender.predict(
            valid_sample
        ).crop

    def test_shap_attributions_are_finite(
        self, engine: ExplainerConsensus, valid_sample: List[float]
    ) -> None:
        report = engine.explain(valid_sample)
        assert all(math.isfinite(v) for v in report.shap_values.values())
        assert all(math.isfinite(v) for v in report.lime_values.values())

    def test_primary_driver_leads_the_shap_ranking(
        self, engine: ExplainerConsensus, valid_sample: List[float]
    ) -> None:
        report = engine.explain(valid_sample)
        assert report.primary_driver == report.shap_top_k[0]
        strongest = max(report.shap_values.items(), key=lambda kv: abs(kv[1]))[0]
        assert report.primary_driver == strongest

    def test_consensus_drivers_are_the_intersection(
        self, engine: ExplainerConsensus, valid_sample: List[float]
    ) -> None:
        report = engine.explain(valid_sample)
        assert set(report.consensus_drivers) == set(report.shap_top_k) & set(
            report.lime_top_k
        )

    def test_attribution_frame_is_well_formed(
        self, engine: ExplainerConsensus, valid_sample: List[float]
    ) -> None:
        frame = engine.explain(valid_sample).to_frame()
        assert len(frame) == N_FEATURES
        assert frame["in_shap_top_k"].sum() == CONSENSUS_TOP_K
        assert frame["in_lime_top_k"].sum() == CONSENSUS_TOP_K

    def test_interpretation_is_non_empty(
        self, engine: ExplainerConsensus, valid_sample: List[float]
    ) -> None:
        report = engine.explain(valid_sample)
        assert report.predicted_crop in report.interpretation()

    def test_batch_audit_returns_the_mean_index(
        self, engine: ExplainerConsensus, valid_sample: List[float]
    ) -> None:
        rows = [valid_sample, [20.0, 130.0, 200.0, 22.0, 92.0, 5.9, 112.0]]
        reports, mean_jaccard = engine.audit_batch(rows)
        assert len(reports) == 2
        assert math.isclose(
            mean_jaccard, sum(r.jaccard for r in reports) / 2, abs_tol=1e-12
        )
