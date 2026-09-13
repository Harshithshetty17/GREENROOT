"""Verification of the statistical machinery behind the architectural claims.

The corrected resampled *t*-test and the calibration metrics are load-bearing:
the project's architectural conclusions are stated in their terms. They are
therefore checked against closed-form values and known invariants rather than
trusted by inspection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from validate_architecture import (
    corrected_resampled_ttest,
    expected_calibration_error,
    multiclass_brier,
)


class TestCorrectedResampledTTest:
    """Nadeau–Bengio correction for dependent cross-validation folds."""

    def test_identical_models_are_not_distinguishable(self) -> None:
        t_statistic, p_value = corrected_resampled_ttest(np.zeros(10), 1760, 440)
        assert t_statistic == 0.0
        assert p_value == 1.0

    def test_constant_nonzero_difference_is_certain(self) -> None:
        """Zero variance with a non-zero mean is an unambiguous separation."""
        t_statistic, p_value = corrected_resampled_ttest(np.full(10, 0.05), 1760, 440)
        assert math.isinf(t_statistic)
        assert p_value == 0.0

    def test_is_more_conservative_than_the_naive_test(self) -> None:
        """The whole point of the correction: larger p than the paired t-test."""
        rng = np.random.default_rng(0)
        differences = rng.normal(0.004, 0.003, size=15)
        _, corrected_p = corrected_resampled_ttest(differences, 1760, 440)
        _, naive_p = stats.ttest_1samp(differences, 0.0)
        assert corrected_p > naive_p

    def test_matches_its_closed_form(self) -> None:
        differences = np.array([0.01, 0.02, -0.005, 0.015, 0.0])
        n_train, n_test = 1760, 440
        k = len(differences)
        expected_t = np.mean(differences) / np.sqrt(
            (1.0 / k + n_test / n_train) * np.var(differences, ddof=1)
        )
        t_statistic, _ = corrected_resampled_ttest(differences, n_train, n_test)
        assert t_statistic == pytest.approx(expected_t)

    def test_sign_follows_the_mean_difference(self) -> None:
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 0.001, size=12)
        positive, _ = corrected_resampled_ttest(0.01 + noise, 1760, 440)
        negative, _ = corrected_resampled_ttest(-0.01 + noise, 1760, 440)
        assert positive > 0 > negative

    def test_p_value_is_a_probability(self) -> None:
        rng = np.random.default_rng(2)
        for _ in range(30):
            differences = rng.normal(0, 0.01, size=15)
            _, p_value = corrected_resampled_ttest(differences, 1760, 440)
            assert 0.0 <= p_value <= 1.0

    def test_too_few_folds_returns_nan(self) -> None:
        t_statistic, p_value = corrected_resampled_ttest(np.array([0.01]), 1760, 440)
        assert math.isnan(t_statistic) and math.isnan(p_value)

    def test_larger_test_fraction_is_more_conservative(self) -> None:
        """A bigger test split means more overlap, so a smaller statistic."""
        differences = np.array([0.01, 0.02, 0.005, 0.015, 0.012])
        small, _ = corrected_resampled_ttest(differences, 1980, 220)
        large, _ = corrected_resampled_ttest(differences, 1100, 1100)
        assert abs(large) < abs(small)


class TestExpectedCalibrationError:
    """ECE must be zero for a perfectly calibrated model and bounded in [0,1]."""

    def test_perfect_confident_model_scores_zero(self) -> None:
        classes = np.array(["a", "b"])
        truth = np.array(["a", "b", "a", "b"])
        probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        assert expected_calibration_error(truth, probabilities, classes) == pytest.approx(0.0)

    def test_fully_confident_and_always_wrong_scores_one(self) -> None:
        classes = np.array(["a", "b"])
        truth = np.array(["b", "a", "b", "a"])
        probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        assert expected_calibration_error(truth, probabilities, classes) == pytest.approx(1.0)

    def test_is_bounded(self) -> None:
        rng = np.random.default_rng(3)
        classes = np.array(["a", "b", "c"])
        for _ in range(20):
            raw = rng.random((60, 3))
            probabilities = raw / raw.sum(axis=1, keepdims=True)
            truth = rng.choice(classes, size=60)
            assert 0.0 <= expected_calibration_error(truth, probabilities, classes) <= 1.0

    def test_detects_overconfidence(self) -> None:
        """90% confident but only 50% correct is a 0.4 calibration gap."""
        classes = np.array(["a", "b"])
        truth = np.array(["a", "a", "b", "b"])
        probabilities = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        assert expected_calibration_error(truth, probabilities, classes) == pytest.approx(0.4)


class TestMulticlassBrier:
    """The Brier score is a strictly proper scoring rule."""

    def test_perfect_prediction_scores_zero(self) -> None:
        classes = np.array(["a", "b"])
        truth = np.array(["a", "b"])
        probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert multiclass_brier(truth, probabilities, classes) == pytest.approx(0.0)

    def test_worst_case_scores_two(self) -> None:
        classes = np.array(["a", "b"])
        truth = np.array(["a", "b"])
        probabilities = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert multiclass_brier(truth, probabilities, classes) == pytest.approx(2.0)

    def test_uniform_prediction_over_k_classes(self) -> None:
        """A uniform posterior over k classes scores 1 - 1/k."""
        k = 4
        classes = np.array(list("abcd"))
        truth = np.array(["a", "b", "c", "d"])
        probabilities = np.full((4, k), 1.0 / k)
        assert multiclass_brier(truth, probabilities, classes) == pytest.approx(1.0 - 1.0 / k)

    def test_rewards_honest_uncertainty_over_confident_error(self) -> None:
        """The property that makes it a proper scoring rule."""
        classes = np.array(["a", "b"])
        truth = np.array(["a"])
        hedged = multiclass_brier(truth, np.array([[0.6, 0.4]]), classes)
        confidently_wrong = multiclass_brier(truth, np.array([[0.05, 0.95]]), classes)
        assert hedged < confidently_wrong

    def test_is_bounded_by_two(self) -> None:
        rng = np.random.default_rng(4)
        classes = np.array(["a", "b", "c"])
        for _ in range(20):
            raw = rng.random((40, 3))
            probabilities = raw / raw.sum(axis=1, keepdims=True)
            truth = rng.choice(classes, size=40)
            assert 0.0 <= multiclass_brier(truth, probabilities, classes) <= 2.0
