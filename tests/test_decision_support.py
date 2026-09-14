"""Verification of the three decision-support layers above the model.

Season fit, intervention simulation and input costing all sit between the
classifier and a farmer spending money. Each is tested for the property that
matters most in that position: that it never fabricates a number, and that it
stays silent rather than guessing when it lacks an input.
"""

from __future__ import annotations

import pytest

from src.core.config import FEATURE_BOUNDS, FEATURE_NAMES
from src.utils import seasons as season_lib
from src.utils.economics import (
    QUINTAL_KG,
    CostEstimate,
    estimate_cost,
    price_per_kg_from_bag,
)
from src.utils.intervention import (
    NEGLIGIBLE_GAIN_PP,
    nutrients_from_plan,
    simulate_advisory,
)
from tests.conftest import artefacts_required


# --------------------------------------------------------------------------- #
# Seasons
# --------------------------------------------------------------------------- #
class TestSeasonTable:
    """The season table must cover the model's whole class space."""

    def test_every_class_has_a_season(self, recommender) -> None:
        missing = [c for c in recommender.class_names if not season_lib.seasons_for(c)]
        assert not missing, f"No sowing season recorded for: {missing}"

    def test_every_season_value_is_known(self) -> None:
        valid = set(season_lib.SEASONS)
        for crop, windows in season_lib.CROP_SEASONS.items():
            assert windows, crop
            assert windows <= valid, f"{crop} has an unknown season: {windows - valid}"

    def test_perennials_are_marked_as_such(self) -> None:
        for crop in ("coconut", "coffee", "mango", "banana", "apple"):
            assert season_lib.is_perennial(crop), crop

    def test_annuals_are_not_perennial(self) -> None:
        for crop in ("rice", "chickpea", "watermelon", "cotton"):
            assert not season_lib.is_perennial(crop), crop


class TestSeasonFit:
    """A mismatch must be reported; a perennial must never be flagged."""

    def test_rabi_pulse_clashes_with_kharif(self) -> None:
        fit = season_lib.assess("chickpea", season_lib.KHARIF)
        assert not fit.suitable
        assert "rabi" in fit.message().lower()

    def test_kharif_crop_fits_kharif(self) -> None:
        assert season_lib.assess("rice", season_lib.KHARIF).suitable

    @pytest.mark.parametrize("season", [season_lib.KHARIF, season_lib.RABI, season_lib.SUMMER])
    def test_perennial_fits_every_season(self, season: str) -> None:
        fit = season_lib.assess("coconut", season)
        assert fit.suitable and fit.is_perennial

    def test_unknown_crop_is_not_flagged(self) -> None:
        """A gap in our table is our problem, not the farmer's."""
        assert season_lib.assess("sorghum", season_lib.RABI).suitable

    def test_mismatch_message_names_the_right_season(self) -> None:
        message = season_lib.assess("watermelon", season_lib.RABI).message()
        assert "summer" in message.lower()

    def test_filter_keeps_only_sowable_crops(self) -> None:
        crops = ["rice", "chickpea", "coconut", "watermelon"]
        kept = season_lib.filter_sowable(crops, season_lib.RABI)
        assert "chickpea" in kept and "coconut" in kept
        assert "watermelon" not in kept

    @pytest.mark.parametrize("month", list(range(1, 13)))
    def test_default_season_is_always_valid(self, month: int) -> None:
        assert season_lib.default_season(month) in season_lib.SEASONS

    def test_default_season_tracks_the_calendar(self) -> None:
        assert season_lib.default_season(7) == season_lib.KHARIF
        assert season_lib.default_season(11) == season_lib.RABI
        assert season_lib.default_season(3) == season_lib.SUMMER


# --------------------------------------------------------------------------- #
# Intervention simulation
# --------------------------------------------------------------------------- #
class TestNutrientConversion:
    """The prescription must invert back to the element it supplies."""

    def test_urea_supplies_46_percent_nitrogen(self) -> None:
        assert nutrients_from_plan({"Urea": 100.0})["N"] == pytest.approx(46.0)

    def test_mop_supplies_60_percent_potassium(self) -> None:
        assert nutrients_from_plan({"Muriate of Potash (MOP)": 100.0})["K"] == (
            pytest.approx(60.0)
        )

    def test_ssp_supplies_16_percent_phosphorus(self) -> None:
        assert nutrients_from_plan({"Single Super Phosphate (SSP)": 100.0})["P"] == (
            pytest.approx(16.0)
        )

    def test_unknown_product_is_skipped_not_guessed(self) -> None:
        assert nutrients_from_plan({"Mystery Mix": 100.0}) == {}

    def test_empty_plan_yields_nothing(self) -> None:
        assert nutrients_from_plan({}) == {}


@artefacts_required
class TestSimulation:
    """Simulating the advice must use the same inference path as the advice."""

    def _advisory(self, recommender, vector):
        from src.utils.agronomy_advisory import generate_advisory

        prediction = recommender.predict(vector)
        advisory = generate_advisory(
            prediction.crop, prediction.raw_features, confidence=prediction.confidence
        )
        return prediction, advisory

    def test_returns_none_when_nothing_is_prescribed(self, recommender) -> None:
        prediction, advisory = self._advisory(
            recommender, [90, 42, 43, 20.88, 82.0, 6.5, 202.94]
        )
        if advisory.fertiliser_plan:
            pytest.skip("This reading does prescribe fertiliser")
        assert simulate_advisory(
            recommender, prediction.raw_features, advisory, before=prediction
        ) is None

    def test_deficit_reading_produces_a_simulation(self, recommender) -> None:
        vector = [20, 20, 20, 28, 80, 6.5, 150]
        prediction, advisory = self._advisory(recommender, vector)
        result = simulate_advisory(
            recommender, prediction.raw_features, advisory, before=prediction
        )
        assert result is not None
        assert result.products
        assert result.nutrients_added

    def test_amended_readings_increase_by_the_supplied_nutrient(
        self, recommender
    ) -> None:
        vector = [20, 20, 20, 28, 80, 6.5, 150]
        prediction, advisory = self._advisory(recommender, vector)
        result = simulate_advisory(
            recommender, prediction.raw_features, advisory, before=prediction
        )
        for nutrient, added in result.nutrients_added.items():
            before = prediction.raw_features[nutrient]
            assert result.amended_features[nutrient] == pytest.approx(
                before + added, abs=1e-6
            )

    def test_untouched_features_are_unchanged(self, recommender) -> None:
        vector = [20, 20, 20, 28, 80, 6.5, 150]
        prediction, advisory = self._advisory(recommender, vector)
        result = simulate_advisory(
            recommender, prediction.raw_features, advisory, before=prediction
        )
        for name in FEATURE_NAMES:
            if name not in result.nutrients_added:
                assert result.amended_features[name] == prediction.raw_features[name]

    def test_amended_readings_stay_inside_the_envelope(self, recommender) -> None:
        """A large dose must be clipped, not pushed past the validator."""
        vector = [20, 20, 20, 28, 80, 6.5, 150]
        prediction, advisory = self._advisory(recommender, vector)
        result = simulate_advisory(
            recommender, prediction.raw_features, advisory, before=prediction
        )
        for name, value in result.amended_features.items():
            low, high = FEATURE_BOUNDS[name]
            assert low <= value <= high

    def test_verdict_is_honest_about_small_gains(self, recommender) -> None:
        vector = [20, 20, 20, 28, 80, 6.5, 150]
        prediction, advisory = self._advisory(recommender, vector)
        result = simulate_advisory(
            recommender, prediction.raw_features, advisory, before=prediction
        )
        verdict = result.verdict(simple=True).lower()
        if result.original_crop_delta < NEGLIGIBLE_GAIN_PP and not result.crop_changed:
            assert "barely" in verdict
            assert not result.is_worthwhile
        else:
            assert result.is_worthwhile

    def test_ph_advice_is_declared_unsimulated(self, recommender) -> None:
        """pH response is not modelled, and the result must say so."""
        vector = [20, 20, 20, 28, 80, 4.5, 150]  # strongly acidic
        prediction, advisory = self._advisory(recommender, vector)
        result = simulate_advisory(
            recommender, prediction.raw_features, advisory, before=prediction
        )
        if result is None:
            pytest.skip("No prescription for this reading")
        assert "Soil Reaction" in result.unsimulated


# --------------------------------------------------------------------------- #
# Economics
# --------------------------------------------------------------------------- #
class TestCosting:
    """No price in, no number out — the module must never invent one."""

    PLAN = {"Urea": 100.0, "Muriate of Potash (MOP)": 50.0}

    def test_no_prices_means_no_cost(self) -> None:
        estimate = estimate_cost(self.PLAN, {})
        assert not estimate.priced
        assert estimate.total_per_acre == 0.0
        assert "Enter" in estimate.summary(simple=True)

    def test_zero_price_is_excluded_not_assumed(self) -> None:
        assert not estimate_cost(self.PLAN, {"Urea": 0.0}).priced

    def test_partial_prices_cost_only_what_is_priced(self) -> None:
        estimate = estimate_cost(self.PLAN, {"Urea": 6.0})
        assert len(estimate.lines) == 1
        assert estimate.lines[0].product == "Urea"

    def test_bag_price_converts_per_kilogram(self) -> None:
        assert price_per_kg_from_bag(300.0) == pytest.approx(6.0)
        assert price_per_kg_from_bag(0.0) == 0.0

    def test_cost_uses_per_acre_not_per_hectare(self) -> None:
        """The farmer works in acres; costing in hectares would be 2.47x wrong."""
        estimate = estimate_cost({"Urea": 100.0}, {"Urea": 10.0})
        assert estimate.lines[0].kg_per_acre == pytest.approx(40.47, abs=0.01)
        assert estimate.total_per_acre == pytest.approx(404.69, abs=0.1)
        assert estimate.total_per_hectare == pytest.approx(1000.0, abs=0.1)

    def test_break_even_is_none_without_a_crop_price(self) -> None:
        estimate = estimate_cost({"Urea": 100.0}, {"Urea": 10.0})
        assert estimate.break_even_kg_per_acre(0.0) is None
        assert estimate.break_even_message("rice", 0.0) is None

    def test_break_even_matches_its_arithmetic(self) -> None:
        estimate = estimate_cost({"Urea": 100.0}, {"Urea": 10.0})
        price_per_quintal = 2000.0
        expected = estimate.total_per_acre / (price_per_quintal / QUINTAL_KG)
        assert estimate.break_even_kg_per_acre(price_per_quintal) == pytest.approx(
            expected
        )

    def test_higher_crop_price_lowers_break_even(self) -> None:
        estimate = estimate_cost({"Urea": 100.0}, {"Urea": 10.0})
        cheap = estimate.break_even_kg_per_acre(1000.0)
        dear = estimate.break_even_kg_per_acre(4000.0)
        assert dear < cheap

    def test_empty_estimate_breaks_even_at_nothing(self) -> None:
        assert CostEstimate().break_even_kg_per_acre(2000.0) is None

    def test_bags_per_acre_is_reported(self) -> None:
        estimate = estimate_cost({"Urea": 100.0}, {"Urea": 10.0})
        assert estimate.lines[0].bags_per_acre == pytest.approx(0.809, abs=0.01)
