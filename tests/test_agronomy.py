"""Crop metadata, the commercial bag converter, and the spray advisory."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.config import DATA_DIR
from src.utils.agronomy import (
    BAG_KG,
    CROPS,
    DAP_N,
    DAP_P2O5,
    MOP_K2O,
    SURVEY_CORROBORATED,
    UREA_N,
    bag_plan,
    bilingual_name,
    coverage,
    kannada_name,
    profile,
    roadmap,
    spray_advice,
    with_overrides,
)


# --------------------------------------------------------------------------- #
# Crop metadata
# --------------------------------------------------------------------------- #
class TestCropCoverage:
    def test_every_model_class_has_a_profile(self, recommender):
        """A crop the model can predict but cannot name breaks the crop card."""
        missing = [c for c in recommender.class_names if c not in CROPS]
        assert not missing, f"no profile for {missing}"

    def test_no_profile_for_a_class_the_model_cannot_predict(self, recommender):
        extra = [c for c in CROPS if c not in recommender.class_names]
        assert not extra, f"profile for unknown class {extra}"

    def test_coverage_is_twenty_two(self):
        assert coverage() == 22

    @pytest.mark.parametrize("crop", sorted(CROPS))
    def test_economics_are_positive_and_finite(self, crop):
        p = CROPS[crop]
        assert p.yield_quintal_per_acre > 0
        assert p.price_per_quintal > 0
        assert p.cost_per_acre > 0
        assert 60 <= p.duration_days <= 365

    @pytest.mark.parametrize("crop", sorted(CROPS))
    def test_kannada_name_is_actually_kannada(self, crop):
        """Guards against an English name left in the Kannada slot."""
        script = CROPS[crop].kannada_script
        assert script, f"{crop} has no Kannada name"
        # Kannada occupies U+0C80..U+0CFF.
        assert any("ಀ" <= ch <= "೿" for ch in script), script

    def test_spec_examples_match(self):
        """The three names the specification named explicitly."""
        assert kannada_name("rice") == "ಭತ್ತ (Bhatta)"
        assert kannada_name("chickpea") == "ಕಡಲೆ (Kadale)"
        assert kannada_name("cotton") == "ಹತ್ತಿ (Hatti)"

    def test_bilingual_form(self):
        assert bilingual_name("rice") == "Rice · ಭತ್ತ (Bhatta)"

    def test_unknown_crop_falls_back_rather_than_raising(self):
        assert profile("dragonfruit") is None
        assert kannada_name("dragonfruit") == "Dragonfruit"
        assert bilingual_name("dragonfruit") == "Dragonfruit"

    def test_lookup_is_case_insensitive(self):
        assert profile("RICE") is profile("rice")


class TestSurveyCorroboration:
    """The claim that these names are backed by the shipped survey.

    Re-derived from the CSV rather than trusted, so the constant cannot drift
    away from the data it claims to be grounded in.
    """

    @staticmethod
    def _survey_kannada() -> str:
        columns = [
            "kharifcrop", "rabicrop", "summercrop",
            "othercrop1", "other_crop2", "other_crop3",
        ]
        frame = pd.read_csv(DATA_DIR / "Cleaned_NFSM_Dataset.csv", usecols=columns)
        labels = set()
        for column in columns:
            labels |= set(frame[column].dropna().astype(str).unique())
        return " | ".join(l.split("/", 1)[1] for l in labels if "/" in l)

    def test_constant_matches_the_data(self):
        blob = self._survey_kannada()
        derived = {c for c, p in CROPS.items() if p.kannada_script in blob}
        assert derived == set(SURVEY_CORROBORATED)

    def test_maize_is_excluded(self):
        """The survey's ``Maize/ಜೋಳ`` is a conflation -- ಜೋಳ is jowar."""
        assert "maize" not in SURVEY_CORROBORATED
        assert CROPS["maize"].kannada_script == "ಮೆಕ್ಕೆಜೋಳ"

    def test_the_flag_is_exposed_per_crop(self):
        assert CROPS["rice"].name_is_corroborated is True
        assert CROPS["maize"].name_is_corroborated is False


class TestOverrides:
    def test_price_override_flows_into_gross(self):
        base = CROPS["rice"]
        mine = with_overrides("rice", price_per_quintal=3000)
        assert mine.price_per_quintal == 3000
        assert mine.gross_per_acre == base.yield_quintal_per_acre * 3000

    def test_untouched_fields_keep_the_benchmark(self):
        mine = with_overrides("rice", price_per_quintal=3000)
        assert mine.cost_per_acre == CROPS["rice"].cost_per_acre
        assert mine.kannada == CROPS["rice"].kannada

    def test_no_overrides_returns_the_benchmark_itself(self):
        assert with_overrides("rice") is CROPS["rice"]

    def test_unknown_crop_returns_none(self):
        assert with_overrides("dragonfruit", price_per_quintal=1) is None


class TestScaledEconomics:
    def test_scales_linearly_with_acreage(self):
        one = CROPS["rice"].economics(1)
        three = CROPS["rice"].economics(3)
        assert three.gross == pytest.approx(one.gross * 3)
        assert three.net == pytest.approx(one.net * 3)

    def test_zero_acres_is_zero_not_negative(self):
        nil = CROPS["rice"].economics(0)
        assert nil.gross == 0 and nil.cost == 0 and nil.net == 0

    def test_negative_acreage_is_clamped(self):
        assert CROPS["rice"].economics(-5).acres == 0

    def test_a_loss_making_benchmark_is_flagged(self):
        thin = with_overrides("rice", price_per_quintal=1)
        assert thin.economics(1).is_loss is True


# --------------------------------------------------------------------------- #
# Commercial bag converter
# --------------------------------------------------------------------------- #
class TestBagPlan:
    def test_dap_covers_the_phosphorus_deficit(self):
        plan = bag_plan(
            n_kg_per_hectare=0, p_kg_per_hectare=46, k_kg_per_hectare=0,
            acres=2.4710538,  # exactly one hectare
        )
        assert plan.dap.kg == pytest.approx(100.0, abs=0.2)

    def test_mop_covers_the_potassium_deficit(self):
        plan = bag_plan(
            n_kg_per_hectare=0, p_kg_per_hectare=0, k_kg_per_hectare=60,
            acres=2.4710538,
        )
        assert plan.mop.kg == pytest.approx(100.0, abs=0.2)

    def test_urea_is_credited_with_the_nitrogen_the_dap_delivers(self):
        """The whole point of the sequence.

        100 kg of DAP carries 18 kg of N. A 50 kg N target must therefore be
        met with 32 kg of urea-nitrogen, not 50.
        """
        plan = bag_plan(
            n_kg_per_hectare=50, p_kg_per_hectare=46, k_kg_per_hectare=0,
            acres=2.4710538,
        )
        assert plan.dap.kg == pytest.approx(100.0, abs=0.2)
        assert plan.nitrogen_from_dap_kg == pytest.approx(18.0, abs=0.1)
        assert plan.urea.kg == pytest.approx((50 - 18) / UREA_N, abs=0.3)

    def test_urea_is_never_negative_when_dap_over_supplies_nitrogen(self):
        """A big phosphorus dose can deliver more N than the crop needs."""
        plan = bag_plan(
            n_kg_per_hectare=5, p_kg_per_hectare=200, k_kg_per_hectare=0,
            acres=2.4710538,
        )
        assert plan.urea.kg == 0.0

    def test_scales_with_acreage(self):
        one = bag_plan(n_kg_per_hectare=50, p_kg_per_hectare=25,
                       k_kg_per_hectare=25, acres=1)
        four = bag_plan(n_kg_per_hectare=50, p_kg_per_hectare=25,
                        k_kg_per_hectare=25, acres=4)
        # Quantities are rounded to 0.1 kg for display, so scaling a rounded
        # single acre cannot match a directly computed four to closer than
        # that rounding.
        assert four.urea.kg == pytest.approx(one.urea.kg * 4, abs=0.4)
        assert four.dap.kg == pytest.approx(one.dap.kg * 4, abs=0.4)

    def test_negative_deficits_are_treated_as_no_requirement(self):
        """A soil already above target must not prescribe negative fertiliser."""
        plan = bag_plan(n_kg_per_hectare=-30, p_kg_per_hectare=-10,
                        k_kg_per_hectare=-5, acres=2)
        assert (plan.urea.kg, plan.dap.kg, plan.mop.kg) == (0.0, 0.0, 0.0)
        assert plan.is_empty is True

    def test_whole_bags_round_up(self):
        """A part sack cannot be bought, and under-dosing wastes the rest."""
        plan = bag_plan(n_kg_per_hectare=0, p_kg_per_hectare=0,
                        k_kg_per_hectare=60 * 0.51, acres=2.4710538)
        assert plan.mop.kg == pytest.approx(51.0, abs=0.2)
        assert plan.mop.whole_bags == 2
        assert plan.mop.bags == pytest.approx(1.0, abs=0.05)

    def test_exactly_one_bag_does_not_round_to_two(self):
        plan = bag_plan(n_kg_per_hectare=0, p_kg_per_hectare=0,
                        k_kg_per_hectare=MOP_K2O * BAG_KG, acres=2.4710538)
        assert plan.mop.kg == pytest.approx(50.0, abs=0.2)
        assert plan.mop.whole_bags == 1

    def test_zero_requirement_is_zero_bags(self):
        plan = bag_plan(n_kg_per_hectare=0, p_kg_per_hectare=0,
                        k_kg_per_hectare=0, acres=3)
        assert plan.total_bags == 0
        assert "none needed" in plan.urea.say()

    def test_rows_are_renderable(self):
        plan = bag_plan(n_kg_per_hectare=60, p_kg_per_hectare=30,
                        k_kg_per_hectare=20, acres=2)
        rows = plan.as_rows()
        assert len(rows) == 3
        assert {"Fertiliser", "Grade", "Bags (50 kg)", "Exact quantity"} == set(rows[0])
        assert rows[1]["Grade"] == "18-46-0"

    def test_grades_state_the_real_analysis(self):
        plan = bag_plan(n_kg_per_hectare=1, p_kg_per_hectare=1,
                        k_kg_per_hectare=1, acres=1)
        assert plan.urea.grade == "46-0-0" and UREA_N == 0.46
        assert plan.dap.grade == "18-46-0" and (DAP_N, DAP_P2O5) == (0.18, 0.46)
        assert plan.mop.grade == "0-0-60" and MOP_K2O == 0.60


# --------------------------------------------------------------------------- #
# Spray advisory
# --------------------------------------------------------------------------- #
class TestSprayAdvice:
    def test_clear_in_benign_weather(self):
        advice = spray_advice(rainfall_mm=0, humidity_pct=60, temperature_c=28)
        assert advice.should_spray is True
        assert advice.headline == "Safe to spray today"

    def test_rain_outranks_everything(self):
        """Runoff wastes the chemical outright, so it wins over damp and heat."""
        advice = spray_advice(rainfall_mm=40, humidity_pct=95, temperature_c=40)
        assert advice.status == "hold"
        assert "wash" in advice.headline

    def test_high_humidity_is_a_caution_not_a_hold(self):
        advice = spray_advice(rainfall_mm=0, humidity_pct=92, temperature_c=28)
        assert advice.status == "caution"
        assert advice.should_spray is False

    def test_heat_is_caught_when_the_air_is_dry(self):
        advice = spray_advice(rainfall_mm=0, humidity_pct=40, temperature_c=39)
        assert advice.status == "caution"
        assert "hot" in advice.headline

    def test_fallback_weather_says_so(self):
        """A verdict about the sky from mock weather must not sound certain."""
        live = spray_advice(rainfall_mm=0, humidity_pct=60, temperature_c=28)
        mock = spray_advice(rainfall_mm=0, humidity_pct=60, temperature_c=28,
                            is_live=False)
        assert "look at the sky" in mock.detail
        assert "look at the sky" not in live.detail

    @pytest.mark.parametrize("status,icon", [("clear", "✅"), ("caution", "⚠️"),
                                             ("hold", "🚫")])
    def test_each_status_has_an_icon(self, status, icon):
        advice = {
            "clear": spray_advice(rainfall_mm=0, humidity_pct=50, temperature_c=25),
            "caution": spray_advice(rainfall_mm=0, humidity_pct=95, temperature_c=25),
            "hold": spray_advice(rainfall_mm=99, humidity_pct=50, temperature_c=25),
        }[status]
        assert advice.icon == icon


# --------------------------------------------------------------------------- #
# Roadmap
# --------------------------------------------------------------------------- #
class TestRoadmap:
    @pytest.mark.parametrize("crop", sorted(CROPS))
    def test_every_crop_gets_four_stages(self, crop):
        stages = roadmap(crop)
        assert len(stages) == 4

    @pytest.mark.parametrize("crop", sorted(CROPS))
    def test_stages_are_chronological(self, crop):
        days = [s.day for s in roadmap(crop)]
        assert days == sorted(days)

    def test_annual_crop_ends_at_its_duration(self):
        assert roadmap("rice")[-1].day == CROPS["rice"].duration_days

    def test_perennial_is_described_as_a_cycle(self):
        stages = roadmap("coconut")
        assert stages[-1].when == "Through the year"
        assert "rounds" in stages[-1].action

    def test_unknown_crop_still_returns_a_plan(self):
        assert len(roadmap("dragonfruit")) == 4
