"""Verification of the design tokens and the plain-language layer.

The farmer-facing register is not decoration: it is the only wording most
users will ever read, and a bad unit conversion in it becomes a wrong
fertiliser dose in a real field. It is therefore tested like any other logic.
"""

from __future__ import annotations

import pytest

from src.core.config import FEATURE_NAMES
from src.core.theme import (
    CATEGORICAL,
    DIVERGING_HIGH,
    DIVERGING_LOW,
    MAX_SERIES,
    STATUS,
    apply_matplotlib_theme,
    diverging_colours,
    series_colour,
    series_palette,
)
from src.utils.plain_language import (
    ACRES_PER_HECTARE,
    COPY,
    category_name,
    confidence_band,
    describe_quantity,
    per_acre,
    severity_name,
    text,
)


class TestPalette:
    """Colour assignment must follow the validated order, never be generated."""

    def test_palette_has_eight_distinct_slots(self) -> None:
        assert len(CATEGORICAL) == MAX_SERIES
        assert len(set(CATEGORICAL)) == MAX_SERIES

    def test_every_slot_is_a_hex_colour(self) -> None:
        for colour in CATEGORICAL:
            assert colour.startswith("#") and len(colour) == 7
            int(colour[1:], 16)  # raises if not hex

    def test_slots_are_assigned_in_fixed_order(self) -> None:
        """Colour follows the entity's slot, never its current rank."""
        assert series_palette(3) == CATEGORICAL[:3]
        assert series_colour(0) == CATEGORICAL[0]

    def test_ninth_series_is_refused_not_generated(self) -> None:
        """A generated ninth hue would be indistinguishable under CVD."""
        with pytest.raises(IndexError, match="fold"):
            series_colour(MAX_SERIES)

    def test_palette_request_is_capped(self) -> None:
        assert len(series_palette(50)) == MAX_SERIES

    def test_diverging_poles_are_distinct(self) -> None:
        assert DIVERGING_HIGH != DIVERGING_LOW

    def test_diverging_maps_by_sign(self) -> None:
        assert diverging_colours([1.0, -1.0, 0.0]) == [
            DIVERGING_HIGH,
            DIVERGING_LOW,
            DIVERGING_HIGH,
        ]

    def test_status_colours_are_not_series_colours(self) -> None:
        """Status is reserved; reusing a series hue for it would mislead."""
        assert STATUS["critical"] not in CATEGORICAL
        assert STATUS["warning"] not in CATEGORICAL

    def test_theme_applies_without_error(self) -> None:
        import matplotlib as mpl

        apply_matplotlib_theme()
        assert mpl.rcParams["axes.grid"] is True
        # Dashed grids read as thresholds; the house style forbids them.
        assert mpl.rcParams["grid.linestyle"] == "-"


class TestUnitConversion:
    """A wrong conversion here becomes a wrong dose in a real field."""

    def test_hectare_to_acre_is_correct(self) -> None:
        assert per_acre(ACRES_PER_HECTARE) == pytest.approx(1.0)
        assert per_acre(100.0) == pytest.approx(40.4686, abs=1e-3)

    def test_conversion_is_monotonic(self) -> None:
        assert per_acre(10) < per_acre(20) < per_acre(30)

    def test_quantity_names_bags_and_acres(self) -> None:
        described = describe_quantity(100.0)
        assert "per acre" in described
        assert "bag" in described

    @pytest.mark.parametrize(
        ("kg_per_ha", "expected_phrase"),
        [
            (5.0, "less than a quarter bag"),
            (60.0, "about half a bag"),
            (120.0, "about 1 bag"),
            (400.0, "bags"),
        ],
    )
    def test_bag_bands(self, kg_per_ha: float, expected_phrase: str) -> None:
        assert expected_phrase in describe_quantity(kg_per_ha)

    def test_quantity_never_reports_a_negative(self) -> None:
        assert "-" not in describe_quantity(0.0)


class TestConfidenceBands:
    """Plain wording must track the number, and never overstate certainty."""

    @pytest.mark.parametrize(
        ("confidence", "label"),
        [
            (99.0, "Very good match"),
            (85.0, "Very good match"),
            (70.0, "Good match"),
            (60.0, "Good match"),
            (50.0, "Fair match"),
            (40.0, "Fair match"),
            (25.0, "Weak match"),
            (0.0, "Weak match"),
        ],
    )
    def test_band_boundaries(self, confidence: float, label: str) -> None:
        assert confidence_band(confidence).label == label

    def test_low_confidence_advises_a_soil_test(self) -> None:
        """The weakest band must send the user to a real test, not guess."""
        assert "soil test" in confidence_band(20.0).detail.lower()

    def test_severity_escalates_as_confidence_falls(self) -> None:
        order = ["good", "good", "warning", "critical"]
        assert [confidence_band(c).severity for c in (95, 70, 45, 20)] == order


class TestCopyRegisters:
    """Both registers must exist for every string, and actually differ."""

    def test_every_entry_has_both_registers(self) -> None:
        for key, entry in COPY.items():
            assert len(entry) == 2, key
            assert entry[0].strip(), f"{key}: empty simple wording"
            assert entry[1].strip(), f"{key}: empty technical wording"

    def test_simple_and_technical_are_selectable(self) -> None:
        assert text("tab_recommend", True) != text("tab_recommend", False)

    def test_missing_key_returns_the_key(self) -> None:
        """A missing string should be visible, not silently blank."""
        assert text("no_such_key") == "no_such_key"

    def test_every_feature_has_a_plain_label(self) -> None:
        for name in FEATURE_NAMES:
            assert f"field_{name}" in COPY, f"no plain label for {name}"

    def test_plain_labels_avoid_jargon(self) -> None:
        """The whole point of the register is that these words are absent."""
        jargon = (
            "posterior", "covariate", "stochastic", "ensemble", "heuristic",
            "attribution", "jaccard", "surrogate",
        )
        for key, (simple, _) in COPY.items():
            lowered = simple.lower()
            for word in jargon:
                assert word not in lowered, f"{key} uses jargon: {word!r}"

    def test_category_and_severity_translate(self) -> None:
        assert category_name("Hydrology", True) == "Water"
        assert category_name("Hydrology", False) == "Hydrology"
        assert severity_name("critical", True) == "Must fix first"
        assert severity_name("critical", False) == "Critical"

    def test_unknown_category_passes_through(self) -> None:
        assert category_name("Pedology", True) == "Pedology"


class TestAdvisoryRegisters:
    """Every advisory item must carry a farmer-readable wording."""

    def test_items_have_plain_text(self, recommender) -> None:
        from src.utils.agronomy_advisory import generate_advisory

        result = recommender.predict([20, 20, 20, 35, 90, 5.0, 300])
        advisory = generate_advisory(
            result.crop,
            result.raw_features,
            confidence=result.confidence,
            ood_features=result.ood_features,
        )
        assert advisory.items
        for item in advisory.items:
            assert item.plain.strip()
            assert item.say(simple=True) == item.plain
            assert item.say(simple=False) == item.message

    def test_plain_wording_differs_from_technical(self, recommender) -> None:
        from src.utils.agronomy_advisory import generate_advisory

        result = recommender.predict([20, 20, 20, 35, 90, 5.0, 300])
        advisory = generate_advisory(result.crop, result.raw_features)
        rewritten = [i for i in advisory.items if i.plain != i.message]
        assert rewritten, "no advisory item carries a plain-language rewrite"

    def test_plain_wording_avoids_jargon(self, recommender) -> None:
        from src.utils.agronomy_advisory import generate_advisory

        result = recommender.predict([20, 20, 20, 35, 90, 5.0, 300])
        advisory = generate_advisory(
            result.crop, result.raw_features, confidence=result.confidence
        )
        jargon = ("envelope", "posterior", "physiological", "interpolation")
        for item in advisory.items:
            for word in jargon:
                assert word not in item.plain.lower(), (
                    f"{item.category} plain text uses {word!r}"
                )
