"""Verification of bulk advisory processing and farmer-facing localisation.

Bulk processing is where a silently wrong recommendation does the most damage:
nobody inspects row 287 of a 500-row survey. These tests therefore concentrate
on lenient-but-correct column resolution, per-row failure isolation, and the
flagging of rows that need human review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.config import FEATURE_NAMES
from src.models.batch import (
    MAX_BATCH_ROWS,
    BatchProcessor,
    build_template,
    resolve_columns,
    resolve_identity_columns,
)
from src.utils.localisation import (
    CROP_NAMES,
    ENGLISH,
    KANNADA,
    bilingual_crop,
    bilingual_feature,
    coverage,
    crop_transliteration,
    label,
    translate_crop,
)
from tests.conftest import artefacts_required


# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #
class TestColumnResolution:
    """Laboratory exports do not agree on header names; the resolver must."""

    def test_canonical_headers_resolve(self) -> None:
        frame = pd.DataFrame({name: [1.0] for name in FEATURE_NAMES})
        resolved, missing = resolve_columns(frame)
        assert not missing
        assert set(resolved) == set(FEATURE_NAMES)

    @pytest.mark.parametrize(
        ("header", "canonical"),
        [
            ("Nitrogen", "N"), ("nitrogen", "N"), ("avl_n", "N"), ("N (kg/ha)", "N"),
            ("Phosphorus", "P"), ("avl_p", "P"),
            ("Potassium", "K"), ("avl_k", "K"),
            ("Temp", "temperature"), ("Temp (C)", "temperature"),
            ("Relative Humidity", "humidity"), ("RH", "humidity"),
            ("Soil pH", "ph"), ("pH", "ph"),
            ("Rainfall (mm)", "rainfall"), ("Precipitation", "rainfall"),
        ],
    )
    def test_aliases_resolve(self, header: str, canonical: str) -> None:
        frame = pd.DataFrame({header: [1.0]})
        resolved, _ = resolve_columns(frame)
        assert resolved.get(canonical) == header

    def test_missing_columns_are_reported(self) -> None:
        frame = pd.DataFrame({"N": [1.0], "P": [1.0]})
        _, missing = resolve_columns(frame)
        assert set(missing) == set(FEATURE_NAMES) - {"N", "P"}

    def test_identity_columns_are_detected(self) -> None:
        frame = pd.DataFrame(
            {"Sample No": ["A"], "Taluk": ["Udupi"], "Farmer Name": ["X"],
             "Village Name": ["Y"]}
        )
        identity = resolve_identity_columns(frame)
        assert identity["sample_id"] == "Sample No"
        assert identity["district"] == "Taluk"
        assert identity["farmer"] == "Farmer Name"
        assert identity["village"] == "Village Name"

    def test_identity_columns_are_optional(self) -> None:
        assert resolve_identity_columns(pd.DataFrame({"N": [1]})) == {}


# --------------------------------------------------------------------------- #
# Template
# --------------------------------------------------------------------------- #
class TestTemplate:
    """The shipped template must itself be valid input."""

    def test_template_has_every_required_column(self) -> None:
        resolved, missing = resolve_columns(build_template())
        assert not missing
        assert len(resolved) == len(FEATURE_NAMES)

    def test_template_carries_identity_columns(self) -> None:
        identity = resolve_identity_columns(build_template())
        assert {"sample_id", "district", "village"} <= set(identity)

    def test_template_is_not_empty(self) -> None:
        assert len(build_template()) >= 1


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def processor() -> BatchProcessor:
    """A batch processor backed by the deployed ensemble."""
    from src.core.config import MODEL_PATH

    if not MODEL_PATH.exists():
        pytest.skip("Pre-trained artefacts absent from models/.")
    return BatchProcessor()


@artefacts_required
class TestBatchProcessing:
    """Every admissible row must be scored; every inadmissible one explained."""

    def test_template_processes_completely(self, processor: BatchProcessor) -> None:
        result = processor.process(build_template())
        assert len(result.recommendations) == len(build_template())
        assert result.rejected.empty
        assert result.success_rate == 1.0

    def test_every_row_gets_a_crop_and_confidence(
        self, processor: BatchProcessor
    ) -> None:
        frame = processor.process(build_template()).recommendations
        assert frame["recommended_crop"].notna().all()
        assert frame["confidence"].between(0, 100).all()

    def test_alternatives_are_recorded(self, processor: BatchProcessor) -> None:
        frame = processor.process(build_template()).recommendations
        assert "alternative_2" in frame.columns
        assert "alternative_3" in frame.columns

    def test_identity_columns_are_carried_through(
        self, processor: BatchProcessor
    ) -> None:
        frame = processor.process(build_template()).recommendations
        assert "district" in frame.columns
        assert "sample_id" in frame.columns

    def test_row_numbers_are_one_based_and_ordered(
        self, processor: BatchProcessor
    ) -> None:
        frame = processor.process(build_template()).recommendations
        assert list(frame["row"]) == list(range(1, len(frame) + 1))

    def test_bad_row_is_isolated_not_fatal(self, processor: BatchProcessor) -> None:
        """One implausible reading must not abort the whole survey."""
        frame = build_template()
        frame.loc[1, "ph"] = 99.0
        result = processor.process(frame)
        assert len(result.recommendations) == len(frame) - 1
        assert len(result.rejected) == 1
        assert "ph" in result.rejected.iloc[0]["reason"]

    @pytest.mark.parametrize("column", ["N", "P", "K"])
    def test_negative_nutrients_are_rejected(
        self, processor: BatchProcessor, column: str
    ) -> None:
        frame = build_template()
        frame.loc[0, column] = -1.0
        result = processor.process(frame)
        assert len(result.rejected) == 1
        assert "outside admissible" in result.rejected.iloc[0]["reason"]

    def test_non_numeric_cell_is_rejected(self, processor: BatchProcessor) -> None:
        frame = build_template().astype({"N": object})
        frame.loc[0, "N"] = "not a number"
        result = processor.process(frame)
        assert len(result.rejected) == 1

    def test_rejected_row_reports_its_source_position(
        self, processor: BatchProcessor
    ) -> None:
        frame = build_template()
        frame.loc[2, "ph"] = 0.0
        result = processor.process(frame)
        assert int(result.rejected.iloc[0]["row"]) == 3  # 1-based

    def test_empty_frame_raises(self, processor: BatchProcessor) -> None:
        with pytest.raises(ValueError, match="no rows"):
            processor.process(pd.DataFrame())

    def test_missing_feature_column_raises(self, processor: BatchProcessor) -> None:
        frame = build_template().drop(columns=["ph"])
        with pytest.raises(ValueError, match="ph"):
            processor.process(frame)

    def test_oversized_batch_raises(self, processor: BatchProcessor) -> None:
        frame = pd.concat(
            [build_template()] * (MAX_BATCH_ROWS // 5 + 2), ignore_index=True
        )
        with pytest.raises(ValueError, match="exceeds"):
            processor.process(frame)

    def test_messy_headers_are_accepted(self, processor: BatchProcessor) -> None:
        frame = pd.DataFrame(
            {
                "Taluk": ["Udupi"],
                "Nitrogen": [90], "Phosphorus": [42], "Potassium": [43],
                "Temp (C)": [20.9], "Relative Humidity": [82.0],
                "Soil pH": [6.5], "Rainfall (mm)": [202.9],
            }
        )
        result = processor.process(frame)
        assert len(result.recommendations) == 1
        assert result.recommendations.iloc[0]["recommended_crop"] == "rice"

    def test_matches_single_prediction(self, processor: BatchProcessor) -> None:
        """A batch of one must agree exactly with the interactive path."""
        from src.models.inference import get_recommender

        vector = [90.0, 42.0, 43.0, 20.88, 82.0, 6.5, 202.94]
        frame = pd.DataFrame([dict(zip(FEATURE_NAMES, vector))])
        batch = processor.process(frame).recommendations.iloc[0]
        single = get_recommender().predict(vector)
        assert batch["recommended_crop"] == single.crop
        assert batch["confidence"] == pytest.approx(single.confidence, abs=0.01)


@artefacts_required
class TestBatchReview:
    """The review queue is the safety net for unattended bulk runs."""

    def test_out_of_distribution_rows_are_flagged(
        self, processor: BatchProcessor
    ) -> None:
        frame = build_template()
        frame.loc[0, "rainfall"] = 1100.0
        result = processor.process(frame)
        flagged = result.recommendations[result.recommendations["out_of_distribution"]]
        assert len(flagged) >= 1
        assert "rainfall" in flagged.iloc[0]["ood_features"]

    def test_in_distribution_rows_are_not_flagged(
        self, processor: BatchProcessor
    ) -> None:
        result = processor.process(build_template())
        assert not result.recommendations["out_of_distribution"].all()

    def test_flagged_view_includes_low_confidence(
        self, processor: BatchProcessor
    ) -> None:
        result = processor.process(build_template())
        flagged = result.flagged()
        expected = (result.recommendations["confidence"] < 50.0) | (
            result.recommendations["out_of_distribution"]
        )
        assert len(flagged) == int(expected.sum())

    def test_summary_counts_are_consistent(self, processor: BatchProcessor) -> None:
        frame = build_template()
        frame.loc[1, "ph"] = 99.0
        result = processor.process(frame)
        summary = result.summary()
        assert summary["processed"] == len(result.recommendations)
        assert summary["rejected"] == len(result.rejected)
        assert summary["processed"] + summary["rejected"] == result.total

    def test_crop_distribution_shares_sum_to_one(
        self, processor: BatchProcessor
    ) -> None:
        distribution = processor.process(build_template()).crop_distribution()
        assert distribution["share"].sum() == pytest.approx(1.0)
        assert distribution["count"].sum() == len(build_template())


# --------------------------------------------------------------------------- #
# Localisation
# --------------------------------------------------------------------------- #
class TestLocalisation:
    """Farmer-facing text must be complete, and English must always survive."""

    def test_every_target_class_has_a_kannada_name(self, recommender) -> None:
        missing = [c for c in recommender.class_names if c.lower() not in CROP_NAMES]
        assert not missing, f"No Kannada name for: {missing}"

    def test_every_crop_has_a_transliteration(self) -> None:
        """The transliteration is what makes native review possible."""
        assert all(crop_transliteration(crop) for crop in CROP_NAMES)

    def test_translations_are_actually_kannada_script(self) -> None:
        """Guard against an English string being left in the Kannada column."""
        kannada_block = range(0x0C80, 0x0D00)
        for crop, (translated, _) in CROP_NAMES.items():
            assert any(ord(ch) in kannada_block for ch in translated), crop

    def test_english_falls_back_to_capitalised_label(self) -> None:
        assert translate_crop("rice", ENGLISH) == "Rice"

    def test_unknown_crop_degrades_gracefully(self) -> None:
        assert translate_crop("sorghum", KANNADA) == "Sorghum"
        assert bilingual_crop("sorghum") == "Sorghum"

    def test_bilingual_always_retains_english(self) -> None:
        """A mistranslation must never silently replace the advice."""
        for crop in CROP_NAMES:
            assert crop.capitalize() in bilingual_crop(crop)

    def test_bilingual_feature_retains_english(self) -> None:
        assert "Nitrogen" in bilingual_feature("N")

    def test_label_falls_back_to_english(self) -> None:
        assert label("district", ENGLISH) == "District"
        assert label("district", KANNADA) != "District"

    def test_unknown_label_returns_its_key(self) -> None:
        assert label("no_such_label", KANNADA) == "no_such_label"

    def test_coverage_is_complete(self) -> None:
        stats = coverage()
        assert stats["crops"] == 22
        assert stats["features"] == len(FEATURE_NAMES)
