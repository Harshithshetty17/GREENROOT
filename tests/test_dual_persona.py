"""District resolution, climate normals, and the two ledger views."""

from __future__ import annotations

import pandas as pd
import pytest

from src.database import db
from src.services.soil_service import (
    DISTRICT_ALIASES,
    KARNATAKA_CLIMATE,
    district_profile,
    get_district_baseline,
    get_district_climate,
    list_districts,
)
from src.core.config import BENCHMARK_DATASET, FEATURE_BOUNDS, FEATURE_NAMES

@pytest.fixture(scope="module")
def crop_frame() -> pd.DataFrame:
    """The training benchmark, for in-distribution assertions."""
    return pd.read_csv(BENCHMARK_DATASET)


#: The districts the brief names explicitly.
REQUIRED = [
    "Udupi", "Dakshina Kannada", "Mysuru",
    "Bengaluru Rural", "Dharwad", "Shimoga",
]


class TestDistrictCoverage:
    @pytest.mark.parametrize("name", REQUIRED)
    def test_every_required_district_resolves(self, name):
        baseline = get_district_baseline(name)
        assert baseline.source in {"nfsm", "fallback"}, (
            f"{name} fell through to the state composite"
        )

    @pytest.mark.parametrize("name", REQUIRED)
    def test_every_required_district_has_climate(self, name):
        assert get_district_climate(name).is_district_specific

    def test_shimoga_and_shivamogga_are_the_same_place(self):
        """Renamed in 2014; both spellings are still in daily use."""
        assert "shimoga" in DISTRICT_ALIASES
        old = district_profile("Shimoga")
        new = district_profile("Shivamogga")
        assert old == new

    @pytest.mark.parametrize("alias,canonical", sorted(DISTRICT_ALIASES.items()))
    def test_each_alias_matches_its_canonical_name(self, alias, canonical):
        assert district_profile(alias) == district_profile(canonical)

    def test_shivamogga_is_listed(self):
        assert "Shivamogga" in list_districts()


class TestDistrictProfile:
    @pytest.mark.parametrize("name", REQUIRED)
    def test_returns_all_seven_model_features(self, name):
        profile = district_profile(name)
        assert set(profile) == set(FEATURE_NAMES)

    @pytest.mark.parametrize("name", REQUIRED)
    def test_every_value_is_within_bounds(self, name):
        """A district load must never push the model outside its input domain."""
        for feature, value in district_profile(name).items():
            low, high = FEATURE_BOUNDS[feature]
            assert low <= value <= high, f"{name}.{feature}={value}"

    @pytest.mark.parametrize("name", REQUIRED)
    def test_climate_sits_inside_the_training_distribution(self, name, crop_frame):
        """A baseline outside the training manifold is an OOD flag on arrival.

        The whole point of a one-tap district load is to give a usable
        starting point, not to hand the model a vector it has never seen.
        """
        profile = district_profile(name)
        for feature in ("temperature", "humidity", "rainfall"):
            column = crop_frame[feature]
            assert column.min() <= profile[feature] <= column.max(), (
                f"{name}.{feature}={profile[feature]} outside "
                f"[{column.min():.1f}, {column.max():.1f}]"
            )

    def test_unknown_district_still_returns_a_usable_profile(self):
        profile = district_profile("Atlantis")
        assert set(profile) == set(FEATURE_NAMES)

    def test_climate_is_not_claimed_to_be_survey_derived(self):
        """The NFSM export has no climate columns; nothing may imply it does."""
        assert get_district_climate("Udupi").source == "normal"
        assert get_district_climate("Atlantis").source == "default"

    def test_coastal_districts_are_wetter_than_the_dry_zone(self):
        """A sanity check that the normals are not copy-paste of each other."""
        assert (
            KARNATAKA_CLIMATE["Udupi"]["rainfall"]
            > KARNATAKA_CLIMATE["Dharwad"]["rainfall"]
        )


# --------------------------------------------------------------------------- #
# Ledger views
# --------------------------------------------------------------------------- #
@pytest.fixture
def ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "timestamp": ["2026-09-14T06:47:40", "not a date"],
            "district": ["Udupi", "Mysuru"],
            "N": [60.4, 30.0], "P": [40.0, 12.0], "K": [45.0, 9.0],
            "pH": [6.24, 7.0], "temp": [26.0, 24.0], "humidity": [78.0, 60.0],
            "rainfall": [190.6, 80.0],
            "recommended_crop": ["jute", "rice"],
            "confidence": [90.855, 71.2],
            "primary_shap_driver": ["rainfall", "N"],
            "jaccard_index": [0.2, 0.5],
        }
    )


class TestFarmerView:
    def test_hides_every_technical_column(self, ledger):
        view = db.farmer_view(ledger)
        for column in db.TECHNICAL_ONLY:
            assert column not in view.columns

    def test_dates_are_localised(self, ledger):
        assert db.farmer_view(ledger)["Saved on"].iloc[0] == "14 Sep 2026"

    def test_unparseable_date_says_so_rather_than_nat(self, ledger):
        assert db.farmer_view(ledger)["Saved on"].iloc[1] == "unknown"

    def test_score_is_an_integer_out_of_a_hundred(self, ledger):
        assert db.farmer_view(ledger)["Match"].iloc[0] == "91 / 100"

    def test_no_raw_schema_names_survive(self, ledger):
        assert list(db.farmer_view(ledger).columns) == db.FARMER_COLUMNS

    def test_crop_is_title_cased(self, ledger):
        assert db.farmer_view(ledger)["Crop"].iloc[0] == "Jute"

    def test_empty_frame_keeps_the_column_shape(self):
        view = db.farmer_view(pd.DataFrame())
        assert view.empty and list(view.columns) == db.FARMER_COLUMNS


class TestExaminerView:
    def test_is_the_stored_row_untouched(self, ledger):
        view = db.examiner_view(ledger)
        assert list(view.columns) == list(ledger.columns)
        for column in db.TECHNICAL_ONLY:
            assert column in view.columns
        assert view["confidence"].iloc[0] == 90.855


class TestRecordDate:
    @pytest.mark.parametrize(
        "simple,expected",
        [(True, "14 Sep 2026"), (False, "2026-09-14")],
    )
    def test_each_persona_gets_its_own_format(self, simple, expected):
        assert db.record_date("2026-09-14T06:47:40", simple) == expected

    def test_technical_mode_keeps_iso_even_when_unparseable(self):
        assert db.record_date("2026-99-99", False) == "2026-99-99"

    def test_farmer_mode_says_unknown(self):
        assert db.record_date("2026-99-99", True) == "unknown date"


class TestTableHeight:
    def test_one_row_does_not_render_nine_blanks(self):
        assert db.table_height(1) < 120

    def test_caps_for_a_long_ledger(self):
        assert db.table_height(500) == 380

    def test_zero_rows_is_still_positive(self):
        assert db.table_height(0) > 0
