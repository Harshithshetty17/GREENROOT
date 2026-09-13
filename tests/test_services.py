"""Verification of the weather and soil micro-services.

The weather client is exercised against mocked transports rather than the live
OpenWeatherMap API, so the suite is deterministic and runs offline. The soil
service is checked against the real NFSM export where present, and against its
curated fallback table otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest
import requests

from src.core.config import FEATURE_BOUNDS, WEATHER_MOCK_DEFAULTS
from src.services import weather_service
from src.services.soil_service import (
    KARNATAKA_FALLBACK,
    SoilBaseline,
    get_district_baseline,
    list_districts,
)
from src.services.weather_service import WeatherReading, clear_cache, get_weather

API_KEY = "test-key-not-a-real-credential"


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeSession:
    """A session whose ``get`` returns a canned payload or raises."""

    def __init__(self, payload: Any = None, error: Optional[Exception] = None) -> None:
        self._payload = payload
        self._error = error
        self.calls: list[Dict[str, Any]] = []

    def get(self, url: str, params: Dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._payload)


def _ok_payload(name: str = "Udupi") -> Dict[str, Any]:
    """A well-formed OpenWeatherMap ``/weather`` response."""
    return {
        "cod": 200,
        "name": name,
        "main": {"temp": 28.4, "humidity": 84},
        "rain": {"1h": 0.25},
    }


@pytest.fixture(autouse=True)
def _isolate_cache() -> None:
    """Ensure no cross-test contamination through the TTL cache."""
    clear_cache()


# --------------------------------------------------------------------------- #
# Weather service
# --------------------------------------------------------------------------- #
class TestWeatherSuccessPath:
    """A well-formed upstream response must be parsed faithfully."""

    def test_parses_temperature_and_humidity(self) -> None:
        session = _FakeSession(payload=_ok_payload())
        reading = get_weather("Udupi", API_KEY, session=session)
        assert reading.source == "api"
        assert reading.is_live
        assert reading.temperature == pytest.approx(28.4)
        assert reading.humidity == pytest.approx(84.0)

    def test_extrapolates_hourly_rain_to_a_monthly_accumulation(self) -> None:
        session = _FakeSession(payload=_ok_payload())
        reading = get_weather("Udupi", API_KEY, session=session)
        assert reading.rainfall == pytest.approx(0.25 * 24 * 30, abs=1e-6)

    def test_absent_rain_block_yields_zero(self) -> None:
        payload = _ok_payload()
        payload.pop("rain")
        reading = get_weather("Udupi", API_KEY, session=_FakeSession(payload=payload))
        assert reading.rainfall == 0.0

    def test_falls_back_to_three_hour_accumulation(self) -> None:
        payload = _ok_payload()
        payload["rain"] = {"3h": 0.5}
        reading = get_weather("Udupi", API_KEY, session=_FakeSession(payload=payload))
        assert reading.rainfall == pytest.approx(0.5 * 24 * 30, abs=1e-6)

    def test_request_carries_the_configured_timeout(self) -> None:
        session = _FakeSession(payload=_ok_payload())
        get_weather("Udupi", API_KEY, session=session)
        assert session.calls[0]["timeout"] == weather_service.WEATHER_TIMEOUT_SECONDS

    def test_request_uses_metric_units_and_the_query_city(self) -> None:
        session = _FakeSession(payload=_ok_payload())
        get_weather("Mysuru", API_KEY, session=session)
        params = session.calls[0]["params"]
        assert params["q"] == "Mysuru"
        assert params["units"] == "metric"
        assert params["appid"] == API_KEY

    def test_as_dict_exposes_the_model_features(self) -> None:
        reading = get_weather("Udupi", API_KEY, session=_FakeSession(payload=_ok_payload()))
        assert set(reading.as_dict()) == {"temperature", "humidity", "rainfall"}


class TestWeatherDegradation:
    """Every failure mode must degrade to a usable mock, never an exception."""

    def test_missing_api_key_short_circuits_to_mock(self, monkeypatch) -> None:
        monkeypatch.setattr(weather_service, "OPENWEATHER_API_KEY", "")
        session = _FakeSession(payload=_ok_payload())
        reading = get_weather("Udupi", None, session=session)
        assert reading.source == "mock"
        assert not session.calls, "No network call should be made without a key"

    def test_empty_city_short_circuits_to_mock(self) -> None:
        assert get_weather("", API_KEY).source == "mock"

    @pytest.mark.parametrize(
        "error",
        [
            requests.exceptions.Timeout("timed out"),
            requests.exceptions.ConnectionError("dns failure"),
            requests.exceptions.RequestException("generic transport failure"),
        ],
    )
    def test_transport_failures_degrade_to_mock(self, error: Exception) -> None:
        reading = get_weather("Udupi", API_KEY, session=_FakeSession(error=error))
        assert reading.source == "mock"
        assert reading.message, "A fallback must explain itself"

    def test_malformed_json_degrades_to_mock(self) -> None:
        session = _FakeSession(payload=ValueError("not json"))
        reading = get_weather("Udupi", API_KEY, session=session)
        assert reading.source == "mock"
        assert "malformed" in reading.message.lower()

    def test_application_level_error_degrades_to_mock(self) -> None:
        payload = {"cod": "404", "message": "city not found"}
        reading = get_weather("Atlantis", API_KEY, session=_FakeSession(payload=payload))
        assert reading.source == "mock"
        assert "city not found" in reading.message

    def test_unexpected_schema_degrades_to_mock(self) -> None:
        payload = {"cod": 200, "name": "Udupi"}  # no "main" block
        reading = get_weather("Udupi", API_KEY, session=_FakeSession(payload=payload))
        assert reading.source == "mock"

    def test_mock_matches_configured_defaults(self) -> None:
        reading = get_weather("Udupi", None)
        assert reading.temperature == WEATHER_MOCK_DEFAULTS["temperature"]
        assert reading.humidity == WEATHER_MOCK_DEFAULTS["humidity"]
        assert reading.rainfall == WEATHER_MOCK_DEFAULTS["rainfall"]
        assert not reading.is_live

    def test_mock_values_are_within_the_physiological_envelope(self) -> None:
        reading = get_weather("Udupi", None)
        for feature, value in reading.as_dict().items():
            low, high = FEATURE_BOUNDS[feature]
            assert low <= value <= high


class TestWeatherCache:
    """The TTL cache must serve repeats without re-hitting the network."""

    def test_second_call_is_served_from_cache(self) -> None:
        session = _FakeSession(payload=_ok_payload())
        first = get_weather("Udupi", API_KEY, session=session)
        second = get_weather("Udupi", API_KEY, session=session)
        assert first.source == "api"
        assert second.source == "cache"
        assert len(session.calls) == 1

    def test_cache_is_case_insensitive(self) -> None:
        session = _FakeSession(payload=_ok_payload())
        get_weather("Udupi", API_KEY, session=session)
        assert get_weather("UDUPI", API_KEY, session=session).source == "cache"
        assert len(session.calls) == 1

    def test_cached_values_match_the_original(self) -> None:
        session = _FakeSession(payload=_ok_payload())
        first = get_weather("Udupi", API_KEY, session=session)
        second = get_weather("Udupi", API_KEY, session=session)
        assert second.as_dict() == first.as_dict()

    def test_cache_can_be_bypassed(self) -> None:
        session = _FakeSession(payload=_ok_payload())
        get_weather("Udupi", API_KEY, session=session)
        reading = get_weather("Udupi", API_KEY, session=session, use_cache=False)
        assert reading.source == "api"
        assert len(session.calls) == 2

    def test_clear_cache_forces_a_refetch(self) -> None:
        session = _FakeSession(payload=_ok_payload())
        get_weather("Udupi", API_KEY, session=session)
        clear_cache()
        assert get_weather("Udupi", API_KEY, session=session).source == "api"
        assert len(session.calls) == 2

    def test_distinct_cities_are_cached_separately(self) -> None:
        session = _FakeSession(payload=_ok_payload())
        get_weather("Udupi", API_KEY, session=session)
        get_weather("Mysuru", API_KEY, session=session)
        assert len(session.calls) == 2


# --------------------------------------------------------------------------- #
# Soil service
# --------------------------------------------------------------------------- #
class TestSoilBaselineRetrieval:
    """District resolution must always yield a usable, admissible prior."""

    def test_returns_a_soil_baseline(self) -> None:
        assert isinstance(get_district_baseline("Udupi"), SoilBaseline)

    @pytest.mark.parametrize("district", sorted(KARNATAKA_FALLBACK))
    def test_every_curated_zone_resolves(self, district: str) -> None:
        baseline = get_district_baseline(district)
        assert baseline.district == district
        for value in baseline.as_dict().values():
            assert value > 0

    @pytest.mark.parametrize(
        "query", ["udupi", "UDUPI", "  Udupi  ", "uDuPi"]
    )
    def test_matching_is_case_and_whitespace_insensitive(self, query: str) -> None:
        assert get_district_baseline(query).district == "Udupi"

    def test_unknown_district_falls_back_to_the_state_composite(self) -> None:
        baseline = get_district_baseline("Atlantis")
        assert baseline.source == "default"
        assert baseline.matched_on == "default"
        assert not baseline.is_survey_backed

    def test_empty_query_is_handled(self) -> None:
        baseline = get_district_baseline("")
        assert baseline.source == "default"
        assert "Karnataka" in baseline.district

    def test_baselines_respect_the_physiological_envelope(self) -> None:
        """A baseline that pre-fills the form must itself be a valid input."""
        for district in list_districts():
            baseline = get_district_baseline(district)
            for feature, value in baseline.as_dict().items():
                low, high = FEATURE_BOUNDS[feature]
                assert low <= value <= high, (
                    f"{district}.{feature}={value} escapes [{low}, {high}] and "
                    f"would be rejected by the inference validator"
                )

    def test_as_dict_keys_match_the_feature_contract(self) -> None:
        assert set(get_district_baseline("Udupi").as_dict()) == {"N", "P", "K", "ph"}

    def test_survey_backed_baselines_report_a_sample_count(self) -> None:
        survey = [
            b
            for b in (get_district_baseline(d) for d in list_districts())
            if b.is_survey_backed
        ]
        if not survey:
            pytest.skip("NFSM dataset unavailable in this checkout")
        assert all(b.sample_count > 0 for b in survey)

    def test_substring_query_resolves_a_composite_label(self) -> None:
        """'Challakere' should resolve even when labelled 'Chitradurga (Challakere)'."""
        districts = list_districts()
        composite = [d for d in districts if "(" in d]
        if not composite:
            pytest.skip("No composite district labels in this dataset")
        inner = composite[0].split("(")[1].rstrip(")")
        assert get_district_baseline(inner).is_survey_backed


class TestDistrictListing:
    """The selector must offer a stable, de-duplicated, sorted list."""

    def test_listing_is_sorted_and_unique(self) -> None:
        districts = list_districts()
        assert districts == sorted(districts)
        assert len(districts) == len(set(districts))

    def test_listing_includes_the_curated_zones(self) -> None:
        assert set(KARNATAKA_FALLBACK).issubset(set(list_districts()))

    def test_listing_can_exclude_the_fallback_table(self) -> None:
        assert len(list_districts(include_fallback=False)) <= len(list_districts())

    def test_every_listed_district_resolves(self) -> None:
        for district in list_districts():
            assert get_district_baseline(district).as_dict()
