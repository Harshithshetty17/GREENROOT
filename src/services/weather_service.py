"""OpenWeatherMap ingestion client with bounded latency and offline degradation.

The dashboard runs in rural-connectivity conditions where an upstream call may
hang, rate-limit, or fail DNS resolution entirely. This client therefore treats
the network as *optional*: every failure mode collapses into a structured
:class:`WeatherReading` carrying ``source='mock'`` and a human-readable reason,
so the recommendation pipeline downstream never blocks and never raises.

Guarantees
----------
* Every request is bounded by :data:`~src.core.config.WEATHER_TIMEOUT_SECONDS`.
* Successful responses are memoised for
  :data:`~src.core.config.WEATHER_CACHE_TTL_SECONDS` under a thread-safe lock,
  which keeps the free API tier within its call budget.
* ``get_weather`` never propagates an exception to the caller.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import requests

from src.core.config import (
    OPENWEATHER_API_KEY,
    OPENWEATHER_ENDPOINT,
    WEATHER_CACHE_TTL_SECONDS,
    WEATHER_MOCK_DEFAULTS,
    WEATHER_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

#: Hours of precipitation reported by OWM's ``rain.1h`` field, extrapolated to a
#: nominal 30-day monsoon accumulation to match the units of the benchmark
#: ``rainfall`` feature (millimetres per growing month).
_RAIN_EXTRAPOLATION_HOURS: int = 24 * 30

_CACHE: Dict[str, Tuple[float, "WeatherReading"]] = {}
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class WeatherReading:
    """A normalised microclimate observation.

    Attributes
    ----------
    temperature:
        Ambient air temperature in °C.
    humidity:
        Relative humidity as a percentage.
    rainfall:
        Monthly-equivalent accumulation in millimetres.
    source:
        ``'api'`` for a live upstream response, ``'cache'`` for a memoised one,
        ``'mock'`` for the deterministic offline fallback.
    city:
        The location the reading is attributed to.
    message:
        Diagnostic detail. Empty on success; the failure reason on fallback.
    raw:
        Untouched upstream payload, retained for auditability.
    """

    temperature: float
    humidity: float
    rainfall: float
    source: str = "mock"
    city: str = ""
    message: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_live(self) -> bool:
        """``True`` when the reading originated from the upstream API."""
        return self.source in {"api", "cache"}

    def as_dict(self) -> Dict[str, float]:
        """Return the three model-relevant fields keyed by feature name."""
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "rainfall": self.rainfall,
        }


def _mock_reading(city: str, message: str) -> WeatherReading:
    """Build the deterministic offline fallback reading."""
    return WeatherReading(
        temperature=float(WEATHER_MOCK_DEFAULTS["temperature"]),
        humidity=float(WEATHER_MOCK_DEFAULTS["humidity"]),
        rainfall=float(WEATHER_MOCK_DEFAULTS["rainfall"]),
        source="mock",
        city=city,
        message=message,
    )


def _parse_payload(payload: Dict[str, Any], city: str) -> WeatherReading:
    """Map an OpenWeatherMap ``/weather`` payload onto a :class:`WeatherReading`.

    Raises
    ------
    KeyError, TypeError, ValueError
        If the payload does not carry the expected ``main`` block; the caller
        converts these into a mock fallback.
    """
    main = payload["main"]
    precipitation = payload.get("rain") or {}
    hourly_mm = precipitation.get("1h", precipitation.get("3h", 0.0)) or 0.0

    return WeatherReading(
        temperature=float(main["temp"]),
        humidity=float(main["humidity"]),
        rainfall=round(float(hourly_mm) * _RAIN_EXTRAPOLATION_HOURS, 2),
        source="api",
        city=str(payload.get("name") or city),
        message="",
        raw=payload,
    )


def get_weather(
    city: str,
    api_key: Optional[str] = None,
    *,
    timeout: float = WEATHER_TIMEOUT_SECONDS,
    use_cache: bool = True,
    session: Optional[requests.Session] = None,
) -> WeatherReading:
    """Fetch current conditions for ``city``, degrading gracefully on failure.

    Parameters
    ----------
    city:
        Location query, e.g. ``"Udupi"`` or ``"Mysuru,IN"``.
    api_key:
        OpenWeatherMap key. Falls back to the ``OPENWEATHER_API_KEY``
        environment variable; if neither is present the mock reading is
        returned immediately without touching the network.
    timeout:
        Per-request ceiling in seconds.
    use_cache:
        Consult and populate the TTL cache. Disable for freshness-critical use.
    session:
        Optional :class:`requests.Session`, chiefly a test seam.

    Returns
    -------
    WeatherReading
        Always a valid reading — live, cached, or mocked. Inspect
        :attr:`WeatherReading.is_live` to distinguish, and
        :attr:`WeatherReading.message` for the fallback reason.
    """
    city = (city or "").strip()
    if not city:
        return _mock_reading(city, "No location supplied; using offline defaults.")

    key = (api_key or OPENWEATHER_API_KEY or "").strip()
    if not key:
        return _mock_reading(
            city, "No API key configured; using calibrated offline defaults."
        )

    cache_key = city.lower()
    if use_cache:
        with _CACHE_LOCK:
            entry = _CACHE.get(cache_key)
        if entry is not None:
            cached_at, reading = entry
            if time.monotonic() - cached_at < WEATHER_CACHE_TTL_SECONDS:
                logger.debug("Weather cache hit for %s", city)
                return WeatherReading(
                    temperature=reading.temperature,
                    humidity=reading.humidity,
                    rainfall=reading.rainfall,
                    source="cache",
                    city=reading.city,
                    message="Served from local cache.",
                    raw=reading.raw,
                )

    params = {"q": city, "appid": key, "units": "metric"}
    getter = session.get if session is not None else requests.get

    try:
        response = getter(OPENWEATHER_ENDPOINT, params=params, timeout=timeout)
        payload = response.json()
    except requests.exceptions.Timeout:
        return _mock_reading(city, f"Upstream timed out after {timeout:.0f}s.")
    except requests.exceptions.RequestException as exc:
        return _mock_reading(city, f"Network unavailable ({type(exc).__name__}).")
    except ValueError:
        return _mock_reading(city, "Upstream returned a malformed response.")

    # OWM signals application-level errors in the body, not only via HTTP status.
    status = payload.get("cod") if isinstance(payload, dict) else None
    if str(status) != "200":
        detail = ""
        if isinstance(payload, dict):
            detail = str(payload.get("message", "")).strip()
        return _mock_reading(city, detail or f"Upstream rejected the request ({status}).")

    try:
        reading = _parse_payload(payload, city)
    except (KeyError, TypeError, ValueError) as exc:
        return _mock_reading(city, f"Unexpected payload schema ({exc}).")

    if use_cache:
        with _CACHE_LOCK:
            _CACHE[cache_key] = (time.monotonic(), reading)
    return reading


def clear_cache() -> None:
    """Evict every memoised weather response."""
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = ["WeatherReading", "get_weather", "clear_cache"]
