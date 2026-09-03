import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.weather.weather_agent import get_weather_advice

REQUIRED_KEYS = {"answer", "confidence", "reason_for_confidence", "source_freshness"}
VALID_CONFIDENCE = {"High", "Medium", "Low"}


def test_returns_required_schema_keys():
    result = get_weather_advice("Hyderabad,IN")
    assert REQUIRED_KEYS.issubset(result.keys())


def test_confidence_is_valid_value():
    result = get_weather_advice("Hyderabad,IN")
    assert result["confidence"] in VALID_CONFIDENCE


def test_answer_is_nonempty_string():
    result = get_weather_advice("Hyderabad,IN")
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0


def test_invalid_location_handled_gracefully():
    result = get_weather_advice("ZzzNotARealPlace123")
    assert REQUIRED_KEYS.issubset(result.keys())
    assert result["confidence"] in VALID_CONFIDENCE


# --- Phase 8 prep: drought signal unit tests (synthetic data, no live API call) ---
from agents.weather.weather_agent import _check_drought_signal, DROUGHT_POP_THRESHOLD, DROUGHT_TEMP_THRESHOLD
from datetime import datetime, timezone, timedelta


def _fake_forecast(daily_pop_temp_pairs):
    """Builds a synthetic OpenWeatherMap-shaped forecast dict.
    daily_pop_temp_pairs: list of (pop, temp) tuples, one per day, applied
    to 3 entries that day (mimicking the real 3-hour-interval structure)."""
    now = datetime.now(timezone.utc)
    entries = []
    for day_idx, (pop, temp) in enumerate(daily_pop_temp_pairs):
        for hour_offset in [0, 8, 16]:
            entry_time = now + timedelta(days=day_idx, hours=hour_offset)
            entries.append({
                "dt": int(entry_time.timestamp()),
                "pop": pop,
                "main": {"temp": temp},
            })
    return {"list": entries}


def test_drought_signal_fires_when_uniformly_dry_and_hot():
    # 5 days, all dry (pop below threshold) and hot (temp above threshold)
    forecast = _fake_forecast([(0.05, 38.0)] * 5)
    assert _check_drought_signal(forecast) is True


def test_drought_signal_does_not_fire_when_any_day_has_rain():
    # 4 dry/hot days + 1 day with real rain probability
    forecast = _fake_forecast([(0.05, 38.0)] * 4 + [(0.8, 30.0)])
    assert _check_drought_signal(forecast) is False


def test_drought_signal_does_not_fire_when_dry_but_not_hot():
    # Dry every day, but temperature stays below the drought threshold
    forecast = _fake_forecast([(0.05, 22.0)] * 5)
    assert _check_drought_signal(forecast) is False


def test_drought_signal_does_not_fire_on_empty_forecast():
    forecast = {"list": []}
    assert _check_drought_signal(forecast) is False


def test_drought_signal_thresholds_are_as_documented():
    # Sanity-check the constants themselves haven't silently drifted
    assert DROUGHT_POP_THRESHOLD == 0.2
    assert DROUGHT_TEMP_THRESHOLD == 35.0
