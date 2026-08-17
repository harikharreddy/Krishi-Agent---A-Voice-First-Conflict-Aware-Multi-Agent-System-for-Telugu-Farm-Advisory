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
