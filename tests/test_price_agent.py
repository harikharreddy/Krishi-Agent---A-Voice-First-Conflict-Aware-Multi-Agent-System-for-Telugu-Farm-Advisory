import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.price.price_agent import get_price_advice

REQUIRED_KEYS = {"answer", "confidence", "reason_for_confidence", "source_freshness"}
VALID_CONFIDENCE = {"High", "Medium", "Low"}


def test_returns_required_schema_keys():
    result = get_price_advice("Telangana", "Tomato")
    assert REQUIRED_KEYS.issubset(result.keys())


def test_confidence_is_valid_value():
    result = get_price_advice("Telangana", "Tomato")
    assert result["confidence"] in VALID_CONFIDENCE


def test_answer_is_nonempty_string():
    result = get_price_advice("Telangana", "Tomato")
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0


def test_specific_market_lookup_works():
    result = get_price_advice("Telangana", "Tomato", market="Warangal APMC")
    assert REQUIRED_KEYS.issubset(result.keys())
    assert result["confidence"] in VALID_CONFIDENCE


def test_nonexistent_commodity_handled_gracefully():
    result = get_price_advice("Telangana", "ZzzNotARealCrop123")
    assert REQUIRED_KEYS.issubset(result.keys())
    assert result["confidence"] in VALID_CONFIDENCE
