"""
Phase 4.2b -- Edge case tests for run_pipeline().

Covers the 3 edge cases named in the project guide's Phase 4.2 step:
  1. No photo attached (disease intent detected, but image_path=None)
  2. Missing/incomplete Farm Profile fields
  3. Agent API failure (weather/price agent raises or returns a failure shape)

All cases assert the pipeline degrades gracefully -- returns SOME answer,
never raises an uncaught exception. A crash here means a farmer's whole
question fails silently on the frontend, which is worse than a low-confidence
"unresolved" answer.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.pipeline import run_pipeline

FULL_PROFILE = {
    "district": "Warangal",
    "mandi": "Warangal APMC",
    "state": "Telangana",
    "crop": "Tomato",
}


def check(name, fn):
    try:
        result = fn()
        assert isinstance(result, dict) and "final_answer" in result and result["final_answer"]
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def test_no_photo_attached():
    """Disease intent detected but image_path=None -- Disease Agent should be
    skipped, not crash trying to run inference on nothing."""
    mock_intents = {"wants_disease": True, "wants_weather": False,
                     "wants_price": False, "wants_soil": False}
    with patch("orchestrator.pipeline.route_intent", return_value=mock_intents):
        result = run_pipeline("నా ఆకులపై మచ్చలు ఉన్నాయి", FULL_PROFILE, image_path=None)
    assert result["disease_raw"] is None, "Disease Agent should not have run"
    return result


def test_missing_farm_profile_fields():
    """Farm Profile missing 'state' -- Price Agent should be skipped with a
    warning, not raise KeyError (real bug found and fixed in Phase 4.2b)."""
    incomplete_profile = {"district": "Warangal", "crop": "Tomato"}  # no mandi, no state
    mock_intents = {"wants_disease": False, "wants_weather": False,
                     "wants_price": True, "wants_soil": False}
    with patch("orchestrator.pipeline.route_intent", return_value=mock_intents):
        result = run_pipeline("ఈ రోజు టమాటా ధర ఎంత?", incomplete_profile, image_path=None)
    assert result["price_raw"] is None, "Price Agent should not have run with incomplete profile"
    return result


def test_empty_farm_profile():
    """Completely empty Farm Profile -- every agent that needs profile data
    should be skipped, still no crash."""
    mock_intents = {"wants_disease": False, "wants_weather": True,
                     "wants_price": True, "wants_soil": False}
    with patch("orchestrator.pipeline.route_intent", return_value=mock_intents):
        result = run_pipeline("వాతావరణం ఎలా ఉంది?", {}, image_path=None)
    assert result["weather_raw"] is None
    assert result["price_raw"] is None
    return result


def test_weather_agent_api_failure():
    """Weather Agent's own internal try/except catches API failures and
    returns a Low-confidence dict (see weather_agent.py) rather than raising --
    confirm the pipeline handles that shape correctly end-to-end."""
    mock_intents = {"wants_disease": False, "wants_weather": True,
                     "wants_price": False, "wants_soil": False}
    api_failure_response = {
        "answer": "వాతావరణ సమాచారం ప్రస్తుతం అందుబాటులో లేదు.",
        "confidence": "Low",
        "reason_for_confidence": "API call failed: simulated timeout",
        "source_freshness": "unavailable",
        # no "rain_expected" key -- matches the real failure shape
    }
    with patch("orchestrator.pipeline.route_intent", return_value=mock_intents), \
         patch("orchestrator.pipeline.get_weather_advice", return_value=api_failure_response):
        result = run_pipeline("వర్షం పడుతుందా?", FULL_PROFILE, image_path=None)
    assert result["weather_state"] == "uncertain", "API failure should map to uncertain, not crash"
    return result


def test_price_agent_raises_exception():
    """Unlike Weather Agent, Price Agent's _fetch_today_price() can raise
    RuntimeError directly (e.g. missing API key) rather than returning a
    failure dict -- confirm this doesn't crash the whole pipeline."""
    mock_intents = {"wants_disease": False, "wants_weather": False,
                     "wants_price": True, "wants_soil": False}
    with patch("orchestrator.pipeline.route_intent", return_value=mock_intents), \
         patch("orchestrator.pipeline.get_price_advice", side_effect=RuntimeError("simulated missing API key")):
        try:
            result = run_pipeline("ధర ఎంత?", FULL_PROFILE, image_path=None)
            return result
        except RuntimeError:
            raise AssertionError(
                "Pipeline crashed on Price Agent RuntimeError -- not caught anywhere"
            )


def main():
    results = [
        check("No photo attached", test_no_photo_attached),
        check("Missing Farm Profile fields (state)", test_missing_farm_profile_fields),
        check("Completely empty Farm Profile", test_empty_farm_profile),
        check("Weather Agent API failure shape", test_weather_agent_api_failure),
        check("Price Agent raises exception", test_price_agent_raises_exception),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
