"""
Phase 4.2a -- End-to-end pipeline tests.

Drives the Phase 3 conflict scenarios through the REAL run_pipeline() flow
(state-mapping functions included), not just resolve_conflict() directly
(that's already covered by test_conflict_resolver.py, 12/12 passing).

Weather, Price, and Disease agents are mocked with synthetic raw output --
their live API calls are exercised separately by test_weather_agent.py /
test_price_agent.py, and real external calls would make this suite flaky
and untestable for specific scenarios on demand.

KNOWN GAP (not hidden): 3 of the 12 Phase 3 scenarios (ids 4, 6, 7) require
weather_state="drought_risk", which _map_weather_state() can never produce --
the Weather Agent has no drought detection, only a 48h rain forecast. These
3 scenarios are SKIPPED here with an explicit reason, not silently mocked
around, since forcing them to "pass" would misrepresent what the pipeline
can actually do today. Revisit if/when real drought detection is built.
"""

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.pipeline import run_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "conflict_scenarios.json"), encoding="utf-8") as f:
    SCENARIOS = json.load(f)

UNREACHABLE_IDS = {4, 6, 7}  # require drought_risk -- see module docstring

TEST_PROFILE = {
    "district": "Warangal",
    "mandi": "Warangal APMC",
    "state": "Telangana",
    "crop": "Tomato",
}


def _mock_disease_output(state):
    """Build a synthetic Disease Agent output that _map_disease_state()
    will map to the given target state."""
    if state == "treat_now":
        return {"predicted_class": "Tomato_Late_blight", "confidence": 0.95}
    if state == "monitor":
        return {"predicted_class": "Tomato_Late_blight", "confidence": 0.5}
    if state == "no_action":
        return {"predicted_class": "Tomato_healthy", "confidence": 0.9}
    return None  # disease_state is None -> intent won't request disease


def _mock_weather_output(state):
    """Build a synthetic Weather Agent output that _map_weather_state()
    will map to the given target state. drought_risk is intentionally
    NOT supported here -- see module docstring."""
    if state == "rain_risk":
        return {"rain_expected": True, "max_pop": 0.8}
    if state == "favorable":
        return {"rain_expected": False, "max_pop": 0.1}
    if state == "uncertain":
        return {}  # simulates API failure -- no rain_expected key
    return None


def _mock_price_output(state):
    """Build a synthetic Price Agent output that _map_price_state()
    will map to the given target state."""
    if state == "sell_now":
        return {"pct_diff": 0.15}
    if state == "hold":
        return {"pct_diff": -0.15}
    if state == "neutral":
        return {"pct_diff": 0.0}
    return None


def run_scenario(scenario):
    inputs = scenario["inputs"]
    disease_state = inputs["disease_state"]
    weather_state = inputs["weather_state"]
    price_state = inputs["price_state"]

    wants_disease = disease_state is not None
    wants_weather = weather_state is not None
    wants_price = price_state is not None

    mock_intents = {
        "wants_disease": wants_disease,
        "wants_weather": wants_weather,
        "wants_price": wants_price,
        "wants_soil": False,
    }

    disease_raw = _mock_disease_output(disease_state)
    weather_raw = _mock_weather_output(weather_state)
    price_raw = _mock_price_output(price_state)

    with patch("orchestrator.pipeline.route_intent", return_value=mock_intents), \
         patch("orchestrator.pipeline.predict_disease", return_value=disease_raw), \
         patch("orchestrator.pipeline.get_weather_advice", return_value=weather_raw), \
         patch("orchestrator.pipeline.get_price_advice", return_value=price_raw):

        image_path = "/fake/path.jpg" if wants_disease else None
        result = run_pipeline(
            question=scenario["description"],
            farm_profile=TEST_PROFILE,
            image_path=image_path,
        )
    return result


def main():
    passed, failed, skipped = 0, 0, 0

    for scenario in SCENARIOS:
        sid = scenario["id"]

        if sid in UNREACHABLE_IDS:
            print(f"[SKIP] Scenario {sid}: requires drought_risk, "
                  f"not producible by current Weather Agent")
            skipped += 1
            continue

        result = run_scenario(scenario)
        expected = scenario["expected"]
        actual = result["resolution"]

        ok = (
            actual["resolution"] == expected["resolution"]
            and actual["confidence"] == expected["confidence"]
            and actual["is_conflict"] == expected["is_conflict"]
        )

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] Scenario {sid}: {scenario['description']}")
        if not ok:
            print(f"         expected: {expected}")
            print(f"         actual:   {actual}")

        if ok:
            passed += 1
        else:
            failed += 1

    total_run = passed + failed
    print(f"\n{passed}/{total_run} passed ({skipped} skipped -- see module docstring)")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
