"""
Phase 8.1, evaluation metric #5 (Conflict-rule table pass rate): packages
tests/test_pipeline.py's 12 scenarios into a clean, citable JSON table --
scenario, inputs, expected vs. actual resolution, and pass/fail -- derived
directly from a live run (not hand-transcribed), so the table can't drift
out of sync with the actual test.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from orchestrator.pipeline import run_pipeline
from test_pipeline import (
    SCENARIOS, TEST_PROFILE, _mock_disease_output, _mock_weather_output, _mock_price_output,
)


def run_scenario_with_trace(scenario):
    inputs = scenario["inputs"]
    disease_state, weather_state, price_state = inputs["disease_state"], inputs["weather_state"], inputs["price_state"]
    mock_intents = {
        "wants_disease": disease_state is not None,
        "wants_weather": weather_state is not None,
        "wants_price": price_state is not None,
        "wants_soil": False,
    }
    disease_raw = _mock_disease_output(disease_state)
    weather_raw = _mock_weather_output(weather_state)
    price_raw = _mock_price_output(price_state)

    with patch("orchestrator.pipeline.route_intent", return_value=mock_intents), \
         patch("orchestrator.pipeline.predict_disease", return_value=disease_raw), \
         patch("orchestrator.pipeline.get_weather_advice", return_value=weather_raw), \
         patch("orchestrator.pipeline.get_price_advice", return_value=price_raw):
        image_path = "/fake/path.jpg" if mock_intents["wants_disease"] else None
        result = run_pipeline(question=scenario["description"], farm_profile=TEST_PROFILE, image_path=image_path)
    return result


def main():
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(HERE)).decode().strip()
    rows = []
    passed = 0

    for scenario in SCENARIOS:
        result = run_scenario_with_trace(scenario)
        expected = scenario["expected"]
        actual = result["resolution"]
        ok = (
            actual["resolution"] == expected["resolution"]
            and actual["confidence"] == expected["confidence"]
            and actual["is_conflict"] == expected["is_conflict"]
        )
        passed += int(ok)
        rows.append({
            "id": scenario["id"],
            "description": scenario["description"],
            "inputs": scenario["inputs"],
            "expected": expected,
            "actual": actual,
            "final_answer_telugu": result["final_answer"],
            "pass": ok,
            "notes": scenario.get("notes"),
        })

    evidence = {
        "metric": "Conflict Resolver rule-table pass rate",
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "git_commit": git_commit,
        "method": (
            "tests/test_pipeline.py -- drives 12 hand-designed scenarios through "
            "the REAL run_pipeline() (state-mapping + resolve_conflict() + "
            "phrase_resolution() all real, unmocked), with only the 4 external "
            "calls (route_intent, predict_disease, get_weather_advice, "
            "get_price_advice) replaced by unittest.mock.patch to inject "
            "controlled synthetic agent states. Exact-match scoring on "
            "(resolution, confidence, is_conflict) against a hand-authored "
            "expected outcome per scenario (tests/conflict_scenarios.json)."
        ),
        "result": f"{passed}/{len(rows)} passed",
        "scenarios": rows,
        "honest_gaps": [
            "These are SYNTHETIC agent outputs (unittest.mock.patch), not live "
            "API data -- this validates the resolution LOGIC's correctness "
            "against its own design spec, not real-world scenario coverage or "
            "how often each scenario type actually occurs for real farmers.",

            "100% here means 100% of the 12 scenarios this test was DESIGNED "
            "to cover -- it does not mean the rule table has zero gaps. A "
            "real, separately-discovered gap exists: (None, rain_risk, "
            "sell_now) has no rule table entry (only (None, rain_risk, hold) "
            "does), found via live-data pipeline characterization in "
            "docs/evaluation_and_validation.md Sec. 2.7, not by this test. "
            "That gap involves a state combination none of these 12 hand-"
            "designed scenarios happens to exercise.",

            "The drought_risk state depends on _check_drought_signal(), which "
            "is an explicitly documented short-range PROXY (uniformly dry+hot "
            "in a 5-day forecast) -- NOT true multi-week drought detection, "
            "which a 5-day forecast cannot observe. Scenarios 4, 6, and 7 "
            "passing means the CONFLICT-RESOLUTION LOGIC correctly handles a "
            "drought_risk signal when one is present, not that the underlying "
            "drought DETECTION itself is validated as accurate.",
        ],
    }

    evidence_dir = os.path.join(os.path.dirname(HERE), "docs", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    out_path = os.path.join(evidence_dir, "metric5_conflict_rule_table_evidence.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    print(f"{passed}/{len(rows)} passed")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
