import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from orchestrator.conflict_resolver import resolve_conflict

def load_test_set():
    path = os.path.join(os.path.dirname(__file__), "conflict_scenarios.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    test_cases = load_test_set()
    correct = 0
    total = len(test_cases)

    for case in test_cases:
        inputs = case["inputs"]
        expected = case["expected"]
        actual = resolve_conflict(
            disease_state=inputs.get("disease_state"),
            weather_state=inputs.get("weather_state"),
            price_state=inputs.get("price_state"),
        )

        keys = ["is_conflict", "resolution", "confidence"]
        match = all(actual.get(k) == expected.get(k) for k in keys)

        status = "PASS" if match else "FAIL"
        if match:
            correct += 1

        print(f"[{status}] #{case['id']}: {case['description']}")
        if not match:
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")

    print(f"\n{correct}/{total} correct ({100*correct/total:.1f}%)")

if __name__ == "__main__":
    main()
