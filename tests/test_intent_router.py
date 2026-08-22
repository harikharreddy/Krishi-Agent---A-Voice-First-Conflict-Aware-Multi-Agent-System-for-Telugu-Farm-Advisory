import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from orchestrator.intent_router import route_intent

def load_test_set():
    path = os.path.join(os.path.dirname(__file__), "intent_router_test_set.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    test_cases = load_test_set()
    correct = 0
    total = len(test_cases)

    for case in test_cases:
        question = case["question"]
        expected = case["expected"]
        actual = route_intent(question)

        keys = ["wants_disease", "wants_weather", "wants_price", "wants_soil"]
        match = all(actual.get(k) == expected.get(k) for k in keys)

        status = "PASS" if match else "FAIL"
        if match:
            correct += 1

        print(f"[{status}] #{case['id']}: {case['gloss']}")
        if not match:
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")

    print(f"\n{correct}/{total} correct ({100*correct/total:.1f}%)")

if __name__ == "__main__":
    main()
