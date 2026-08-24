import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from orchestrator.conflict_resolver import resolve_conflict
from orchestrator.phrasing_templates import phrase_resolution, FALLBACK_PHRASE

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
        resolved = resolve_conflict(
            disease_state=inputs.get("disease_state"),
            weather_state=inputs.get("weather_state"),
            price_state=inputs.get("price_state"),
        )
        sentence = phrase_resolution(resolved["resolution"], resolved["confidence"])

        # A scenario "passes" here if it produces a non-empty sentence,
        # and only hits FALLBACK_PHRASE when the resolution itself was
        # unresolved_conflict (i.e. fallback is expected, not a bug).
        is_empty_or_broken = not sentence or "{" in sentence
        unexpected_fallback = (
            sentence == FALLBACK_PHRASE and resolved["resolution"] != "unresolved_conflict"
        )
        match = not is_empty_or_broken and not unexpected_fallback

        status = "PASS" if match else "FAIL"
        if match:
            correct += 1

        print(f"[{status}] #{case['id']}: {sentence}")
        if not match:
            print(f"    resolution: {resolved['resolution']}, confidence: {resolved['confidence']}")

    print(f"\n{correct}/{total} correct ({100*correct/total:.1f}%)")

if __name__ == "__main__":
    main()
