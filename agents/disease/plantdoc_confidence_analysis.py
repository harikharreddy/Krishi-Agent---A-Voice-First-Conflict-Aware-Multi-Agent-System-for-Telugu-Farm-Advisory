"""
Confidence-calibration spot-check (Phase 8, evaluation metric #6):
is the Disease Agent's 0.7 treat_now/monitor cutoff actually supported
by real accuracy-vs-confidence data on PlantDoc?
"""

import json
import os

IN_PATH = os.path.join(os.path.dirname(__file__), "results", "plantdoc_perimage_confidence.json")

BUCKETS = [
    (0.9, 1.01, "0.9 - 1.0"),
    (0.8, 0.9, "0.8 - 0.9"),
    (0.7, 0.8, "0.7 - 0.8"),
    (0.6, 0.7, "0.6 - 0.7"),
    (0.5, 0.6, "0.5 - 0.6"),
    (0.0, 0.5, "below 0.5"),
]

def main():
    with open(IN_PATH) as f:
        records = json.load(f)

    print(f"Total records: {len(records)}\n")
    print(f"{'Confidence range':<15} {'Count':>7} {'Correct':>8} {'Accuracy':>9}")
    print("-" * 42)

    for lo, hi, label in BUCKETS:
        bucket = [r for r in records if lo <= r["confidence"] < hi]
        count = len(bucket)
        correct = sum(1 for r in bucket if r["correct"])
        acc = correct / count if count else float("nan")
        print(f"{label:<15} {count:>7} {correct:>8} {acc:>9.2%}")

    # Direct check of the 0.7 cutoff specifically
    above = [r for r in records if r["confidence"] >= 0.7]
    below = [r for r in records if r["confidence"] < 0.7]

    above_acc = sum(1 for r in above if r["correct"]) / len(above) if above else float("nan")
    below_acc = sum(1 for r in below if r["correct"]) / len(below) if below else float("nan")

    print("\n--- 0.7 cutoff check (treat_now vs monitor) ---")
    print(f"Confidence >= 0.7 ('treat_now'): {len(above)} images, accuracy {above_acc:.2%}")
    print(f"Confidence <  0.7 ('monitor'):   {len(below)} images, accuracy {below_acc:.2%}")

    if above and below:
        gap = above_acc - below_acc
        print(f"\nAccuracy gap (above minus below): {gap:+.2%}")
        if gap > 0.10:
            print("=> 0.7 cutoff shows a meaningful, positive separation. Reasonably supported by data.")
        elif gap > 0:
            print("=> 0.7 cutoff shows a small positive separation, but weak. Marginal support.")
        else:
            print("=> 0.7 cutoff shows NO positive separation (higher confidence is not more accurate here).")
            print("   This is a real, citable finding: the confidence score is poorly calibrated on")
            print("   real-world (PlantDoc) images, even though it may be well-calibrated on PlantVillage.")

if __name__ == "__main__":
    main()
