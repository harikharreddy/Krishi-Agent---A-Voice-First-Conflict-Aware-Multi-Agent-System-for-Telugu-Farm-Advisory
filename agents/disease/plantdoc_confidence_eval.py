"""
Standalone script: runs the trained hierarchical Disease Agent over the
PlantDoc dataset, logging per-image confidence (not just aggregate accuracy),
so we can check whether the 0.7 treat_now/monitor cutoff is actually
supported by the data. Reuses disease_agent.py's predict_disease() directly
-- same models, same confidence definition, no retraining.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from disease_agent import predict_disease

PLANTDOC_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "plantdoc_raw"
)

# Same mapping the Phase 1 notebook used: PlantDoc's folder names -> our 13-class labels
PLANTDOC_MAP = {
    "Tomato leaf": "Tomato_healthy",
    "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato leaf bacterial spot": "Tomato_Bacterial_spot",
    "Tomato leaf late blight": "Tomato_Late_blight",
    "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato mold leaf": "Tomato_Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
}

def main():
    records = []
    total, correct = 0, 0
    missing_dirs = []

    for split in ["train", "test"]:
        for plantdoc_class, true_label in PLANTDOC_MAP.items():
            src_dir = os.path.join(PLANTDOC_ROOT, split, plantdoc_class)
            if not os.path.isdir(src_dir):
                missing_dirs.append(src_dir)
                continue

            for fname in os.listdir(src_dir):
                fpath = os.path.join(src_dir, fname)
                if not os.path.isfile(fpath):
                    continue

                result = predict_disease(fpath)
                pred_class = result.get("predicted_class")
                pred_crop = result.get("predicted_crop")
                confidence = result.get("confidence", 0.0)

                is_correct = (pred_class == true_label)
                total += 1
                correct += int(is_correct)

                true_crop = "Tomato" if true_label.startswith("Tomato") else "Potato"

                records.append({
                    "image": f"{split}/{plantdoc_class}/{fname}",
                    "true_class": true_label,
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "correct": is_correct,
                    "stage1_correct_crop": (pred_crop == true_crop),
                })

                if total % 50 == 0:
                    print(f"...{total} images processed, running acc: {correct/total:.4f}")

    if missing_dirs:
        print(f"\nWARNING: {len(missing_dirs)} expected folders were missing, e.g.:")
        for d in missing_dirs[:5]:
            print(f"  {d}")

    acc = correct / total if total else 0.0
    print(f"\nTotal images evaluated: {total}")
    print(f"Hierarchical accuracy: {acc:.4f} ({correct}/{total})")

    out_path = os.path.join(os.path.dirname(__file__), "results", "plantdoc_perimage_confidence.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved {len(records)} per-image records to {out_path}")


if __name__ == "__main__":
    main()
