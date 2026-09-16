"""
Phase 8.1, model lineage row 1 close-out (2026-09-16): PlantDoc
evaluation of the flat baseline checkpoint
(agents/disease/checkpoints/flat_baseline_best.pt), which had only an
aggregate accuracy (22.00%, n=968) on record -- no per-image predictions
existed anywhere in this repo, so no confusion matrix / per-class P/R/F1
was ever computable for it, unlike rows 2 and 6. Checked whether this
gap was genuinely closeable (the checkpoint and raw PlantDoc data both
exist) rather than assumed unrecoverable, per the same standard applied
to every other "is this actually true" check this project runs.

Architecture: flat 13-class EfficientNetB0, identical head shape/class
list to row 6's v2 checkpoint (agents/disease/results/
flat_baseline_results.json's own "classes" field -- verified in-file,
not assumed -- and already the canonical 13-class ordering this session
uses everywhere; row 1's checkpoint is a bare state_dict, not a
dict-with-metadata like v2's, so this file supplies the class order
instead of an embedded class_names list). Uses PLANTDOC_MAP from
plantdoc_confidence_eval.py (the SAME mapping row 2 uses) rather than
PLANTDOC_MAP_V2, since row 1 already uses the canonical single/double-
underscore naming convention -- no cross-checkpoint name translation
needed here, unlike row 6.

Zero-shot, train+test combined is the valid comparison (same reasoning
as row 6): this checkpoint was trained on PlantVillage only, never
touched PlantDoc, so no train/test contamination risk exists -- verified
below via the same train-vs-test accuracy gap check used for every other
checkpoint this session, not assumed.
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from plantdoc_confidence_eval import PLANTDOC_MAP

CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "flat_baseline_best.pt")
CLASSES_SOURCE_PATH = os.path.join(HERE, "results", "flat_baseline_results.json")
PLANTDOC_ROOT = os.path.join(ROOT, "data", "plantdoc_raw")
EVIDENCE_DIR = os.path.join(ROOT, "docs", "evidence")

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_flat_baseline_model():
    with open(CLASSES_SOURCE_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta["classes"]
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, class_names


def main():
    model, class_names = load_flat_baseline_model()
    print(f"Loaded row 1 (flat baseline) checkpoint: {len(class_names)} classes, device={device}")
    print(f"class_names (from flat_baseline_results.json): {class_names}")

    records = []
    missing_dirs = []
    for split in ["train", "test"]:
        for plantdoc_class, true_label in PLANTDOC_MAP.items():
            src_dir = os.path.join(PLANTDOC_ROOT, split, plantdoc_class)
            if not os.path.isdir(src_dir):
                missing_dirs.append(src_dir)
                continue
            for fname in sorted(os.listdir(src_dir)):
                fpath = os.path.join(src_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                img = Image.open(fpath).convert("RGB")
                x = eval_transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = F.softmax(model(x), dim=1)[0]
                pred_idx = probs.argmax().item()
                pred_class = class_names[pred_idx]
                confidence = probs[pred_idx].item()
                records.append({
                    "image": f"{split}/{plantdoc_class}/{fname}",
                    "split": split,
                    "true_class": true_label,
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "correct": pred_class == true_label,
                })

    if missing_dirs:
        print(f"WARNING: {len(missing_dirs)} expected folders missing, e.g. {missing_dirs[0]}")

    n = len(records)
    correct = sum(r["correct"] for r in records)
    train_records = [r for r in records if r["split"] == "train"]
    test_records = [r for r in records if r["split"] == "test"]
    train_acc = sum(r["correct"] for r in train_records) / len(train_records) if train_records else None
    test_acc = sum(r["correct"] for r in test_records) / len(test_records) if test_records else None
    overall_acc = correct / n if n else None

    # Contamination sanity check -- same logic every other checkpoint this
    # session was checked with: a POSITIVE train-over-test gap would be the
    # signature of contamination (the model has seen and fit to the train
    # images); this checkpoint should show none, since it was never
    # fine-tuned on PlantDoc at all.
    gap = (train_acc - test_acc) if (train_acc is not None and test_acc is not None) else None
    if gap is not None and gap > 0.03:
        contamination_verdict = f"UNEXPECTED positive train-over-test gap ({gap:+.4f}) -- investigate before trusting this checkpoint's zero-shot claim."
    else:
        contamination_verdict = f"NO EVIDENCE OF CONTAMINATION (gap={gap:+.4f}) -- consistent with a genuinely zero-shot checkpoint, as expected."
    print(f"\n{contamination_verdict}")

    # Prediction distribution, same attractor-bias check as rows 2 and 6.
    pred_counts = {}
    for r in records:
        pred_counts[r["predicted_class"]] = pred_counts.get(r["predicted_class"], 0) + 1
    true_counts = {}
    for r in records:
        true_counts[r["true_class"]] = true_counts.get(r["true_class"], 0) + 1
    top_predicted = sorted(pred_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]

    print(f"\nOverall: n={n}, accuracy={overall_acc:.4f} ({correct}/{n})")
    print(f"Train-only: n={len(train_records)}, accuracy={train_acc:.4f}" if train_records else "No train records")
    print(f"Test-only: n={len(test_records)}, accuracy={test_acc:.4f}" if test_records else "No test records")
    print(f"\nTop predicted classes: {top_predicted}")
    for cls, cnt in top_predicted:
        true_rate = true_counts.get(cls, 0) / n
        pred_rate = cnt / n
        print(f"  {cls}: predicted {pred_rate:.1%} of the time, true rate {true_rate:.1%}")

    evidence = {
        "metric": "Row 1 (flat baseline) PlantDoc zero-shot evaluation -- closing the per-image-data gap flagged in the confusion-matrix work",
        "checkpoint": "agents/disease/checkpoints/flat_baseline_best.pt",
        "checkpoint_training_provenance": "PlantVillage only (99.72% val accuracy per flat_baseline_results.json), never fine-tuned on PlantDoc -- genuinely zero-shot, same status as rows 2 and 6.",
        "class_names_source": "agents/disease/results/flat_baseline_results.json's 'classes' field (this checkpoint is a bare state_dict with no embedded class metadata, unlike row 6's checkpoint format)",
        "n_total": n,
        "n_train_split": len(train_records),
        "n_test_split": len(test_records),
        "overall_accuracy_train_plus_test_combined": overall_acc,
        "train_only_accuracy": train_acc,
        "test_only_accuracy": test_acc,
        "train_test_gap": gap,
        "contamination_sanity_check": contamination_verdict,
        "prediction_distribution_top3": [{"class": c, "n_predicted": n_, "predicted_rate": n_ / n, "true_rate": true_counts.get(c, 0) / n} for c, n_ in top_predicted],
        "missing_folders": missing_dirs,
        "records": records,
    }

    out_path = os.path.join(EVIDENCE_DIR, "metric1_row1_flat_baseline_evidence.json")
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
