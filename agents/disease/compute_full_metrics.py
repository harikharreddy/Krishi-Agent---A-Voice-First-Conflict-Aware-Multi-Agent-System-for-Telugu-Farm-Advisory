"""
Full verification & validation metrics for the Disease Agent, matching a
standard engineering-report outcome matrix: precision/recall/F1 (not just
accuracy), per-sample inference latency, throughput, and model footprint
(parameter count, checkpoint size, peak RAM during inference).

Runs the CURRENT deployed checkpoints (via disease_agent.predict_disease(),
the real production path) over PlantDoc's held-out test split, recording
every prediction -- not just aggregate correct/incorrect -- so a real
confusion matrix and per-class precision/recall/F1 can be computed.

IoU / Dice coefficient are deliberately NOT reported: this is an image
CLASSIFICATION task (whole-image label, no bounding box or pixel mask),
not segmentation or detection, so those metrics don't apply here --
noted explicitly rather than silently omitted.
"""

import json
import os
import sys
import time

import psutil
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from disease_agent import predict_disease, _get_models, STAGE1_CLASSES, STAGE2_TOMATO_CLASSES, STAGE2_POTATO_CLASSES

HERE = os.path.dirname(os.path.abspath(__file__))
PLANTDOC_ROOT = os.path.join(HERE, "..", "..", "data", "plantdoc_raw", "test")
RESULTS_DIR = os.path.join(HERE, "results")

PLANTDOC_MAP = {
    "Tomato leaf": "Tomato_healthy",
    "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato leaf bacterial spot": "Tomato_Bacterial_spot",
    "Tomato leaf late blight": "Tomato_Late_blight",
    "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato mold leaf": "Tomato_Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
}

ALL_CLASSES = sorted(set(STAGE2_TOMATO_CLASSES) | set(STAGE2_POTATO_CLASSES))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    process = psutil.Process()
    ram_before_mb = process.memory_info().rss / (1024**2)

    # Warm the models once (matches real server behavior -- models cached
    # after first request), then measure footprint.
    predict_disease(next(
        os.path.join(PLANTDOC_ROOT, cls, f)
        for cls, _ in PLANTDOC_MAP.items()
        if os.path.isdir(os.path.join(PLANTDOC_ROOT, cls))
        for f in os.listdir(os.path.join(PLANTDOC_ROOT, cls))
        if os.path.isfile(os.path.join(PLANTDOC_ROOT, cls, f))
    ))
    models = _get_models()
    param_counts = {name: count_params(m) for name, m in models.items()}
    total_params = sum(param_counts.values())

    checkpoint_sizes_mb = {}
    for name, fname in [("stage1", "stage1_crop_best.pt"), ("stage2_tomato", "stage2_tomato_best.pt"), ("stage2_potato", "stage2_potato_best.pt")]:
        path = os.path.join(HERE, "checkpoints", fname)
        checkpoint_sizes_mb[name] = round(os.path.getsize(path) / (1024**2), 2)

    ram_after_load_mb = process.memory_info().rss / (1024**2)

    # Full run over PlantDoc test, recording every prediction + per-sample latency.
    y_true, y_pred, latencies_ms = [], [], []
    for plantdoc_class, true_label in PLANTDOC_MAP.items():
        src_dir = os.path.join(PLANTDOC_ROOT, plantdoc_class)
        if not os.path.isdir(src_dir):
            continue
        for fname in sorted(os.listdir(src_dir)):
            fpath = os.path.join(src_dir, fname)
            if not os.path.isfile(fpath):
                continue
            t0 = time.perf_counter()
            result = predict_disease(fpath)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            y_true.append(true_label)
            y_pred.append(result.get("predicted_class"))

    peak_ram_mb = process.memory_info().rss / (1024**2)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL_CLASSES, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL_CLASSES, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL_CLASSES, average="weighted", zero_division=0
    )
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    per_class = {
        cls: {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4), "support": int(s)}
        for cls, p, r, f, s in zip(ALL_CLASSES, precision, recall, f1, support)
        if s > 0  # only classes actually present in this test set
    }

    mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
    throughput_ips = 1000 / mean_latency_ms  # images per second, single-threaded CPU/MPS inference

    result = {
        "n": len(y_true),
        "accuracy": round(accuracy, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "latency_ms": {
            "mean": round(mean_latency_ms, 2),
            "min": round(min(latencies_ms), 2),
            "max": round(max(latencies_ms), 2),
            "std": round((sum((l - mean_latency_ms) ** 2 for l in latencies_ms) / len(latencies_ms)) ** 0.5, 2),
        },
        "throughput_images_per_sec": round(throughput_ips, 2),
        "model_footprint": {
            "total_parameters": total_params,
            "parameters_by_stage": param_counts,
            "checkpoint_size_mb": checkpoint_sizes_mb,
            "total_checkpoint_size_mb": round(sum(checkpoint_sizes_mb.values()), 2),
        },
        "memory_rss_mb": {
            "before_model_load": round(ram_before_mb, 1),
            "after_model_load": round(ram_after_load_mb, 1),
            "peak_during_inference": round(peak_ram_mb, 1),
        },
        "note_iou_dice": "Not applicable -- this is an image classification task (whole-image label), not segmentation or detection, so IoU/Dice coefficient don't apply. Precision/recall/F1 above serve the equivalent role of proving low false-positive/false-negative rates for a classifier.",
    }

    print(f"n={result['n']}  accuracy={result['accuracy']:.4f}")
    print(f"macro P/R/F1: {result['macro_precision']:.4f} / {result['macro_recall']:.4f} / {result['macro_f1']:.4f}")
    print(f"weighted P/R/F1: {result['weighted_precision']:.4f} / {result['weighted_recall']:.4f} / {result['weighted_f1']:.4f}")
    print(f"latency: mean={result['latency_ms']['mean']:.1f}ms  std={result['latency_ms']['std']:.1f}ms")
    print(f"throughput: {result['throughput_images_per_sec']:.2f} images/sec")
    print(f"total params: {total_params:,}  total checkpoint size: {result['model_footprint']['total_checkpoint_size_mb']:.1f}MB")
    print(f"peak RSS during inference: {peak_ram_mb:.1f}MB")

    out_path = os.path.join(RESULTS_DIR, "full_verification_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
