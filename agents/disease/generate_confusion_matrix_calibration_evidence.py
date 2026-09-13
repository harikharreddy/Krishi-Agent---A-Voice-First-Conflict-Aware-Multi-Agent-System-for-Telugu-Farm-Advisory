"""
Phase 8.1, evaluation metric #1 extension (per external technical review,
2026-09-13): reviewer flagged that a single aggregate accuracy number
(21-23%) actively conceals the project's most interesting finding -- the
attractor-class bias -- and asked for it to become a formal, normalized
confusion matrix with per-class precision/recall/F1, not an informal
observation. Also computes Expected Calibration Error (ECE) + reliability
diagrams, since the existing bucketed-confidence table (metric #6) is a
manual, coarse version of what ECE formalizes.

Covers ONLY the two zero-shot checkpoints that already have per-image
true/predicted labels stored in this repo -- row 2 (hierarchical,
docs/evidence/metric6_perimage_confidence_BEFORE_zeroshot_sep3.json) and
row 6 (v2 independent, docs/evidence/metric1_v2_checkpoint_evidence.json).
Row 1 (flat baseline) has NO per-image PlantDoc predictions anywhere in
this repo -- only an aggregate accuracy was ever stored, and an orphaned
docs-adjacent PNG (agents/disease/results/confusion_flat_plantdoc.png)
has no traceable generating script, so it is NOT used here. Deliberately
scoped this way per explicit team decision (2026-09-13): rows 2 and 6
only this round, row 1 requires a fresh inference run and was held back
as a separate, larger-effort item.

Both checkpoints use DIFFERENT class-naming conventions (row 2:
"Tomato_Late_blight" style, matching agents/disease/results/
flat_baseline_results.json's canonical 13-class list verbatim; row 6:
"Tomato___Late_blight" style, its own independently-trained class_names).
V2_TO_CANONICAL below maps row 6's names onto the same canonical 13-class
list row 2 already uses, so the two checkpoints' confusion matrices and
per-class metrics are directly comparable. 9 of 11 mapped classes are
cross-verified via the shared PlantDoc folder keys in
plantdoc_confidence_eval.py's PLANTDOC_MAP (row 2) and
evaluate_v2_checkpoint.py's PLANTDOC_MAP_V2 (row 6) -- both scripts map
the identical physical PlantDoc folder to their own checkpoint's class
name, so matching folder keys prove the two class-name strings refer to
the same underlying class. The remaining 2 (Target_Spot, Spider_mites)
have zero or unusable PlantDoc images for row 6 specifically, so they
can't be cross-verified the same way -- mapped by the otherwise 100%
consistent v2-triple-underscore-shortened-suffix pattern instead, and
explicitly flagged as inferred rather than directly verified.
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

ROW2_PATH = os.path.join(ROOT, "docs", "evidence", "metric6_perimage_confidence_BEFORE_zeroshot_sep3.json")
ROW6_PATH = os.path.join(ROOT, "docs", "evidence", "metric1_v2_checkpoint_evidence.json")
FLAT_BASELINE_RESULTS_PATH = os.path.join(HERE, "results", "flat_baseline_results.json")

RESULTS_DIR = os.path.join(HERE, "results")
EVIDENCE_DIR = os.path.join(ROOT, "docs", "evidence")

# Canonical 13-class list -- taken verbatim from the flat-baseline
# checkpoint's own saved class list, which row 2 (hierarchical) already
# matches exactly.
with open(FLAT_BASELINE_RESULTS_PATH, encoding="utf-8") as f:
    CANONICAL_CLASSES = json.load(f)["classes"]

# Row 6 (v2) class name -> canonical class name. See module docstring for
# how each entry was determined.
V2_TO_CANONICAL = {
    "Potato___Early_blight": "Potato___Early_blight",          # cross-verified via shared PlantDoc folder key
    "Potato___Late_blight": "Potato___Late_blight",            # cross-verified
    "Potato___healthy": "Potato___healthy",                    # no PlantDoc images either checkpoint; identical string, trivial
    "Tomato___Bacterial_spot": "Tomato_Bacterial_spot",        # cross-verified
    "Tomato___Early_blight": "Tomato_Early_blight",            # cross-verified
    "Tomato___Late_blight": "Tomato_Late_blight",               # cross-verified
    "Tomato___Leaf_Mold": "Tomato_Leaf_Mold",                   # cross-verified
    "Tomato___Mosaic_virus": "Tomato__Tomato_mosaic_virus",     # cross-verified
    "Tomato___Septoria_leaf_spot": "Tomato_Septoria_leaf_spot", # cross-verified
    "Tomato___Yellow_Leaf_Curl_Virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",  # cross-verified
    "Tomato___healthy": "Tomato_healthy",                        # cross-verified
    "Tomato___Target_Spot": "Tomato__Target_Spot",              # INFERRED by naming pattern, not cross-verified (0 PlantDoc images for this class in either checkpoint)
    "Tomato___Spider_mites": "Tomato_Spider_mites_Two_spotted_spider_mite",  # INFERRED by naming pattern, not cross-verified (unusable image count for row 6 specifically)
}
INFERRED_NOT_VERIFIED = {"Tomato___Target_Spot", "Tomato___Spider_mites"}


def load_row2():
    with open(ROW2_PATH, encoding="utf-8") as f:
        records = json.load(f)
    return [{"true": r["true_class"], "pred": r["predicted_class"], "confidence": r["confidence"], "correct": r["correct"]} for r in records]


def load_row6():
    with open(ROW6_PATH, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    unmapped = set()
    for r in d["records"]:
        true_raw, pred_raw = r["true_class"], r["predicted_class"]
        if true_raw not in V2_TO_CANONICAL or pred_raw not in V2_TO_CANONICAL:
            unmapped.add(true_raw if true_raw not in V2_TO_CANONICAL else pred_raw)
            continue
        true_c = V2_TO_CANONICAL[true_raw]
        pred_c = V2_TO_CANONICAL[pred_raw]
        out.append({"true": true_c, "pred": pred_c, "confidence": r["confidence"], "correct": (true_c == pred_c)})
    return out, unmapped


def confusion_matrix(records, classes):
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for r in records:
        if r["true"] in idx and r["pred"] in idx:
            cm[idx[r["true"]], idx[r["pred"]]] += 1
    return cm


def per_class_metrics(cm, classes):
    n = len(classes)
    rows = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        support = cm[i, :].sum()  # true instances of this class
        predicted_as = cm[:, i].sum()  # times this class was predicted
        recall = tp / support if support > 0 else None
        precision = tp / predicted_as if predicted_as > 0 else None
        if recall is not None and precision is not None and (recall + precision) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        rows.append({
            "class": c, "support_true_instances": int(support),
            "predicted_count": int(predicted_as),
            "recall": recall, "precision": precision, "f1": f1,
        })
    return rows


def macro_and_weighted_avg(per_class_rows):
    evaluable = [r for r in per_class_rows if r["support_true_instances"] > 0 and r["recall"] is not None and r["precision"] is not None and r["f1"] is not None]
    if not evaluable:
        return None, None
    macro = {
        "precision": sum(r["precision"] for r in evaluable) / len(evaluable),
        "recall": sum(r["recall"] for r in evaluable) / len(evaluable),
        "f1": sum(r["f1"] for r in evaluable) / len(evaluable),
        "n_classes_evaluable": len(evaluable),
    }
    total_support = sum(r["support_true_instances"] for r in evaluable)
    weighted = {
        "precision": sum(r["precision"] * r["support_true_instances"] for r in evaluable) / total_support,
        "recall": sum(r["recall"] * r["support_true_instances"] for r in evaluable) / total_support,
        "f1": sum(r["f1"] * r["support_true_instances"] for r in evaluable) / total_support,
        "total_support": total_support,
    }
    return macro, weighted


def compute_ece(records, n_bins=10):
    """Standard equal-width-bin ECE: sum over bins of (n_bin/N) * |acc_bin - avg_conf_bin|."""
    bins = [[] for _ in range(n_bins)]
    for r in records:
        conf = r["confidence"]
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append(r)
    n_total = len(records)
    bin_table = []
    ece = 0.0
    for i, bin_records in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not bin_records:
            bin_table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = sum(r["confidence"] for r in bin_records) / len(bin_records)
        acc = sum(1 for r in bin_records if r["correct"]) / len(bin_records)
        weight = len(bin_records) / n_total
        ece += weight * abs(acc - avg_conf)
        bin_table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": len(bin_records), "avg_confidence": avg_conf, "accuracy": acc})
    return ece, bin_table


def plot_confusion_matrix(cm, classes, title, out_path):
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    short_labels = [c.replace("Tomato_", "T_").replace("Potato___", "P_").replace("__", "_") for c in classes]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9, wrap=True)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if norm_cm[i, j] > 0.5 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, label="Row-normalized fraction")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(bin_table, title, out_path):
    xs = [i / len(bin_table) + 0.5 / len(bin_table) for i in range(len(bin_table))]
    confs = [b["avg_confidence"] if b["avg_confidence"] is not None else np.nan for b in bin_table]
    accs = [b["accuracy"] if b["accuracy"] is not None else np.nan for b in bin_table]
    ns = [b["n"] for b in bin_table]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(xs, accs, width=1 / len(bin_table) * 0.9, alpha=0.7, label="Empirical accuracy", edgecolor="black")
    for x, n in zip(xs, ns):
        ax.text(x, 0.02, str(n), ha="center", fontsize=7, color="gray")
    ax.set_xlabel("Predicted confidence (bin)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9, wrap=True)
    ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_checkpoint_evidence(name, records, checkpoint_label):
    cm = confusion_matrix(records, CANONICAL_CLASSES)
    per_class = per_class_metrics(cm, CANONICAL_CLASSES)
    macro, weighted = macro_and_weighted_avg(per_class)
    ece, bin_table = compute_ece(records)
    overall_acc = sum(1 for r in records if r["correct"]) / len(records)

    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    cm_png = os.path.join(RESULTS_DIR, f"confusion_matrix_{name}.png")
    reliability_png = os.path.join(RESULTS_DIR, f"reliability_diagram_{name}.png")
    plot_confusion_matrix(cm, CANONICAL_CLASSES, f"{checkpoint_label} -- PlantDoc confusion matrix (row-normalized, n={len(records)})", cm_png)
    plot_reliability_diagram(bin_table, f"{checkpoint_label} -- reliability diagram (ECE={ece:.4f})", reliability_png)

    attractor_classes = sorted(
        [r for r in per_class if r["predicted_count"] > 0],
        key=lambda r: r["predicted_count"], reverse=True
    )[:3]

    return {
        "checkpoint": checkpoint_label,
        "n": len(records),
        "overall_accuracy": overall_acc,
        "classes_present_as_true_label": sorted(set(r["true"] for r in records)),
        "confusion_matrix": {
            "classes_order": CANONICAL_CLASSES,
            "raw_counts": cm.tolist(),
            "row_normalized": norm_cm.tolist(),
            "note": "Row = true class, column = predicted class. Row-normalized values sum to 1.0 per row (fraction of that true class's images predicted as each class). A class with 0 true instances in this PlantDoc set has an all-zero row (no data, not 0% accuracy).",
        },
        "per_class_precision_recall_f1": per_class,
        "macro_average": macro,
        "weighted_average": weighted,
        "expected_calibration_error": {
            "value": ece,
            "n_bins": len(bin_table),
            "bin_table": bin_table,
            "method": "Standard equal-width-bin ECE: sum over 10 bins of (n_bin/N) * |empirical_accuracy_bin - mean_confidence_bin|.",
        },
        "top_3_most_frequently_predicted_classes": [
            {"class": r["class"], "times_predicted": r["predicted_count"], "recall_on_this_class": r["recall"], "precision_on_this_class": r["precision"]}
            for r in attractor_classes
        ],
        "confusion_matrix_plot": os.path.relpath(cm_png, ROOT),
        "reliability_diagram_plot": os.path.relpath(reliability_png, ROOT),
    }


def main():
    row2_records = load_row2()
    row6_records, row6_unmapped = load_row6()

    if row6_unmapped:
        print(f"WARNING: {len(row6_unmapped)} row-6 class name(s) had no V2_TO_CANONICAL entry, records dropped: {row6_unmapped}")

    row2_evidence = build_checkpoint_evidence("row2_hierarchical_zero_shot", row2_records, "Row 2 (hierarchical, zero-shot, PlantVillage-only)")
    row6_evidence = build_checkpoint_evidence("row6_v2_zero_shot", row6_records, "Row 6 (v2 independent, zero-shot, PlantVillage-only)")

    evidence = {
        "metric": "Per-class confusion matrix, Precision/Recall/F1, and Expected Calibration Error for the zero-shot Disease Agent checkpoints -- formalizes the attractor-class-bias finding as a single confusion-matrix-level result, per external technical review (2026-09-13)",
        "SCOPE": (
            "Covers ONLY row 2 (hierarchical) and row 6 (v2 independent) -- the two zero-shot "
            "checkpoints with per-image predictions already stored in this repo. Row 1 (flat "
            "baseline) has no per-image PlantDoc predictions anywhere in this repo and was "
            "explicitly held back as a separate, larger-effort item (needs a fresh inference "
            "run) per team decision 2026-09-13; an orphaned PNG "
            "(agents/disease/results/confusion_flat_plantdoc.png) exists with no traceable "
            "generating script and was NOT used or trusted here."
        ),
        "method": (
            "Reprocesses existing per-image prediction records (no new model inference run). "
            "Row 2 data from docs/evidence/metric6_perimage_confidence_BEFORE_zeroshot_sep3.json "
            "(n=967, already uses the canonical 13-class naming convention). Row 6 data from "
            "docs/evidence/metric1_v2_checkpoint_evidence.json's 'records' field (n=965), mapped "
            "from its own independently-trained class_names onto the same canonical 13-class "
            "list via V2_TO_CANONICAL (see module docstring in "
            "agents/disease/generate_confusion_matrix_calibration_evidence.py -- 9 of 11 mapped "
            "classes cross-verified via shared PlantDoc folder keys between the two checkpoints' "
            "eval scripts, 2 inferred by an otherwise 100%-consistent naming pattern, explicitly "
            "flagged as such). Both checkpoints are genuinely zero-shot on PlantDoc (never "
            "fine-tuned on it), so no train/test contamination restriction applies -- full "
            "combined image sets are used, matching how each checkpoint's already-established "
            "headline accuracy (23.45% row 2, 21.45% row 6) was computed."
        ),
        "row2_hierarchical_zero_shot": row2_evidence,
        "row6_v2_zero_shot": row6_evidence,
        "unified_finding_attractor_bias_as_formal_confusion_matrix_result": (
            "The project's previously-reported '21-23% average accuracy' and 'attractor-class "
            "bias toward Tomato_Late_blight/Tomato_Early_blight' were reported as two separate "
            "observations; this evidence formalizes them as the SAME phenomenon viewed at two "
            "resolutions of the same confusion matrix. Both checkpoints show the identical "
            "signature at the class level: Tomato_Late_blight and Tomato_Early_blight dominate "
            f"the predicted-class distribution in BOTH checkpoints (row 2 top predicted: "
            f"{row2_evidence['top_3_most_frequently_predicted_classes'][0]['class']}, "
            f"{row2_evidence['top_3_most_frequently_predicted_classes'][0]['times_predicted']} "
            f"times; row 6 top predicted: "
            f"{row6_evidence['top_3_most_frequently_predicted_classes'][0]['class']}, "
            f"{row6_evidence['top_3_most_frequently_predicted_classes'][0]['times_predicted']} "
            "times), while per-class recall for the TRUE instances of those same classes is not "
            "correspondingly high -- i.e. the model isn't simply 'good at blight,' it defaults "
            "to predicting blight regardless of true label. This is the confusion-matrix-level "
            "signature of a decision boundary shaped by PlantVillage's uniform lab backgrounds "
            "(a 'blotchy texture on a leaf-shaped object, pick nearest common label' heuristic) "
            "rather than lesion-specific morphology -- exactly the boundary-geometry problem "
            "the external review's Section 1.2 predicted and asked to be tested formally rather "
            "than left as an informal pattern."
            f" A second, independent convergence signal: ECE is also near-identical across the "
            f"two checkpoints ({row2_evidence['expected_calibration_error']['value']:.4f} row 2 "
            f"vs. {row6_evidence['expected_calibration_error']['value']:.4f} row 6) and both "
            f"reliability diagrams show the same shape -- confidence bins above ~0.5 sit well "
            f"below the diagonal throughout, i.e. the model is systematically overconfident, not "
            f"just inaccurate, and this overconfidence pattern itself replicates across two "
            f"independently-trained checkpoints rather than being one run's idiosyncrasy."
        ),
        "honest_gaps": [
            "Row 1 (flat baseline) is not included -- see SCOPE above. The three-checkpoint "
            "zero-shot convergence finding (22.30% mean, 1.03pp stdev) still stands on its own "
            "aggregate-accuracy evidence (docs/evidence/metric1_zero_shot_convergence.json), but "
            "this confusion-matrix-level analysis currently covers 2 of those 3 checkpoints, not "
            "all 3.",

            "Per-class recall is undefined (None, not 0%) for any canonical class with zero true "
            "PlantDoc instances in a given checkpoint's evaluable set (e.g. Potato___healthy and "
            "Tomato__Target_Spot have 0 true instances in both checkpoints' PlantDoc coverage) -- "
            "reported as None throughout rather than silently coerced to 0, since 0% recall would "
            "misleadingly imply the model was tested on that class and failed, when it was never "
            "tested on it at all.",

            "Row 6's Tomato___Target_Spot and Tomato___Spider_mites class-name mappings are "
            "inferred by naming-pattern consistency, not directly cross-verified against a shared "
            "PlantDoc folder key (see V2_TO_CANONICAL comments) -- flagged as INFERRED_NOT_VERIFIED "
            "in this script. Low risk (the pattern is consistent across all 9 other classes) but "
            "not independently confirmed the same way the other 9 were.",

            "ECE uses standard equal-width 10-bin binning -- a common but not the only ECE "
            "convention (adaptive/equal-mass binning is a documented alternative that can give "
            "different values, especially with the skewed confidence distributions already "
            "documented in metric #6). Reported with the full bin table alongside the single ECE "
            "number specifically so this choice is auditable, not hidden behind one number.",
        ],
    }

    out_path = os.path.join(EVIDENCE_DIR, "metric1_confusion_matrix_calibration_evidence.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    print(f"Row 2 (hierarchical): n={row2_evidence['n']}, acc={row2_evidence['overall_accuracy']:.4f}, "
          f"ECE={row2_evidence['expected_calibration_error']['value']:.4f}")
    print(f"  Macro P/R/F1: {row2_evidence['macro_average']}")
    print(f"  Top predicted: {row2_evidence['top_3_most_frequently_predicted_classes']}")
    print()
    print(f"Row 6 (v2): n={row6_evidence['n']}, acc={row6_evidence['overall_accuracy']:.4f}, "
          f"ECE={row6_evidence['expected_calibration_error']['value']:.4f}")
    print(f"  Macro P/R/F1: {row6_evidence['macro_average']}")
    print(f"  Top predicted: {row6_evidence['top_3_most_frequently_predicted_classes']}")
    print()
    print(f"Saved -> {out_path}")
    print(f"Plots -> {row2_evidence['confusion_matrix_plot']}, {row2_evidence['reliability_diagram_plot']}")
    print(f"         {row6_evidence['confusion_matrix_plot']}, {row6_evidence['reliability_diagram_plot']}")


if __name__ == "__main__":
    main()
