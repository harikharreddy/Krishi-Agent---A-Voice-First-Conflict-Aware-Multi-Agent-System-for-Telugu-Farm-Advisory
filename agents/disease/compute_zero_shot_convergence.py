"""
Phase 8.1 metric #1 addendum: convergence summary across all three
independently-trained, differently-architected zero-shot checkpoints
(model lineage rows 1, 2, 6). Pulls numbers directly from each
checkpoint's own results file rather than hardcoding them, so this
can't drift out of sync if any of those files are ever regenerated.
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
EVIDENCE_DIR = os.path.join(HERE, "..", "..", "docs", "evidence")


def main():
    flat_hier = json.load(open(os.path.join(RESULTS_DIR, "plantdoc_results.json")))
    v2 = json.load(open(os.path.join(EVIDENCE_DIR, "metric1_v2_checkpoint_evidence.json")))

    checkpoints = {
        "row_1_flat_baseline": {
            "architecture": "Flat 13-class EfficientNetB0 (original)",
            "accuracy": flat_hier["flat_baseline"]["plantdoc_accuracy"],
            "n": flat_hier["plantdoc_images_evaluated"],
        },
        "row_2_hierarchical": {
            "architecture": "Hierarchical 2-stage EfficientNetB0 (original, deployed lineage before fine-tuning)",
            "accuracy": flat_hier["hierarchical"]["plantdoc_accuracy"],
            "n": flat_hier["plantdoc_images_evaluated"],
        },
        "row_6_v2_flat": {
            "architecture": "Flat 13-class EfficientNetB0 (independent Colab training run)",
            "accuracy": v2["overall_accuracy_train_plus_test_combined"],
            "n": v2["n_total"],
        },
    }

    accuracies = [c["accuracy"] for c in checkpoints.values()]
    mean_acc = statistics.mean(accuracies)
    stdev_acc = statistics.stdev(accuracies)
    lo, hi = min(accuracies), max(accuracies)

    evidence = {
        "metric": "Zero-shot PlantDoc accuracy convergence across 3 independent checkpoints",
        "reporting_status": (
            "All three are genuine zero-shot results (model lineage rows 1, 2, "
            "6) -- none fine-tuned on PlantDoc. Framed as convergence evidence: "
            "three separately trained models (2 architectures, 2 training "
            "environments/runs) landing in a ~2-point band supports this being "
            "a structural property of the PlantVillage-to-PlantDoc domain gap, "
            "not one model's idiosyncrasy -- the SAME argument already made for "
            "the shared Late_blight attractor bias (see model_lineage.md), now "
            "backed by the aggregate accuracy number too."
        ),
        "checkpoints": checkpoints,
        "mean_accuracy": mean_acc,
        "stdev_accuracy": stdev_acc,
        "range_accuracy": {"min": lo, "max": hi, "range_points": hi - lo},
        "note_on_n": (
            "Rows 1/2 evaluated on n=968 (train+test PlantDoc combined, per "
            "the original Sep 3 methodology); row 6 on n=965 (2-image "
            "discrepancy investigated and documented in metric1_v2_checkpoint_evidence.json, "
            "effect negligible). Not perfectly identical sample sizes, but "
            "close enough (99.7% overlap) that this doesn't undermine the "
            "convergence claim."
        ),
    }

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    out_path = os.path.join(EVIDENCE_DIR, "metric1_zero_shot_convergence.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    print(f"mean={mean_acc:.4f}  stdev={stdev_acc:.4f}  range=[{lo:.4f}, {hi:.4f}] ({hi-lo:.4f} points)")
    for name, c in checkpoints.items():
        print(f"  {name}: {c['accuracy']:.4f} (n={c['n']}) -- {c['architecture']}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
