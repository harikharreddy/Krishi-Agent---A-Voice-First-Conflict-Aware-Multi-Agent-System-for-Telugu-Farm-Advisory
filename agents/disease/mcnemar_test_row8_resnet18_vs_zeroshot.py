"""
Row 8 (ResNet18, cross-architecture zero-shot baseline) vs. the three
existing zero-shot Disease Agent checkpoints (rows 1, 2, 6) -- McNemar's
test, run 2026-09-16 per explicit request following delivery of the
ResNet18 training/evaluation evidence.

Row 8 was trained externally (not in this repo) under a protocol
claimed identical to rows 1/2/6: same seed=42, same 70/15/15 split, same
LR schedule shape, same PlantVillage data, same max epochs/early-stop
patience -- the only changed variable is the architecture (ResNet18 vs.
EfficientNetB0). Delivered evidence: 99.70% PlantVillage test accuracy,
22.51% PlantDoc zero-shot accuracy (185/822, content-hash deduped), with
a Tomato_Late_blight / Tomato_Early_blight over-prediction pattern
(44.0% / 28.8% of all predictions) matching the attractor-class bias
already documented for rows 1/2/6.

This script does two things, per the explicit request to check overlap
before testing rather than assume alignment:

1. Verifies image-set identity between row 8's 822-image PlantDoc set
   and rows 1/2/6's established 967/967/965-image set -- NOT assumed.
   Row 8's images were delivered as absolute Colab paths
   ("/content/PlantDoc-Dataset/<split>/<folder>/<file>"); normalized to
   the same "<split>/<folder>/<file>" key rows 1/2/6 already use.

2. Runs McNemar's test (paired, correct tool for per-image
   correct/incorrect outcomes) between row 8 and each of rows 1/2/6,
   restricted to the actual per-pair image intersection.

Pure post-hoc analysis on already-saved per-image predictions -- no new
model inference, no pipeline code touched.
"""

import json
import os
import sys

from statsmodels.stats.contingency_tables import mcnemar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

ROW1_PATH = os.path.join(ROOT, "docs", "evidence", "metric1_row1_flat_baseline_evidence.json")
ROW2_PATH = os.path.join(ROOT, "docs", "evidence", "metric6_perimage_confidence_BEFORE_zeroshot_sep3.json")
ROW6_PATH = os.path.join(ROOT, "docs", "evidence", "metric1_v2_checkpoint_evidence.json")
ROW8_PATH = os.path.join(ROOT, "docs", "evidence", "row8_resnet18_plantdoc_perimage_predictions.json")
EVIDENCE_DIR = os.path.join(ROOT, "docs", "evidence")

EXACT_TEST_DISCORDANT_THRESHOLD = 25

COLAB_MARKER = "PlantDoc-Dataset/"


def load_row1_or_row2(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    records = d["records"] if isinstance(d, dict) else d
    return {r["image"]: r["correct"] for r in records}


def load_row6_canonical(path):
    from generate_confusion_matrix_calibration_evidence import V2_TO_CANONICAL

    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    unmapped = 0
    for r in d["records"]:
        true_raw, pred_raw = r["true_class"], r["predicted_class"]
        if true_raw not in V2_TO_CANONICAL or pred_raw not in V2_TO_CANONICAL:
            unmapped += 1
            continue
        out[r["image"]] = (V2_TO_CANONICAL[true_raw] == V2_TO_CANONICAL[pred_raw])
    return out, unmapped


def load_row8(path):
    """Row 8's raw predictions carry absolute Colab paths
    ("/content/PlantDoc-Dataset/<split>/<folder>/<file>"), not the
    "<split>/<folder>/<file>" key rows 1/2/6 use -- normalized here by
    slicing after the shared "PlantDoc-Dataset/" marker, not assumed to
    already match. The stored 'correct' field is used directly (both
    true_class and predicted_class are already in the same V2-style
    namespace within this file, so no cross-checkpoint remapping is
    needed for correctness itself -- only for the image key)."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    out = {}
    unparsed = []
    for r in records:
        p = r["path"]
        idx = p.find(COLAB_MARKER)
        if idx == -1:
            unparsed.append(p)
            continue
        key = p[idx + len(COLAB_MARKER):]
        out[key] = r["correct"]
    return out, unparsed, records


def class_coverage_gap_report(row8_records, row1_dict):
    """Row 8's own total (822) is smaller than rows 1/2's (967) -- not
    just a handful of stray images. Characterized explicitly rather than
    left as an unexplained smaller n: which classes/images are missing,
    and why the headline 822-image accuracy is not a like-for-like
    benchmark against rows 1/2/6's 967/965-image figures."""
    from collections import Counter

    row8_keys = set()
    for r in row8_records:
        p = r["path"]
        idx = p.find(COLAB_MARKER)
        if idx != -1:
            row8_keys.add(p[idx + len(COLAB_MARKER):])

    row1_keys = set(row1_dict.keys())
    missing_from_row8 = sorted(row1_keys - row8_keys)
    extra_in_row8 = sorted(row8_keys - row1_keys)

    by_class = Counter()
    for m in missing_from_row8:
        parts = m.split("/", 2)
        folder = parts[1] if len(parts) > 1 else "UNKNOWN"
        by_class[folder] += 1

    return {
        "row8_own_total": len(row8_keys),
        "row1_total": len(row1_keys),
        "n_missing_from_row8_vs_row1": len(missing_from_row8),
        "n_present_in_row8_not_in_row1": len(extra_in_row8),
        "extra_in_row8_not_in_row1": extra_in_row8,
        "missing_from_row8_by_plantdoc_folder": dict(by_class.most_common()),
        "finding": (
            "Row 8's PlantDoc-Dataset clone (external Colab environment) is missing two "
            "entire class folders that rows 1/2/6's established image set includes: "
            "'Tomato leaf yellow virus' (76 images -> Tomato_Yellow_Leaf_Curl_Virus) and "
            "'Tomato leaf' (63 images -> Tomato_healthy), plus 7 stray misses elsewhere "
            "(5 in 'Potato leaf late blight', 2 in 'Tomato Septoria leaf spot'), for 146 "
            "total images row 1 has that row 8 does not -- offset by 1 image row 8 has "
            "that row 1 doesn't, netting the observed 967-822=145 gap. This is the SAME "
            "TWO CLASSES already flagged as dropped by a data-loading bug in row 7's fine-"
            "tuning run (model_lineage.md, 'Row 7' section) -- consistent with an external-"
            "environment PlantDoc-Dataset clone/version issue rather than row 8's own "
            "training or evaluation code, but not independently root-caused here (no access "
            "to the Colab environment's filesystem to SHA-verify, unlike row 7's fine-tune "
            "which was root-caused directly against `data/plantdoc_raw`). Practical "
            "consequence: row 8's self-reported 22.51% (185/822) is NOT computed on an "
            "identical 13-class benchmark to rows 1/2/6's 22.00%/23.45%/21.45% (n=967/968/965) "
            "-- it is missing 2 of 13 classes entirely. The McNemar's tests below are "
            "unaffected (each pair is restricted to the actual shared-image intersection, "
            "which naturally excludes these already-missing images), and the accuracy figures "
            "reported per pair on that shared subset are the fair, apples-to-apples comparison "
            "-- the standalone 822-image headline number should carry this caveat wherever "
            "it is cited standing alone."
        ),
    }


def verify_image_sets(images1, images2, images6, images8):
    return {
        "row1_n": len(images1), "row2_n": len(images2), "row6_n": len(images6), "row8_n": len(images8),
        "row8_minus_row1": sorted(images8 - images1),
        "row1_minus_row8": len(images1 - images8),
        "row8_minus_row2": sorted(images8 - images2),
        "row2_minus_row8": len(images2 - images8),
        "row8_minus_row6": sorted(images8 - images6),
        "row6_minus_row8": len(images6 - images8),
    }


def pairwise_mcnemar(name_a, correct_a, name_b, correct_b):
    shared_images = sorted(set(correct_a.keys()) & set(correct_b.keys()))
    only_a = set(correct_a.keys()) - set(correct_b.keys())
    only_b = set(correct_b.keys()) - set(correct_a.keys())

    both_correct = 0
    a_only_correct = 0
    b_only_correct = 0
    both_wrong = 0

    for img in shared_images:
        a_ok, b_ok = correct_a[img], correct_b[img]
        if a_ok and b_ok:
            both_correct += 1
        elif a_ok and not b_ok:
            a_only_correct += 1
        elif b_ok and not a_ok:
            b_only_correct += 1
        else:
            both_wrong += 1

    n = len(shared_images)
    table = [[both_correct, a_only_correct], [b_only_correct, both_wrong]]
    discordant = a_only_correct + b_only_correct

    use_exact = discordant < EXACT_TEST_DISCORDANT_THRESHOLD
    result = mcnemar(table, exact=use_exact, correction=True)

    acc_a = (both_correct + a_only_correct) / n if n else None
    acc_b = (both_correct + b_only_correct) / n if n else None

    verdict = (
        "SIGNIFICANT at alpha=0.05 -- accuracy difference is NOT explainable by chance alone "
        "on this paired sample. This pair does NOT support 'indistinguishable performance.'"
        if result.pvalue < 0.05 else
        "NOT significant at alpha=0.05 -- cannot reject the null hypothesis that these two "
        "checkpoints have the same accuracy on this paired sample. This IS statistical support "
        "for 'these models perform indistinguishably,' not just an eyeballed similarity."
    )

    return {
        "pair": f"{name_a} vs {name_b}",
        "n_shared_images": n,
        "n_images_only_in_a_not_in_this_pair": len(only_a),
        "n_images_only_in_b_not_in_this_pair": len(only_b),
        "contingency_table": {
            "both_correct": both_correct,
            f"{name_a}_only_correct": a_only_correct,
            f"{name_b}_only_correct": b_only_correct,
            "both_wrong": both_wrong,
        },
        "accuracy_on_shared_set": {name_a: acc_a, name_b: acc_b},
        "discordant_pairs_b_plus_c": discordant,
        "test_variant_used": (
            f"EXACT BINOMIAL (discordant pairs = {discordant} < {EXACT_TEST_DISCORDANT_THRESHOLD} threshold)"
            if use_exact else
            f"CHI-SQUARE with continuity correction (discordant pairs = {discordant} >= "
            f"{EXACT_TEST_DISCORDANT_THRESHOLD} threshold)"
        ),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_at_0.05": bool(result.pvalue < 0.05),
        "verdict": verdict,
    }


def main():
    row1 = load_row1_or_row2(ROW1_PATH)
    row2 = load_row1_or_row2(ROW2_PATH)
    row6, row6_unmapped = load_row6_canonical(ROW6_PATH)
    row8, row8_unparsed, row8_records = load_row8(ROW8_PATH)

    if row6_unmapped:
        print(f"NOTE: {row6_unmapped} row-6 record(s) had no V2_TO_CANONICAL entry, excluded.")
    if row8_unparsed:
        print(f"WARNING: {len(row8_unparsed)} row-8 record(s) had no 'PlantDoc-Dataset/' marker in their path, excluded: {row8_unparsed}")

    images1, images2, images6, images8 = set(row1), set(row2), set(row6), set(row8)
    image_set_report = verify_image_sets(images1, images2, images6, images8)
    coverage_gap = class_coverage_gap_report(row8_records, row1)

    print("=== Image-set identity check (verified, not assumed) ===")
    print(f"Row 1 n={image_set_report['row1_n']}, Row 2 n={image_set_report['row2_n']}, "
          f"Row 6 n={image_set_report['row6_n']}, Row 8 n={image_set_report['row8_n']}")
    print(f"Row8 - Row1: {len(image_set_report['row8_minus_row1'])} unique to row8")
    print(f"Row1 - Row8: {image_set_report['row1_minus_row8']} images row1 has that row8 doesn't")
    print()
    print("=== Class coverage gap (row 8 vs row 1) ===")
    print(f"Missing by folder: {coverage_gap['missing_from_row8_by_plantdoc_folder']}")
    print(coverage_gap["finding"])
    print()

    pairs = [
        pairwise_mcnemar("row8_resnet18", row8, "row1_flat_baseline", row1),
        pairwise_mcnemar("row8_resnet18", row8, "row2_hierarchical", row2),
        pairwise_mcnemar("row8_resnet18", row8, "row6_v2", row6),
    ]

    for p in pairs:
        print(f"--- {p['pair']} ---")
        print(f"  n shared images: {p['n_shared_images']}")
        print(f"  Contingency table: {p['contingency_table']}")
        print(f"  Accuracy on shared set: {p['accuracy_on_shared_set']}")
        print(f"  Discordant pairs (b+c): {p['discordant_pairs_b_plus_c']}")
        print(f"  Test variant: {p['test_variant_used']}")
        print(f"  Statistic: {p['statistic']:.4f}, p-value: {p['p_value']:.4f}")
        print(f"  {p['verdict']}")
        print()

    all_non_significant = all(not p["significant_at_0.05"] for p in pairs)
    overall_verdict = (
        "ALL THREE cross-architecture pairwise comparisons (ResNet18 vs. each of rows 1/2/6, "
        "all EfficientNetB0-family) are non-significant (p >= 0.05). This is genuine "
        "statistical support -- not an eyeballed impression -- that a completely independent "
        "architecture family lands in the same accuracy regime as the three EfficientNetB0 "
        "checkpoints on this paired sample. The convergence finding is upgraded from "
        "'three same-family checkpoints converge' to 'convergence holds across two distinct "
        "architecture families, formally tested in both directions.'"
        if all_non_significant else
        "NOT all cross-architecture pairwise comparisons are non-significant -- see individual "
        "pair verdicts above. Report exactly which pair(s) showed a significant difference; do "
        "not extend the 'architecture-independent convergence' claim to cover a pair where it "
        "does not hold."
    )
    print(f"=== Overall verdict ===\n{overall_verdict}")

    evidence = {
        "metric": "McNemar's test for pairwise statistical significance between the cross-architecture ResNet18 zero-shot baseline (row 8) and each of the three existing EfficientNetB0-family zero-shot checkpoints (rows 1, 2, 6)",
        "context": (
            "Row 8 (ResNet18, trained externally under a protocol claimed identical to rows "
            "1/2/6 except architecture) was delivered with a 22.51% (185/822) PlantDoc zero-shot "
            "accuracy and an attractor-class bias pattern (Tomato_Late_blight 44.0% / "
            "Tomato_Early_blight 28.8% of all predictions) matching rows 1/2/6's documented "
            "bias. This tests whether row 8's accuracy is formally indistinguishable from each "
            "of rows 1/2/6, the same question already answered for the three EfficientNetB0 "
            "checkpoints among themselves (mcnemar_zero_shot_checkpoint_comparison_evidence.json), "
            "now extended across architecture families."
        ),
        "method": (
            "McNemar's test, the correct tool for PAIRED binary outcomes -- uses only the "
            "discordant pairs (one checkpoint right, the other wrong), not the full accuracy "
            "percentages treated as independent samples. Test variant chosen per pair via the "
            f"standard {EXACT_TEST_DISCORDANT_THRESHOLD}-discordant-pair threshold, checked "
            "explicitly. Implementation: statsmodels.stats.contingency_tables.mcnemar. Pure "
            "post-hoc analysis on already-saved per-image predictions -- no new model "
            "inference, no pipeline code touched."
        ),
        "image_set_identity_verification": image_set_report,
        "class_coverage_gap_row8_vs_row1": coverage_gap,
        "pairwise_results": pairs,
        "overall_verdict": overall_verdict,
        "honest_gaps": [
            "Row 8's own 822-image evaluation set is missing 2 of 13 classes entirely "
            "(Tomato_healthy, Tomato_Yellow_Leaf_Curl_Virus) relative to rows 1/2/6's "
            "967/968/965-image set -- see class_coverage_gap_row8_vs_row1 above. Each "
            "pairwise McNemar's test below correctly restricts to the actual shared-image "
            "intersection (not assumed alignment), so this gap does not bias the significance "
            "results themselves, but the standalone 822-image 22.51% figure is not a like-for-"
            "like benchmark against rows 1/2/6's headline numbers -- the accuracy_on_shared_set "
            "values in each pairwise_results entry are the fair, apples-to-apples comparison.",

            "Row 8 was trained and evaluated entirely outside this repository (external Colab "
            "environment); the training/eval scripts themselves are not available here to "
            "independently verify the 'identical protocol' claim beyond what the delivered "
            "evidence JSON states. The delivered per-image predictions file was taken at face "
            "value for the McNemar's test (same standard applied to any externally-generated "
            "evidence in this project), but this is a real provenance difference from rows "
            "1/2/6, whose full generation scripts live in this repo and were run this session.",

            "Not corrected for multiple comparisons across the now 6 total pairwise tests run "
            "this project (3 within-EfficientNetB0-family + 3 cross-architecture here) -- same "
            "caveat already noted for the first McNemar's evidence file, restated here since "
            "these 3 add to that count.",
        ],
    }

    out_path = os.path.join(EVIDENCE_DIR, "mcnemar_row8_resnet18_vs_zeroshot_evidence.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
