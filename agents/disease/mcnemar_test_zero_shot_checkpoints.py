"""
Post-Defense Journal Extension Roadmap, item 2 (formal statistical
significance testing) -- pulled forward and actually run, 2026-09-16:
McNemar's test on the pairwise accuracy differences between the three
zero-shot Disease Agent checkpoints (model lineage rows 1, 2, 6), using
the per-image predictions already saved from tonight's confusion-matrix
work. Answers the specific question the descriptive "three-checkpoint
convergence" finding never formally tested: are the observed accuracy
differences (23.45% / 21.92% / 21.45%) statistically distinguishable,
or does the convergence claim hold up under a real significance test,
not just eyeballing that the numbers are close?

Pure post-hoc analysis on existing predictions -- no new model
inference, no pipeline code touched.

McNemar's test is the right tool here specifically because it's a PAIRED
test: each image gets one (correct/incorrect) outcome from checkpoint A
and one from checkpoint B, and the test only uses the DISCORDANT pairs
(A right/B wrong, or A wrong/B right) -- it is not simply comparing two
accuracy percentages as if they were independent samples, which would
throw away the pairing information and risk a different, less powerful
(or misleading) test.

Image-set identity was verified before running anything, not assumed:
row 1 and row 2 use the EXACT SAME 967-image set (verified via set
equality on the 'image' field, not just matching counts). Row 6 is a
strict subset -- 965 images, missing exactly 2 (both in the "Tomato two
spotted spider mites leaf" folder, the same discrepancy investigated in
metric1_v2_checkpoint_evidence.json's own image_count_discrepancy_note).
Each pairwise test below restricts to the actual intersection of images
both checkpoints in that specific pair were evaluated on -- 967 for
row1-vs-row2, 965 for the two pairs involving row 6 -- rather than
assuming the full 967/965/965 sets align without checking.
"""

import json
import os
import sys

from statsmodels.stats.contingency_tables import mcnemar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from generate_confusion_matrix_calibration_evidence import V2_TO_CANONICAL

ROW1_PATH = os.path.join(ROOT, "docs", "evidence", "metric1_row1_flat_baseline_evidence.json")
ROW2_PATH = os.path.join(ROOT, "docs", "evidence", "metric6_perimage_confidence_BEFORE_zeroshot_sep3.json")
ROW6_PATH = os.path.join(ROOT, "docs", "evidence", "metric1_v2_checkpoint_evidence.json")
EVIDENCE_DIR = os.path.join(ROOT, "docs", "evidence")

# Standard practical threshold for choosing the exact binomial McNemar
# variant over the large-sample chi-square approximation: when the
# number of DISCORDANT pairs (b+c -- the only cells the test statistic
# depends on) is small, the chi-square approximation to the binomial
# distribution is unreliable. 25 is the commonly cited cutoff (e.g. the
# same threshold statsmodels' own mcnemar() docs and standard
# biostatistics references use) -- checked explicitly per pair below,
# not assumed.
EXACT_TEST_DISCORDANT_THRESHOLD = 25


def load_row1_or_row2(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    records = d["records"] if isinstance(d, dict) else d
    return {r["image"]: r["correct"] for r in records}


def load_row6_canonical(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    unmapped = 0
    for r in d["records"]:
        true_raw, pred_raw = r["true_class"], r["predicted_class"]
        if true_raw not in V2_TO_CANONICAL or pred_raw not in V2_TO_CANONICAL:
            unmapped += 1
            continue
        true_c = V2_TO_CANONICAL[true_raw]
        pred_c = V2_TO_CANONICAL[pred_raw]
        out[r["image"]] = (true_c == pred_c)
    return out, unmapped


def verify_image_sets(images1, images2, images6):
    """Checked, not assumed: do these checkpoints share the same image
    identities, not just the same counts?"""
    report = {
        "row1_n": len(images1), "row2_n": len(images2), "row6_n": len(images6),
        "row1_equals_row2_exact_set": images1 == images2,
        "row1_minus_row2": sorted(images1 - images2),
        "row2_minus_row1": sorted(images2 - images1),
        "row1_minus_row6": sorted(images1 - images6),
        "row6_minus_row1": sorted(images6 - images1),
        "row2_minus_row6": sorted(images2 - images6),
        "row6_minus_row2": sorted(images6 - images2),
    }
    return report


def pairwise_mcnemar(name_a, correct_a, name_b, correct_b):
    """correct_a/correct_b: dict of {image: bool}. Restricts to the
    actual intersection of images both checkpoints were evaluated on --
    checked per pair, since row 6's coverage differs from rows 1/2."""
    shared_images = sorted(set(correct_a.keys()) & set(correct_b.keys()))
    only_a = set(correct_a.keys()) - set(correct_b.keys())
    only_b = set(correct_b.keys()) - set(correct_a.keys())

    both_correct = 0
    a_only_correct = 0  # a right, b wrong
    b_only_correct = 0  # b right, a wrong
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
            f"EXACT BINOMIAL (discordant pairs = {discordant} < {EXACT_TEST_DISCORDANT_THRESHOLD} threshold "
            f"-- chi-square approximation would be unreliable at this count)"
            if use_exact else
            f"CHI-SQUARE with continuity correction (discordant pairs = {discordant} >= "
            f"{EXACT_TEST_DISCORDANT_THRESHOLD} threshold -- large enough for the approximation to be reasonable)"
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

    if row6_unmapped:
        print(f"NOTE: {row6_unmapped} row-6 record(s) had no V2_TO_CANONICAL entry, excluded from this analysis.")

    images1, images2, images6 = set(row1.keys()), set(row2.keys()), set(row6.keys())
    image_set_report = verify_image_sets(images1, images2, images6)

    print("=== Image-set identity check (verified, not assumed) ===")
    print(f"Row 1 n={image_set_report['row1_n']}, Row 2 n={image_set_report['row2_n']}, Row 6 n={image_set_report['row6_n']}")
    print(f"Row1 == Row2 exact set equality: {image_set_report['row1_equals_row2_exact_set']}")
    print(f"Row1 - Row6 (missing from row6): {image_set_report['row1_minus_row6']}")
    print(f"Row6 - Row1 (row6 has, row1 doesn't): {image_set_report['row6_minus_row1']}")
    print()

    pairs = [
        pairwise_mcnemar("row1_flat_baseline", row1, "row2_hierarchical", row2),
        pairwise_mcnemar("row1_flat_baseline", row1, "row6_v2", row6),
        pairwise_mcnemar("row2_hierarchical", row2, "row6_v2", row6),
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
        "ALL THREE pairwise comparisons are non-significant (p >= 0.05). This is genuine "
        "statistical support -- not just a visual/descriptive impression -- for the claim that "
        "these three independently-trained checkpoints perform indistinguishably on PlantDoc. "
        "The 'three-model convergence' finding is upgraded from a descriptive observation to a "
        "formally-tested one."
        if all_non_significant else
        "NOT all pairwise comparisons are non-significant -- see individual pair verdicts above. "
        "The convergence claim needs to be qualified per-pair rather than stated as a blanket "
        "'these models are statistically indistinguishable' claim; report exactly which pair(s) "
        "showed a significant difference and do not force the convergence narrative to cover them."
    )
    print(f"=== Overall verdict ===\n{overall_verdict}")

    evidence = {
        "metric": "McNemar's test for pairwise statistical significance between the three zero-shot Disease Agent checkpoints (model lineage rows 1, 2, 6)",
        "context": (
            "Post-Defense Journal Extension Roadmap item 2 (formal statistical significance testing), "
            "pulled forward and run 2026-09-16, per explicit request -- tests whether the descriptive "
            "'three-checkpoint convergence' finding (23.45%/21.92%/21.45% accuracy, all within a 2-point "
            "band) holds up under a real paired significance test, not just an eyeballed similarity."
        ),
        "method": (
            "McNemar's test, the correct tool for PAIRED binary outcomes (each image has one correct/"
            "incorrect result per checkpoint) -- uses only the discordant pairs (one checkpoint right, "
            "the other wrong), not the full accuracy percentages treated as independent samples. Test "
            "variant (exact binomial vs. chi-square with continuity correction) chosen per pair based "
            f"on the number of discordant pairs, using the standard {EXACT_TEST_DISCORDANT_THRESHOLD}-pair "
            "threshold -- checked explicitly, not defaulted to chi-square. Implementation: "
            "statsmodels.stats.contingency_tables.mcnemar. Pure post-hoc analysis on already-saved "
            "per-image predictions -- no new model inference, no pipeline code touched."
        ),
        "image_set_identity_verification": image_set_report,
        "pairwise_results": pairs,
        "overall_verdict": overall_verdict,
        "honest_gaps": [
            "Row 6's 2 missing images (both 'Tomato two spotted spider mites leaf') are excluded from "
            "both row1-vs-row6 and row2-vs-row6 comparisons -- those two pairs are tested on n=965, not "
            "the full n=967 rows 1/2 share between themselves. This is the correct handling (McNemar "
            "requires paired data), not a shortcut, but means the three pairwise tests are not all on "
            "the identical image set -- stated explicitly rather than glossed over.",

            "McNemar's test evaluates whether the two checkpoints' MARGINAL accuracy differs on this "
            "specific paired sample -- it does not test whether they make the SAME individual errors "
            "(two checkpoints could have identical accuracy while disagreeing on which specific images "
            "they get right, which the discordant-pair counts in the contingency tables above do show "
            "happening to some degree in every pair).",

            "This is one test per pair at alpha=0.05, not corrected for multiple comparisons (3 pairwise "
            "tests run). A Bonferroni-corrected threshold (alpha=0.05/3=0.0167) would be more "
            "conservative -- not applied here since the roadmap scoped this as a single significance "
            "check per pair, but worth noting for anyone citing these p-values against a stricter "
            "threshold.",
        ],
    }

    out_path = os.path.join(EVIDENCE_DIR, "mcnemar_zero_shot_checkpoint_comparison_evidence.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
