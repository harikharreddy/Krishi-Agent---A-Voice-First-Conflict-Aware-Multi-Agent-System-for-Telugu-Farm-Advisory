"""
Phase 8.1, evaluation metric #? extension (per external technical review,
2026-09-13, Sec 2.3): the Intent Router's existing 87.5% (14/16) result
has no per-intent breakdown -- with 4 target labels and only 16 test
questions, it was previously impossible to tell whether the 2 failures
concentrate in one intent category (e.g. ambiguous multi-intent
questions) or spread evenly. This script re-runs the same 16-question
test set through the real, unmocked route_intent() (live Ollama call,
qwen2.5:7b-instruct) and reports both the aggregate exact-match result
and a per-label (wants_disease/wants_weather/wants_price/wants_soil)
precision/recall/F1 breakdown, treating each label as its own binary
classification task across all 16 questions.

Also flags external-validity limitation the review raised: this 16-
question set is small and was used as both the few-shot example source
(SYSTEM_PROMPT in orchestrator/intent_router.py) and the test set for
the same intent categories -- any accuracy figure from it should not be
read as an unbiased estimate of field performance on novel phrasing.

Observation only -- orchestrator/intent_router.py is not modified.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from orchestrator.intent_router import route_intent

TEST_SET_PATH = os.path.join(HERE, "intent_router_test_set.json")
LABELS = ["wants_disease", "wants_weather", "wants_price", "wants_soil"]


def main():
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)

    rows = []
    exact_match_count = 0
    per_label_counts = {label: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for label in LABELS}

    for item in test_set:
        qid = item["id"]
        question = item["question"]
        expected = item["expected"]

        actual = route_intent(question)

        label_results = {}
        for label in LABELS:
            exp_v = bool(expected[label])
            act_v = bool(actual.get(label, False))
            label_results[label] = {"expected": exp_v, "actual": act_v, "correct": exp_v == act_v}
            if exp_v and act_v:
                per_label_counts[label]["tp"] += 1
            elif not exp_v and act_v:
                per_label_counts[label]["fp"] += 1
            elif not exp_v and not act_v:
                per_label_counts[label]["tn"] += 1
            else:
                per_label_counts[label]["fn"] += 1

        exact_match = all(label_results[label]["correct"] for label in LABELS)
        if exact_match:
            exact_match_count += 1

        wrong_labels = [label for label in LABELS if not label_results[label]["correct"]]

        row = {
            "id": qid,
            "gloss": item["gloss"],
            "question": question,
            "expected": expected,
            "actual": {label: actual.get(label) for label in LABELS},
            "exact_match": exact_match,
            "wrong_labels": wrong_labels,
            "n_intents_expected_true": sum(1 for label in LABELS if expected[label]),
        }
        rows.append(row)

        print(f"q{qid:<2} exact_match={exact_match}  wrong_labels={wrong_labels or 'none'}")
        print(f"    expected: {expected}")
        print(f"    actual:   {actual}")

    per_label_metrics = {}
    for label in LABELS:
        c = per_label_counts[label]
        n = c["tp"] + c["fp"] + c["tn"] + c["fn"]
        accuracy = (c["tp"] + c["tn"]) / n if n else None
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else None
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision is not None and recall is not None and (precision + recall) > 0) else None
        per_label_metrics[label] = {
            "n_positive_in_test_set": c["tp"] + c["fn"],
            "tp": c["tp"], "fp": c["fp"], "tn": c["tn"], "fn": c["fn"],
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        }

    # Are the 2 failures concentrated in one intent category, or spread evenly?
    failure_rows = [r for r in rows if not r["exact_match"]]
    label_failure_counts = {label: 0 for label in LABELS}
    for r in failure_rows:
        for label in r["wrong_labels"]:
            label_failure_counts[label] += 1
    multi_intent_failure_count = sum(1 for r in failure_rows if r["n_intents_expected_true"] >= 2)
    n_labels_with_failures = sum(1 for v in label_failure_counts.values() if v > 0)
    if not failure_rows:
        failure_verdict = "NO FAILURES THIS RUN"
    elif n_labels_with_failures <= 1:
        failure_verdict = "CONCENTRATED"
    else:
        failure_verdict = "SPREAD ACROSS MULTIPLE INTENT CATEGORIES"

    evidence = {
        "metric": "Intent Router per-intent breakdown -- are the known 87.5% (14/16) result's failures concentrated in one intent category or spread evenly?",
        "method": (
            "Re-runs the same 16-question test set (tests/intent_router_test_set.json) through "
            "the real, unmocked route_intent() -- live Ollama call, qwen2.5:7b-instruct, same "
            "model and prompt as production. Each of the 4 target labels "
            "(wants_disease/wants_weather/wants_price/wants_soil) is scored independently across "
            "all 16 questions as its own binary classification task (TP/FP/TN/FN), in addition to "
            "the aggregate exact-match (all 4 labels correct) result already reported elsewhere."
        ),
        "aggregate_result": {
            "exact_match": f"{exact_match_count}/{len(rows)}",
            "exact_match_pct": exact_match_count / len(rows),
        },
        "per_intent_label_breakdown": per_label_metrics,
        "failure_concentration_analysis": {
            "n_questions_with_at_least_1_wrong_label": len(failure_rows),
            "wrong_label_counts_by_intent": label_failure_counts,
            "verdict": failure_verdict,
            "n_failures_that_were_multi_intent_questions": multi_intent_failure_count,
            "note": (
                "A question is 'multi-intent' if the ground truth expects 2+ of the 4 labels "
                "true simultaneously (e.g. a combined disease+weather question). If failures "
                "cluster on multi-intent questions specifically, that points to a different root "
                "cause (the LLM struggling to hold multiple simultaneous judgments) than if they "
                "cluster on a single intent category regardless of question complexity."
            ),
        },
        "raw_comparison_table": rows,
        "external_validity_limitation": (
            "This same 16-question set supplies the few-shot examples baked into "
            "orchestrator/intent_router.py's SYSTEM_PROMPT (5 of these exact questions appear "
            "verbatim as worked examples in the prompt) AND is the test set used here and in the "
            "original 87.5% result -- the model is being tested partly on phrasing patterns it "
            "was directly shown. This is a real external-validity limitation, not just a small-n "
            "caveat: an accuracy figure from this set should not be read as an unbiased estimate "
            "of field performance on genuinely novel farmer phrasing. Phase 6 already found a "
            "real-world misrouting failure on unvalidated phrasing outside this set "
            "('ఈ ఆకుకు ఏమి జబ్బు?', see orchestrator/intent_router_results.md) -- consistent with "
            "this concern, not just a hypothetical one. Flagged explicitly per the external "
            "review's request, rather than left as an implicit assumption behind the 87.5% number."
        ),
        "honest_gaps": [
            "n=16 questions total, and per-label positive-class counts are small for the "
            "less-frequent intents (see per_intent_label_breakdown's n_positive_in_test_set per "
            "label) -- precision/recall at this n should be read as directional, not tightly "
            "estimated.",

            "The LLM-based router is not deterministic between calls in general (temperature=0.1, "
            "not 0) -- this is ONE run, not a repeated-trials stability check. A different run "
            "could plausibly produce a different exact_match count on the same 16 questions; this "
            "was not tested here (out of scope for this pass, could be added as a future "
            "stability check analogous to the Metric #6 cutoff bootstrap).",
        ],
    }

    out_dir = os.path.join(ROOT, "docs", "evidence")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metric_intent_router_per_intent_evidence.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"Exact match: {exact_match_count}/{len(rows)}")
    print(f"Per-intent breakdown:")
    for label, m in per_label_metrics.items():
        print(f"  {label:<15} acc={m['accuracy']:.3f}  precision={m['precision']}  recall={m['recall']}  f1={m['f1']}  n_pos={m['n_positive_in_test_set']}")
    print(f"\nFailure concentration verdict: {evidence['failure_concentration_analysis']['verdict']}")
    print(f"Wrong-label counts by intent: {label_failure_counts}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
