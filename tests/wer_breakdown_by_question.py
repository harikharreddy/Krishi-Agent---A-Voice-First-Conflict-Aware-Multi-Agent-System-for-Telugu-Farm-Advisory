"""
Phase 8.1 metric #2 addendum: per-QUESTION WER breakdown, distinct from
check_all_wer.py's per-SPEAKER view. Answers a different question: not
"how did each speaker do overall" but "which of the 16 specific questions
(farming vocabulary terms) are hardest to recognize, across whichever
speakers have been recorded so far."

Reads tests/audio/wer_eval/wer_results.json (written by check_all_wer.py)
-- run that first. Designed to be run as-is once all 4 speakers are done,
not rewritten; with only 1 speaker present it still runs correctly, just
with n=1 per question (each question's "mean" trivially equals that one
speaker's WER) -- flagged explicitly in the output so it isn't mistaken
for a real cross-speaker result.
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
WER_RESULTS_PATH = os.path.join(HERE, "audio", "wer_eval", "wer_results.json")


def main():
    if not os.path.isfile(WER_RESULTS_PATH):
        print(f"ERROR: {WER_RESULTS_PATH} not found -- run check_all_wer.py first.")
        return

    with open(WER_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    by_question = {}
    for r in data["per_question"]:
        by_question.setdefault(r["id"], []).append(r)

    n_speakers = len(data["by_speaker"])
    rows = []
    for qid in sorted(by_question.keys()):
        recs = by_question[qid]
        wers = [r["wer"] for r in recs]
        rows.append({
            "id": qid,
            "reference": recs[0]["reference"],
            "n_speakers_recorded": len(recs),
            "mean_wer": statistics.mean(wers),
            "min_wer": min(wers),
            "max_wer": max(wers),
            "per_speaker": {r["speaker"]: r["wer"] for r in recs},
        })

    rows_sorted_hardest_first = sorted(rows, key=lambda r: -r["mean_wer"])

    is_complete = data.get("is_complete", False)
    print(f"Speakers present: {n_speakers}/4  |  {'FINAL' if is_complete else 'PRELIMINARY -- single-speaker view only, not a real cross-speaker comparison yet' if n_speakers == 1 else 'PRELIMINARY'}")
    print(f"\n{'Q':<4}{'n':<4}{'mean WER':<10}{'min':<8}{'max':<8}Question")
    for r in rows_sorted_hardest_first:
        print(f"{r['id']:<4}{r['n_speakers_recorded']:<4}{r['mean_wer']:<10.3f}{r['min_wer']:<8.3f}{r['max_wer']:<8.3f}{r['reference']}")

    evidence = {
        "metric": "ASR WER breakdown by question (which farming-vocabulary questions are hardest, across speakers)",
        "reporting_status": (
            "FINAL" if is_complete else
            f"PRELIMINARY -- {n_speakers}/4 speakers. With n=1, 'mean/min/max "
            "across speakers' trivially collapses to that one speaker's own "
            "WER per question -- this is a mechanism dry-run, not yet a real "
            "cross-speaker hardest-question finding. Re-run once all 4 "
            "speakers are recorded."
        ),
        "n_speakers": n_speakers,
        "by_question_hardest_first": rows_sorted_hardest_first,
    }

    out_path = os.path.join(HERE, "audio", "wer_eval", "wer_breakdown_by_question.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
