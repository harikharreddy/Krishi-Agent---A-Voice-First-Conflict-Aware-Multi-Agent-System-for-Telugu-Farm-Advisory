"""
Phase 8.1, evaluation metric #2: ASR Word Error Rate on farming vocabulary.

Runs indic-conformer-600m against every recorded speaker's 16 samples
(tests/audio/wer_eval/speaker<N>/q1.wav..q16.wav), scoring each against
its reference transcript from tests/intent_router_test_set.json (same
16-question set used by Phase 2.1-2.2 and Phase 4.2c's pipeline sanity
test -- q<id>.wav was recorded reading question `id` from that file, in
order). Auto-discovers however many of the 4 planned speakers actually
have recordings present -- designed to be run as-is once all 4 are done,
not rewritten.

Supersedes check_q1_wer.py (q1-only) -- kept alongside it since removing a
previously-committed file isn't this script's job.
"""

import glob
import json
import os
import re

import jiwer
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO_ROOT = os.path.join(HERE, "audio", "wer_eval")
PLANNED_SPEAKERS = 4
QUESTIONS_PER_SPEAKER = 16


def normalize(text):
    text = re.sub(r"[.,!?।]", "", text)
    return " ".join(text.split())


def discover_speakers():
    """Any tests/audio/wer_eval/speaker*/ directory containing at least one
    q*.wav file counts as present -- doesn't assume all 4 exist."""
    speaker_dirs = sorted(glob.glob(os.path.join(AUDIO_ROOT, "speaker*")))
    return [d for d in speaker_dirs if glob.glob(os.path.join(d, "q*.wav"))]


def main():
    with open(os.path.join(HERE, "intent_router_test_set.json"), encoding="utf-8") as f:
        test_set = json.load(f)

    speaker_dirs = discover_speakers()
    speaker_names = [os.path.basename(d) for d in speaker_dirs]
    print(f"Speakers found: {speaker_names} ({len(speaker_dirs)}/{PLANNED_SPEAKERS} planned)")
    if not speaker_dirs:
        print("No speaker recordings found -- nothing to evaluate.")
        return

    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
    )
    model.eval()

    by_speaker = {}
    all_results = []

    for speaker_dir in speaker_dirs:
        speaker = os.path.basename(speaker_dir)
        results = []
        missing = []
        for item in test_set:
            qid = item["id"]
            reference = item["question"]
            wav_path = os.path.join(speaker_dir, f"q{qid}.wav")
            if not os.path.isfile(wav_path):
                missing.append(qid)
                continue

            wav_np, sr = sf.read(wav_path, dtype="float32")
            if wav_np.ndim == 1:
                wav_np = wav_np[np.newaxis, :]
            wav = torch.from_numpy(wav_np)

            with torch.no_grad():
                transcription = model(wav, "te", "rnnt")

            ref_norm = normalize(reference)
            hyp_norm = normalize(transcription)
            wer = jiwer.wer(ref_norm, hyp_norm)
            record = {"speaker": speaker, "id": qid, "reference": ref_norm, "hypothesis": hyp_norm, "wer": wer}
            results.append(record)
            all_results.append(record)

            print(f"[{speaker}] q{qid:<2} WER={wer:.3f}  ref: {ref_norm}")
            print(f"{'':>{len(speaker)+4}}      hyp: {hyp_norm}")

        speaker_mean = sum(r["wer"] for r in results) / len(results) if results else None
        by_speaker[speaker] = {
            "n_recordings": len(results),
            "n_expected": QUESTIONS_PER_SPEAKER,
            "missing_question_ids": missing,
            "mean_wer": speaker_mean,
        }
        if speaker_mean is not None:
            print(f"\n[{speaker}] mean WER across {len(results)}/{QUESTIONS_PER_SPEAKER}: {speaker_mean:.4f}\n")

    overall_mean = sum(r["wer"] for r in all_results) / len(all_results) if all_results else None
    total_expected = PLANNED_SPEAKERS * QUESTIONS_PER_SPEAKER
    is_complete = len(speaker_dirs) == PLANNED_SPEAKERS and len(all_results) == total_expected

    print(f"\n{'='*70}")
    print(f"OVERALL: {len(all_results)}/{total_expected} recordings evaluated "
          f"({len(speaker_dirs)}/{PLANNED_SPEAKERS} speakers)")
    print(f"Mean WER (all recordings evaluated so far): {overall_mean:.4f}" if overall_mean is not None else "No recordings.")
    if not is_complete:
        print(f"⚠ PRELIMINARY -- only {len(speaker_dirs)}/{PLANNED_SPEAKERS} speakers present. "
              f"This mean WER reflects {'1 speaker only' if len(speaker_dirs) == 1 else f'{len(speaker_dirs)} speakers only'}, "
              f"not the full multi-speaker methodology (16 questions x 4 speakers). "
              f"Do not report as final until all {PLANNED_SPEAKERS} speakers are recorded.")

    output = {
        "metric": "ASR Word Error Rate on farming vocabulary",
        "method": "jiwer, RNNT decoding via ai4bharat/indic-conformer-600m-multilingual, 16 questions per speaker",
        "planned_speakers": PLANNED_SPEAKERS,
        "planned_questions_per_speaker": QUESTIONS_PER_SPEAKER,
        "is_complete": is_complete,
        "overall_mean_wer": overall_mean,
        "by_speaker": by_speaker,
        "per_question": all_results,
        "reporting_status": (
            "FINAL" if is_complete else
            f"PRELIMINARY -- {len(speaker_dirs)}/{PLANNED_SPEAKERS} speakers recorded. "
            "Do not cite overall_mean_wer as the project's ASR result until all "
            f"{PLANNED_SPEAKERS} speakers are complete; single-speaker WER does not "
            "measure cross-speaker/accent robustness, which is the actual point of "
            "the 4-speaker design."
        ),
    }

    out_path = os.path.join(AUDIO_ROOT, "wer_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
