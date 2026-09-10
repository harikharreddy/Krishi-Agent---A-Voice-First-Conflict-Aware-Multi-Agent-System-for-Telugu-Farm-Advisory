"""
Phase 8.1 ASR WER evaluation -- runs indic-conformer-600m against all 16
recorded speaker1 samples (tests/audio/wer_eval/speaker1/q1.wav..q16.wav),
scoring each against its reference transcript from
tests/intent_router_test_set.json (same 16-question set used by Phase 2.1-2.2
and Phase 4.2c's pipeline sanity test -- q<id>.wav was recorded reading
question `id` from that file, in order).

Supersedes check_q1_wer.py (q1-only) -- kept alongside it since removing a
previously-committed file isn't this script's job.
"""

import json
import re

import jiwer
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel


def normalize(text):
    text = re.sub(r"[.,!?।]", "", text)
    return " ".join(text.split())


def main():
    with open("tests/intent_router_test_set.json", encoding="utf-8") as f:
        test_set = json.load(f)

    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
    )
    model.eval()

    results = []
    for item in test_set:
        qid = item["id"]
        reference = item["question"]
        wav_path = f"tests/audio/wer_eval/speaker1/q{qid}.wav"

        wav_np, sr = sf.read(wav_path, dtype="float32")
        if wav_np.ndim == 1:
            wav_np = wav_np[np.newaxis, :]
        wav = torch.from_numpy(wav_np)

        with torch.no_grad():
            transcription = model(wav, "te", "rnnt")

        ref_norm = normalize(reference)
        hyp_norm = normalize(transcription)
        wer = jiwer.wer(ref_norm, hyp_norm)
        results.append({"id": qid, "reference": ref_norm, "hypothesis": hyp_norm, "wer": wer})

        print(f"q{qid:<2} WER={wer:.3f}  ref: {ref_norm}")
        print(f"        hyp: {hyp_norm}")

    mean_wer = sum(r["wer"] for r in results) / len(results)
    print(f"\nMean WER across {len(results)} questions: {mean_wer:.4f}")

    with open("tests/audio/wer_eval/speaker1_results.json", "w", encoding="utf-8") as f:
        json.dump({"mean_wer": mean_wer, "per_question": results}, f, ensure_ascii=False, indent=2)
    print("Saved -> tests/audio/wer_eval/speaker1_results.json")


if __name__ == "__main__":
    main()
