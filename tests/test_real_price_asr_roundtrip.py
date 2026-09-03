"""
Phase 8 prep: ASR round-trip check on REAL live Price Agent output
(paired with test_tts_real_price_output.py), closing the gap Phase 5.2
flagged as open -- previous roundtrip testing used only synthetic sentences.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
import soundfile as sf
from transformers import AutoModel

AUDIO_DIR = "tests/audio/real_price_review"
LANGUAGE = "te"


def main():
    expected_path = os.path.join(AUDIO_DIR, "expected_texts.json")
    with open(expected_path, encoding="utf-8") as f:
        expected_texts = json.load(f)

    print("Loading ASR model...")
    asr_model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
    )
    asr_model.eval()
    print("Model loaded.\n")

    for filename, expected in expected_texts.items():
        path = os.path.join(AUDIO_DIR, filename)
        wav_np, sr = sf.read(path, dtype="float32")
        if wav_np.ndim == 1:
            wav_np = wav_np[np.newaxis, :]
        wav = torch.from_numpy(wav_np)

        if sr != 16000:
            wav = torch.from_numpy(
                np.interp(
                    np.linspace(0, len(wav_np[0]), int(len(wav_np[0]) * 16000 / sr)),
                    np.arange(len(wav_np[0])),
                    wav_np[0],
                ).astype(np.float32)
            ).unsqueeze(0)

        with torch.no_grad():
            transcription = asr_model(wav, LANGUAGE, "rnnt")

        print(f"=== {filename} ===")
        print(f"Expected:     {expected}")
        print(f"ASR heard:    {transcription}")
        print()


if __name__ == "__main__":
    main()
