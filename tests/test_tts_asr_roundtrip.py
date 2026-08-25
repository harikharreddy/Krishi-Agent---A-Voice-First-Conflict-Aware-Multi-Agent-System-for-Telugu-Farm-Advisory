"""
Phase 5.2 -- TTS -> ASR round-trip check.
Feeds our numeral test audio files back through the ASR model and
prints what it transcribes, as an automated cross-check on whether
the numbers came through correctly -- independent of human listening.

NOT a perfect ground truth (ASR can mishear too), but a useful
second opinion that's repeatable without needing the team present.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import soundfile as sf
from transformers import AutoModel

AUDIO_DIR = "tests/audio/numeral_review"
LANGUAGE = "te"

EXPECTED_TEXT = {
    "price_sell_4digit.wav": "ఈరోజు Tomato ధర 2450 రూ., ఈ నెలలో సాధారణ ధర (1900 రూ.) కంటే 29% ఎక్కువ. ఇప్పుడు అమ్మడం మంచిది.",
    "price_hold_3digit.wav": "ఈరోజు Potato ధర 820 రూ., ఈ నెలలో సాధారణ ధర (980 రూ.) కంటే 16% తక్కువ. వీలైతే ఆగడం మంచిది.",
    "weather_3hour.wav": "రాబోయే 3 గంటల్లో వర్షం పడే అవకాశం ఉంది (వర్షం సంభావ్యత 65%).",
    "weather_24hour.wav": "రాబోయే 24 గంటల్లో వర్షం పడే అవకాశం ఉంది (వర్షం సంభావ్యత 45%).",
}


def main():
    print("Loading ASR model...")
    asr_model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
    )
    asr_model.eval()
    print("Model loaded.\n")

    for filename, expected in EXPECTED_TEXT.items():
        path = os.path.join(AUDIO_DIR, filename)
        wav_np, sr = sf.read(path, dtype="float32")
        if wav_np.ndim == 1:
            wav_np = wav_np[np.newaxis, :]
        wav = torch.from_numpy(wav_np)

        if sr != 16000:
            print(f"[{filename}] Sample rate is {sr}, resampling to 16000...")
            import torch.nn.functional as F
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
