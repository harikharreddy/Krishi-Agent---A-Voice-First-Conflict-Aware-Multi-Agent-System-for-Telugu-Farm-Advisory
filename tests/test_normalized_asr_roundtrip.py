"""
Phase 5.2 -- ASR round-trip check on NORMALIZED TTS output.
Same method as test_tts_asr_roundtrip.py, but on the
numeral_review_normalized/ files, to compare against the raw-digit
baseline and see whether text normalization actually helped.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import soundfile as sf
from transformers import AutoModel

AUDIO_DIR = "tests/audio/numeral_review_normalized"
LANGUAGE = "te"

EXPECTED_NORMALIZED_TEXT = {
    "price_sell_4digit.wav": "ఈరోజు Tomato ధర రెండు వేయి నాలుగు వందల యాభై రూ., ఈ నెలలో సాధారణ ధర (ఒకటి వేయి తొమ్మిది వంద రూ.) కంటే ఇరవై తొమ్మిది శాతం ఎక్కువ. ఇప్పుడు అమ్మడం మంచిది.",
    "price_hold_3digit.wav": "ఈరోజు Potato ధర ఎనిమిది వందల ఇరవై రూ., ఈ నెలలో సాధారణ ధర (తొమ్మిది వందల ఎనభై రూ.) కంటే పదహారు శాతం తక్కువ. వీలైతే ఆగడం మంచిది.",
    "weather_3hour.wav": "రాబోయే మూడు గంటల్లో వర్షం పడే అవకాశం ఉంది (వర్షం సంభావ్యత అరవై అయిదు శాతం).",
    "weather_24hour.wav": "రాబోయే ఇరవై నాలుగు గంటల్లో వర్షం పడే అవకాశం ఉంది (వర్షం సంభావ్యత నలభై అయిదు శాతం).",
}


def main():
    print("Loading ASR model...")
    asr_model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
    )
    asr_model.eval()
    print("Model loaded.\n")

    for filename, expected in EXPECTED_NORMALIZED_TEXT.items():
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
