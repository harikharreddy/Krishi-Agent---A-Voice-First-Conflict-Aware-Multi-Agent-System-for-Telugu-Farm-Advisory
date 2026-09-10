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
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
    )
    model.eval()

    wav_np, sr = sf.read("tests/audio/wer_eval/speaker1/q1.wav", dtype="float32")
    if wav_np.ndim == 1:
        wav_np = wav_np[np.newaxis, :]
    wav = torch.from_numpy(wav_np)
    print(f"Sample rate: {sr}")

    with torch.no_grad():
        transcription = model(wav, "te", "rnnt")
    print(f"Raw transcription: {transcription}")

    reference = "నా టమాటా ఆకులపై మచ్చలు ఉన్నాయి, ఇది ఏ వ్యాధి?"
    ref_norm = normalize(reference)
    hyp_norm = normalize(transcription)

    print(f"Reference (norm):  {ref_norm}")
    print(f"Hypothesis (norm): {hyp_norm}")
    print(f"WER: {jiwer.wer(ref_norm, hyp_norm):.4f}")


if __name__ == "__main__":
    main()
