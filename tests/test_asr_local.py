"""
Standalone ASR sanity check — Phase 5.1.
Loads ai4bharat/indic-conformer-600m-multilingual and transcribes
tests/audio/test_recording.wav (Telugu, 16kHz mono, from Phase 0).
No pipeline wiring — just proving the model loads and runs locally.
"""

import torch
import soundfile as sf
import numpy as np
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
AUDIO_PATH = "tests/audio/test_recording.wav"
LANGUAGE = "te"  # Telugu

def main():
    print(f"Loading {MODEL_ID} ...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    print("Model loaded.")

    print(f"Loading audio: {AUDIO_PATH}")
    wav_np, sr = sf.read(AUDIO_PATH, dtype="float32")
    print(f"Sample rate: {sr}, shape: {wav_np.shape}")

    if wav_np.ndim == 1:
        wav_np = wav_np[np.newaxis, :]

    wav = torch.from_numpy(wav_np)

    if sr != 16000:
        raise ValueError(f"Expected 16000 Hz, got {sr} Hz")

    print("Running RNNT decoding...")
    with torch.no_grad():
        transcription_rnnt = model(wav, LANGUAGE, "rnnt")
    print(f"RNNT transcription: {transcription_rnnt}")

    print("Running CTC decoding...")
    with torch.no_grad():
        transcription_ctc = model(wav, LANGUAGE, "ctc")
    print(f"CTC transcription:  {transcription_ctc}")

if __name__ == "__main__":
    main()
