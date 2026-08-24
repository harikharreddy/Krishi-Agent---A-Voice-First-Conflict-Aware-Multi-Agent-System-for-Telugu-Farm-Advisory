"""
Voice pipeline wiring test — Phase 5.1.
audio in -> ASR -> run_pipeline() [Phase 4, untouched] -> TTS -> audio out
No Streamlit/mic involved yet -- that's Phase 5.2.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import soundfile as sf
from transformers import AutoModel, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

from orchestrator.pipeline import run_pipeline

INPUT_AUDIO = "tests/audio/test_recording.wav"
OUTPUT_AUDIO = "tests/audio/voice_pipeline_output.wav"
LANGUAGE = "te"

TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
TTS_DESCRIPTION = "Lalitha's voice is calm and clear, with a natural Telugu accent, at a moderate speed with high quality recording."

test_farm_profile = {
    "district": "Warangal",
    "mandi": "Warangal APMC",
    "state": "Telangana",
    "crop": "Tomato",
}


def main():
    # --- Step 1: ASR ---
    print("Loading ASR model...")
    asr_model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
    )
    asr_model.eval()

    print(f"Transcribing {INPUT_AUDIO} ...")
    wav_np, sr = sf.read(INPUT_AUDIO, dtype="float32")
    if wav_np.ndim == 1:
        import numpy as np
        wav_np = wav_np[np.newaxis, :]
    wav = torch.from_numpy(wav_np)
    if sr != 16000:
        raise ValueError(f"Expected 16000 Hz, got {sr} Hz")

    with torch.no_grad():
        question = asr_model(wav, LANGUAGE, "rnnt")
    print(f"Transcribed question: {question}")

    # --- Step 2: Text pipeline (Phase 4, untouched) ---
    print("Running text pipeline...")
    result = run_pipeline(question, test_farm_profile, image_path=None)
    answer_text = result["final_answer"]
    print(f"Pipeline answer: {answer_text}")

    # --- Step 3: TTS ---
    print("Loading TTS model...")
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL_ID).to("cpu")
    tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    description_tokenizer = AutoTokenizer.from_pretrained(
        tts_model.config.text_encoder._name_or_path
    )

    input_ids = description_tokenizer(TTS_DESCRIPTION, return_tensors="pt").input_ids
    prompt_input_ids = tts_tokenizer(answer_text, return_tensors="pt").input_ids

    print("Synthesizing answer audio...")
    generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    sf.write(OUTPUT_AUDIO, audio_arr, tts_model.config.sampling_rate)
    print(f"Saved spoken answer to {OUTPUT_AUDIO}")
    print(f"Duration: {len(audio_arr) / tts_model.config.sampling_rate:.2f}s")


if __name__ == "__main__":
    main()
