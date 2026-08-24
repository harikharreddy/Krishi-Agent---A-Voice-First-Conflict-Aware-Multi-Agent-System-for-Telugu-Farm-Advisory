"""
Standalone TTS sanity check — Phase 5.1.
Loads ai4bharat/indic-parler-tts and synthesizes a short Telugu sentence
to a local WAV file. No pipeline wiring — just proving it works locally.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

MODEL_ID = "ai4bharat/indic-parler-tts"
OUTPUT_PATH = "tests/audio/tts_output.wav"

# Same test sentence as the ASR check, so the round-trip is comparable
TEXT = "నా పంట ఆకులపై మచ్చలు కనిపిస్తున్నాయి"

# Phase 0 recommended voices: "Lalitha" or "Prakash"
DESCRIPTION = "Lalitha's voice is calm and clear, with a natural Telugu accent, at a moderate speed with high quality recording."

def main():
    device = "cpu"
    print(f"Loading {MODEL_ID} ...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    print("Model loaded.")

    input_ids = description_tokenizer(DESCRIPTION, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(TEXT, return_tensors="pt").input_ids.to(device)

    print("Generating audio (this may take a minute on CPU)...")
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    sf.write(OUTPUT_PATH, audio_arr, model.config.sampling_rate)
    print(f"Saved audio to {OUTPUT_PATH}")
    print(f"Sample rate: {model.config.sampling_rate}, duration: {len(audio_arr) / model.config.sampling_rate:.2f}s")

if __name__ == "__main__":
    main()
