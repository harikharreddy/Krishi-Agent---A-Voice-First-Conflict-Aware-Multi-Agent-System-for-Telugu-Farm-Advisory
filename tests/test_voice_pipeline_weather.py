"""
Voice pipeline wiring test #2 -- Phase 5.1.
Tests a question that should get a REAL agent answer (not the
unresolved-conflict fallback), to confirm pipeline->TTS handles
substantive Telugu text correctly, not just the fallback sentence.
Skips ASR here (already proven standalone) -- goes straight from
a known-good text question into run_pipeline() -> TTS.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

from orchestrator.pipeline import run_pipeline

QUESTION = "రేపు వర్షం పడుతుందా?"
OUTPUT_AUDIO = "tests/audio/voice_pipeline_weather_output.wav"

TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
TTS_DESCRIPTION = "Lalitha's voice is calm and clear, with a natural Telugu accent, at a moderate speed with high quality recording."

test_farm_profile = {
    "district": "Warangal",
    "mandi": "Warangal APMC",
    "state": "Telangana",
    "crop": "Tomato",
}


def main():
    print(f"Question (text, ASR skipped): {QUESTION}")

    print("Running text pipeline...")
    result = run_pipeline(QUESTION, test_farm_profile, image_path=None)
    answer_text = result["final_answer"]
    print(f"Pipeline answer: {answer_text}")

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
