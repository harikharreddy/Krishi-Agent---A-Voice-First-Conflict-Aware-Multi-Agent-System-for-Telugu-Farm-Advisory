"""
Phase 5.2 -- Batch TTS synthesis of all Phrasing Template sentences
(Phase 3.2), for fluent-Telugu-speaker pronunciation review.
Each template gets its own output file, named by (resolution, confidence).
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

from orchestrator.phrasing_templates import TEMPLATES, FALLBACK_PHRASE

OUTPUT_DIR = "tests/audio/phrasing_review"
TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
TTS_DESCRIPTION = "Lalitha's voice is calm and clear, with a natural Telugu accent, at a moderate speed with high quality recording."


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading TTS model...")
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL_ID).to("cpu")
    tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    description_tokenizer = AutoTokenizer.from_pretrained(
        tts_model.config.text_encoder._name_or_path
    )
    input_ids = description_tokenizer(TTS_DESCRIPTION, return_tensors="pt").input_ids
    print("Model loaded.\n")

    items = list(TEMPLATES.items())
    items.append((("FALLBACK", "n/a"), FALLBACK_PHRASE))

    for (resolution, confidence), text in items:
        safe_name = f"{resolution}_{confidence}".replace(" ", "_").replace("/", "-")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.wav")

        print(f"[{resolution} / {confidence}] {text}")
        prompt_input_ids = tts_tokenizer(text, return_tensors="pt").input_ids
        generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(out_path, audio_arr, tts_model.config.sampling_rate)
        print(f"  -> saved {out_path}\n")

    print(f"Done. {len(items)} files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
