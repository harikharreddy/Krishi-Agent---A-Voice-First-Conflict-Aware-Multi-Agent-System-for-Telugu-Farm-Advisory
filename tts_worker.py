"""
Phase 6 -- Standalone TTS worker, run as a subprocess from ui/app.py.
Isolates TTS generation from Streamlit's process, since generation was
observed to take 5-7x longer when run in-process inside Streamlit
versus as a standalone script (root cause not identified after
eliminating thread count, network I/O, file-watching, and sentence
length as causes across multiple systematic tests; most pronounced on
macOS, where in-process generation was taking ~3 minutes per reply).

The input text is expected to already be normalized (e.g. via
shared.text_normalization.normalize_numerals_te) by the caller.

Usage: python3 tts_worker.py <text_file_path> <output_wav_path>
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import sys
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

TTS_DESCRIPTION = "Lalitha's voice is calm and clear, with a natural Telugu accent, at a moderate speed with high quality recording."


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 tts_worker.py <text_file_path> <output_wav_path>", file=sys.stderr)
        sys.exit(1)

    text_file_path, output_wav_path = sys.argv[1], sys.argv[2]

    with open(text_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to("cpu")
    tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)

    input_ids = description_tokenizer(TTS_DESCRIPTION, return_tensors="pt").input_ids
    prompt_input_ids = tts_tokenizer(text, return_tensors="pt").input_ids

    generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    sf.write(output_wav_path, audio_arr, tts_model.config.sampling_rate)
    print("DONE")


if __name__ == "__main__":
    main()
