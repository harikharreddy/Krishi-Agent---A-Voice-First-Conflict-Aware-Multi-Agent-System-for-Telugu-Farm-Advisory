"""
Phase 5.2 -- TTS numeral test, WITH normalization applied.
Same 4 sentences as test_tts_numerals.py, but numbers are converted
to Telugu words first via shared/text_normalization.py, to compare
against the raw-digit version.
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

from shared.text_normalization import normalize_numerals_te

OUTPUT_DIR = "tests/audio/numeral_review_normalized"
TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
TTS_DESCRIPTION = "Lalitha's voice is calm and clear, with a natural Telugu accent, at a moderate speed with high quality recording."

RAW_CASES = {
    "price_sell_4digit": "ఈరోజు Tomato ధర 2450 రూ., ఈ నెలలో సాధారణ ధర (1900 రూ.) కంటే 29% ఎక్కువ. ఇప్పుడు అమ్మడం మంచిది.",
    "price_hold_3digit": "ఈరోజు Potato ధర 820 రూ., ఈ నెలలో సాధారణ ధర (980 రూ.) కంటే 16% తక్కువ. వీలైతే ఆగడం మంచిది.",
    "weather_3hour": "రాబోయే 3 గంటల్లో వర్షం పడే అవకాశం ఉంది (వర్షం సంభావ్యత 65%).",
    "weather_24hour": "రాబోయే 24 గంటల్లో వర్షం పడే అవకాశం ఉంది (వర్షం సంభావ్యత 45%).",
}


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

    for name, raw_text in RAW_CASES.items():
        normalized_text = normalize_numerals_te(raw_text)
        out_path = os.path.join(OUTPUT_DIR, f"{name}.wav")

        print(f"[{name}]")
        print(f"  raw:        {raw_text}")
        print(f"  normalized: {normalized_text}")

        prompt_input_ids = tts_tokenizer(normalized_text, return_tensors="pt").input_ids
        generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(out_path, audio_arr, tts_model.config.sampling_rate)
        print(f"  -> saved {out_path}\n")

    print(f"Done. {len(RAW_CASES)} files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
