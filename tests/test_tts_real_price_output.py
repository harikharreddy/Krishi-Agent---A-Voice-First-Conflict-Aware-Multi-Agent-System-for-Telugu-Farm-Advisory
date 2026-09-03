"""
Phase 8 prep: closes a gap flagged in Phase 5.2 -- "worth re-running the
same round-trip check against a real Price Agent answer once live data is
available." Uses REAL live get_price_advice() output (not hand-constructed
synthetic sentences like test_tts_numerals_v2_fixed.py), normalized and
synthesized, to confirm the numeral fix holds on genuine agent output.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from shared.text_normalization import normalize_numerals_te
from agents.price.price_agent import get_price_advice

OUTPUT_DIR = "tests/audio/real_price_review"
TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
TTS_DESCRIPTION = "Lalitha's voice is calm and clear, with a natural Telugu accent, at a moderate speed with high quality recording."

# Same 3 real cases you just pulled live from data.gov.in / Agmarknet
REAL_CASES = [
    ("telangana_tomato_any", "Telangana", "Tomato", None),
    ("telangana_tomato_warangal", "Telangana", "Tomato", "Warangal APMC"),
    ("ap_potato_any", "Andhra Pradesh", "Potato", None),
]


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

    expected_texts = {}

    for name, state, commodity, market in REAL_CASES:
        result = get_price_advice(state, commodity, market)
        raw_text = result["answer"]
        normalized_text = normalize_numerals_te(raw_text)

        out_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        print(f"[{name}] ({state} / {commodity} / {market or 'any market'})")
        print(f"  RAW:        {raw_text}")
        print(f"  NORMALIZED: {normalized_text}")

        prompt_input_ids = tts_tokenizer(normalized_text, return_tensors="pt").input_ids
        generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(out_path, audio_arr, tts_model.config.sampling_rate)
        print(f"  -> saved {out_path}\n")

        expected_texts[f"{name}.wav"] = normalized_text

    # Save expected texts so the ASR round-trip script can compare against them
    with open(os.path.join(OUTPUT_DIR, "expected_texts.json"), "w", encoding="utf-8") as f:
        json.dump(expected_texts, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(REAL_CASES)} files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
