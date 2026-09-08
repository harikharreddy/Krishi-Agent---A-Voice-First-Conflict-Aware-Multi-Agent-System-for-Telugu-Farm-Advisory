"""
Phase 6 -- Standalone TTS worker, run as a subprocess from ui/app.py.
Isolates TTS generation from Streamlit's process, since generation was
observed to take 5-7x longer when run in-process inside Streamlit
versus as a standalone script (root cause not identified after
eliminating thread count, network I/O, file-watching, and sentence
length as causes across multiple systematic tests; most pronounced on
macOS, where in-process generation was taking ~3 minutes per reply).

Two modes:
  - One-shot: `python3 tts_worker.py <text_file_path> <output_wav_path>`
    Loads the model, synthesizes once, exits. Simple but pays the full
    model-load cost (several seconds to tens of seconds) on every call.
  - Serve: `python3 tts_worker.py --serve`
    Loads the model once, then reads one JSON request per line from
    stdin: {"text_path": ..., "output_path": ...}. For each request it
    writes the synthesized audio to output_path and replies on stdout
    with {"status": "ok"} or {"status": "error", "message": ...}. Prints
    {"status": "ready"} once, right after the model finishes loading.
    ui/app.py keeps one of these processes alive for the life of a
    session so repeat replies skip the model-load cost entirely.

The input text is expected to already be normalized (e.g. via
shared.text_normalization.normalize_numerals_te) by the caller.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import sys
import json
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

TTS_DESCRIPTION = "Lalitha's voice is calm and clear, with a natural Telugu accent, at a moderate speed with high quality recording."


def _load_model():
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to("cpu")
    tts_model.eval()
    tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)
    return tts_model, tts_tokenizer, description_tokenizer


def _synthesize_one(tts_model, tts_tokenizer, description_tokenizer, text, output_wav_path):
    input_ids = description_tokenizer(TTS_DESCRIPTION, return_tensors="pt").input_ids
    prompt_input_ids = tts_tokenizer(text, return_tensors="pt").input_ids

    generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    sf.write(output_wav_path, audio_arr, tts_model.config.sampling_rate)


def _run_one_shot(text_file_path, output_wav_path):
    with open(text_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tts_model, tts_tokenizer, description_tokenizer = _load_model()
    _synthesize_one(tts_model, tts_tokenizer, description_tokenizer, text, output_wav_path)
    print("DONE")


def _run_serve():
    tts_model, tts_tokenizer, description_tokenizer = _load_model()
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            with open(request["text_path"], "r", encoding="utf-8") as f:
                text = f.read()
            _synthesize_one(tts_model, tts_tokenizer, description_tokenizer, text, request["output_path"])
            print(json.dumps({"status": "ok"}), flush=True)
        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}), flush=True)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--serve":
        _run_serve()
        return

    if len(sys.argv) != 3:
        print(
            "Usage: python3 tts_worker.py <text_file_path> <output_wav_path>\n"
            "       python3 tts_worker.py --serve",
            file=sys.stderr,
        )
        sys.exit(1)

    _run_one_shot(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
