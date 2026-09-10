"""
Standalone mic recorder for Phase 8.1 ASR WER evaluation.

Records a fixed-duration clip from the default input device, saves as
16kHz mono WAV (matching the format ai4bharat/indic-conformer-600m-multilingual
requires, per Phase 0/5's test_asr_local.py).

Usage:
    python3 tests/record_wer_sample.py <output_path.wav> [--seconds N]

Example:
    python3 tests/record_wer_sample.py tests/audio/wer_eval/speaker1/q1.wav --seconds 6
"""

import argparse
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", help="Where to save the WAV file")
    parser.add_argument("--seconds", type=float, default=6.0, help="Recording duration in seconds")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Will record for {args.seconds} seconds at {SAMPLE_RATE} Hz mono.")
    print(f"Output: {args.output_path}")
    print()
    for i in (3, 2, 1):
        print(f"Starting in {i}...")
        time.sleep(1)
    print(">>> SPEAK NOW <<<")

    recording = sd.rec(
        int(args.seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Done recording.")

    # Basic sanity check: warn if the clip looks silent (helps catch a wrong
    # input device or muted mic before you've recorded all 16 questions)
    peak = float(np.max(np.abs(recording)))
    if peak < 0.01:
        print(f"WARNING: peak amplitude is very low ({peak:.4f}) — clip may be silent. "
              f"Check your mic input device before continuing.")

    sf.write(args.output_path, recording, SAMPLE_RATE)
    print(f"Saved: {args.output_path} (peak amplitude: {peak:.4f})")


if __name__ == "__main__":
    main()
