# ASR WER Recording Instructions — Speakers 2, 3, 4

For whoever is recording the remaining speakers' samples. Speaker 1's 16
recordings are already done and match this exact process — follow the
same steps so all 4 speakers stay comparable.

## What you're recording

The same 16 Telugu farmer questions, read aloud, once each — the same
questions already in `tests/intent_router_test_set.json`. Don't paraphrase
or add words; read exactly what's shown, naturally, at a normal speaking
pace.

## How to record

1. Activate the project environment: `source .venv-voice/bin/activate`
   (from the repo root).
2. For each question 1-16, run:
   ```bash
   python3 tests/record_wer_sample.py tests/audio/wer_eval/speaker2/q1.wav --seconds 6
   ```
   Replace `speaker2` with your assigned number (`speaker3`, `speaker4`),
   and `q1` with the question number. Repeat for all 16 (`q1.wav` through
   `q16.wav`).
3. The script gives a 3-2-1 countdown, then "SPEAK NOW" — start speaking
   right after that appears. 6 seconds is enough for any of these 16
   questions; if you need more time for a longer one (e.g. question 10 or
   12), add `--seconds 8`.
4. The script warns if your recording looks silent (very low peak
   amplitude) — if you see that warning, check your mic input device and
   re-record that one file before moving to the next question.
5. When all 16 are done, you should have exactly 16 files in
   `tests/audio/wer_eval/speaker<N>/`, named `q1.wav` through `q16.wav`.

## Why this matters for the evidence

`tests/check_all_wer.py` auto-discovers whichever `speaker*/` folders
exist and scores whatever's there — it doesn't need to be re-run or
re-written when you add your files, it just needs the files to actually
be there with the right names. If a file is missing or misnamed
(`q1.wav` vs `Q1.wav` vs `question1.wav`), the script will silently skip
that question for your speaker rather than fail loudly — so please
double-check the filenames match exactly before saying you're done.

## Status as of this evidence run

| Speaker | Recordings | Status |
|---|---|---|
| speaker1 | 16/16 | Done — mean WER 10.96% (preliminary, 1/4 speakers) |
| speaker2 | 0/16 | Not started |
| speaker3 | 0/16 | Not started |
| speaker4 | 0/16 | Not started |

Once all 4 are in, re-run `python3 tests/check_all_wer.py` — it will
automatically report `is_complete: true` and the real multi-speaker mean
WER instead of the current preliminary single-speaker number.
