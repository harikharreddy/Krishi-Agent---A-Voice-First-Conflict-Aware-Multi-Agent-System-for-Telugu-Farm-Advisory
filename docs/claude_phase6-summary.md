# Phase 6 Summary — Krishi-Agent (completed Sep 3, 2026)

## Starting point
Phase 5 delivered a working voice pipeline verified only via standalone test scripts (`tests/test_voice_pipeline.py` etc.) — never through the actual Streamlit app. `ui/app.py` going into Phase 6 had only the Phase 2 Farm Profile form; no question input, no audio, no photo upload existed in the UI at all.

## 6.1 — Typed question wired to `run_pipeline()`
- Added a text input + Ask button to `ui/app.py`, calling `run_pipeline(question, farm_profile)` and displaying `trace["final_answer"]`.
- Found and fixed three real environment bugs on first run, none of them pipeline logic bugs: missing `PYTHONPATH` (Streamlit only adds `ui/` to the import path, not the repo root, so `orchestrator` wasn't importable), no `.venv` activated (`torch` not found), and a stale shell PATH cache for `streamlit` after activating a venv.
- **`run.sh` created**: `cd` to repo root, activate the venv, set `PYTHONPATH`, launch Streamlit — removes the need to remember this multi-step incantation every session.

## 6.2 — Mic recorder added
- `st.audio_input` added **alongside** the typed box (not replacing it) — deliberate choice so voice-only farmers never need to read/type, while the team keeps typing as a fast debugging path.
- Verified recording, playback, and waveform display all work correctly on desktop first.

## 6.3 — Environment consolidation: switched `run.sh` to `.venv-voice`
- Since ASR/TTS need `.venv-voice` (Python 3.12) and it already had `torch`/`torchvision` from Phase 5, switched `run.sh` to activate `.venv-voice` instead of maintaining two separate launch paths.
- Found `streamlit` itself wasn't installed inside `.venv-voice` (only `.venv` had it) — installed it, which triggered a **protobuf version conflict**: Streamlit needs `protobuf>=5.26.1`, `descript-audiotools` (a `parler-tts` dependency) needs `protobuf<5.0.0`. Resolved by pinning to `protobuf==4.25.9` — both libraries import and run correctly despite pip's static conflict warning (confirmed empirically, not just by silencing the warning).

## 6.4 — Mic recording wired through ASR into `run_pipeline()`
- Added `transcribe_audio()`: saves `st.audio_input`'s in-memory bytes to a temp WAV, reads via `soundfile`, checks for 16kHz, runs the Phase 5-proven `AutoModel(...)(wav, "te", "rnnt")` call.
- ASR model loading wrapped in `@st.cache_resource` so it only loads once per session, not per question.
- **First-ever real test of browser-recorded audio through ASR** (Phase 5 only ever tested a pre-recorded file) — worked correctly on the first real attempt: recorded a live Telugu question, "Heard:" showed an accurate transcription, pipeline produced the correct weather answer.
- On the "Ask" button: recorded audio takes priority over the typed box when both are present.

## 6.5 — TTS wired into `run_pipeline()`'s output
- Added `synthesize_speech()`, reusing Phase 5's proven `ParlerTTSForConditionalGeneration` + "Lalitha" voice pattern, cached via `@st.cache_resource`.
- **Real bug found and fixed**: raw pipeline answer text was sent to TTS without calling Phase 5.2's `normalize_numerals_te()` — numbers were spoken as raw digits. Fixed by calling normalization before synthesis.
- **Second, deeper bug found and fixed**: even with Phase 5.2's fix applied, `num2words`'s Telugu output was literal/unnatural for hundreds and thousands (e.g. 1051 → "ఒకటి వేయిల యాభై ఒకటి", literally "one-thousand fifty-one", audibly rough) — this was the actual root cause of a live TTS clip sounding garbled, not a TTS artifact as first suspected.
  - Built `_telugu_number_words()` in `shared/text_normalization.py`: groups by thousands/hundreds using natural Telugu ("వెయ్యి" not "ఒకటి వేయిల" for exactly 1000s place, "వేలు"/"వందలు" plural forms otherwise), falls back to `num2words` only for the 0-99 remainder (already confirmed correct).
  - Also fixed a latent `num2words` quirk: `num2words(50, lang='te')` returns `"యాభై "` with a trailing space baked in, causing double-spaces in normalized output — fixed with `.strip()` on each word part.
  - Verified via `repr()` on 9 test cases (16, 20, 48, 51, 100, 200, 820, 980, 1051, 2450, 3000, 9999, 500) and via actual TTS audio playback on the previously-broken 1051 case (confirmed clean).
  - Re-ran Phase 5.2's original 4 numeral test sentences through the new formatter (`tests/test_tts_numerals_v2_fixed.py`), saved to `tests/audio/numeral_review_v2_fixed/` as a documented before/after record.

## 6.6 — TTS non-determinism confirmed as a real, separate finding
- Isolated two cases (a "48" mispronunciation, later a "29" mispronunciation) where the *exact same* correctly-normalized sentence sometimes produced an unrelated/garbled audio onset and sometimes didn't, across repeated identical runs.
- Systematically isolated: single words in isolation were always clean; only full-sentence context occasionally triggered it; not reproducible on demand (same text, same code, different outcome across runs).
- Concluded: inherent TTS generation stochasticity in `indic-parler-tts`, not a text-normalization or code bug — a genuine, citable model limitation, distinct from and found only after fixing the real normalization bug above.

## 6.7 — Photo upload added (real architecture gap, not originally listed in Phase 6 scope)
- Discovered mid-phase that `ui/app.py` had never had an image upload widget at all, despite the Disease Agent (`predict_disease(image_path)`) and `run_pipeline(..., image_path=...)` being fully built and wired since Phase 4. Every disease-intent question tested through Phase 5/6 up to this point had silently fallen through to "no photo attached, skipping Disease Agent."
- Added `st.file_uploader` (jpg/jpeg/png) + `st.image` preview; uploaded file saved to a temp path and passed as `image_path` to `run_pipeline()`.
- **Verified working end-to-end** with a real tomato leaf photo (sourced from the web, since no local PlantVillage/PlantDoc images exist on this machine — `data/` is gitignored per Phase 0) and a Phase 2.2-validated disease question: correctly returned `Tomato_Septoria_leaf_spot` (conf 0.94).
- First test with an *improvised* (non-validated) disease question ("ఈ ఆకుకు ఏమి జబ్బు?") misrouted to `wants_disease: False` — a real, natural-language example of Phase 2.2's known 87.5% Intent Router accuracy ceiling, not a new bug.
- **Explicitly verified crop-identification independence**: intentionally set Farm Profile to "Potato," uploaded the tomato leaf photo, asked a crop-agnostic question ("ఈ ఆకు ఏమి సమస్య ఉంది" — no crop name mentioned). Disease Agent still correctly returned `Tomato_Septoria_leaf_spot`, proving Stage-1 crop classification is driven purely by image content and is fully independent of the Farm Profile's Crop field (confirmed by code inspection — `predict_disease()` takes only `image_path`, never touches `farm_profile`).

## 6.8 — TTS latency investigation (open, not resolved)
- **Finding:** TTS generation takes ~15-40s standalone (isolated terminal script) but ~200-280s (3.5-4.5 min) when run from inside the Streamlit app — a consistent ~5-7x slowdown, reproduced across many repeated tests, both on laptop (localhost) and phone (via ngrok).
- **Eight hypotheses tested and ruled out with direct evidence**, not assumption:
  1. Sentence length (short/medium/long standalone sentences all fast: 15-30s)
  2. Thread count (`OMP_NUM_THREADS=8` in `run.sh` — no meaningful change, 237s vs 209s)
  3. Streamlit's file-watcher (`watchdog` installed — no change, 212.6s)
  4. Network calls (`HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` standalone — still fast, 37.5s, ruling out network I/O as inherently slow)
  5. Single-threaded execution (`torch.set_num_threads(1)` standalone — still fast, 40.9s, ruling out thread-context issues)
  6. Memory pressure (Activity Monitor: green/no pressure during a slow run)
  7. CPU load (Activity Monitor CPU tab during a slow run: 96.46% idle — process was *not* computing during most of the delay, pointing to I/O/blocking, not compute)
  8. Full process isolation (`tts_worker.py`, a standalone subprocess launched via `subprocess.run()` from inside Streamlit) — still slow (252-281s), ruling out shared-process/thread contention as the cause
- **One conclusive, unexplained result**: the identical subprocess mechanism, launched from a plain terminal Python script instead of from inside Streamlit, took 43.6s — fast. The only variable that changed was "is the parent process Streamlit." Also tried redirecting subprocess output to a file instead of capturing via pipes (`stdout=log_file` instead of `capture_output=True`) in case of pipe-buffering interaction — still slow (280.9s).
- `py-spy` (sampling profiler) attempted to get a real call-stack snapshot during a slow run — blocked by macOS SIP even under `sudo` ("Operation not permitted").
- **Status: unresolved.** Something specific to being a descendant of the Streamlit process is the cause, mechanism unidentified. `tts_worker.py` kept in the repo for a planned future test: reproduce on a Windows or Linux machine to determine if this is macOS-specific.
- **Mitigation applied:** spinner message updated to set expectations ("this can take a few minutes on this device") rather than looking stuck; reverted to the simpler in-process `synthesize_speech()` (subprocess isolation added complexity without fixing anything).

## 6.9 — Confidence-tone phrasing polish
- Found: Weather and Price Agents built their Telugu `answer` text *before* computing `confidence`, so Medium/Low-confidence answers were phrased identically to High-confidence ones — the confidence label existed but never affected what the farmer actually heard.
- Fixed both agents: confidence now computed first, then a confidence-appropriate hedge is appended — High: no hedge (unchanged), Medium: "కానీ ఖచ్చితంగా చెప్పలేం" (but can't say for certain), Low (Price Agent only): "అయితే ఇది స్పష్టమైన సూచన కాదు" (though this isn't a clear signal).
- All existing tests re-verified passing after the change: Weather Agent 4/4, Price Agent 5/5, pipeline edge cases 5/5, pipeline conflict scenarios 9/9 (3 skipped, documented drought_risk limitation, unchanged from Phase 4).
- Phrasing Templates (`orchestrator/phrasing_templates.py`, Phase 3.2) were reviewed and already had reasonable tone variation by resolution+confidence for multi-agent conflict cases — no changes needed there.

## Commits (all pushed to `main`)
- `d15db76` — 6.1/6.2: typed question + mic recorder in `ui/app.py`, `run.sh` created
- `c6b44bd` — 6.4: mic recording wired through ASR into `run_pipeline()`
- `46f8180` — 6.5: TTS wired in; natural Telugu number phrasing fix (hundreds/thousands + trailing-whitespace fix)
- `2252201` — 6.7: leaf photo upload added; timing instrumentation added for latency diagnosis; `OMP_NUM_THREADS=8` (later found not to help, kept harmlessly)
- `719009a` — 6.8: reverted to in-process TTS after subprocess isolation experiment; `tts_worker.py` kept for future cross-platform testing; expectation-setting spinner message
- `ae649d4` — 6.9: confidence-appropriate hedging in Weather/Price Agent answers

## Known limitations going into Phase 7/8 (documented, not hidden)
- **TTS latency inside Streamlit is unresolved** (~5-7x slower than standalone, cause not identified, eight hypotheses ruled out, macOS SIP blocked deeper profiling). Real risk for a live demo — mitigate with pre-generated answers before the panel arrives and/or a recorded backup video, per the project guide's existing Section 11 risk mitigation. Follow-up: test on a Windows/Linux machine to check if this is macOS-specific.
- **TTS generation is occasionally non-deterministic** even on correctly-normalized text — same sentence can rarely produce a garbled onset on one run and not another. Confirmed via repeated isolated testing (the "48" and "29" cases), not fixable at the text-normalization layer since it's a property of the neural generation process itself.
- **Intent Router misroutes on natural/improvised phrasing remain real** (consistent with Phase 2.2's 87.5% figure) — e.g. "ఈ ఆకుకు ఏమి జబ్బు?" and "ఈ రోజు నా ఆకుకు ఏమైంది" both failed to set `wants_disease: True`, while Phase 2.2's validated phrasings succeeded. Farmers using natural, unrehearsed phrasing may occasionally hit the fallback response.
- **UI is English-only** (labels, headings, buttons) — acceptable for team testing but a real accessibility gap for a farmer who can't read English; noted as a scope boundary for the report, not something this capstone claims to have solved. The mic-first design means a farmer never *needs* to read anything to ask a question, but does need someone literate to help with the one-time Farm Profile setup as currently built.
- **`num2words`'s remaining 0-99 output** is grammatically correct but occasionally slightly less idiomatic than an ideal native phrasing would choose — not flagged as wrong, low priority.
- Two Python environments (`.venv`, `.venv-voice`) remain unmerged — not urgent, both work correctly side by side, `.venv-voice` now handles all `run.sh`-launched work.

## State going into Phase 7

Per the guide, Phase 6's three scoped items (phrasing tone, mobile testing, latency) are done or honestly documented as attempted-and-open. Two items outside the original Phase 6 scope were found and fixed along the way (photo upload, TTS number-phrasing correctness) — both real gaps that would have surfaced awkwardly at demo time otherwise.

The full voice loop — record in Telugu → ASR → Intent Router → Agents (including Disease Agent via photo) → Conflict Resolver/passthrough → TTS → spoken answer out — is verified working end-to-end on both laptop and Android phone (via ngrok), with the caveat that TTS is currently slow (~3-5 min) rather than broken.

**Phase 7** (stretch scope: Soil Agent, Pepper as a 3rd crop) is next, gated behind Phase 4 being solid, which it is. Alternatively, given TTS latency's demo-day risk, it may be worth a short dedicated session before Phase 7 to test on a teammate's Windows/Linux machine and/or prepare the pre-generated-answers demo mitigation, before moving to new stretch scope.
