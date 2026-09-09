# Phase 5 Summary — Krishi-Agent (completed Aug 25, 2026)

## 5.1 — Wiring STT/TTS onto the proven Phase 4 pipeline

**Environment setup (blocking issue, resolved)**
- ASR/TTS dependencies (`transformers==4.46.1`, `parler-tts`, `tokenizers==0.20.3`) failed to build on Python 3.14, the version used by the main project `.venv` — `tokenizers` has no prebuilt wheel for 3.14 and its Rust build tool (PyO3 0.22.5) doesn't support 3.14 yet.
- **Fix:** created a separate local environment, `.venv-voice` (Python 3.12), specifically for voice work. Installed `torch`, `transformers==4.46.1`, `torchaudio`, `onnxruntime`, `soundfile`, `torchvision`, and `python-dotenv` (the last two needed because `.venv-voice` also has to import the full `orchestrator/pipeline.py`, which pulls in the Weather/Price/Disease agents).
- `parler-tts` installs via `pip install git+https://github.com/huggingface/parler-tts.git` — required `git-lfs` (installed via `brew install git-lfs` + `git lfs install`) since one of its dependencies (`descript-audiotools`) uses Git LFS.
- HF login (`hf auth login`) and gated-model access for both `ai4bharat/indic-conformer-600m-multilingual` and `ai4bharat/indic-parler-tts` — access carried over from Phase 0's HF account, no new request needed.
- `torchaudio.load()` defaults to a `torchcodec` backend not installed here — switched to `soundfile` for reading/writing WAV files throughout, since all our audio is plain 16kHz mono WAV.

**ASR verification (`tests/test_asr_local.py`)**
- Loaded `ai4bharat/indic-conformer-600m-multilingual` locally, transcribed the Phase 0 Telugu test recording (`tests/audio/test_recording.wav`, reused from Phase 0/1, confirmed to still be the correct sentence).
- **RNNT decoding: exact match** — "నా పంట ఆకులపై మచ్చలు కనిపిస్తున్నాయి", identical to Phase 0's Colab result.
- **CTC decoding:** same minor repeated-character artifact Phase 0 already documented (a doubled vowel sign) — confirmed consistent, not a new local-environment bug.

**TTS verification (`tests/test_tts_local.py`)**
- Loaded `ai4bharat/indic-parler-tts`, synthesized the same test sentence with the "Lalitha" voice description (per Phase 0's recommendation).
- Produced valid, audible, correct Telugu speech (confirmed by listening) — 3.36s, 44.1kHz, healthy amplitude (no clipping, not silent).
- `USE_TF=0` / `USE_FLAX=0` set before any `transformers` import, per Phase 0's documented gotcha — no protobuf conflict encountered this time.

**Full voice pipeline wiring (`tests/test_voice_pipeline.py`, `tests/test_voice_pipeline_weather.py`)**
- Built `audio in → ASR → run_pipeline() [Phase 4, untouched] → TTS → audio out`, reusing `orchestrator/pipeline.py` exactly as Phase 4 left it — no changes to pipeline logic.
- **Test case 1 (no photo, disease-only question):** correctly detected `wants_disease`, correctly skipped the Disease Agent (no photo attached), correctly fell through to the `unresolved_conflict` fallback, correctly synthesized the fallback sentence as audio. Confirmed by listening.
- **Test case 2 (weather-only question, "రేపు వర్షం పడుతుందా?"):** correctly detected `wants_weather` only, Weather Agent hit the real OpenWeatherMap API, returned a genuine data-backed answer ("rain expected in the next 1 hour, 100% probability"), routed through the Phase 4.2c single-agent-passthrough fix (High confidence, no false arbitration), synthesized correctly as audio.
- Both cases confirmed by listening as correct, audible Telugu speech.

**Commits:** `4426d26` (ASR), `3dee05d` (TTS), `4315083` (full voice pipeline wiring)

---

## 5.2 — Pronunciation validation with the team (fluent Telugu speakers)

**Phrasing Template review (`tests/test_tts_phrasing_templates.py`)**
- Batch-synthesized all 10 distinct Phrasing Templates from Phase 3.2 (`orchestrator/phrasing_templates.py`) plus the fallback sentence, saved as individually labeled audio files (`tests/audio/phrasing_review/`).
- Team listened to all 11 files: **clear and natural, no issues reported.**
- Notable: none of these templates contain numeric content (they're hand-written prose), which turned out to matter — see below.

**Numeral mispronunciation — found (ad hoc, then confirmed systematically)**
- First surfaced by accident: the Test Case 2 weather answer above ("1 hour... 100%") was heard by the team as "10 hours" with an unclear percentage.
- Investigated systematically with 4 synthetic sentences matching the *real* output format of `price_agent.py` and `weather_agent.py` (`tests/test_tts_numerals.py`): a 4-digit price + percentage, a 3-digit price + percentage, and two weather sentences with different hour counts/percentages.
- Live price data wasn't available that day (data.gov.in returned no records for Tomato/Potato at Warangal APMC — a known Phase 1 API gap, not a new bug), so these 4 cases used realistic synthetic numbers matching the agents' actual answer-string format exactly.
- **Team listening confirmed:** numbers in all 4 cases were "not clear."
- **Objective cross-check:** built an automated TTS→ASR round-trip test (`tests/test_tts_asr_roundtrip.py`) — fed the synthesized audio back through the ASR model and compared the transcription to the intended text, as a second, repeatable opinion independent of human listening.
  - Round-trip confirmed the pattern precisely: simple two-digit numbers and round hundreds (16%, 24 hours, 29%, "nineteen hundred") came through correctly; **multi-digit currency figures (2450, 820, 980) and percentages in the 40–60s range (65%, 45%) came through as garbled/unintelligible nonsense.**

**Fix built and verified**
- Checked whether `num2words` supports Telugu (`lang="te"`) — confirmed yes, with correct grammatical output (e.g. `2450` → "రెండు వేయి నాలుగు వందల యాభై", 16 → "పదహారు").
- Built `shared/text_normalization.py`: a `normalize_numerals_te()` function that regex-matches digit sequences (optionally followed by `%`) and replaces them with Telugu words via `num2words`, appending "శాతం" for percentages. Leaves all other text untouched.
- Re-ran the same 4 numeral test cases through TTS *with* normalization applied first (`tests/test_tts_numerals_normalized.py`), then re-ran the same ASR round-trip check (`tests/test_normalized_asr_roundtrip.py`).
- **Result: all 4 cases now transcribe back essentially word-for-word correctly** — a dramatic improvement over the raw-digit baseline (e.g. "2450" went from complete gibberish to an exact match; "65%" and "45%" both landed correctly).
- Team listened to the 4 normalized files and confirmed they sound clear and natural, numbers included.
- Added `num2words==0.5.14` to `requirements.txt`.

**Commit:** `86505f0` — includes `shared/text_normalization.py`, all phrasing-review and numeral-review audio (raw and normalized), the two ASR round-trip test scripts, and the `requirements.txt` update.

---

## Known limitations going into Phase 6 (documented, not hidden)

- **Text normalization covers only digit sequences and `%`.** It does not handle dates, decimals, ranges, or other numeric formats not yet seen in agent output — fine for now since it matches every real pattern currently produced by `weather_agent.py` and `price_agent.py`, but should be revisited if new numeric formats are introduced later (e.g. a Soil Agent in Phase 7).
- **`num2words`'s Telugu output is grammatically correct but not always the most natural phrasing** a native speaker would choose (e.g. "100" → "ఒకటి వంద" — literally "one hundred" — rather than the more idiomatic "వంద" alone). Not flagged as wrong by the team's review, but a possible Phase 6 polish item.
- **Two separate Python environments now exist:** `.venv` (Python 3.14, main pipeline/agents) and `.venv-voice` (Python 3.12, ASR/TTS). `.venv-voice` needed the full `requirements.txt` plus `torchvision` installed separately, since it also imports `orchestrator/pipeline.py`. Not unified — a possible future cleanup, not urgent, since both work correctly side by side.
- **ASR round-trip cross-check is a useful proxy, not ground truth.** The ASR model can itself mishear things; it complements team listening rather than replacing it. Both methods agreed in every case tested so far.
- **Numeral testing so far used synthetic sentences**, not live agent output, because live price data wasn't available on test day. Worth re-running the same round-trip check against a real Price Agent answer once live data is available, to confirm the fix holds on genuine (not hand-constructed) output.
- Phase 5.2's original done-condition ("whether the ASR captures common ways farmers might phrase questions") was not tested this phase — only the one Phase 0 test sentence was used for ASR. Broader ASR phrasing-robustness testing (e.g. against the Phase 2.2 16-question test set, spoken aloud) is still open.

## Documentation updates made this phase

- `krish_agent_project.md` (Complete Project Guide) updated: architecture diagram now shows the `[5.5] Numeral Text Normalization` step; Section 5 tools table includes `num2words` and the `.venv-voice` environment note; Section 7 Phase 5 marked complete with findings; Section 11 risks table includes the TTS numeral issue as a resolved risk with evidence.

## State going into Phase 6

Per the guide, Phase 5's done condition — record a Telugu sentence, transcribe it, run it through the pipeline, hear a synthesized reply — is met, and additionally stress-tested against numeral-heavy agent answers rather than just a single demo case. A real TTS limitation was found through systematic testing (not by luck) and fixed with a verified, working solution, not just documented as a gap.

**Phase 6 (Polish)** is next: refining the Orchestrator's phrasing so confidence differences sound different in tone, testing the full app on the actual demo phone, and measuring end-to-end latency (quantizing the local LLM if needed). The `st.audio_input` mic widget from Phase 0 has not yet been wired into `ui/app.py` — that wiring, plus real mobile testing, is Phase 6 scope per the guide, not something Phase 5 skipped.
