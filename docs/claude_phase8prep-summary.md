# Phase 8 Prep Summary — Krishi-Agent (completed Sep 3, 2026)

Standalone session between Phase 6 and Phase 7/8, closing three items flagged as open across earlier phase summaries: the Disease Agent's unvalidated 0.7 confidence cutoff, the missing drought detection, and Phase 5.2's untested-on-live-data numeral TTS fix. All work done directly on the local MacBook (no Colab), reusing already-trained checkpoints found locally in `agents/disease/checkpoints/`.

## 1 — Disease-confidence cutoff (0.7): confidence-calibration spot-check

**Problem:** the pipeline's `treat_now` (≥0.7) vs `monitor` (<0.7) split was a placeholder, never validated against real accuracy-vs-confidence data.

**What was built:**
- `agents/disease/plantdoc_confidence_eval.py` — re-runs the trained hierarchical model (`stage1_crop_best.pt`, `stage2_tomato_best.pt`, `stage2_potato_best.pt`) over all 967 available PlantDoc images (1 image missing vs. Phase 1's 968 — one `test` split folder absent, harmless), reusing `disease_agent.py`'s `predict_disease()` directly. No retraining. Logs per-image confidence + correctness to `agents/disease/results/plantdoc_perimage_confidence.json`.
- Sanity check: 23.37% accuracy (967 images) essentially matches Phase 1's originally reported 23.45% (968 images) — confirms this is the same evaluation, just with confidence now captured.
- `agents/disease/plantdoc_confidence_analysis.py` — buckets predictions by confidence range and computes accuracy per bucket.

**Finding (honest, not hidden):**
| Confidence range | Count | Accuracy |
|---|---|---|
| 0.9–1.0 | 304 | 28.62% |
| 0.8–0.9 | 110 | 24.55% |
| 0.7–0.8 | 91 | 21.98% |
| 0.6–0.7 | 113 | 23.01% |
| 0.5–0.6 | 111 | 20.72% |
| below 0.5 | 238 | 18.07% |

- Confidence ≥0.7 ("treat_now"): 505 images, **26.53%** accuracy
- Confidence <0.7 ("monitor"): 462 images, **19.91%** accuracy
- Gap: **+6.62%** — a real but weak, non-monotonic positive separation (0.6–0.7 bucket is actually higher than 0.7–0.8, a sign of noisy calibration).

**Conclusion:** the 0.7 cutoff carries real but weak signal on real-world (PlantDoc) images. Kept as-is (not tuned to this one sample — would risk overfitting). Documented for Phase 8 evaluation metric #6 (confidence-calibration spot-check) with the table above ready to drop into the report.

**Commit:** `d81846a`

---

## 2 — Drought detection: short-range proxy signal

**Problem:** Weather Agent only ever computed a 48h rain forecast; `weather_state="drought_risk"` was structurally unreachable, and 3 of 12 Phase 3 conflict scenarios (ids 4, 6, 7) could never fire through the real pipeline.

**What was built (`agents/weather/weather_agent.py`):**
- `_daily_summaries()` — groups the full 5-day/3-hour OpenWeatherMap forecast (already fetched, previously discarded past 48h) into per-day max-rain-probability and mean-temperature summaries.
- `_check_drought_signal()` — flags `drought_signal=True` only when **every** forecasted day is both dry (max pop ≤0.2) and hot (mean temp ≥35°C).
- **Explicitly documented as a short-range PROXY, not true drought detection** — true drought is a multi-week/seasonal phenomenon a 5-day forecast cannot observe. Comments and Telugu answer phrasing both flag this honestly ("ఇది స్వల్పకాలిక సూచన మాత్రమే" — this is a short-term signal only).
- New `drought_signal` field added to the returned schema dict; existing `rain_expected`/48h logic completely unchanged.
- New drought-specific Telugu phrasing branch, consistent with Phase 6.9's confidence-hedging pattern.

**Pipeline wiring (`orchestrator/pipeline.py`):**
- `_map_weather_state()` now maps `drought_signal=True` → `weather_state="drought_risk"`.
- Module docstring updated to reflect the fix (previously described drought as an open, unaddressed gap).

**Test suite updates (`tests/test_pipeline.py`):**
- Removed `UNREACHABLE_IDS = {4, 6, 7}` skip logic.
- Added a `drought_risk` branch to the mock Weather Agent output builder.
- **Result: 12/12 Phase 3 conflict scenarios now pass through the real pipeline, up from 9/9 (3 skipped).**

**New unit tests (`tests/test_weather_agent.py`)** — added after initially only verifying against live weather (which happened to stay `False` during monsoon season, proving nothing about whether the rule actually fires correctly):
- Fires when all 5 forecasted days are uniformly dry + hot
- Does not fire when any single day has real rain
- Does not fire when dry but below the temperature threshold
- Does not fire on an empty forecast (defensive check)
- Sanity-checks the threshold constants haven't silently drifted
- **9/9 passing** (4 original + 5 new)

**Re-verified after all changes:** `test_weather_agent.py` 9/9, `test_pipeline_edge_cases.py` 5/5, `test_pipeline.py` 12/12 (0 skipped).

**Commits:** `866716c` (feature), `3d084ad` (unit tests), `2ede5f0` (docstring cleanup)

---

## 3 — TTS numeral fix: confirmed on real live data

**Problem:** Phase 5.2 found and fixed a TTS numeral mispronunciation bug (later deepened by Phase 6.5's natural-phrasing fix), but all round-trip verification had used hand-constructed synthetic sentences — live price data wasn't available on either testing day. Flagged as explicitly open: *"worth re-running the same round-trip check against a real Price Agent answer once live data is available."*

**What was done:**
- Confirmed data.gov.in/Agmarknet had live data today by running `price_agent.py` directly.
- `tests/test_tts_real_price_output.py` — calls `get_price_advice()` for real (not synthetic) Telangana/Tomato and Andhra Pradesh/Potato cases, normalizes the genuine answer text, synthesizes via TTS, saves audio to `tests/audio/real_price_review/`.
- `tests/test_real_price_asr_roundtrip.py` — feeds that audio back through ASR and compares to the intended text.

**Result — every number in both real cases transcribed back correctly:**
- Warangal APMC Tomato: **1750 / 1253 / 40%** — all correct
- Andhra Pradesh Potato: **800 / 1354 / 41%** — all correct
- (A third case, "any market" Tomato, returned "price not reported today" due to live data flakiness between two runs minutes apart — not a bug, just no numbers to test that round.)
- Only differences between expected and ASR-heard text were non-numeric (English loanword transliteration `Tomato`→`టొమాటో`, missing punctuation) — both expected, harmless ASR behavior.

**Conclusion:** Phase 6.5's number-phrasing fix is now confirmed to hold on genuine, non-synthetic agent output, not just hand-picked test sentences. Gap fully closed.

**Commit:** `b5afc08`

---

## Commits this session (all pushed to `main`)
- `d81846a` — Disease confidence-calibration spot-check (0.7 cutoff analysis)
- `866716c` — Drought detection short-range proxy, wired end-to-end
- `3d084ad` — Drought signal unit tests (synthetic forecast data)
- `2ede5f0` — Pipeline docstring updated to reflect both fixes
- `b5afc08` — TTS numeral fix confirmed on real live Price Agent output

## State going into Phase 7 / Phase 8

Two of the three domain-logic review items flagged after Phase 6 are now closed with real evidence, not just documentation:
- ✅ Disease-confidence cutoff — analyzed, honest finding ready for Phase 8 metric #6
- ✅ Drought detection — real, tested, wired end-to-end; conflict-handling pass rate now 12/12 (Phase 8 metric #5)
- ⏳ **Still open:** Telugu disease-name labels (`disease_agent.py` lines 42–56) — needs a fluent-Telugu teammate's review, no code work required.

**Phase 7 (optional stretch scope)** — Soil Agent only, per the guide's Section 7a, gated behind Phase 4 (solid). Not yet started; a fresh chat/prompt was prepared separately for this.

**Phase 8 remaining work:**
- Full evaluation run (disease PlantVillage-vs-PlantDoc — done via Phase 1; ASR WER on farming vocabulary — not yet measured on the full Phase 2.2 question set spoken aloud; end-to-end latency — TTS-in-Streamlit latency still unresolved, ~5-7x slower than standalone, flagged for a Windows/Linux cross-platform test; voice-vs-text ablation; conflict-rule pass rate — now 12/12; confidence-calibration spot-check — done this session)
- Report write-up (8.2)
- Demo rehearsal (8.3)

**Known limitations still open (unchanged from Phase 6, honestly documented, not fixable by more engineering):**
- TTS latency inside Streamlit (~200–280s vs ~40s standalone), cause unidentified, macOS SIP blocked deeper profiling
- TTS generation is occasionally non-deterministic even on correctly-normalized text
- Intent Router's 87.5% ceiling on natural/improvised Telugu phrasing
- UI is English-only (accessibility gap, noted as scope boundary)
- Two Python environments (`.venv`, `.venv-voice`) remain unmerged, both work correctly side by side
