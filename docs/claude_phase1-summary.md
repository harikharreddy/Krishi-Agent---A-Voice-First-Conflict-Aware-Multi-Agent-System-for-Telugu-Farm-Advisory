# Phase 1 Summary — Krishi-Agent (completed Aug 21, 2026)

## What was built (all committed and pushed to `main`)

**1.1 Weather Agent** (`agents/weather/weather_agent.py`, commit `5a78071`)
- `get_weather_advice(location)` — OpenWeatherMap free-tier 5-day/3-hour forecast, 48-hour window, rain-probability threshold 0.5
- Returns the shared schema in Telugu; confidence High when forecast is clearly wet or clearly dry, Medium otherwise
- 4 pytest tests passing (`tests/test_weather_agent.py`)

**1.2 Price Agent** (`agents/price/price_agent.py` + `build_price_history.py`, commit `95e69d9`)
- Live "today's price" comes from the data.gov.in Mandi API — **key finding: this API only ever returns today's snapshot, no history**, despite an `arrival_date` filter parameter that silently has no effect
- Historical seasonal baseline instead sourced from a Kaggle Agmarknet archive (2001–2026), filtered to Telangana + Andhra Pradesh, Tomato + Potato → `data/price_history_ap_telangana.csv` (155,546 rows, gitignored, built locally by `build_price_history.py`)
- Sell/hold rule: ±10% vs seasonal average triggers a recommendation, ±20% triggers High confidence
- **Gotcha carried forward:** `requests` calls to data.gov.in hang indefinitely unless a real browser `User-Agent` header is set — the default `python-requests` UA gets silently dropped
- 5 pytest tests passing (`tests/test_price_agent.py`)

**1.3 Disease Agent — flat baseline** (commit `8a6982e`, notebook `notebooks/phase1_disease_flat_baseline.ipynb`)
- EfficientNetB0 (ImageNet pretrained, fine-tuned), 13 classes (10 Tomato + 3 Potato), class-weighted loss for PlantVillage's severe imbalance
- Best val accuracy: **99.72%** (`agents/disease/results/flat_baseline_results.json`)
- Checkpoint `agents/disease/checkpoints/flat_baseline_best.pt` (gitignored, local only — 16.4MB)

**1.4 Disease Agent — hierarchical** (commit `bd68f37`)
- Stage 1: crop classifier (Tomato/Potato). Stage 2: per-crop disease classifier. Each an independent EfficientNetB0.
- End-to-end val accuracy: **99.70%** (`agents/disease/results/hierarchical_results.json`) — statistically tied with the flat baseline on clean PlantVillage data
- Checkpoints gitignored, local only

**1.5 PlantDoc validation** (commits `3ebd446`, `deb795c`, `15af3e9`)
- Zero-shot evaluation of both models on 968 real-world PlantDoc images (11/13 classes have real-world equivalents; missing: `Tomato__Target_Spot`, `Potato___healthy`)
- **Flat baseline: 22.00% accuracy (77.72-point drop from PlantVillage)**
- **Hierarchical: 23.45% accuracy (76.25-point drop from PlantVillage)**
- Both well below the guide's 65-80% target range — attributed honestly to a larger-than-typical visual domain gap between PlantVillage's clean lab photos and PlantDoc's cluttered real-world scraped images, plus a strict pure zero-shot methodology (no PlantDoc fine-tuning)
- **Real finding worth citing:** hierarchical modestly but consistently outperforms flat on the harder real-world set (e.g. `Tomato__Tomato_YellowLeaf__Curl_Virus` diagonal: 20/76 hierarchical vs 12/76 flat) even though they're tied on clean data — consistent with literature paper [5]'s claim that hierarchical's robustness advantage shows up under harder conditions
- Results JSON + confusion matrices (`agents/disease/results/plantdoc_results.json`, `confusion_flat_plantdoc.png`, `confusion_hier_plantdoc.png`) and notebook (`notebooks/phase1_disease_hierarchical_plantdoc.ipynb`) all committed
- Explicit decision made: did **not** retroactively fine-tune on PlantDoc just to hit the target number, to keep the generalization test honest

## Standing operational lessons (apply to future Colab work too)
- Never batch multiple `files.download()` calls in one Colab cell — the second one is silently blocked as a popup; always download one file per cell, and check the Chrome omnibox popup-blocker icon if a download doesn't land
- Colab sessions disconnect/reset without warning, wiping installed packages and in-memory variables — verify liveness with `!nvidia-smi` + checking a known in-memory variable before trusting anything is still there
- Git workflow used consistently throughout: `git status` → `git add <specific files>` (never `-A`) → `git status` again → `git commit` → `git push origin main`

## Loose ends not yet cleaned up
- `test_recording.wav` and `ui/mic_test.py.save` are untracked leftovers in the repo root/`ui/` — harmless, not yet deleted or gitignored

## State going into Phase 2
- `ui/` currently only has the Phase 0 `mic_test.py` — no real Streamlit app entry point exists yet
- No LLM has been chosen yet for the Phase 2.2 Intent Router — this needs to be decided as part of Phase 2, not assumed
