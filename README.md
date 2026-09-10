# Krishi-Agent

Voice-first, conflict-aware multi-agent system for Telugu farm advisory — combines crop disease detection, weather, and market price into one spoken answer.

## About

Krishi-Agent is a multi-agent AI system that lets a farmer ask a question in Telugu and get back a single, spoken answer that combines crop disease diagnosis (from a photo), weather conditions, and market price trends. A conflict-aware orchestrator resolves contradicting advice (e.g. "spray now" vs. "rain coming") and communicates uncertainty honestly instead of overclaiming.

Built for Tomato and Potato (13 disease classes total), using free and open-source tools throughout — AI4Bharat for Telugu speech, PlantVillage/PlantDoc for disease data, and a local open-source LLM (Ollama) as orchestrator.

FastAPI backend + a static HTML/CSS/JS frontend (`static/`), replacing an earlier Streamlit prototype — Streamlit's threading model was adding 150-250s of TTS latency that isn't present in the current backend (see `latency_proof/`).

## Running it

**Prerequisites:**
- Python 3.12 (the ASR/TTS stack doesn't build on 3.13+ — see `requirements-voice.txt`)
- [Git LFS](https://git-lfs.com/) (`git lfs install`, once per machine) — the trained Disease Agent checkpoints (~200MB) are stored via LFS; a plain clone without it will fetch pointer files, not the real model weights, and the Disease Agent will fail to load
- [Ollama](https://ollama.com/), running locally, with `qwen2.5:7b-instruct` pulled (`ollama pull qwen2.5:7b-instruct`) — this is the Intent Router's model
- API keys: [OpenWeatherMap](https://openweathermap.org/api) and [data.gov.in](https://www.data.gov.in/) (Agmarknet), both free tier

**Setup:**
```bash
git clone <this-repo-url>
cd krishi-agent
python3.12 -m venv .venv-voice
source .venv-voice/bin/activate
pip install -r requirements-voice.txt
```

Create a `.env` file in the repo root:
```
OPENWEATHERMAP_API_KEY=<your key>
DATA_GOV_IN_API_KEY=<your key>
```

**Run:**
```bash
source .venv-voice/bin/activate && uvicorn backend.main:app --reload --port 8000
```
Open `http://localhost:8000` in a browser. First request will be slow while Ollama loads the model and ASR/TTS models load into memory; subsequent requests are faster (see `backend/main.py`'s `_unload_intent_router_model()` docstring for a documented latency/accuracy tradeoff in that path).

## Known limitations (honest, not hidden)

- **Disease Agent real-world accuracy**: ~38% end-to-end on real-world PlantDoc test photos (up from ~23% before fine-tuning), varying 20-93% by class — `Potato___healthy` is reliable (93%), `Tomato__Target_Spot` is weak (~20%). Full numbers and methodology in `agents/disease/results/` and the docstrings in `agents/disease/disease_agent.py` and `finetune_plantdoc.py`.
- **Weather drought signal** is a short-range proxy (5-day forecast), not true multi-week drought detection — see `agents/weather/weather_agent.py`.
- **Intent Router latency**: every question pays a 5-12s model-reload cost to avoid a memory collision with TTS on 8GB RAM machines — a deliberate, documented tradeoff (kept the more accurate 7B model over a faster 3B one).
- **Telugu translations** (UI place names) have been carefully reviewed but not signed off by a native speaker — see `static/js/districts.js`.

## Status

Working end-to-end: voice/typed question in, spoken Telugu answer out, disease photo upload, farm profile setup. See `docs/` for phase-by-phase build history.

## Team

4-person team.
