"""
FastAPI backend for Krishi-Agent (Phase 6/8 UI migration off Streamlit).

Wires the proven Phase 4/5/6 logic -- run_pipeline(), transcribe_audio(),
synthesize_speech() -- behind one HTTP endpoint for the static frontend in
static/. Does not change orchestrator/pipeline.py or any agent logic.

Run from the repo root:
    source .venv-voice/bin/activate && uvicorn backend.main:app --reload --port 8000
"""
import io
import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import soundfile as sf

load_dotenv()

import logging

import requests

from orchestrator.pipeline import run_pipeline
from orchestrator.intent_router import MODEL as INTENT_ROUTER_MODEL, OLLAMA_URL
from backend.voice import transcribe_audio, synthesize_speech
from backend.geocoding_overrides import weather_location_for

logger = logging.getLogger(__name__)

app = FastAPI(title="Krishi-Agent")


def _unload_intent_router_model():
    """
    Ollama keeps a model resident in RAM for 5 minutes after each call by
    default. route_intent() (orchestrator/intent_router.py) loads
    qwen2.5:7b-instruct (~5GB) on every question, right before TTS needs its
    own ~1.5GB -- on an 8GB machine that collision is what was actually
    causing multi-minute TTS times (not Streamlit vs. FastAPI; confirmed
    Sep 9 by watching `top` mid-request and finding llama-server resident
    during a 191s generate() call). Force-unloading it here, backend-layer
    only, keeps orchestrator/intent_router.py untouched per the migration's
    no-pipeline-changes constraint.

    Honest tradeoff, not a free fix (measured Sep 10): this defeats Ollama's
    own warm-reuse, so every question now pays a full cold-load penalty for
    the Intent Router step -- measured 5.6s/12.1s/10.7s across 3 consecutive
    real route_intent() calls with this unload active, vs. ~3.6s warm.
    Considered switching to qwen2.5:3b-instruct (loads faster, smaller) to
    cut that cost, but Phase 2.2 already tested 3B for this exact task and
    it scored 50% (8/16) vs. 7B's 87.5% (14/16) -- reintroducing that
    accuracy regression to save a few seconds isn't a good trade, so 7B
    stays and the reload cost is accepted as the honest price of avoiding
    the 191s collision. This is a mitigation for the 8GB RAM ceiling, not a
    fix for it -- resurfaces if the pipeline ever needs the LLM and TTS
    resident at the same time (e.g. concurrent requests).
    """
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": INTENT_ROUTER_MODEL, "keep_alive": 0},
            timeout=5,
        )
    except requests.exceptions.RequestException:
        logger.warning("Could not reach Ollama to unload intent-router model (non-fatal)", exc_info=True)


@app.post("/api/ask")
def ask(
    state: str = Form(...),
    district: str = Form(...),
    mandi: str = Form(...),
    crop: str = Form(...),
    question: str = Form(None),
    audio: UploadFile = File(None),
    photo: UploadFile = File(None),
):
    # weather_location_for() substitutes a geocoding-friendly place name only
    # for the Weather Agent's live lookup (see backend/geocoding_overrides.py)
    # -- the farmer's actual saved district is untouched everywhere else.
    farm_profile = {
        "state": state,
        "district": weather_location_for(district),
        "mandi": mandi,
        "crop": crop,
    }

    heard_text = None
    if audio is not None:
        audio_bytes = audio.file.read()
        try:
            heard_text = transcribe_audio(audio_bytes)
        except ValueError as e:
            return JSONResponse({"error": "bad_audio", "detail": str(e)}, status_code=400)
        final_question = heard_text
    elif question:
        final_question = question
    else:
        return JSONResponse({"error": "no_question"}, status_code=400)

    image_path = None
    if photo is not None and photo.filename:
        suffix = os.path.splitext(photo.filename)[1] or ".jpg"
        photo_bytes = photo.file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(photo_bytes)
            image_path = tmp.name

    try:
        trace = run_pipeline(final_question, farm_profile, image_path=image_path)
    finally:
        if image_path:
            os.unlink(image_path)

    answer_text = trace["final_answer"]
    _unload_intent_router_model()

    disease_raw = trace.get("disease_raw")
    detected_crop = disease_raw.get("predicted_crop") if disease_raw else None

    disease_mismatch = None
    if detected_crop and detected_crop != crop:
        disease_mismatch = {
            "detected_crop": detected_crop,
            "profile_crop": crop,
        }

    audio_arr, sample_rate = synthesize_speech(answer_text)
    buf = io.BytesIO()
    sf.write(buf, audio_arr, sample_rate, format="WAV")

    return {
        "heard_text": heard_text,
        "answer_text": answer_text,
        "detected_crop": detected_crop,
        "disease_mismatch": disease_mismatch,
        "sample_rate": sample_rate,
        "audio_wav_b64": _b64(buf.getvalue()),
    }


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


app.mount("/", StaticFiles(directory="static", html=True), name="static")
