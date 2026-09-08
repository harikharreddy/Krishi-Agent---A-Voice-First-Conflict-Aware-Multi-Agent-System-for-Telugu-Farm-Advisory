import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import sys
import time
import json
import queue
import logging
import tempfile
import threading
import subprocess
import torch
import soundfile as sf
import numpy as np
from transformers import AutoModel
import streamlit as st

from orchestrator.pipeline import run_pipeline
from shared.text_normalization import normalize_numerals_te

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TIMING] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TTS_WORKER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tts_worker.py")
TTS_WARMUP_TIMEOUT_SECONDS = 300
TTS_REQUEST_TIMEOUT_SECONDS = 180


class TTSWorker:
    """Keeps a single tts_worker.py --serve subprocess alive for the life of
    the session so the (slow) model load only happens once instead of on
    every reply. Self-heals: a request that times out or a worker that dies
    gets a fresh subprocess spawned for the next request rather than wedging
    the UI forever."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.stdout_queue = None
        self._start()

    def _start(self):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            [sys.executable, TTS_WORKER_PATH, "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.stdout_queue = queue.Queue()
        threading.Thread(target=self._pump_stdout, daemon=True).start()

        try:
            line = self.stdout_queue.get(timeout=TTS_WARMUP_TIMEOUT_SECONDS)
        except queue.Empty:
            self.process.kill()
            raise RuntimeError(f"TTS worker did not become ready within {TTS_WARMUP_TIMEOUT_SECONDS}s")
        if line is None:
            stderr_output = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"TTS worker exited during startup: {stderr_output.strip()}")

        ready = json.loads(line)
        if ready.get("status") != "ready":
            raise RuntimeError(f"TTS worker failed to start: {ready}")

    def _pump_stdout(self):
        for line in self.process.stdout:
            self.stdout_queue.put(line)
        self.stdout_queue.put(None)

    def synthesize(self, text_path, output_path):
        with self.lock:
            if self.process.poll() is not None:
                logger.warning("TTS worker had exited; restarting.")
                self._start()

            request = json.dumps({"text_path": text_path, "output_path": output_path})
            try:
                self.process.stdin.write(request + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                logger.warning("TTS worker pipe was broken; restarting.")
                self._start()
                self.process.stdin.write(request + "\n")
                self.process.stdin.flush()

            try:
                line = self.stdout_queue.get(timeout=TTS_REQUEST_TIMEOUT_SECONDS)
            except queue.Empty:
                logger.error(f"TTS worker timed out after {TTS_REQUEST_TIMEOUT_SECONDS}s; restarting for next request.")
                self.process.kill()
                self._start()
                raise RuntimeError(f"TTS generation timed out after {TTS_REQUEST_TIMEOUT_SECONDS}s")

            if line is None:
                stderr_output = self.process.stderr.read() if self.process.stderr else ""
                self._start()
                raise RuntimeError(f"TTS worker exited unexpectedly: {stderr_output.strip()}")

            response = json.loads(line)
            if response.get("status") != "ok":
                raise RuntimeError(f"TTS worker error: {response.get('message', 'unknown error')}")


@st.cache_resource
def load_asr_model():
    t0 = time.time()
    logger.info("Loading ASR model...")
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
    )
    model.eval()
    logger.info(f"ASR model loaded in {time.time()-t0:.1f}s")
    return model


def transcribe_audio(audio_bytes):
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    wav_np, sr = sf.read(tmp_path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {sr} Hz")

    if wav_np.ndim == 1:
        wav_np = wav_np[np.newaxis, :]
    wav = torch.from_numpy(wav_np)

    asr_model = load_asr_model()
    logger.info(f"ASR model ready (from cache or fresh load) at {time.time()-t0:.1f}s")
    with torch.no_grad():
        transcription = asr_model(wav, "te", "rnnt")
    logger.info(f"ASR transcription done, total {time.time()-t0:.1f}s")
    return transcription


@st.cache_resource
def get_tts_worker():
    logger.info("Starting persistent TTS worker (loads the model once)...")
    t0 = time.time()
    worker = TTSWorker()
    logger.info(f"TTS worker ready in {time.time()-t0:.1f}s")
    return worker


def synthesize_speech(text):
    """Run TTS generation in a persistent standalone subprocess (tts_worker.py)
    rather than in-process. Generation was observed to take 5-7x longer
    in-process inside Streamlit, especially on macOS, likely due to
    thread/BLAS contention between Streamlit's script-runner threads and
    PyTorch's CPU inference threads. A clean subprocess avoids that
    contention. The subprocess is kept alive and reused across replies
    (via get_tts_worker's st.cache_resource caching) so the model-load cost
    is paid once per session instead of on every reply, and it auto-restarts
    if a request times out or the process dies, so a single bad reply can't
    permanently wedge the app."""
    t0 = time.time()
    normalized_text = normalize_numerals_te(text)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as text_tmp:
        text_tmp.write(normalized_text)
        text_path = text_tmp.name
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    worker = get_tts_worker()
    logger.info(f"TTS worker ready (from cache or fresh load) at {time.time()-t0:.1f}s")
    worker.synthesize(text_path, audio_path)
    logger.info(f"TTS generation finished, total synth: {time.time()-t0:.1f}s")

    audio_arr, sample_rate = sf.read(audio_path, dtype="float32")
    return audio_arr, sample_rate


st.set_page_config(page_title="Krishi-Agent", page_icon="🌾")

st.title("Krishi-Agent")

if "farm_profile" not in st.session_state:
    st.session_state.farm_profile = None

with st.form("farm_profile_form"):
    district = st.text_input("District")
    mandi = st.text_input("Nearest Mandi")
    state = st.selectbox("State", ["Telangana", "Andhra Pradesh"])
    crop = st.selectbox("Crop", ["Tomato", "Potato"])
    submitted = st.form_submit_button("Save Profile")

    if submitted:
        if not district or not mandi:
            st.error("Please fill in both District and Mandi.")
        else:
            st.session_state.farm_profile = {
                "district": district,
                "mandi": mandi,
                "state": state,
                "crop": crop,
            }
            st.success("Farm profile saved for this session.")

if st.session_state.farm_profile:
    st.subheader("Current Profile")
    st.json(st.session_state.farm_profile)

st.divider()
st.subheader("Ask a Question")

if not st.session_state.farm_profile:
    st.info("Save your Farm Profile above before asking a question.")
else:
    audio_value = st.audio_input("Or record your question in Telugu")

    if audio_value is not None:
        st.success("Recording captured.")
        st.audio(audio_value)

    question = st.text_input("Type your question in Telugu")

    uploaded_photo = st.file_uploader(
        "Optional: attach a photo of the crop leaf (for disease questions)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_photo is not None:
        st.image(uploaded_photo, caption="Uploaded leaf photo", width=250)

    ask_submitted = st.button("Ask")

    if ask_submitted:
        t_start = time.time()
        final_question = None

        if audio_value is not None:
            with st.spinner("Transcribing your recording..."):
                final_question = transcribe_audio(audio_value.getvalue())
            logger.info(f"ASR stage total: {time.time()-t_start:.1f}s")
            st.write(f"Heard: {final_question}")
        elif question:
            final_question = question

        if not final_question:
            st.error("Please record or type a question.")
        else:
            image_path = None
            if uploaded_photo is not None:
                suffix = os.path.splitext(uploaded_photo.name)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_photo.getvalue())
                    image_path = tmp.name

            t_pipeline = time.time()
            with st.spinner("Thinking..."):
                trace = run_pipeline(final_question, st.session_state.farm_profile, image_path=image_path)
            logger.info(f"Pipeline stage took {time.time()-t_pipeline:.1f}s")
            answer_text = trace["final_answer"]
            st.subheader("Answer")
            st.write(answer_text)

            t_tts = time.time()
            try:
                with st.spinner("Generating spoken reply..."):
                    audio_arr, sample_rate = synthesize_speech(answer_text)
                logger.info(f"TTS stage took {time.time()-t_tts:.1f}s")
                st.audio(audio_arr, sample_rate=sample_rate)
            except RuntimeError as e:
                logger.error(f"TTS stage failed after {time.time()-t_tts:.1f}s: {e}")
                st.warning(f"Could not generate the spoken reply ({e}). The text answer above is still available.")
            logger.info(f"TOTAL end-to-end: {time.time()-t_start:.1f}s")
