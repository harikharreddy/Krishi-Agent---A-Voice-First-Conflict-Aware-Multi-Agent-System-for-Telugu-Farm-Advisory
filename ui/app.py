import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import ctypes
import sys
import threading
import time
import logging
import tempfile
import torch
import soundfile as sf
import numpy as np
from transformers import AutoModel, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import streamlit as st

from orchestrator.pipeline import run_pipeline
from shared.text_normalization import normalize_numerals_te

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TIMING] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TTS_DESCRIPTION = "Lalitha's voice is calm and clear, with a natural Telugu accent, at a moderate speed with high quality recording."

# macOS pins a thread's CPU-bound work to slow efficiency cores whenever that
# thread's QoS class drops below "user initiated" -- Streamlit runs each script
# rerun on its own worker thread, and once Terminal loses focus (e.g. the user
# switches to the browser to use the app) that thread's QoS gets demoted,
# producing a 5-10x slowdown for CPU-bound generation that never shows up when
# running the same code as a plain foreground script. Explicitly boosting the
# calling thread's QoS class right before heavy compute (measured to fully
# reverse the slowdown, in-process, no subprocess/root needed) works around it.
if sys.platform == "darwin":
    _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    _libc.pthread_self.restype = ctypes.c_void_p
    _libc.pthread_set_qos_class_self_np.argtypes = [ctypes.c_int, ctypes.c_int]
    _libc.pthread_set_qos_class_self_np.restype = ctypes.c_int
    _libc.pthread_get_qos_class_np.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    _libc.pthread_get_qos_class_np.restype = ctypes.c_int
    _QOS_CLASS_USER_INITIATED = 0x19
    _QOS_NAMES = {
        0x21: "USER_INTERACTIVE",
        0x19: "USER_INITIATED",
        0x15: "DEFAULT",
        0x11: "UTILITY",
        0x09: "BACKGROUND",
        0x00: "UNSPECIFIED",
    }

    def current_thread_qos():
        qos_out = ctypes.c_int(0)
        rel_out = ctypes.c_int(0)
        _libc.pthread_get_qos_class_np(_libc.pthread_self(), ctypes.byref(qos_out), ctypes.byref(rel_out))
        return qos_out.value, _QOS_NAMES.get(qos_out.value, hex(qos_out.value))

    def boost_thread_qos():
        before_val, before_name = current_thread_qos()
        rc = _libc.pthread_set_qos_class_self_np(_QOS_CLASS_USER_INITIATED, 0)
        after_val, after_name = current_thread_qos()
        logger.info(
            f"boost_thread_qos() called on thread={threading.current_thread().name} "
            f"tid={threading.get_ident()} before={before_name} rc={rc} after={after_name}"
        )
        if rc != 0:
            logger.warning(f"pthread_set_qos_class_self_np failed rc={rc} errno={ctypes.get_errno()}")
else:
    def current_thread_qos():
        return None, "n/a"

    def boost_thread_qos():
        pass


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


def select_tts_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource
def load_tts_model():
    t0 = time.time()
    device = select_tts_device()
    logger.info(f"Loading TTS model onto device={device}...")
    try:
        tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/indic-parler-tts"
        ).to(device)
    except Exception:
        logger.exception(f"Failed to load TTS model on device={device}, falling back to cpu")
        device = "cpu"
        tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/indic-parler-tts"
        ).to(device)
    tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(
        tts_model.config.text_encoder._name_or_path
    )
    logger.info(f"TTS model loaded on device={device} in {time.time()-t0:.1f}s")
    return tts_model, tts_tokenizer, description_tokenizer, device


def transcribe_audio(audio_bytes):
    t0 = time.time()
    logger.info(f"transcribe_audio() entered on thread={threading.current_thread().name} tid={threading.get_ident()}")
    boost_thread_qos()
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
    _, qos_name = current_thread_qos()
    logger.info(f"QoS immediately before ASR inference: {qos_name}")
    with torch.no_grad():
        transcription = asr_model(wav, "te", "rnnt")
    logger.info(f"ASR transcription done, total {time.time()-t0:.1f}s")
    return transcription


def synthesize_speech(text):
    t0 = time.time()
    logger.info(f"synthesize_speech() entered on thread={threading.current_thread().name} tid={threading.get_ident()}")
    boost_thread_qos()
    normalized_text = normalize_numerals_te(text)
    tts_model, tts_tokenizer, description_tokenizer, _ = load_tts_model()
    device = str(next(tts_model.parameters()).device)
    logger.info(f"TTS model ready (from cache or fresh load) at {time.time()-t0:.1f}s, device={device}")

    input_ids = description_tokenizer(TTS_DESCRIPTION, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tts_tokenizer(normalized_text, return_tensors="pt").input_ids.to(device)

    _, qos_name = current_thread_qos()
    logger.info(
        f"QoS immediately before tts_model.generate(): {qos_name} "
        f"(thread={threading.current_thread().name} tid={threading.get_ident()}, "
        f"device={device}, torch.get_num_threads()={torch.get_num_threads()})"
    )
    t1 = time.time()
    try:
        generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    except Exception:
        if device == "cpu":
            raise
        logger.exception(f"generate() failed on device={device}, retrying on cpu")
        tts_model = tts_model.to("cpu")
        generation = tts_model.generate(
            input_ids=input_ids.to("cpu"), prompt_input_ids=prompt_input_ids.to("cpu")
        )
    logger.info(f"TTS generation took {time.time()-t1:.1f}s (total synth: {time.time()-t0:.1f}s)")

    audio_arr = generation.cpu().float().numpy().squeeze()
    return audio_arr, tts_model.config.sampling_rate


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
            with st.spinner("Generating spoken reply... (this can take a few minutes on this device)"):
                audio_arr, sample_rate = synthesize_speech(answer_text)
            logger.info(f"TTS stage took {time.time()-t_tts:.1f}s")
            st.audio(audio_arr, sample_rate=sample_rate)
            logger.info(f"TOTAL end-to-end: {time.time()-t_start:.1f}s")
