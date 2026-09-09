"""
ASR/TTS logic for the FastAPI backend -- ported from ui/app.py (Phase 5/6)
without changing any of the actual model-call logic, only the caching
mechanism (module-level globals + a lock instead of @st.cache_resource,
since there's no Streamlit runtime here).

Confirmed via latency_proof/ (Sep 9): running this logic through FastAPI
instead of Streamlit brings TTS generation back to the standalone baseline
(~47-53s on this machine, vs 200-280s inside Streamlit) -- the whole reason
for this migration.
"""
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

from shared.text_normalization import normalize_numerals_te

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TIMING] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ASR (ai4bharat/indic-conformer, trust_remote_code) creates multiple
# onnxruntime.InferenceSession instances internally, each defaulting to a
# thread pool sized to CPU count -- ~85 mostly-idle OS threads once ASR is
# loaded, which was found to slow the LATER TTS generate() call 5-10x
# (commit 62124b4). Capping each session to 1 thread fixes it.
try:
    import onnxruntime as _ort

    _orig_ort_session_init = _ort.InferenceSession.__init__

    def _capped_ort_session_init(self, *args, sess_options=None, **kwargs):
        if sess_options is None:
            sess_options = _ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        return _orig_ort_session_init(self, *args, sess_options=sess_options, **kwargs)

    _ort.InferenceSession.__init__ = _capped_ort_session_init
except ImportError:
    pass

TTS_DESCRIPTION = "Lalitha's voice is calm and clear, with a natural Telugu accent, at a moderate speed with high quality recording."

# macOS pins CPU-bound work on a thread to slow efficiency cores whenever
# that thread's QoS class drops below "user initiated" (commit 154f726).
# Boosting it right before heavy compute reverses that, in-process.
if sys.platform == "darwin":
    _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    _libc.pthread_self.restype = ctypes.c_void_p
    _libc.pthread_set_qos_class_self_np.argtypes = [ctypes.c_int, ctypes.c_int]
    _libc.pthread_set_qos_class_self_np.restype = ctypes.c_int
    _QOS_CLASS_USER_INITIATED = 0x19

    def boost_thread_qos():
        rc = _libc.pthread_set_qos_class_self_np(_QOS_CLASS_USER_INITIATED, 0)
        if rc != 0:
            logger.warning(f"pthread_set_qos_class_self_np failed rc={rc} errno={ctypes.get_errno()}")
else:
    def boost_thread_qos():
        pass


# select_tts_device() always returns "cpu": MPS was tried and reverted
# (commit 944ac49) -- Apple Silicon has no separate GPU memory, so MPS adds
# Metal allocator overhead on top of an already-tight budget instead of
# relieving it, and hit a hard OOM on the real 8GB dev machine.
def select_tts_device():
    return "cpu"


_asr_model = None
_asr_lock = threading.Lock()
_tts_cache = None
_tts_lock = threading.Lock()


def load_asr_model():
    global _asr_model
    if _asr_model is None:
        with _asr_lock:
            if _asr_model is None:
                t0 = time.time()
                logger.info("Loading ASR model...")
                model = AutoModel.from_pretrained(
                    "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
                )
                model.eval()
                logger.info(f"ASR model loaded in {time.time() - t0:.1f}s")
                _asr_model = model
    return _asr_model


def load_tts_model():
    global _tts_cache
    if _tts_cache is None:
        with _tts_lock:
            if _tts_cache is None:
                t0 = time.time()
                device = select_tts_device()
                logger.info(f"Loading TTS model onto device={device}...")
                tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
                    "ai4bharat/indic-parler-tts"
                ).to(device)
                tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
                description_tokenizer = AutoTokenizer.from_pretrained(
                    tts_model.config.text_encoder._name_or_path
                )
                logger.info(f"TTS model loaded on device={device} in {time.time() - t0:.1f}s")
                _tts_cache = (tts_model, tts_tokenizer, description_tokenizer, device)
    return _tts_cache


def transcribe_audio(audio_bytes: bytes) -> str:
    t0 = time.time()
    boost_thread_qos()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        wav_np, sr = sf.read(tmp_path, dtype="float32")
    finally:
        os.unlink(tmp_path)

    if sr != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {sr} Hz")

    if wav_np.ndim == 1:
        wav_np = wav_np[np.newaxis, :]
    wav = torch.from_numpy(wav_np)

    asr_model = load_asr_model()
    logger.info(f"ASR model ready (from cache or fresh load) at {time.time() - t0:.1f}s")
    with torch.no_grad():
        transcription = asr_model(wav, "te", "rnnt")
    logger.info(f"ASR transcription done, total {time.time() - t0:.1f}s")
    return transcription


def synthesize_speech(text: str):
    t0 = time.time()
    boost_thread_qos()
    normalized_text = normalize_numerals_te(text)
    tts_model, tts_tokenizer, description_tokenizer, _ = load_tts_model()
    device = str(next(tts_model.parameters()).device)
    logger.info(f"TTS model ready (from cache or fresh load) at {time.time() - t0:.1f}s, device={device}")

    input_ids = description_tokenizer(TTS_DESCRIPTION, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tts_tokenizer(normalized_text, return_tensors="pt").input_ids.to(device)

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
    logger.info(f"TTS generation took {time.time() - t1:.1f}s (total synth: {time.time() - t0:.1f}s)")

    audio_arr = generation.cpu().float().numpy().squeeze()
    return audio_arr, tts_model.config.sampling_rate
