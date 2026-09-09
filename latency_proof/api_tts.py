"""
Latency-proof FastAPI app (throwaway, not the final backend) -- serves the
identical synthesize_speech() logic as standalone_tts.py through one HTTP
endpoint, so its generate() timing can be compared directly against the
standalone script's. This is the hard-constraint check from the migration
plan: if this number isn't close to standalone_tts.py's, the framework swap
hasn't fixed the Phase 6.8 latency bug and we stop before building anything
else.

Run (in one terminal):
    source .venv-voice/bin/activate && uvicorn latency_proof.api_tts:app --port 8000

Then, in a second terminal:
    curl -X POST http://127.0.0.1:8000/synthesize
"""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import ctypes
import sys
import time
import logging

import psutil
from fastapi import FastAPI
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.text_normalization import normalize_numerals_te

logging.basicConfig(level=logging.INFO, format="%(asctime)s [LATENCY-PROOF-API] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

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


def log_memory(label):
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    logger.info(
        f"[MEM {label}] RAM used={vm.percent}% available={vm.available / 1e9:.2f}GB "
        f"| swap used={sm.used / 1e9:.2f}GB ({sm.percent}%)"
    )


app = FastAPI()
_model_cache = {}


def load_tts_model():
    if "model" not in _model_cache:
        log_memory("before model load")
        t0 = time.time()
        tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to("cpu")
        tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)
        logger.info(f"Model load took {time.time() - t0:.1f}s")
        log_memory("after model load")
        _model_cache["model"] = (tts_model, tts_tokenizer, description_tokenizer)
    return _model_cache["model"]


@app.post("/synthesize")
def synthesize(text: str = "నమస్తే, ఈ రోజు వాతావరణం 65% వర్షం అవకాశం ఉంది, 1250 రూపాయలకు టమాటా అమ్మవచ్చు."):
    t0 = time.time()
    boost_thread_qos()
    tts_model, tts_tokenizer, description_tokenizer = load_tts_model()
    normalized_text = normalize_numerals_te(text)

    input_ids = description_tokenizer(TTS_DESCRIPTION, return_tensors="pt").input_ids.to("cpu")
    prompt_input_ids = tts_tokenizer(normalized_text, return_tensors="pt").input_ids.to("cpu")

    log_memory("before generate()")
    t1 = time.time()
    generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    gen_time = time.time() - t1
    log_memory("after generate()")
    total_time = time.time() - t0
    logger.info(f"TTS generation took {gen_time:.1f}s (total request: {total_time:.1f}s)")

    return {"gen_time_seconds": round(gen_time, 1), "total_request_seconds": round(total_time, 1)}
