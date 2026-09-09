"""
Latency-proof script (throwaway, not the final backend) -- reproduces the
Phase 6.8 standalone TTS baseline outside any web framework, with memory/swap
logging added per the Sep 9 git-log investigation (commit 62124b4's
uneliminated confound: is a slow result memory-bound, or something else).

Run:
    source .venv-voice/bin/activate && python3 latency_proof/standalone_tts.py
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
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.text_normalization import normalize_numerals_te

logging.basicConfig(level=logging.INFO, format="%(asctime)s [LATENCY-PROOF] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Same onnxruntime thread cap as ui/app.py (commit 62124b4) -- kept so this
# script's import-time behavior matches the real app even though it never
# loads the ASR model itself.
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


def synthesize_speech(text, tts_model, tts_tokenizer, description_tokenizer):
    t0 = time.time()
    boost_thread_qos()
    normalized_text = normalize_numerals_te(text)
    device = str(next(tts_model.parameters()).device)

    input_ids = description_tokenizer(TTS_DESCRIPTION, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tts_tokenizer(normalized_text, return_tensors="pt").input_ids.to(device)

    log_memory("before generate()")
    t1 = time.time()
    generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    gen_time = time.time() - t1
    log_memory("after generate()")
    logger.info(f"TTS generation took {gen_time:.1f}s (total incl. tokenization: {time.time() - t0:.1f}s)")

    audio_arr = generation.cpu().float().numpy().squeeze()
    return audio_arr, tts_model.config.sampling_rate, gen_time


if __name__ == "__main__":
    log_memory("process start")
    t_load = time.time()
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to("cpu")
    tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)
    logger.info(f"Model load took {time.time() - t_load:.1f}s")
    log_memory("after model load")

    test_text = "నమస్తే, ఈ రోజు వాతావరణం 65% వర్షం అవకాశం ఉంది, 1250 రూపాయలకు టమాటా అమ్మవచ్చు."
    audio_arr, sr, gen_time = synthesize_speech(test_text, tts_model, tts_tokenizer, description_tokenizer)

    out_path = os.path.join(os.path.dirname(__file__), "standalone_output.wav")
    sf.write(out_path, audio_arr, sr)
    logger.info(f"RESULT: standalone generate() = {gen_time:.1f}s. Saved to {out_path}")
