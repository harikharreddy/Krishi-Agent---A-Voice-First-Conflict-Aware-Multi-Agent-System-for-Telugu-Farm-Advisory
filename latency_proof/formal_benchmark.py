"""
Formal N-trial latency benchmark -- closes the "single/few-run measurement,
no reported variance" gap flagged in docs/evaluation_and_validation.md Sec 5.

Benchmarks the CURRENT deployed paths with real repeated-trial statistics
(mean, std, min, max), not the historical Streamlit comparison (that code
was deliberately removed this session -- see git history -- so it can't be
re-run; the original single-measurement numbers, 200-280s / one measured
191s case, stay as a historical data point, already caveated as such).

Two things benchmarked:
1. TTS generation (synthesize_speech()) -- N calls, model stays warm after
   the first (matches real server behavior across requests).
2. Intent Router with the unload-after-use fix active (route_intent() +
   _unload_intent_router_model()) -- N calls, each paying the real reload
   cost, extending the original 3-trial anecdote (5.6/12.1/10.7s) into
   real statistics.

Run:
    source .venv-voice/bin/activate && python3 latency_proof/formal_benchmark.py
"""
import json
import os
import platform
import statistics
import sys
import time

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import psutil

from backend.voice import synthesize_speech
from backend.main import _unload_intent_router_model
from orchestrator.intent_router import route_intent

N_TRIALS = 8

TTS_SENTENCES = [
    "రాబోయే 3 గంటల్లో వర్షం పడే అవకాశం ఉంది.",
    "ఈరోజు టమాటా ధర 1500 రూ., ఈ నెలలో సాధారణ ధర కంటే 20% ఎక్కువ.",
    "Tomato ఆకులో 'లేట్ బ్లైట్' కనిపిస్తోంది.",
    "వెంటనే పంటకు చికిత్స చేయండి, వీలైతే నీరు కూడా పెట్టండి.",
    "ఇప్పుడు చికిత్స చేయడానికి మంచి సమయం — వాతావరణం కూడా అనుకూలంగా ఉంది.",
    "నా మట్టిలో నత్రజని తక్కువగా ఉందా, ఎరువు వేయాలా?",
    "ఈ విషయంలో మాకు స్పష్టమైన సమాధానం లేదు — దయచేసి మీ స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.",
    "పంట ఆకులు ఎండిపోతున్నాయి, వర్షం లేకపోవడం వల్లనా?",
]

ROUTER_QUESTIONS = [
    "నా టమాటా ఆకులపై మచ్చలు ఉన్నాయి, ఇది ఏ వ్యాధి?",
    "రేపు వర్షం పడుతుందా?",
    "ఈ రోజు టమాటా ధర ఎంత?",
    "నా పొలంలో మట్టి బాగుందా?",
    "బంగాళదుంప ఆకులు పసుపు రంగులోకి మారుతున్నాయి, ఎందుకు?",
    "ఈ వారం వాతావరణం ఎలా ఉంటుంది?",
    "మార్కెట్‌లో బంగాళదుంప ధర పెరుగుతుందా?",
    "నేల pH ఎంత ఉండాలి టమాటా కోసం?",
]


def summarize(label, timings):
    mean = statistics.mean(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0
    print(f"\n{label}: n={len(timings)}")
    print(f"  mean={mean:.2f}s  std={std:.2f}s  min={min(timings):.2f}s  max={max(timings):.2f}s")
    print(f"  all trials: {[round(t, 2) for t in timings]}")
    return {"n": len(timings), "mean": mean, "std": std, "min": min(timings), "max": max(timings), "trials": timings}


def main():
    hw = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
    }
    print("=== Hardware ===")
    for k, v in hw.items():
        print(f"  {k}: {v}")

    print(f"\n=== TTS generation, {N_TRIALS} trials (model warm after trial 1) ===")
    tts_timings = []
    for i in range(N_TRIALS):
        text = TTS_SENTENCES[i % len(TTS_SENTENCES)]
        t0 = time.time()
        synthesize_speech(text)
        elapsed = time.time() - t0
        tts_timings.append(elapsed)
        print(f"  trial {i+1}/{N_TRIALS}: {elapsed:.2f}s")
    tts_summary = summarize("TTS generation (synthesize_speech)", tts_timings)
    tts_summary_warm = summarize("TTS generation, EXCLUDING first (cold-load) call", tts_timings[1:])

    print(f"\n=== Intent Router with unload-after-use fix, {N_TRIALS} trials ===")
    router_timings = []
    for i in range(N_TRIALS):
        q = ROUTER_QUESTIONS[i % len(ROUTER_QUESTIONS)]
        t0 = time.time()
        route_intent(q)
        _unload_intent_router_model()
        elapsed = time.time() - t0
        router_timings.append(elapsed)
        print(f"  trial {i+1}/{N_TRIALS}: {elapsed:.2f}s")
    router_summary = summarize("Intent Router + unload (per-question cost)", router_timings)

    result = {
        "hardware": hw,
        "n_trials": N_TRIALS,
        "tts_generation": tts_summary,
        "tts_generation_excluding_cold_start": tts_summary_warm,
        "intent_router_with_unload": router_summary,
    }
    out_path = os.path.join(os.path.dirname(__file__), "formal_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
