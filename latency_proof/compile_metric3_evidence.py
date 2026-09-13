"""
Phase 8.1, evaluation metric #3: end-to-end latency, question to spoken
answer, broken down per pipeline stage.

Compiles the 6 real HTTP requests made against the running FastAPI
backend (each request's full "timing" block, returned by the API itself
via the instrumentation added to orchestrator/pipeline.py and
backend/main.py) into one evidence file. Not synthetic timing -- every
number here is from an actual /api/ask call, covering different request
shapes (weather-only, price-only, disease-with-photo, multi-agent,
voice-with-real-audio) specifically so a slow stage is identifiable
regardless of which agents a given question happens to trigger.
"""
import glob
import json
import os
import statistics

TRIALS_DIR = "/tmp/latency_trials"
EVIDENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "evidence")

TRIAL_LABELS = {
    "t1_weather.json": "weather-only (typed), 1st",
    "t2_price.json": "price-only (typed)",
    "t3_disease.json": "disease-with-photo (typed)",
    "t4_multiagent.json": "multi-agent: weather+price (typed)",
    "t5_voice.json": "voice question, real audio (weather)",
    "t6_weather2.json": "weather-only (typed), 2nd",
}


def main():
    trials = []
    for fname, label in TRIAL_LABELS.items():
        path = os.path.join(TRIALS_DIR, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        trials.append({"trial": label, "timing": data["timing"]})

    # Per-stage stats across whichever trials actually exercised that stage
    # (e.g. asr_s only appears for the voice trial; disease_agent_s only for
    # the photo trial).
    all_stage_keys = sorted({k for t in trials for k in t["timing"]})
    per_stage = {}
    for key in all_stage_keys:
        values = [t["timing"][key] for t in trials if key in t["timing"]]
        per_stage[key] = {
            "n": len(values),
            "mean_s": statistics.mean(values),
            "median_s": statistics.median(values),
            "min_s": min(values),
            "max_s": max(values),
            "std_s": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95_s": None,
        }

    evidence = {
        "metric": "End-to-end latency, question to spoken answer, per pipeline stage",
        "method": (
            "Instrumented orchestrator/pipeline.py (per-agent + intent router + "
            "conflict resolver timing added to trace['timing']) and "
            "backend/main.py (ASR, intent-router-unload, TTS timing added, "
            "combined into the /api/ask response's 'timing' field). 6 real "
            "HTTP requests against the running FastAPI backend (not synthetic "
            "calls), covering different request shapes so a slow stage is "
            "identifiable regardless of which agents a given question happens "
            "to trigger."
        ),
        "trials": trials,
        "per_stage_summary": per_stage,
        "distributional_statistics_note": (
            "Per external technical review (2026-09-13, Sec 2.4): a single-run point estimate "
            "doesn't show distributional properties, so median is now reported alongside mean/"
            "min/max/std for every stage. p95_s is intentionally left as null (not fabricated) "
            "for every stage -- with n=6 total trials (and n=1-2 for several individual stages, "
            "e.g. asr_s, disease_agent_s), a 95th-percentile estimate would need roughly 20+ "
            "independent samples to mean anything; computing 'p95' from 6 points would just be "
            "reporting the max relabeled with false precision. Request_total_s and "
            "pipeline_total_s (the two stages with the full n=6) show the widest spread already "
            "visible in min/max/std (request_total_s: 38.4s-80.9s, std 16.5s) -- that spread IS "
            "the honest signal about tail latency at this sample size; a fabricated percentile "
            "would not add information beyond what min/max/std already show. More trials (ideally "
            "20+) would be needed before a real p95 claim is defensible."
        ),
        "honest_gaps": [
            (
                "n=6 total, and most individual stages have n=1-2 -- this is "
                "enough to identify WHICH stage dominates (TTS, overwhelmingly) "
                "and roughly how per-agent costs compare, but not enough for "
                "tight confidence intervals per stage. Sec 2.4/2.6's formal "
                "8-trial benchmark remains the statistically stronger source "
                "for TTS-alone and intent-router-alone numbers; this evidence "
                "adds the STAGE BREAKDOWN and realistic full-request context "
                "those isolated benchmarks didn't capture."
            ),
            (
                "TTS generation time varies with the LENGTH of the answer text "
                "being synthesized, not just model warmth -- trial 2 (price "
                "answer, longer text with numbers) took 67.9s vs trial 3's "
                "23.1s (short disease diagnosis sentence). Reporting a single "
                "'TTS latency' number without accounting for this conflates two "
                "different things."
            ),
            (
                "Intent Router timing appears to increase as more models are "
                "concurrently resident in memory -- trial 5 (after ASR AND TTS "
                "models were both already loaded) showed 23.7s vs. the ~12-14s "
                "seen in trials 1-4 (TTS loaded, ASR not). This is consistent "
                "with the already-documented 8GB RAM memory-collision issue "
                "(Sec 2.6) but was not isolated as a controlled variable here -- "
                "flagged as an observation, not a proven causal claim."
            ),
            (
                "The very first request of a cold server pays additional "
                "one-time model-load costs (observed: TTS model load added "
                "~15.6s to the first request) that don't recur on subsequent "
                "requests. All 6 trials here were run after that first warm-up, "
                "so these numbers reflect a WARM server, not first-request "
                "latency -- which matters for a real deployment's first user "
                "of a session."
            ),
        ],
        "headline_finding": (
            "TTS dominates end-to-end latency in every trial (23-68s of the "
            "38-81s total per request), 2-6x longer than the Intent Router "
            "stage (12-24s) and 10-100x longer than any single agent call "
            "(0.18-1.2s each). Conflict Resolver itself is effectively free "
            "(microseconds) in every trial -- the bottleneck is unambiguously "
            "TTS generation, not orchestration logic."
        ),
    }

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    out_path = os.path.join(EVIDENCE_DIR, "metric3_latency_evidence.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    print(f"n trials: {len(trials)}")
    print("\nPer-stage summary:")
    for key, stats in per_stage.items():
        print(f"  {key:<28} n={stats['n']}  mean={stats['mean_s']:.2f}s  median={stats['median_s']:.2f}s  "
              f"min={stats['min_s']:.2f}s  max={stats['max_s']:.2f}s  p95=not-meaningful-at-this-n")
    print(f"\n{evidence['headline_finding']}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
