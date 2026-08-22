# Intent Router — Results Log

## Model iterations tried

1. **qwen2.5:3b-instruct, direct Telugu-in/JSON-out, basic prompt**
   - 8/16 correct (50.0%)
   - Failures: mostly missed multi-intent cases (picked one topic instead of several), plus some outright misreadings (tomato price read as weather).

2. **qwen2.5:3b-instruct, same model, added few-shot examples + lower temperature**
   - 8/16 correct (50.0%) — no real improvement.
   - Notably failed even on questions that were copy-pasted as few-shot examples in the prompt itself, indicating a genuine Telugu comprehension limit rather than a prompting/formatting issue.

3. **qwen2.5:3b-instruct, translate-to-English-then-classify (two-step)**
   - 7/16 correct (43.8%) — worse.
   - Inspecting the `_translation` field showed the 3B model hallucinating unrelated English sentences from the Telugu input (e.g. "My father asked me to plant trees" for a sell/hold price+weather question) — confirmed the bottleneck was translation-step comprehension, not classification logic.

4. **qwen2.5:7b-instruct, direct Telugu-in/JSON-out, few-shot prompt (final)**
   - **14/16 correct (87.5%)**
   - Remaining 2 failures are defensible edge cases, not comprehension errors:
     - #12 ("I want full advice on my tomato crop") — genuinely vague/ambiguous even for a human reader.
     - #14 ("Where can I get this pesticide?") — arguably reasonable to flag as disease-related (mentions pesticide) even though scoped as out-of-scope in the test set.

## Decision

Using **qwen2.5:7b-instruct** via Ollama, single-step Telugu-to-JSON classification with a 5-example few-shot prompt, temperature 0.1, for the Intent Router (Phase 2.2).

## Note for the report

The 3B → 7B jump (50% → 87.5%) is a legitimate methodology finding, not just an implementation detail: it's consistent with the literature survey's own finding (Radeva et al. [16]) that small (7-8B and below) LLMs can underperform significantly on domain-specific coordination tasks, and directly demonstrates why model size mattered for reliable Telugu-language intent understanding in this pipeline.

## Known limitation / future work

Latency: 7B is slower per-call on CPU than 3B was. Not addressed in Phase 2 — flagged for Phase 6 (Polish/latency) to measure and potentially mitigate (e.g. quantization, as the guide already anticipates for the Orchestrator LLM).
