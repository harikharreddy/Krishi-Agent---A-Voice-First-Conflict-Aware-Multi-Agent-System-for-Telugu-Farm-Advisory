# Phase 2 Summary — Krishi-Agent (completed Aug 22, 2026)

## 2.1 Farm Profile
- `ui/app.py` — first real Streamlit entry point (previously only had Phase 0's `mic_test.py`)
- Form with District (text), Nearest Mandi (text), Crop (dropdown: Tomato/Potato)
- Stored in `st.session_state` (session-only, resets on refresh — a deliberate simplicity choice, not persisted to disk)
- Validation: empty District/Mandi blocks save with an error message
- Confirmed working end-to-end via browser testing (fill form → save → JSON preview shown → refresh clears it as expected)

## 2.2 Intent Router
- `orchestrator/intent_router.py` — takes a typed Telugu question, returns `wants_disease` / `wants_weather` / `wants_price` / `wants_soil` booleans
- Built a 16-question Telugu test set first (`tests/intent_router_test_set.json`), per the guide's "test cases before logic" habit — flagged as a first draft, not yet reviewed by a fluent Telugu speaker
- **Real finding, iterated through 4 versions:**
  1. `qwen2.5:3b-instruct`, basic prompt — 50% (8/16)
  2. Same model + few-shot examples + low temperature — still 50%, failed even on its own few-shot examples, proving it was a genuine Telugu comprehension limit, not a prompting issue
  3. Same model, added a translate-to-English-first step — 43.8%, *worse* — the 3B model was hallucinating unrelated English translations (e.g. "My father asked me to plant trees" for a sell/hold price+weather question)
  4. Switched to `qwen2.5:7b-instruct`, direct Telugu-in/JSON-out, few-shot — **87.5% (14/16)**, remaining 2 failures are defensible edge cases (a vague "full advice" question, a debatable pesticide/disease scope call), not comprehension errors
- Full iteration log saved in `orchestrator/intent_router_results.md` — worth citing in the Phase 8 report, since it mirrors literature paper [16]'s (Radeva et al.) finding that small 7-8B LLMs underperform on domain-specific coordination tasks
- **Known limitation flagged for Phase 6:** 7B is slower per-call than 3B was — latency not addressed yet, deliberately deferred

## Testing environment
- Ollama (v0.32.15) installed locally on the dev MacBook, running `qwen2.5:7b-instruct`
- All testing done via typed input in a local terminal/browser — no phone/mic involved (voice integration is Phase 5, not Phase 2)

## Committed and pushed to `main`
Commit `8831c86`: `ui/app.py`, `orchestrator/intent_router.py`, `orchestrator/intent_router_results.md`, `tests/intent_router_test_set.json`, `tests/test_intent_router.py`

## Loose ends not yet cleaned up
- `test_recording.wav` and `ui/mic_test.py.save` remain as untracked leftovers from Phase 0/1 — still harmless, still not deleted or gitignored

## State going into Phase 3
- No LLM has been chosen yet for the Orchestrator's phrasing layer (3.2) — the guide says not to assume, so this needs a real decision, same as was done for the Intent Router
- Given Phase 2's finding, `qwen2.5:7b-instruct` (already pulled via Ollama) is a reasonable starting candidate to test first for the Orchestrator, rather than assuming it will work without verification
- Conflict-rule table (3.1) has not been started — needs 10-15 realistic conflict scenarios generated first, per the guide's "test cases before logic" pattern already used successfully in 2.2
