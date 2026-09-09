# Phase 3 Summary — Krishi-Agent (completed Aug 24, 2026)

## 3.1 — Conflict-rule table (commits `e454de6` scenarios, `5bac241` logic)

- Defined 3 action-level states per agent, chosen to keep the rule table small, enumerable, and testable while preserving the one real-world distinction that changes farmer action (rain vs. drought):
  - **Disease Agent** → `treat_now`, `monitor`, `no_action`
  - **Weather Agent** → `rain_risk`, `drought_risk`, `favorable`, `uncertain`
  - **Price Agent** → `sell_now`, `hold`, `neutral`
- 12 hand-designed conflict scenarios drafted in `tests/conflict_scenarios.json` (schema: `id`, `description`, `inputs` [disease_state/weather_state/price_state], `expected` [is_conflict/resolution/action/confidence], `notes`) — following the "test cases before logic" pattern already used successfully for the Intent Router.
- Scenarios deliberately include: genuine conflicts (spray vs. rain, harvest vs. price-hold, treat vs. drought), non-conflicting "agents agree" cases (to test the table doesn't over-trigger), a false-alarm case (apparent conflict that resolves into synergy), and one deliberate edge case with no strong signal from any agent.
- Deterministic rule table implemented in `orchestrator/conflict_resolver.py` — `resolve_conflict(disease_state, weather_state, price_state)` returns `{is_conflict, resolution, action, confidence}`, keyed on a `(disease_state, weather_state, price_state)` dict lookup.
- **Fallback decision — locked:** any combination not explicitly covered by a rule returns a catch-all `unresolved_conflict` response (honest low-confidence phrasing, suggests confirming with a local agriculture officer) rather than silently guessing. Chosen over "default-to-no-conflict" and "default-to-most-cautious" because it directly extends the project's confidence-aware novelty claim (Section 4, claim #3) and gives a second clean, defensible Phase 8 metric.
- **Known limitation (honest, not hidden):** the table only covers the 12 explicit scenarios out of 36 possible state combinations. Some genuinely harmless/non-conflicting combinations outside those 12 will currently be flagged as "unresolved" rather than correctly identified as fine — conservative but not wrong. Worth expanding later or noting as a limitation in the Phase 8 report.
- `tests/test_conflict_resolver.py` — mirrors `test_intent_router.py`'s plain-script pattern (no pytest). **12/12 (100%) passing.**
- Fallback path manually verified working via a standalone sanity check (an unmatched combination correctly returned `unresolved_conflict`).

## 3.2 — LLM phrasing layer (commit `07f513a`)

- **Real finding:** tested `qwen2.5:7b-instruct` (the same model that reached 87.5% on Intent Router classification) for open-ended Telugu *sentence generation* — a different task. It failed badly on both a free-form greeting and a structured phrasing prompt: garbled, non-fluent output, including a stray English word ("Jaguars") with no relevance. Confirmed by the team (all fluent Telugu speakers) as unusable, not just awkward.
- **Interpretation:** classification (picking one of a few labels) and fluent generation (producing grammatical prose) are very different tasks for a small model — structured-task competence doesn't transfer to generation. This is a citable finding for Phase 8, consistent with literature anchor [16] (Radeva et al.) on small-LLM task-specific limits, and extends the Phase 2.2 finding (model scale matters for Telugu) into a new failure mode.
- **Decision:** pivoted to the template library as the *primary* phrasing path (not just a fallback), matching the project guide's original architecture (Section 3: "template library first, LLM fallback with guardrails, always reviewed by a fluent Telugu speaker"). This was treated as implementing the intended design, not settling for a workaround.
- Templates built in `orchestrator/phrasing_templates.py`, keyed on `(resolution, confidence)` tuples. English structure/hedging logic was drafted first (High → direct statement, Medium → soft qualifier "it's advisable to," Low → explicit uncertainty + defer to local agri officer), then the team wrote and approved the actual Telugu wording for all 10 resolution types used across the 12 scenarios.
- **Bug caught and fixed during testing:** `single_agent_action` template originally used an `{agent_advice}` slot-fill mechanism that was never populated by the test harness, producing an empty-slot sentence. Fixed by converting it to one fixed, fully-Telugu sentence (since only one scenario — drought → irrigate — currently maps to this resolution type). Design note for later: if more single-agent scenarios are added, each should get its own fixed template entry rather than reintroducing slot-filling (which risks mixed-language output).
- `tests/test_phrasing_templates.py` — chains `resolve_conflict()` → `phrase_resolution()` and asserts every scenario produces a complete, non-fallback (unless genuinely unresolved), non-empty Telugu sentence. **12/12 (100%) passing** after the fix.
- **LLM fallback deferred as a named follow-up, not abandoned:** an Indic-tuned model (e.g. Sarvam AI, mentioned in the project guide) is a real candidate to test later for richer/less repetitive phrasing — untested so far, since generic Qwen models (3B and 7B) have now both shown Telugu weaknesses (3B on comprehension in Phase 2.2, 7B on generation in Phase 3.2). Tracked for Phase 6 polish or later revisit, not blocking Phase 4.

## Loose ends not yet cleaned up

- `test_recording.wav` and `ui/mic_test.py.save` — untracked leftovers since Phase 0/1, still harmless, still not deleted or gitignored.

## Commits (all pushed to `main`)

- `e454de6` — 12 conflict scenarios (`tests/conflict_scenarios.json`)
- `5bac241` — Conflict-rule table + fallback (`orchestrator/conflict_resolver.py`, `tests/test_conflict_resolver.py`)
- `07f513a` — Telugu phrasing template library (`orchestrator/phrasing_templates.py`, `tests/test_phrasing_templates.py`)

## State going into Phase 4

All core pieces now exist **standalone** but have never been wired together into one pipeline:
- Intent Router (Phase 2.2)
- Farm Profile (Phase 2.1)
- Weather / Price / Disease Agents (Phase 1)
- Conflict Resolver (Phase 3.1)
- Phrasing Templates (Phase 3.2)

**Phase 4.1** is connecting all of these into one pipeline: typed question in → one combined Telugu text answer out, with logging at each stage.
**Phase 4.2** is end-to-end testing against the Phase 2 question set, the Phase 3 conflict scenarios, and edge cases (no photo attached, missing farm profile, agent API failure).

Per the project guide, Phase 4 is the checkpoint that matters most — it should be protected above Phase 5 (voice) and Phase 6 (polish). A working, honest, text-only pipeline is worth more than a half-working voice demo.
