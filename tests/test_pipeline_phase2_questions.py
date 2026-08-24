"""
Phase 4.2c -- Broad sanity pass: run the Phase 2 Intent Router question set
(16 real Telugu farmer questions) through the FULL pipeline, live.

Unlike 4.2a/4.2b, nothing is mocked here -- this hits the real Ollama
Intent Router, the real OpenWeatherMap API, and the real data.gov.in API.
The goal is not to assert exact correctness (routing accuracy was already
measured in Phase 2.2 at 87.5%, and conflict resolution was tested in 4.2a)
-- it's to confirm the wired-together pipeline survives realistic, varied
input without crashing. A crash here on a real farmer question is the
highest-value bug this suite can find.

No photo is attached for any question (per the guide, voice/photo
integration is Phase 5) -- so disease intent, when detected, is expected
to be skipped every time here. That's correct behavior, not a failure.

Uses the fixed Warangal/Telangana/Tomato profile throughout -- this is
NOT testing whether the Intent Router's classification is correct (that's
Phase 2.2's job), only whether the full pipeline handles each question
type without crashing.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.pipeline import run_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "intent_router_test_set.json"), encoding="utf-8") as f:
    QUESTIONS = json.load(f)

TEST_PROFILE = {
    "district": "Warangal",
    "mandi": "Warangal APMC",
    "state": "Telangana",
    "crop": "Tomato",
}


def main():
    crashed, survived = 0, 0

    for q in QUESTIONS:
        qid = q["id"]
        question = q["question"]
        gloss = q["gloss"]

        try:
            result = run_pipeline(question, TEST_PROFILE, image_path=None)
            print(f"[OK]   Q{qid} ({gloss}): {result['final_answer']}")
            survived += 1
        except Exception as e:
            print(f"[CRASH] Q{qid} ({gloss}): {type(e).__name__}: {e}")
            crashed += 1

    total = len(QUESTIONS)
    print(f"\n{survived}/{total} survived without crashing ({crashed} crashed)")
    return crashed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
