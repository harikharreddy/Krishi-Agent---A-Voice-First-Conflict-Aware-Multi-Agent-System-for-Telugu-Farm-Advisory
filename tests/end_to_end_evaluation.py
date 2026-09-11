"""
End-to-end pipeline evaluation for the research-paper evaluation package
(docs/evaluation_and_validation.md). Two parts:

1. Text-only characterization (all 16 questions, no photo -- matches
   test_pipeline_phase2_questions.py's setup) -- captures full traces, not
   just crash/no-crash, and classifies every fallback answer by root cause
   (structural gap vs. expected-by-design vs. a genuine resolver coverage
   gap) rather than reporting a bare pass/fail number.

2. Disease-with-photo end-to-end accuracy -- extends the Disease Agent's
   already-measured PER-COMPONENT accuracy (agents/disease/results/) to a
   FULL-SYSTEM check: real PlantDoc test photos with known ground truth,
   run all the way through Intent Router -> Disease Agent -> Conflict
   Resolver -> Phrasing, checking whether the final Telugu answer names
   the correct disease. This is new -- nothing before this measured
   accuracy past the Disease Agent's own predict_disease() call.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.pipeline import run_pipeline
from agents.disease.disease_agent import TELUGU_DISEASE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "agents", "disease", "results")

TEST_PROFILE = {
    "district": "Warangal",
    "mandi": "Warangal APMC",
    "state": "Telangana",
    "crop": "Tomato",
}

FALLBACK_TEXT = (
    "ఈ విషయంలో మాకు స్పష్టమైన సమాధానం లేదు — దయచేసి మీ స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి."
)

# Manual root-cause classification of why each question might fall back,
# based on what's actually implemented (no Soil Agent exists at all; a
# disease intent with no photo is a documented, correct skip -- Phase 5
# is text-only for these 16 by design). Anything NOT explained by one of
# these is flagged as a genuine resolver coverage gap worth fixing.
def classify_fallback(question_item, intents):
    if not any(intents.values()):
        return "expected (no intent detected -- e.g. a greeting)"
    if intents.get("wants_soil"):
        return "structural gap -- no Soil Agent is implemented"
    if intents.get("wants_disease") and not (intents.get("wants_weather") or intents.get("wants_price")):
        return "expected -- disease intent with no photo attached is a documented skip"
    return "UNEXPLAINED -- possible Conflict Resolver coverage gap"


def run_text_only_characterization():
    with open(os.path.join(HERE, "intent_router_test_set.json"), encoding="utf-8") as f:
        questions = json.load(f)

    records = []
    fallback_count = 0
    for q in questions:
        result = run_pipeline(q["question"], TEST_PROFILE, image_path=None)
        is_fallback = result["final_answer"].strip() == FALLBACK_TEXT
        if is_fallback:
            fallback_count += 1
        resolution_info = result.get("resolution") or {}
        records.append({
            "id": q["id"],
            "gloss": q["gloss"],
            "intents_detected": result["intents"],
            "resolution": resolution_info.get("resolution"),
            "confidence": resolution_info.get("confidence"),
            "is_fallback": is_fallback,
            "fallback_reason": classify_fallback(q, result["intents"]) if is_fallback else None,
            "final_answer": result["final_answer"],
        })

    unexplained = [r for r in records if r.get("fallback_reason") == "UNEXPLAINED -- possible Conflict Resolver coverage gap"]

    summary = {
        "n": len(records),
        "substantive_answer_rate": 1 - fallback_count / len(records),
        "fallback_count": fallback_count,
        "fallback_breakdown": {
            reason: sum(1 for r in records if r.get("fallback_reason") == reason)
            for reason in set(r["fallback_reason"] for r in records if r["fallback_reason"])
        },
        "unexplained_fallbacks": unexplained,
        "records": records,
    }
    return summary


def run_disease_photo_end_to_end():
    """Real PlantDoc test photos, known ground truth, through the FULL
    pipeline (not just predict_disease() directly)."""
    plantdoc_root = os.path.join(os.path.dirname(HERE), "data", "plantdoc_raw", "test")
    # Same folder->label map used throughout agents/disease/ eval scripts.
    plantdoc_map = {
        "Tomato leaf": "Tomato_healthy",
        "Tomato Early blight leaf": "Tomato_Early_blight",
        "Tomato leaf bacterial spot": "Tomato_Bacterial_spot",
        "Tomato leaf late blight": "Tomato_Late_blight",
        "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus",
        "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato mold leaf": "Tomato_Leaf_Mold",
        "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
        "Potato leaf early blight": "Potato___Early_blight",
        "Potato leaf late blight": "Potato___Late_blight",
    }

    question = "ఈ ఆకుకు ఏమైంది?"
    records = []
    for plantdoc_class, true_label in plantdoc_map.items():
        src_dir = os.path.join(plantdoc_root, plantdoc_class)
        if not os.path.isdir(src_dir):
            continue
        crop = "Tomato" if true_label.startswith("Tomato") else "Potato"
        profile = {**TEST_PROFILE, "crop": crop}
        for fname in sorted(os.listdir(src_dir))[:3]:  # 3 per class -> ~30 images
            fpath = os.path.join(src_dir, fname)
            if not os.path.isfile(fpath):
                continue
            result = run_pipeline(question, profile, image_path=fpath)
            disease_raw = result.get("disease_raw") or {}
            pred_class = disease_raw.get("predicted_class")
            true_telugu = TELUGU_DISEASE_NAMES.get(true_label, true_label)
            # The final answer should name the disease correctly whenever
            # the component-level prediction itself was correct AND the
            # pipeline didn't drop/alter it downstream. Checks BOTH crop
            # name and disease name text -- several Telugu disease labels
            # are reused across crops (e.g. Tomato_Early_blight and
            # Potato___Early_blight both render as "ఎర్లీ బ్లైట్"), so
            # disease-name-only substring matching would silently accept
            # a correct disease name paired with the wrong crop.
            component_correct = pred_class == true_label
            answer_names_it = (crop in result["final_answer"]) and (true_telugu in result["final_answer"])
            records.append({
                "image": f"{plantdoc_class}/{fname}",
                "true_class": true_label,
                "predicted_class": pred_class,
                "component_correct": component_correct,
                "final_answer_names_correct_disease": answer_names_it,
                "final_answer": result["final_answer"],
            })

    n = len(records)
    component_correct_n = sum(r["component_correct"] for r in records)
    e2e_correct_n = sum(r["final_answer_names_correct_disease"] for r in records)
    # Wiring integrity: among cases where the component got it right, did
    # the final answer also get it right? Should be ~100% -- anything less
    # means the pipeline is losing/corrupting correct predictions downstream.
    wiring_checks = [r for r in records if r["component_correct"]]
    wiring_intact = sum(r["final_answer_names_correct_disease"] for r in wiring_checks)

    return {
        "n": n,
        "component_level_accuracy": component_correct_n / n,
        "end_to_end_accuracy": e2e_correct_n / n,
        "wiring_integrity": wiring_intact / len(wiring_checks) if wiring_checks else None,
        "wiring_integrity_note": "Fraction of component-correct predictions that survived intact through Conflict Resolver + Phrasing into the final answer.",
        "records": records,
    }


def main():
    print("=== Part 1: Text-only characterization (16 questions, no photo) ===")
    text_summary = run_text_only_characterization()
    print(f"Substantive answer rate: {text_summary['substantive_answer_rate']*100:.1f}% "
          f"({text_summary['n'] - text_summary['fallback_count']}/{text_summary['n']})")
    print("Fallback breakdown:")
    for reason, n in text_summary["fallback_breakdown"].items():
        print(f"  {n}x: {reason}")

    print("\n=== Part 2: Disease-with-photo end-to-end accuracy ===")
    disease_summary = run_disease_photo_end_to_end()
    print(f"n={disease_summary['n']}")
    print(f"Component-level accuracy (predict_disease() alone): {disease_summary['component_level_accuracy']*100:.1f}%")
    print(f"End-to-end accuracy (full pipeline's final Telugu answer): {disease_summary['end_to_end_accuracy']*100:.1f}%")
    print(f"Wiring integrity (correct predictions surviving to final answer): {disease_summary['wiring_integrity']*100:.1f}%")

    out_path = os.path.join(os.path.dirname(HERE), "docs", "end_to_end_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"text_only_characterization": text_summary, "disease_photo_end_to_end": disease_summary}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
