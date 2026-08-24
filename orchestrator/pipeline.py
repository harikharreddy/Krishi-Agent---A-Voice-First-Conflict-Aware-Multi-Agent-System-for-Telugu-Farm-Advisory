"""
Krishi-Agent Pipeline -- Phase 4.1

Wires together the standalone Phase 1-3 pieces into one flow:
typed question + Farm Profile (+ optional leaf photo) -> one Telugu answer.

    Intent Router -> [Disease Agent, Weather Agent, Price Agent] (conditional)
                  -> state mapping (raw agent output -> Conflict Resolver's states)
                  -> Conflict Resolver -> Phrasing Templates -> final answer

Does not touch voice (Phase 5) -- text question in, text answer out.

KNOWN LIMITATIONS (honest, not hidden -- see Phase 8 report):
- Disease state mapping uses a 0.7 confidence cutoff (treat_now vs monitor)
  that is an unvalidated placeholder, not derived from a real PlantDoc
  confidence-calibration analysis. Flagged for a Phase 8 revisit.
- Weather state mapping has no real drought detection -- the Weather Agent
  only computes a 48h rain forecast. Ambiguous/non-rain cases map to
  "uncertain" rather than a fabricated "drought_risk"/"favorable" call.
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.intent_router import route_intent
from orchestrator.conflict_resolver import resolve_conflict
from orchestrator.phrasing_templates import phrase_resolution
from agents.weather.weather_agent import get_weather_advice
from agents.price.price_agent import get_price_advice
from agents.disease.disease_agent import predict_disease

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("krishi_pipeline")

# Draft threshold, not yet calibrated against real PlantDoc confidence data.
DISEASE_CONFIDENCE_CUTOFF = 0.7
HEALTHY_CLASSES = {"Tomato_healthy", "Potato___healthy"}


def _map_disease_state(disease_result: dict) -> str:
    """Map Disease Agent output to a Conflict Resolver state."""
    pred_class = disease_result.get("predicted_class")
    confidence = disease_result.get("confidence", 0.0)

    if pred_class is None:
        return "monitor"  # image read failed etc. -- don't claim treat_now on no data
    if pred_class in HEALTHY_CLASSES:
        return "no_action"
    if confidence >= DISEASE_CONFIDENCE_CUTOFF:
        return "treat_now"
    return "monitor"


def _map_weather_state(weather_result: dict) -> str:
    """Map Weather Agent output to a Conflict Resolver state.
    No real drought detection exists yet -- see module docstring."""
    if "rain_expected" not in weather_result:
        return "uncertain"  # API failure / no forecast data
    return "rain_risk" if weather_result["rain_expected"] else "favorable"


def _map_price_state(price_result: dict) -> str:
    """Map Price Agent output to a Conflict Resolver state,
    reusing the agent's own SELL/HOLD thresholds (0.10 / -0.10)."""
    if "pct_diff" not in price_result:
        return "neutral"  # API failure / no live price / insufficient history
    pct_diff = price_result["pct_diff"]
    if pct_diff >= 0.10:
        return "sell_now"
    if pct_diff <= -0.10:
        return "hold"
    return "neutral"


def run_pipeline(question: str, farm_profile: dict, image_path: str = None) -> dict:
    """
    Main entry point. farm_profile must have: district, mandi, state, crop
    (Phase 2.1 Farm Profile schema). image_path is optional -- Disease Agent
    only runs if both wants_disease is true AND a photo was actually given.

    Returns a dict with the final answer plus every intermediate stage,
    for logging/debugging (Phase 4.2 will lean on this for test failures).
    """
    trace = {"question": question, "farm_profile": farm_profile, "image_path": image_path}

    logger.info(f"Question: {question}")
    intents = route_intent(question)
    trace["intents"] = intents
    logger.info(f"Intents: {intents}")

    disease_state = None
    disease_raw = None
    if intents.get("wants_disease"):
        if image_path:
            disease_raw = predict_disease(image_path)
            disease_state = _map_disease_state(disease_raw)
            logger.info(f"Disease Agent: {disease_raw.get('predicted_class')} "
                        f"(conf {disease_raw.get('confidence', 0):.2f}) -> state={disease_state}")
        else:
            logger.info("Disease intent detected but no photo attached -- skipping Disease Agent.")
    trace["disease_raw"] = disease_raw
    trace["disease_state"] = disease_state

    weather_state = None
    weather_raw = None
    if intents.get("wants_weather"):
        if "district" not in farm_profile or not farm_profile["district"]:
            logger.warning("Weather intent detected but Farm Profile is missing 'district' -- skipping Weather Agent.")
        else:
            location = f"{farm_profile['district']},IN"
            weather_raw = get_weather_advice(location)
            weather_state = _map_weather_state(weather_raw)
            logger.info(f"Weather Agent ({location}): rain_expected="
                        f"{weather_raw.get('rain_expected')} -> state={weather_state}")
    trace["weather_raw"] = weather_raw
    trace["weather_state"] = weather_state

    price_state = None
    price_raw = None
    if intents.get("wants_price"):
        missing = [k for k in ("state", "crop") if k not in farm_profile or not farm_profile[k]]
        if missing:
            logger.warning(f"Price intent detected but Farm Profile is missing {missing} -- skipping Price Agent.")
        else:
            try:
                price_raw = get_price_advice(
                    farm_profile["state"], farm_profile["crop"], farm_profile.get("mandi")
                )
                price_state = _map_price_state(price_raw)
                logger.info(f"Price Agent: pct_diff={price_raw.get('pct_diff')} -> state={price_state}")
            except Exception as e:
                # Unlike Weather Agent, Price Agent's _fetch_today_price() can
                # raise directly (e.g. RuntimeError on a missing API key)
                # rather than always returning a failure-shaped dict.
                logger.warning(f"Price Agent raised an exception -- treating as unavailable: {e}")
                price_raw = None
                price_state = None
    trace["price_raw"] = price_raw
    trace["price_state"] = price_state

    resolution = resolve_conflict(disease_state, weather_state, price_state)
    trace["resolution"] = resolution
    logger.info(f"Conflict Resolver: {resolution}")

    final_answer = phrase_resolution(resolution["resolution"], resolution["confidence"])
    trace["final_answer"] = final_answer
    logger.info(f"Final answer: {final_answer}")

    return trace


if __name__ == "__main__":
    test_profile = {
        "district": "Warangal",
        "mandi": "Warangal APMC",
        "state": "Telangana",
        "crop": "Tomato",
    }
    test_question = "వర్షం వస్తే మందు కొట్టాలా?"
    result = run_pipeline(test_question, test_profile, image_path=None)
    print("\n--- Final answer ---")
    print(result["final_answer"])
