"""
Conflict-aware rule table for Krishi-Agent's Orchestrator (Phase 3.1).

Takes the three agents' action-level states and returns a deterministic
resolution — no LLM involved here. Phase 3.2 will phrase this resolution
into a fluent Telugu sentence.

Valid states:
  disease_state: "treat_now" | "monitor" | "no_action" | None
  weather_state: "rain_risk" | "drought_risk" | "favorable" | "uncertain" | None
  price_state:   "sell_now" | "hold" | "neutral" | None

  (None means that agent was not called for this question, per the
  Intent Router's wants_* flags — Phase 2.2.)
"""

# Explicit rule table, keyed on (disease_state, weather_state, price_state).
# Each entry mirrors a scenario in tests/conflict_scenarios.json.
RULES = {
    ("treat_now", "rain_risk", None): {
        "is_conflict": True,
        "resolution": "delay_treatment",
        "action": "Wait until rain clears, then spray — do not skip treatment",
        "confidence": "Medium",
    },
    (None, "rain_risk", "hold"): {
        "is_conflict": True,
        "resolution": "harvest_now",
        "action": "Harvest early to protect crop from storm; note price is trending up",
        "confidence": "High",
    },
    # Fixed Sep 2026 (Phase 8.1, metric #5 evidence work): this combination
    # was missing entirely -- a real farmer question ("Should I sell or
    # hold, considering price and weather?" with no disease intent) fell
    # through to FALLBACK's generic unresolved-conflict answer instead of
    # a real one, found via live-data pipeline testing (see
    # docs/evaluation_and_validation.md Sec. 2.7). Unlike the sibling
    # (None, rain_risk, hold) entry above -- which IS a real conflict
    # (weather urgency overriding a price signal to wait) -- weather and
    # price AGREE here (both point to harvesting now), so this is a
    # synergy case, not a conflict: is_conflict=False, matching the
    # existing treat_now_synergy pattern's precedent.
    (None, "rain_risk", "sell_now"): {
        "is_conflict": False,
        "resolution": "harvest_now",
        "action": "Harvest now — storm risk and a good sell price both point the same way, no real conflict here",
        "confidence": "High",
    },
    ("treat_now", None, "sell_now"): {
        "is_conflict": True,
        "resolution": "treat_first",
        "action": "Treat the crop first — untreated disease compounds losses faster than a price dip",
        "confidence": "Medium",
    },
    (None, "drought_risk", "sell_now"): {
        "is_conflict": True,
        "resolution": "protect_crop",
        "action": "Irrigate/protect crop if not yet ready for harvest — price advice assumes a harvestable crop exists",
        "confidence": "Medium",
    },
    ("monitor", "favorable", "hold"): {
        "is_conflict": False,
        "resolution": "no_conflict",
        "action": "Continue monitoring crop; no urgent action needed",
        "confidence": "High",
    },
    ("no_action", "drought_risk", "neutral"): {
        "is_conflict": False,
        "resolution": "single_agent_action",
        "action": "Irrigate due to drought risk — no disease/price conflict involved",
        "confidence": "High",
    },
    ("treat_now", "drought_risk", "hold"): {
        "is_conflict": True,
        "resolution": "treat_and_irrigate",
        "action": "Treat disease immediately (fastest-compounding risk); irrigate if feasible; price trend is secondary info",
        "confidence": "Medium",
    },
    ("treat_now", "uncertain", None): {
        "is_conflict": True,
        "resolution": "treat_with_caution",
        "action": "Treat now with a caution note about unclear weather — do not wait indefinitely for forecast clarity",
        "confidence": "Low",
    },
    ("monitor", "rain_risk", "sell_now"): {
        "is_conflict": True,
        "resolution": "harvest_now",
        "action": "Harvest now to capture price and avoid rain damage; treatment can wait since disease is mild",
        "confidence": "Medium",
    },
    ("treat_now", "favorable", "sell_now"): {
        "is_conflict": False,
        "resolution": "treat_now_synergy",
        "action": "Treat now — favorable weather is actually the best spray window available",
        "confidence": "High",
    },
    ("no_action", "rain_risk", "hold"): {
        "is_conflict": True,
        "resolution": "protect_crop",
        "action": "Check harvest readiness and protect crop from storm; price optimization is secondary",
        "confidence": "Medium",
    },
    ("monitor", "uncertain", "neutral"): {
        "is_conflict": None,
        "resolution": "unresolved_conflict",
        "action": "Advise the farmer that guidance is uncertain here; suggest confirming with a local agriculture officer",
        "confidence": "Low",
    },
}

# Fallback for any (disease_state, weather_state, price_state) combination
# not explicitly covered above. Locked decision (Phase 3.1): catch-all
# unresolved-conflict flag, not a silent guess.
FALLBACK = {
    "is_conflict": None,
    "resolution": "unresolved_conflict",
    "action": "Advise the farmer that guidance is uncertain here; suggest confirming with a local agriculture officer",
    "confidence": "Low",
}


def resolve_conflict(disease_state=None, weather_state=None, price_state=None):
    """
    Look up the deterministic resolution for a given combination of
    agent states. Returns the FALLBACK dict for any unmatched combination.
    """
    key = (disease_state, weather_state, price_state)
    return RULES.get(key, FALLBACK)
