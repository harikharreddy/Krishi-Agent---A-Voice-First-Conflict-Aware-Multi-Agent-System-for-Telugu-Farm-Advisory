"""
Weather-lookup-only district name overrides.

orchestrator/pipeline.py passes farm_profile["district"] straight into
OpenWeatherMap's live geocoder as "<district>,IN" (agents/weather/weather_agent.py).
Tested live against the real API on 2026-09-09: 25 of the 59 current AP +
Telangana districts (Requirement 1's lists) return 404 "city not found" --
mostly post-2016/2022 administrative names (person names like "NTR" or
"Alluri Sitharama Raju", or compound names like "Medchal-Malkajgiri") that
aren't in OpenWeatherMap's static city database, plus a few real towns
("Kamareddy", "Wanaparthy") that simply aren't in it either.

Each entry here is a nearby place name that DID resolve, verified live
against the same API (not guessed) -- used only to build the Weather
Agent's location query. The farmer's actual saved district (exact,
official name) is untouched everywhere else: display, and the Price Agent
(which doesn't use district at all -- see agents/price/price_agent.py).

This is backend/UI-layer only; orchestrator/ and agents/ are unchanged.
"""

WEATHER_GEOCODE_OVERRIDES = {
    # Andhra Pradesh
    "Alluri Sitharama Raju": "Paderu",
    "Anakapalli": "Anakapalle",
    "Annamayya": "Rajampet",
    "East Godavari": "Rajahmundry",
    "Konaseema": "Amalapuram",
    "NTR": "Vijayawada",
    "Palnadu": "Narasaraopet",
    "Parvathipuram Manyam": "Parvathipuram",
    "Prakasam": "Ongole",
    "Sri Potti Sriramulu Nellore": "Nellore",
    "Sri Sathya Sai": "Puttaparthi",
    "West Godavari": "Bhimavaram",
    "YSR Kadapa": "Kadapa",
    # Telangana
    "Bhadradri Kothagudem": "Kothagudem",
    "Jayashankar Bhupalapally": "Warangal",
    "Jogulamba Gadwal": "Gadwal",
    "Kamareddy": "Kamareddi",
    "Kumuram Bheem Asifabad": "Asifabad",
    "Mahabubnagar": "Mahbubnagar",
    "Medchal-Malkajgiri": "Medchal",
    "Mulugu": "Warangal",
    "Rajanna Sircilla": "Karimnagar",
    "Ranga Reddy": "Shamshabad",
    "Wanaparthy": "Mahbubnagar",
    "Yadadri Bhuvanagiri": "Bhongir",
}


def weather_location_for(district: str) -> str:
    return WEATHER_GEOCODE_OVERRIDES.get(district, district)
