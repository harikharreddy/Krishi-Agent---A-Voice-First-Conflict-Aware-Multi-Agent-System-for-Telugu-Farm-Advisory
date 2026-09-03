"""
Weather Agent -- Krishi-Agent Phase 1.1 (+ Phase 8 prep: drought signal)

Calls OpenWeatherMap's free 5-day/3-hour forecast API for a given location
and applies a simple rule: is rain likely in the next 24-48 hours.
Returns output in the shared agent schema:

    {
        "answer": str,
        "confidence": "High" | "Medium" | "Low",
        "reason_for_confidence": str,
        "source_freshness": str,
    }

Standalone module -- no UI, no dependency on other agents.
"""

import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

RAIN_POP_THRESHOLD = 0.5   # probability of precipitation (0-1) counted as "likely rain"
FORECAST_WINDOW_HOURS = 48

# Drought signal thresholds -- NOTE: this is a short-range PROXY only.
# True drought is a multi-week/seasonal phenomenon that a 5-day forecast
# cannot actually observe. Flags when every forecasted day is both dry and
# hot, documented honestly as a limited proxy, not real drought detection.
DROUGHT_POP_THRESHOLD = 0.2      # max rain probability per day counted as "dry"
DROUGHT_TEMP_THRESHOLD = 35.0    # mean daily temp (C) counted as "hot"


def _fetch_forecast(location: str) -> dict:
    """location like 'Hyderabad,IN' or 'Warangal,IN'."""
    if not API_KEY:
        raise RuntimeError("OPENWEATHERMAP_API_KEY not set -- check your .env file")
    resp = requests.get(
        FORECAST_URL,
        params={"q": location, "appid": API_KEY, "units": "metric"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _entries_within_window(forecast: dict, hours: int) -> list:
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours)
    entries = []
    for entry in forecast.get("list", []):
        entry_time = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
        if now <= entry_time <= cutoff:
            entries.append(entry)
    return entries


def _daily_summaries(forecast: dict) -> list:
    """Groups the full 5-day/3-hour forecast into per-day summaries
    (max rain probability, mean temperature). Used only for the drought
    signal below -- does not affect the existing 48h rain_expected logic."""
    by_day = defaultdict(list)
    for entry in forecast.get("list", []):
        entry_time = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
        day_key = entry_time.date()
        by_day[day_key].append(entry)

    summaries = []
    for day, entries in sorted(by_day.items()):
        max_pop = max(e.get("pop", 0.0) for e in entries)
        mean_temp = sum(e["main"]["temp"] for e in entries) / len(entries)
        summaries.append({"date": day, "max_pop": max_pop, "mean_temp": mean_temp})
    return summaries


def _check_drought_signal(forecast: dict) -> bool:
    """Short-range proxy only -- see module-level threshold comments."""
    daily = _daily_summaries(forecast)
    if not daily:
        return False
    return all(
        d["max_pop"] <= DROUGHT_POP_THRESHOLD and d["mean_temp"] >= DROUGHT_TEMP_THRESHOLD
        for d in daily
    )


def get_weather_advice(location: str) -> dict:
    """Main entry point. Returns advice dict in the shared agent schema."""
    try:
        forecast = _fetch_forecast(location)
    except requests.exceptions.RequestException as e:
        return {
            "answer": "వాతావరణ సమాచారం ప్రస్తుతం అందుబాటులో లేదు.",
            "confidence": "Low",
            "reason_for_confidence": f"API call failed: {e}",
            "source_freshness": "unavailable",
        }

    window_entries = _entries_within_window(forecast, FORECAST_WINDOW_HOURS)

    if not window_entries:
        return {
            "answer": "రాబోయే 48 గంటల్లో వాతావరణ సూచన అందుబాటులో లేదు.",
            "confidence": "Low",
            "reason_for_confidence": "No forecast entries returned within the 48-hour window.",
            "source_freshness": "unavailable",
        }

    max_pop = max(e.get("pop", 0.0) for e in window_entries)
    rain_entries = [e for e in window_entries if e.get("pop", 0.0) >= RAIN_POP_THRESHOLD]
    rain_expected = len(rain_entries) > 0
    drought_signal = _check_drought_signal(forecast) if not rain_expected else False

    if max_pop >= 0.7 or max_pop <= 0.2:
        confidence = "High" if not drought_signal else "Medium"
    else:
        confidence = "Medium"

    # Phase 6 -- confidence-appropriate hedging, so Medium-confidence
    # answers audibly sound less certain than High-confidence ones,
    # not just carry a different internal label.
    hedge = "" if confidence == "High" else ", అయితే ఖచ్చితంగా చెప్పలేం"

    if drought_signal:
        answer = (
            f"రాబోయే రోజుల్లో వర్షం లేకుండా, ఎండ ఎక్కువగా ఉండే అవకాశం ఉంది{hedge} "
            f"(ఇది స్వల్పకాలిక సూచన మాత్రమే)."
        )
    elif rain_expected:
        first_rain = rain_entries[0]
        rain_time = datetime.fromtimestamp(first_rain["dt"], tz=timezone.utc)
        hours_until = round((rain_time - datetime.now(timezone.utc)).total_seconds() / 3600)
        answer = (
            f"రాబోయే {hours_until} గంటల్లో వర్షం పడే అవకాశం ఉంది{hedge} "
            f"(వర్షం సంభావ్యత {round(max_pop * 100)}%)."
        )
    else:
        answer = (
            f"రాబోయే 48 గంటల్లో వర్షం పడే అవకాశం తక్కువ{hedge} "
            f"(గరిష్ట సంభావ్యత {round(max_pop * 100)}%)."
        )

    reason = (
        f"Based on {len(window_entries)} forecast points from OpenWeatherMap "
        f"over the next {FORECAST_WINDOW_HOURS}h; max precipitation probability {max_pop:.2f}."
    )
    if drought_signal:
        reason += " Drought proxy signal: all forecasted days dry+hot (short-range only, not true drought detection)."

    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "answer": answer,
        "confidence": confidence,
        "reason_for_confidence": reason,
        "source_freshness": f"OpenWeatherMap forecast retrieved {retrieved_at}",
        # Added Phase 4.1b: structured fields for the Orchestrator's
        # conflict-state mapping (avoids parsing the Telugu answer text).
        "max_pop": max_pop,
        "rain_expected": rain_expected,
        "drought_signal": drought_signal,  # NEW -- short-range proxy, see _check_drought_signal docstring
    }


if __name__ == "__main__":
    test_locations = ["Hyderabad,IN", "Warangal,IN", "Vijayawada,IN"]
    for loc in test_locations:
        print(f"\n--- {loc} ---")
        result = get_weather_advice(loc)
        for k, v in result.items():
            print(f"{k}: {v}")
