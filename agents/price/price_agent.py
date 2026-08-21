"""
Price Agent -- Krishi-Agent Phase 1.2

Combines a live "today's" mandi price (data.gov.in Current Daily Price API)
with a seasonal baseline computed from a filtered multi-year historical
archive (data/price_history_ap_telangana.csv, built by
build_price_history.py from a Kaggle-hosted Agmarknet dataset -- the live
API itself only retains today's snapshot, no history).

Rule: compare today's price against the average price for this calendar
month across historical years for the same state/commodity. Returns
output in the shared agent schema:

    {
        "answer": str,
        "confidence": "High" | "Medium" | "Low",
        "reason_for_confidence": str,
        "source_freshness": str,
    }

Standalone module -- no UI, no dependency on other agents.

NOTE: first-draft domain heuristic, not verified agricultural-economics
advice -- sanity-check the thresholds with someone who knows mandis.
"""

import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATA_GOV_IN_API_KEY")
LIVE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(HERE, "..", "..", "data", "price_history_ap_telangana.csv")

SELL_THRESHOLD = 0.10
HOLD_THRESHOLD = -0.10
MIN_HISTORICAL_SAMPLES = 5

_history_cache = None


def _load_history() -> pd.DataFrame:
    global _history_cache
    if _history_cache is None:
        df = pd.read_csv(HISTORY_PATH, low_memory=False)
        df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")
        df = df.dropna(subset=["Arrival_Date", "Modal_Price"])
        df = df[df["Modal_Price"] > 0]
        _history_cache = df
    return _history_cache


def _seasonal_baseline(state: str, commodity: str, month: int):
    df = _load_history()
    subset = df[
        (df["State"] == state)
        & (df["Commodity"] == commodity)
        & (df["Arrival_Date"].dt.month == month)
    ]
    if subset.empty:
        return None, 0
    return float(subset["Modal_Price"].mean()), len(subset)


def _fetch_today_price(state: str, commodity: str, market: str = None):
    if not API_KEY:
        raise RuntimeError("DATA_GOV_IN_API_KEY not set -- check your .env file")

    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 50,
        "filters[state]": state,
        "filters[commodity]": commodity,
    }
    resp = requests.get(
        LIVE_URL, params=params, headers={"User-Agent": BROWSER_UA}, timeout=15
    )
    resp.raise_for_status()
    records = resp.json().get("records", [])
    if not records:
        return None

    if market:
        for r in records:
            if r.get("market", "").strip().lower() == market.strip().lower():
                return r

    prices = [float(r["modal_price"]) for r in records if float(r.get("modal_price", 0)) > 0]
    if not prices:
        return None
    prices.sort()
    median_price = prices[len(prices) // 2]
    return {
        "modal_price": median_price,
        "market": "state-wide median",
        "arrival_date": records[0].get("arrival_date"),
    }


def get_price_advice(state: str, commodity: str, market: str = None) -> dict:
    """Main entry point. Returns advice dict in the shared agent schema."""
    try:
        today_record = _fetch_today_price(state, commodity, market)
    except requests.exceptions.RequestException as e:
        return {
            "answer": "ధరల సమాచారం ప్రస్తుతం అందుబాటులో లేదు.",
            "confidence": "Low",
            "reason_for_confidence": f"API call failed: {e}",
            "source_freshness": "unavailable",
        }

    if today_record is None:
        return {
            "answer": f"{market or state}లో {commodity} ధర ఈరోజు నివేదించబడలేదు.",
            "confidence": "Low",
            "reason_for_confidence": "No live price record found for today for this market/commodity.",
            "source_freshness": "unavailable",
        }

    today_price = float(today_record["modal_price"])
    current_month = datetime.now(timezone.utc).month
    baseline, sample_count = _seasonal_baseline(state, commodity, current_month)

    if baseline is None or sample_count < MIN_HISTORICAL_SAMPLES:
        return {
            "answer": f"ఈరోజు {commodity} ధర {int(today_price)} రూ. కానీ పోల్చడానికి సరిపడా చరిత్ర డేటా లేదు.",
            "confidence": "Low",
            "reason_for_confidence": f"Only {sample_count} historical records for this state/commodity/month -- too few for a reliable seasonal baseline.",
            "source_freshness": f"Live price from {today_record.get('arrival_date')}",
        }

    pct_diff = (today_price - baseline) / baseline

    if pct_diff >= SELL_THRESHOLD:
        answer = (
            f"ఈరోజు {commodity} ధర {int(today_price)} రూ., ఈ నెలలో సాధారణ ధర ({int(baseline)} రూ.) కంటే "
            f"{round(pct_diff * 100)}% ఎక్కువ. ఇప్పుడు అమ్మడం మంచిది."
        )
    elif pct_diff <= HOLD_THRESHOLD:
        answer = (
            f"ఈరోజు {commodity} ధర {int(today_price)} రూ., ఈ నెలలో సాధారణ ధర ({int(baseline)} రూ.) కంటే "
            f"{round(abs(pct_diff) * 100)}% తక్కువ. వీలైతే ఆగడం మంచిది."
        )
    else:
        answer = (
            f"ఈరోజు {commodity} ధర {int(today_price)} రూ., ఇది ఈ నెలలో సాధారణ ధర ({int(baseline)} రూ.)కి "
            f"దగ్గరగా ఉంది."
        )

    if abs(pct_diff) >= 0.20:
        confidence = "High"
    elif abs(pct_diff) >= 0.08:
        confidence = "Medium"
    else:
        confidence = "Low"

    reason = (
        f"Today's modal price {today_price:.0f} compared to a {sample_count}-record seasonal average "
        f"({baseline:.0f}) for {commodity} in {state} during month {current_month}; "
        f"deviation {pct_diff*100:.1f}%."
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "reason_for_confidence": reason,
        "source_freshness": (
            f"Live price from {today_record.get('arrival_date')}; "
            f"seasonal baseline from {sample_count} historical records (2002-2026 archive)"
        ),
    }


if __name__ == "__main__":
    test_cases = [
        ("Telangana", "Tomato", None),
        ("Telangana", "Tomato", "Warangal APMC"),
        ("Andhra Pradesh", "Potato", None),
    ]
    for state, commodity, market in test_cases:
        print(f"\n--- {state} / {commodity} / {market or 'any market'} ---")
        result = get_price_advice(state, commodity, market)
        for k, v in result.items():
            print(f"{k}: {v}")
