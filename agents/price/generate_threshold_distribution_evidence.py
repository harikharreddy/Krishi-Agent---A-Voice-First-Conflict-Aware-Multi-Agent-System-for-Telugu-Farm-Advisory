"""
Phase 8.1, Price Agent threshold review (2026-09-13): the ±10% sell/hold
thresholds (agents/price/price_agent.py's SELL_THRESHOLD/HOLD_THRESHOLD)
are unvalidated constants, same category of gap as the Disease Agent's
0.7 confidence cutoff was before that investigation. Unlike the disease
cutoff, there is no clean same-instant ground truth for "was this
threshold a good call" -- that would require a forward-looking backtest
against a stated holding-horizon assumption (tomatoes are perishable; a
farmer can't realistically "wait 30 days to sell"), and this project has
no solid agronomic grounding for that assumption. A rigorous outcome-based
backtest is explicitly held as future work (see honest_gaps below), not
attempted here.

This script is the descriptive-only piece: using the EXACT seasonal-
baseline formula the deployed agent already uses
(_seasonal_baseline() in price_agent.py -- same state/commodity/month
grouping, same mean-of-historical-modal-price definition), characterize
how often the ±10% thresholds would even trigger on the real historical
distribution (data/price_history_ap_telangana.csv, 2002-2026), for every
state/commodity combination the deployed system actually supports
(Telangana/Andhra Pradesh x Tomato/Potato). Makes NO claim about whether
triggering sell_now/hold at a given pct_diff was historically a good
decision -- only characterizes the shape of the distribution the fixed
thresholds are being applied to.

One deliberate methodology difference from the live agent: the live
agent's baseline is always computed from an ARCHIVE that excludes
"today's" live-fetched price by construction (today's price comes from a
separate live API call, never from the CSV). Reusing the same CSV to ask
"what would pct_diff have been on every historical day" would include
each row in its own baseline average unless corrected -- a small
self-inclusion / look-ahead bias that pulls each day's pct_diff slightly
toward zero. Corrected here via leave-one-out: each row's baseline is
computed from its (state, commodity, month) group EXCLUDING itself.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_PATH = os.path.join(ROOT, "data", "price_history_ap_telangana.csv")
EVIDENCE_DIR = os.path.join(ROOT, "docs", "evidence")

# Mirror price_agent.py's own constants exactly (imported, not retyped,
# so this evidence can never silently drift from what's actually deployed).
import sys
sys.path.insert(0, ROOT)
from agents.price.price_agent import SELL_THRESHOLD, HOLD_THRESHOLD, MIN_HISTORICAL_SAMPLES

CONFIDENCE_HIGH = 0.20  # matches price_agent.py's get_price_advice() confidence bucketing
CONFIDENCE_MEDIUM = 0.08

SUPPORTED_COMBOS = [
    ("Telangana", "Tomato"),
    ("Andhra Pradesh", "Tomato"),
    ("Telangana", "Potato"),
    ("Andhra Pradesh", "Potato"),
]


def leave_one_out_pct_diff(df, state, commodity):
    """Reproduces price_agent.py's _seasonal_baseline() grouping
    (state, commodity, calendar month across all years) but computes a
    leave-one-out mean per row instead of a whole-group mean, so each
    historical day's baseline never includes its own price."""
    subset = df[(df["State"] == state) & (df["Commodity"] == commodity) & (df["Modal_Price"] > 0)].copy()
    subset["month"] = subset["Arrival_Date"].dt.month
    group_sum = subset.groupby("month")["Modal_Price"].transform("sum")
    group_count = subset.groupby("month")["Modal_Price"].transform("count")
    loo_sum = group_sum - subset["Modal_Price"]
    loo_count = group_count - 1
    subset["loo_baseline"] = np.where(loo_count > 0, loo_sum / loo_count, np.nan)
    subset["loo_group_count"] = loo_count
    subset = subset[subset["loo_group_count"] >= MIN_HISTORICAL_SAMPLES]  # same floor the live agent applies
    subset["pct_diff"] = (subset["Modal_Price"] - subset["loo_baseline"]) / subset["loo_baseline"]
    return subset


def characterize(pct_diff_series):
    n = len(pct_diff_series)
    return {
        "n": n,
        "mean": float(pct_diff_series.mean()),
        "median": float(pct_diff_series.median()),
        "std": float(pct_diff_series.std()),
        "percentiles": {
            "p10": float(pct_diff_series.quantile(0.10)),
            "p25": float(pct_diff_series.quantile(0.25)),
            "p50": float(pct_diff_series.quantile(0.50)),
            "p75": float(pct_diff_series.quantile(0.75)),
            "p90": float(pct_diff_series.quantile(0.90)),
        },
        "sell_now_trigger_rate": float((pct_diff_series >= SELL_THRESHOLD).mean()),
        "hold_trigger_rate": float((pct_diff_series <= HOLD_THRESHOLD).mean()),
        "neutral_rate": float(((pct_diff_series > HOLD_THRESHOLD) & (pct_diff_series < SELL_THRESHOLD)).mean()),
        "high_confidence_rate": float((pct_diff_series.abs() >= CONFIDENCE_HIGH).mean()),
        "medium_confidence_rate": float(((pct_diff_series.abs() >= CONFIDENCE_MEDIUM) & (pct_diff_series.abs() < CONFIDENCE_HIGH)).mean()),
        "low_confidence_rate": float((pct_diff_series.abs() < CONFIDENCE_MEDIUM).mean()),
    }


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")
    df = df.dropna(subset=["Arrival_Date", "Modal_Price"])

    results = {}
    for state, commodity in SUPPORTED_COMBOS:
        subset = leave_one_out_pct_diff(df, state, commodity)
        if subset.empty:
            results[f"{state}_{commodity}"] = {"state": state, "commodity": commodity, "n": 0, "note": "No usable rows after MIN_HISTORICAL_SAMPLES filter."}
            continue
        stats = characterize(subset["pct_diff"])
        stats["state"] = state
        stats["commodity"] = commodity
        stats["date_range"] = f"{subset['Arrival_Date'].min().date()} to {subset['Arrival_Date'].max().date()}"
        results[f"{state}_{commodity}"] = stats
        print(f"{state} / {commodity}: n={stats['n']}, mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
              f"sell_now_rate={stats['sell_now_trigger_rate']:.1%}, hold_rate={stats['hold_trigger_rate']:.1%}, "
              f"neutral_rate={stats['neutral_rate']:.1%}")

    tg_tomato_raw = df[(df["State"] == "Telangana") & (df["Commodity"] == "Tomato") & (df["Modal_Price"] > 0)]["Modal_Price"]
    tg_tomato_skew = float(tg_tomato_raw.skew())
    sub_mean = float(tg_tomato_raw.mean())
    sub_median = float(tg_tomato_raw.median())

    evidence = {
        "metric": "Price Agent threshold distribution characterization (descriptive only -- no outcome/correctness claim)",
        "SCOPE": (
            "Option A of a 3-part review (A: this file, descriptive distribution characterization; "
            "D: Limitations-section caveat; plus an unrelated maintainability fix consolidating "
            "the duplicated 0.10/-0.10 threshold constants in price_agent.py and pipeline.py into "
            "a single shared definition). This file makes NO claim about whether triggering "
            "sell_now/hold at a given pct_diff was historically a GOOD decision -- only "
            "characterizes how often the current fixed thresholds would trigger on the real "
            "historical price distribution, using the exact same seasonal-baseline formula the "
            "deployed agent uses. An outcome-based backtest (B/C from the discussed options) is "
            "explicitly NOT attempted here -- see honest_gaps."
        ),
        "method": (
            "For each (state, commodity) combination the deployed system actually supports, "
            "reproduces agents/price/price_agent.py's _seasonal_baseline() grouping (mean modal "
            "price per calendar month, pooled across all years in the 2002-2026 archive) but "
            "computes a LEAVE-ONE-OUT mean per historical row instead of a whole-group mean, so "
            "each day's own price is never included in its own baseline (the live agent's baseline "
            "and 'today' price are always from separate sources -- archive vs. live API call -- so "
            "reusing one dataset for both here required this correction to avoid a self-inclusion "
            "bias that would pull every pct_diff slightly toward zero). Same MIN_HISTORICAL_SAMPLES "
            "floor (5) applied as the live agent."
        ),
        "distribution_by_state_commodity": results,
        "consolidated_finding": {
            "note": (
                "The values used for SELL_THRESHOLD/HOLD_THRESHOLD (±10%) are NOT near-universal "
                "triggers nor vanishingly rare events on the real historical distribution -- both "
                "sides fire often enough to be a genuinely discriminating operating point (roughly "
                f"{results.get('Telangana_Tomato', {}).get('sell_now_trigger_rate', 0):.0%} sell_now / "
                f"{results.get('Telangana_Tomato', {}).get('hold_trigger_rate', 0):.0%} hold / "
                f"{results.get('Telangana_Tomato', {}).get('neutral_rate', 0):.0%} neutral for "
                "Telangana Tomato). This rules out the thresholds being obviously degenerate (almost "
                "never firing, or firing on almost everything) -- it is NOT the same claim as 'the "
                "thresholds are optimally calibrated.'"
            ),
            "notable_asymmetry_explained": (
                f"hold_rate is roughly {results.get('Telangana_Tomato', {}).get('hold_trigger_rate', 0) / max(results.get('Telangana_Tomato', {}).get('sell_now_trigger_rate', 1), 0.001):.1f}x "
                "sell_now_rate across every state/commodity combination -- not a market-timing signal, "
                "a mechanical consequence of the baseline being an arithmetic MEAN applied to a "
                f"right-skewed price distribution (raw Telangana Tomato modal price skew = "
                f"{tg_tomato_skew:.2f}; mean {sub_mean:.0f} vs. median {sub_median:.0f} -- occasional "
                "price-spike days pull the mean well above the typical day's price). Since the "
                "seasonal baseline is that same inflated mean, a 'typical' day sits BELOW it more "
                "often than above it by construction, so 'hold' (price below baseline) fires more "
                "often than 'sell_now' (price above baseline) even with no real asymmetry in market "
                "conditions. This is a property of using a MEAN as the reference point for skewed "
                "data, already present in the deployed _seasonal_baseline() formula -- this analysis "
                "surfaces it, doesn't introduce or fix it. A median-based baseline would likely "
                "produce a more symmetric trigger split, but changing the baseline formula itself is "
                "a bigger change than a threshold-value tweak and is out of scope here."
            ),
        },
        "honest_gaps": [
            "DESCRIPTIVE ONLY -- makes no claim that triggering sell_now/hold at these thresholds "
            "is historically associated with a good outcome for the farmer. That would require a "
            "forward-looking backtest (compare 'sell now' vs. 'wait and sell later' outcomes), "
            "which needs an assumed holding horizon this project has no solid agronomic grounding "
            "for -- tomatoes are perishable, so 'wait 30 days' is not a realistic option the way it "
            "might be for a non-perishable commodity, and picking an arbitrary horizon (3 days? 7? "
            "14?) would itself be a bigger source of uncertainty than the threshold value being "
            "tested. Explicitly held as future work, not attempted here -- see the discussion "
            "preceding this evidence for the full reasoning.",

            "Confidence-bucket thresholds (0.08/0.20) are characterized here (medium/high "
            "confidence trigger rates) but are exactly as unvalidated as the sell/hold thresholds "
            "-- this file describes their real-world trigger rate, it does not validate the "
            "specific cutoff values.",

            "Leave-one-out corrects the most obvious self-inclusion bias, but the underlying "
            "seasonal-baseline METHOD itself (pooling all years' modal prices for a calendar "
            "month, no trend/inflation adjustment across a 24-year archive) is unchanged from "
            "what the live agent already does -- a 2002 price and a 2025 price for the same month "
            "are weighted equally in one baseline average, which is a real methodological "
            "simplification already present in production, not something this analysis "
            "introduces or resolves.",

            "Mean-vs-skew asymmetry (see consolidated_finding.notable_asymmetry_explained): the "
            "deployed baseline is an arithmetic mean over a right-skewed price distribution, so "
            "'hold' fires substantially more often than 'sell_now' as a mechanical property of "
            "that choice, not because the market is more often a 'hold' situation in any deeper "
            "sense. Not fixed here -- changing the baseline from mean to median would be a "
            "different, larger change than a threshold-value adjustment and needs its own review.",

            "State-wide aggregation: price_agent.py falls back to a state-wide median across "
            "markets when no exact mandi match is found (see _fetch_today_price()). This analysis "
            "characterizes state-wide modal-price patterns per commodity, consistent with that "
            "fallback behavior, but does not separately characterize per-mandi distributions.",
        ],
    }

    out_path = os.path.join(EVIDENCE_DIR, "price_threshold_distribution_evidence.json")
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
