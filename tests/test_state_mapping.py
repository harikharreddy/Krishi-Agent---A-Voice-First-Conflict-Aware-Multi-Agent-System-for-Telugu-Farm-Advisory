"""
Boundary-value unit tests for orchestrator/pipeline.py's deterministic
state-mapping functions (_map_weather_state, _map_price_state).

These are simple threshold-based decision functions (e.g. pct_diff >= 0.10
-> sell_now), and existing coverage (tests/test_pipeline.py's synthetic
helpers) only exercises comfortably-inside values (0.15, -0.15, 0.0) --
never the exact boundary (0.10, -0.10) or values just either side of it,
which is exactly where an off-by-one comparison-operator bug (>= vs >,
<= vs <) would actually show up. Closes that gap.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.pipeline import _map_weather_state, _map_price_state


def test_price_exact_sell_threshold_is_sell_now():
    # pct_diff >= 0.10 -> sell_now (boundary itself included)
    assert _map_price_state({"pct_diff": 0.10}) == "sell_now"


def test_price_just_below_sell_threshold_is_neutral():
    assert _map_price_state({"pct_diff": 0.0999999}) == "neutral"


def test_price_exact_hold_threshold_is_hold():
    # pct_diff <= -0.10 -> hold (boundary itself included)
    assert _map_price_state({"pct_diff": -0.10}) == "hold"


def test_price_just_above_hold_threshold_is_neutral():
    assert _map_price_state({"pct_diff": -0.0999999}) == "neutral"


def test_price_zero_is_neutral():
    assert _map_price_state({"pct_diff": 0.0}) == "neutral"


def test_price_missing_key_is_neutral_not_a_crash():
    # API failure / no live price / insufficient history
    assert _map_price_state({}) == "neutral"


def test_price_far_above_sell_threshold_is_sell_now():
    assert _map_price_state({"pct_diff": 0.99}) == "sell_now"


def test_price_far_below_hold_threshold_is_hold():
    assert _map_price_state({"pct_diff": -0.99}) == "hold"


def test_weather_missing_key_is_uncertain_not_a_crash():
    # API failure / no forecast data
    assert _map_weather_state({}) == "uncertain"


def test_weather_rain_expected_true_is_rain_risk():
    assert _map_weather_state({"rain_expected": True}) == "rain_risk"


def test_weather_rain_expected_false_is_favorable():
    assert _map_weather_state({"rain_expected": False}) == "favorable"


def test_weather_drought_signal_overrides_rain_expected_false():
    # drought_signal takes priority over the rain_expected check
    assert _map_weather_state({"rain_expected": False, "drought_signal": True}) == "drought_risk"


def test_weather_drought_signal_false_falls_through_to_rain_check():
    assert _map_weather_state({"rain_expected": True, "drought_signal": False}) == "rain_risk"


def main():
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} correct ({100*passed/len(tests):.1f}%)")


if __name__ == "__main__":
    main()
