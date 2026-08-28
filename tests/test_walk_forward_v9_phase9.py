"""
tests/test_walk_forward_v9_phase9.py — V9 Phase 9 regression tests
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.walk_forward import (
    split_train_test, rolling_folds, evaluate_setup_stability,
    walk_forward_readiness, get_observation_date_range, MIN_OBSERVATIONS_PER_FOLD,
)
from datetime import datetime, timedelta


def _obs(ticker, setup_id, day_offset, forward_return_20d):
    ts = (datetime(2025, 1, 1) + timedelta(days=day_offset)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "ticker": ticker, "setup_id": setup_id, "stage": "2 ✅ Uptrend",
        "timestamp": ts,
        "outcomes": {"20D": {"forward_return_%": forward_return_20d,
                             "excess_return_%": forward_return_20d - 1,
                             "mfe_%": 1, "mae_%": -1, "benchmark_return_%": 1.0}},
        "technical_features": {}, "fundamental_features": {}, "valuation_features": {},
        "market_features": {},
    }


def test_get_observation_date_range_empty():
    assert get_observation_date_range([]) == (None, None)


def test_get_observation_date_range_ignores_unparseable():
    obs = [{"timestamp": "not-a-date"}, _obs("A", "S1-BASE", 5, 1.0)]
    lo, hi = get_observation_date_range(obs)
    assert lo == hi == datetime(2025, 1, 6)


def test_split_train_test_respects_cutoff():
    obs = [_obs("A", "S1-BASE", 1, 1.0), _obs("B", "S1-BASE", 10, 2.0), _obs("C", "S1-BASE", 20, 3.0)]
    train, test = split_train_test(obs, datetime(2025, 1, 15))
    assert {o["ticker"] for o in train} == {"A", "B"}
    assert {o["ticker"] for o in test} == {"C"}


def test_rolling_folds_empty_input():
    assert rolling_folds([], n_folds=3) == []


def test_rolling_folds_preserves_chronological_order():
    obs = [_obs(f"T{i}", "S1-BASE", i, float(i)) for i in range(100)]
    folds = rolling_folds(obs, n_folds=3)
    assert len(folds) <= 3
    for train, test in folds:
        max_train_day = max(int(o["ticker"][1:]) for o in train)
        min_test_day = min(int(o["ticker"][1:]) for o in test)
        # Every TEST observation must come chronologically after every TRAIN one.
        assert min_test_day > max_train_day


def test_evaluate_setup_stability_insufficient_history():
    result = evaluate_setup_stability([], "20D")
    assert result["status"] == "insufficient_history"
    assert result["folds_run"] == 0


def test_evaluate_setup_stability_detects_a_setup_that_holds_up():
    # A single setup, consistently profitable across the whole timeline —
    # should be found as best-in-train and "held up" in test in every fold.
    obs = [_obs(f"T{i}", "S2-HIGH-RS", i, 5.0) for i in range(200)]
    result = evaluate_setup_stability(obs, "20D", n_folds=3, min_n=10)
    assert result["status"] == "ok"
    assert result["folds_usable"] >= 1
    for r in result["fold_results"]:
        if r.get("held_up") is not None:
            assert r["held_up"] is True


def test_evaluate_setup_stability_detects_a_setup_that_does_not_hold_up():
    # Profitable in the first half of history, then flips negative in the
    # second half — the "best train setup" should NOT hold up out of sample.
    obs = ([_obs(f"E{i}", "S2-HIGH-RS", i, 8.0) for i in range(100)] +
          [_obs(f"L{i}", "S2-HIGH-RS", 100 + i, -8.0) for i in range(100)])
    result = evaluate_setup_stability(obs, "20D", n_folds=3, min_n=10)
    held_up_flags = [r["held_up"] for r in result["fold_results"] if r.get("held_up") is not None]
    assert False in held_up_flags


def test_walk_forward_readiness_not_ready_on_thin_data():
    obs = [_obs(f"A{i}", "S1-BASE", i, 1.0) for i in range(5)]
    result = walk_forward_readiness(obs, "20D")
    assert result["ready"] is False
    assert result["passed"] is False
    assert result["resolved_observations"] == 5


def test_walk_forward_readiness_ready_and_passes_on_ample_stable_data():
    needed = MIN_OBSERVATIONS_PER_FOLD * 4  # comfortably over the (min_n * (n_folds+1)) bar
    obs = [_obs(f"T{i}", "S2-HIGH-RS", i, 5.0) for i in range(needed)]
    result = walk_forward_readiness(obs, "20D")
    assert result["ready"] is True
    assert result["passed"] is True


def test_walk_forward_readiness_empty_input_does_not_raise():
    result = walk_forward_readiness([], "20D")
    assert result["ready"] is False
    assert result["resolved_observations"] == 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
