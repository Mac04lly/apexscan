"""
tests/test_apex10_baseline_stage_b.py — Apex the Great X Stage B tests
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.apex10_baseline import (
    compute_baseline_snapshot, get_observation_type, BASELINE_MODEL_VERSION, BASELINE_HORIZONS,
)


def _discovery_obs(ticker, forward_return_20d):
    return {
        "ticker": ticker, "observation_type": "discovery",
        "outcomes": {"20D": {"forward_return_%": forward_return_20d,
                             "excess_return_%": forward_return_20d - 1,
                             "mfe_%": abs(forward_return_20d), "mae_%": -abs(forward_return_20d),
                             "benchmark_return_%": 1.0}},
    }


def _radar_obs(ticker):
    return {"ticker": ticker, "observation_type": "apex10_radar", "outcomes": {}}


def test_get_observation_type_defaults_to_discovery_for_legacy_rows():
    legacy_row = {"ticker": "AAA"}  # no observation_type key at all, like the 599 real rows
    assert get_observation_type(legacy_row) == "discovery"


def test_get_observation_type_respects_explicit_tag():
    assert get_observation_type(_radar_obs("BBB")) == "apex10_radar"


def test_baseline_excludes_radar_observations():
    obs = [_discovery_obs("A", 5.0), _discovery_obs("B", -2.0), _radar_obs("C")]
    snap = compute_baseline_snapshot(obs, horizons=["20D"])
    assert snap["total_discovery_observations"] == 2
    assert snap["horizons"]["20D"]["n"] == 2


def test_baseline_model_version_is_fixed_label():
    snap = compute_baseline_snapshot([], horizons=["20D"])
    assert snap["model_version"] == BASELINE_MODEL_VERSION


def test_baseline_empty_input_does_not_raise():
    snap = compute_baseline_snapshot([], horizons=BASELINE_HORIZONS)
    assert snap["total_discovery_observations"] == 0
    for h in BASELINE_HORIZONS:
        assert snap["horizons"][h]["n"] == 0


def test_baseline_computes_all_default_horizons():
    obs = [_discovery_obs("A", 3.0)]
    snap = compute_baseline_snapshot(obs)
    assert set(snap["horizons"].keys()) == set(BASELINE_HORIZONS)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
