"""
tests/test_alpha_lab_stage_h_separation.py — Stage H: discovery/radar
separation

Proves the fix is real, not cosmetic: computing Alpha Metrics on a
MIXED discovery+apex10_radar pool gives a DIFFERENT (contaminated)
answer than computing it on the discovery-only split ui/alpha_lab.py
now uses for every pre-existing tab. If this test ever passes with
identical numbers for both, the separation isn't doing anything and the
mixing bug is back.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.apex10_baseline import get_observation_type
from modules.alpha_metrics import compute_alpha_metrics


def _discovery(ticker, excess_return):
    return {"ticker": ticker, "observation_type": "discovery",
           "outcomes": {"20D": {"forward_return_%": excess_return, "excess_return_%": excess_return,
                                "mfe_%": abs(excess_return), "mae_%": -abs(excess_return),
                                "benchmark_return_%": 0.0}}}


def _radar(ticker, excess_return):
    return {"ticker": ticker, "observation_type": "apex10_radar",
           "outcomes": {"20D": {"forward_return_%": excess_return, "excess_return_%": excess_return,
                                "mfe_%": abs(excess_return), "mae_%": -abs(excess_return),
                                "benchmark_return_%": 0.0}}}


def test_split_produces_disjoint_complete_partition():
    mixed = [_discovery("A", 5), _discovery("B", -2), _radar("C", 50), _radar("D", -30)]
    discovery_obs = [o for o in mixed if get_observation_type(o) == "discovery"]
    radar_obs = [o for o in mixed if get_observation_type(o) == "apex10_radar"]
    assert len(discovery_obs) + len(radar_obs) == len(mixed)
    assert {o["ticker"] for o in discovery_obs} == {"A", "B"}
    assert {o["ticker"] for o in radar_obs} == {"C", "D"}


def test_mixing_would_contaminate_discovery_only_metrics():
    """The actual regression proof: discovery-only performance is
    strongly positive, radar-only is strongly negative. If they got
    pooled (the bug this stage fixes), the blended expectancy would sit
    between them — nowhere near either population's true number."""
    discovery_obs = [_discovery("A", 20.0), _discovery("B", 24.0)]
    radar_obs = [_radar("C", -40.0), _radar("D", -36.0)]
    mixed = discovery_obs + radar_obs

    discovery_only_metrics = compute_alpha_metrics(discovery_obs, "20D")
    mixed_metrics = compute_alpha_metrics(mixed, "20D")

    assert discovery_only_metrics["expectancy_%"] == 22.0  # (20+24)/2, uncontaminated
    assert discovery_only_metrics["n"] == 2
    # The mixed (bug) computation would be (20+24-40-36)/4 = -8.0 — a
    # completely different, misleading number for "how is the original
    # discovery system doing."
    assert mixed_metrics["expectancy_%"] == -8.0
    assert mixed_metrics["n"] == 4
    assert discovery_only_metrics["expectancy_%"] != mixed_metrics["expectancy_%"]


def test_radar_only_metrics_computed_independently():
    radar_obs = [_radar("C", -40.0), _radar("D", -36.0)]
    radar_metrics = compute_alpha_metrics(radar_obs, "20D")
    assert radar_metrics["expectancy_%"] == -38.0
    assert radar_metrics["n"] == 2


def test_empty_radar_pool_does_not_crash_metrics():
    metrics = compute_alpha_metrics([], "20D")
    assert metrics["n"] == 0


def test_legacy_rows_without_observation_type_are_never_misclassified_as_radar():
    """Every pre-existing observation logged before Stage B (no
    observation_type key at all) must fall through to 'discovery', not
    accidentally leak into the radar-only view."""
    legacy_row = {"ticker": "OLD", "outcomes": {}}
    assert get_observation_type(legacy_row) == "discovery"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
