"""
tests/test_short_horizons.py — 1D/2D/3D/4D horizon addition

Verifies the new short horizons are correctly wired into the single
source of truth (modules/outcome_engine.py) and correctly derived
everywhere else, rather than re-duplicated — and that the "fully
resolved" threshold used by both the long and short radar trackers
correctly grew from 5 to 9 required horizons as a result.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.outcome_engine import HORIZONS, SHORT_NOISY_HORIZONS, _HORIZON_CALENDAR_BUFFER
from modules.apex10_baseline import BASELINE_HORIZONS
import ui.alpha_lab as alpha_lab
from modules.apex10_tracker import _is_fully_resolved as _is_fully_resolved_long
from modules.apex10_short_tracker import _is_fully_resolved as _is_fully_resolved_short


def test_outcome_engine_has_nine_horizons_including_short_ones():
    assert HORIZONS == [1, 2, 3, 4, 5, 10, 20, 40, 60]
    assert SHORT_NOISY_HORIZONS == [1, 2, 3, 4]


def test_every_horizon_has_a_calendar_buffer_entry():
    assert set(_HORIZON_CALENDAR_BUFFER.keys()) == set(HORIZONS)


def test_short_horizon_buffers_are_proportionally_generous():
    # A 1-trading-day buffer must be smaller than a 5-day one, but still
    # comfortably cover a weekend + a possible holiday.
    assert _HORIZON_CALENDAR_BUFFER[1] < _HORIZON_CALENDAR_BUFFER[5]
    assert _HORIZON_CALENDAR_BUFFER[1] >= 3  # at least covers a Fri->Mon gap


def test_baseline_horizons_derive_from_outcome_engine_not_a_separate_copy():
    assert BASELINE_HORIZONS == [f"{h}D" for h in HORIZONS]
    assert "1D" in BASELINE_HORIZONS and "60D" in BASELINE_HORIZONS


def test_alpha_lab_horizons_derive_from_outcome_engine_not_a_separate_copy():
    assert alpha_lab.HORIZONS == [f"{h}D" for h in HORIZONS]


def test_alpha_lab_default_horizon_is_still_20d_after_list_grew():
    """The critical regression this stage was built to prevent: adding
    horizons to the FRONT of the list must not silently change the
    default selection away from 20D."""
    assert alpha_lab.HORIZONS[alpha_lab._DEFAULT_HORIZON_INDEX] == "20D"


def test_short_noisy_horizon_set_matches_source():
    assert alpha_lab._SHORT_NOISY_HORIZONS == {"1D", "2D", "3D", "4D"}
    assert "5D" not in alpha_lab._SHORT_NOISY_HORIZONS
    assert "20D" not in alpha_lab._SHORT_NOISY_HORIZONS


# ── "Fully resolved" threshold correctly grew from 5 to 9 ──────────────

def test_long_radar_fully_resolved_now_requires_all_nine_horizons():
    entry_with_only_old_five = {
        "outcomes": {f"{h}D": {"forward_return_%": 1.0} for h in [5, 10, 20, 40, 60]}
    }
    assert _is_fully_resolved_long(entry_with_only_old_five) is False  # missing 1D-4D now

    entry_with_all_nine = {
        "outcomes": {f"{h}D": {"forward_return_%": 1.0} for h in HORIZONS}
    }
    assert _is_fully_resolved_long(entry_with_all_nine) is True


def test_short_radar_fully_resolved_now_requires_all_nine_horizons():
    entry_with_only_old_five = {
        "outcomes": {f"{h}D": {"forward_return_%": 1.0} for h in [5, 10, 20, 40, 60]}
    }
    assert _is_fully_resolved_short(entry_with_only_old_five) is False

    entry_with_all_nine = {
        "outcomes": {f"{h}D": {"forward_return_%": 1.0} for h in HORIZONS}
    }
    assert _is_fully_resolved_short(entry_with_all_nine) is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
