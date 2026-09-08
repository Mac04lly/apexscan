"""
tests/test_short_horizons.py — 19-horizon expansion

Verifies the fixed, pre-registered horizon list is correctly wired into
the single source of truth (modules/outcome_engine.py) and correctly
derived everywhere else, rather than re-duplicated — and that the
"fully resolved" threshold used by both the long and short radar
trackers correctly grew to require all 19 horizons.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.outcome_engine import HORIZONS, SHORT_NOISY_HORIZONS, _HORIZON_CALENDAR_BUFFER
from modules.apex10_baseline import BASELINE_HORIZONS
import ui.alpha_lab as alpha_lab
from modules.apex10_tracker import _is_fully_resolved as _is_fully_resolved_long

# modules/apex10_short_tracker.py is a separate, earlier deliverable —
# these tests should still run (and everything above should still be
# verifiable) even for a repo that hasn't applied it yet.
try:
    from modules.apex10_short_tracker import _is_fully_resolved as _is_fully_resolved_short
    _HAS_SHORT_TRACKER = True
except ImportError:
    _HAS_SHORT_TRACKER = False

EXPECTED_HORIZONS = [1, 3, 5, 7, 10, 13, 15, 17, 20, 23, 27, 30, 35, 40, 45, 50, 53, 57, 60]


def test_outcome_engine_has_the_full_19_horizon_list():
    assert HORIZONS == EXPECTED_HORIZONS
    assert len(HORIZONS) == 19


def test_short_noisy_horizons_are_only_those_below_five():
    # 5D itself has always been treated as a legitimate first real
    # horizon in this project, not "noisy" — only 1D and 3D sit below it
    # now that 2D/4D no longer exist in the list.
    assert SHORT_NOISY_HORIZONS == [1, 3]


def test_every_horizon_has_a_calendar_buffer_entry():
    assert set(_HORIZON_CALENDAR_BUFFER.keys()) == set(HORIZONS)


def test_calendar_buffers_are_monotonically_increasing_with_horizon():
    for i in range(len(HORIZONS) - 1):
        h_now, h_next = HORIZONS[i], HORIZONS[i + 1]
        assert _HORIZON_CALENDAR_BUFFER[h_now] < _HORIZON_CALENDAR_BUFFER[h_next]


def test_baseline_horizons_derive_from_outcome_engine_not_a_separate_copy():
    assert BASELINE_HORIZONS == [f"{h}D" for h in HORIZONS]
    assert len(BASELINE_HORIZONS) == 19


def test_alpha_lab_horizons_derive_from_outcome_engine_not_a_separate_copy():
    assert alpha_lab.HORIZONS == [f"{h}D" for h in HORIZONS]


def test_alpha_lab_default_horizon_is_still_20d_after_list_grew_again():
    """The exact regression this design was built to prevent, tested a
    second time now that the list changed shape again: the default
    selection must still land on 20D, at whatever index it now sits."""
    assert alpha_lab.HORIZONS[alpha_lab._DEFAULT_HORIZON_INDEX] == "20D"
    assert alpha_lab._DEFAULT_HORIZON_INDEX == 8


def test_short_noisy_horizon_set_matches_source():
    assert alpha_lab._SHORT_NOISY_HORIZONS == {"1D", "3D"}
    assert "5D" not in alpha_lab._SHORT_NOISY_HORIZONS
    assert "7D" not in alpha_lab._SHORT_NOISY_HORIZONS


# ── "Fully resolved" threshold correctly grew to 19 ─────────────────────

def test_long_radar_fully_resolved_now_requires_all_nineteen_horizons():
    entry_with_only_old_nine = {
        "outcomes": {f"{h}D": {"forward_return_%": 1.0} for h in [1, 2, 3, 4, 5, 10, 20, 40, 60]}
    }
    assert _is_fully_resolved_long(entry_with_only_old_nine) is False  # old set is no longer complete

    entry_with_all_nineteen = {
        "outcomes": {f"{h}D": {"forward_return_%": 1.0} for h in HORIZONS}
    }
    assert _is_fully_resolved_long(entry_with_all_nineteen) is True


def test_short_radar_fully_resolved_now_requires_all_nineteen_horizons():
    if not _HAS_SHORT_TRACKER:
        import pytest
        pytest.skip("modules/apex10_short_tracker.py not present in this repo yet — "
                   "a separate, earlier deliverable.")
    entry_with_only_old_nine = {
        "outcomes": {f"{h}D": {"forward_return_%": 1.0} for h in [1, 2, 3, 4, 5, 10, 20, 40, 60]}
    }
    assert _is_fully_resolved_short(entry_with_only_old_nine) is False

    entry_with_all_nineteen = {
        "outcomes": {f"{h}D": {"forward_return_%": 1.0} for h in HORIZONS}
    }
    assert _is_fully_resolved_short(entry_with_all_nineteen) is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
