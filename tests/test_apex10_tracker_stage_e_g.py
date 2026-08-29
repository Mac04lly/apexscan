"""
tests/test_apex10_tracker_stage_e_g.py — Apex the Great X: Stage E + G tests

Covers create-vs-update logic, no-duplicate-entries, append-only
score_history across simulated days, the immutable timestamp/entry_price
anchors, breakout-status transition, and — the core Stage G claim —
that a radar entry has exactly the shape modules.outcome_engine needs
to freeze its forward returns with zero new code.
"""
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.apex10_tracker import (
    create_or_update_radar_entry, run_daily_radar_update, get_radar_table,
    _find_active_radar_entry, RADAR_OBSERVATION_TYPE,
)
from modules.outcome_engine import compute_outcomes_for_observation, HORIZONS
from modules.apex10_baseline import get_observation_type


def _minimal_features():
    return {
        "breakout_proximity": {"resistance_price": 105.0, "distance_to_resistance_pct": 2.0},
        "relative_strength": {"rs_current": 120.0, "rs_5d_change": 5.0},
        "structure": {}, "volatility": {}, "volume": {}, "moving_averages": {},
        "sector_confirmation": {}, "market_regime": {},
    }


def _score(value=65.0, quality="MEDIUM"):
    return {"score": value, "evidence_quality": quality, "components": {}}


# ── Creation / duplicate prevention ──────────────────────────────────

def test_first_sighting_creates_new_entry():
    obs = []
    entry = create_or_update_radar_entry("XYZ", 100.0, _minimal_features(), _score(),
                                         observations=obs, persist=False)
    assert len(obs) == 1
    assert entry["observation_type"] == RADAR_OBSERVATION_TYPE
    assert entry["ticker"] == "XYZ"
    assert entry["first_radar_price"] == 100.0
    assert entry["breakout_status"] == "PRE_BREAKOUT"


def test_second_sighting_same_day_updates_not_duplicates():
    obs = []
    as_of = datetime(2026, 1, 5, 9, 30)
    create_or_update_radar_entry("XYZ", 100.0, _minimal_features(), _score(60.0),
                                 observations=obs, persist=False, as_of=as_of)
    create_or_update_radar_entry("XYZ", 101.0, _minimal_features(), _score(62.0),
                                 observations=obs, persist=False, as_of=as_of)
    assert len(obs) == 1, "must never create a second entry for the same active ticker"
    assert obs[0]["current_price"] == 101.0
    assert obs[0]["current_score"] == 62.0


def test_different_tickers_create_separate_entries():
    obs = []
    create_or_update_radar_entry("AAA", 50.0, _minimal_features(), _score(), observations=obs, persist=False)
    create_or_update_radar_entry("BBB", 75.0, _minimal_features(), _score(), observations=obs, persist=False)
    assert len(obs) == 2


# ── Append-only score_history across simulated days ──────────────────

def test_score_history_appends_across_multiple_days_never_edits_past_entries():
    obs = []
    day1 = datetime(2026, 1, 5)
    day5 = datetime(2026, 1, 9)
    day18 = datetime(2026, 1, 22)

    create_or_update_radar_entry("XYZ", 100.0, _minimal_features(), _score(61.0),
                                 observations=obs, persist=False, as_of=day1)
    create_or_update_radar_entry("XYZ", 103.0, _minimal_features(), _score(68.0),
                                 observations=obs, persist=False, as_of=day5)
    entry = create_or_update_radar_entry("XYZ", 110.0, _minimal_features(), _score(92.0),
                                         observations=obs, persist=False, as_of=day18)

    assert len(obs) == 1  # still one entry, not three
    history = entry["score_history"]
    assert len(history) == 3
    assert [h["score"] for h in history] == [61.0, 68.0, 92.0]
    # Earlier entries must be byte-for-byte unchanged after later updates.
    assert history[0] == {"date": "2026-01-05", "score": 61.0, "state": "WATCH", "evidence_quality": "MEDIUM"}
    assert history[1] == {"date": "2026-01-09", "score": 68.0, "state": "WATCH", "evidence_quality": "MEDIUM"}


def test_same_calendar_day_does_not_append_twice():
    obs = []
    morning = datetime(2026, 1, 5, 9, 30)
    afternoon = datetime(2026, 1, 5, 15, 45)
    create_or_update_radar_entry("XYZ", 100.0, _minimal_features(), _score(60.0),
                                 observations=obs, persist=False, as_of=morning)
    entry = create_or_update_radar_entry("XYZ", 102.0, _minimal_features(), _score(63.0),
                                         observations=obs, persist=False, as_of=afternoon)
    assert len(entry["score_history"]) == 1  # same calendar day -> one entry, latest values win
    assert entry["score_history"][0]["score"] == 63.0


# ── Immutable anchors ─────────────────────────────────────────────────

def test_timestamp_and_entry_price_never_change_after_creation():
    obs = []
    day1 = datetime(2026, 1, 5)
    day10 = datetime(2026, 1, 15)
    entry1 = create_or_update_radar_entry("XYZ", 100.0, _minimal_features(), _score(),
                                          observations=obs, persist=False, as_of=day1)
    original_timestamp, original_entry_price = entry1["timestamp"], entry1["entry_price"]

    entry2 = create_or_update_radar_entry("XYZ", 150.0, _minimal_features(), _score(90.0),
                                          observations=obs, persist=False, as_of=day10)
    assert entry2["timestamp"] == original_timestamp
    assert entry2["entry_price"] == original_entry_price == 100.0
    assert entry2["current_price"] == 150.0  # current_price DOES change — that's expected


# ── Breakout status transition ────────────────────────────────────────

def test_breakout_status_transitions_once_and_stays():
    obs = []
    day1 = datetime(2026, 1, 5)
    day10 = datetime(2026, 1, 15)
    day15 = datetime(2026, 1, 20)

    create_or_update_radar_entry("XYZ", 100.0, _minimal_features(), _score(70.0),
                                 observations=obs, persist=False, as_of=day1)
    entry = create_or_update_radar_entry("XYZ", 106.0, _minimal_features(), _score(93.0),
                                         observations=obs, persist=False, as_of=day10,
                                         trigger_gates={"confirmed_breakout": True})
    assert entry["breakout_status"] == "CONFIRMED_BREAKOUT"
    assert entry["breakout_date"] == "2026-01-15"
    assert entry["breakout_price"] == 106.0

    # A later update, even without confirmed_breakout again, must not revert the status.
    entry2 = create_or_update_radar_entry("XYZ", 108.0, _minimal_features(), _score(91.0),
                                          observations=obs, persist=False, as_of=day15,
                                          trigger_gates={"confirmed_breakout": False})
    assert entry2["breakout_status"] == "CONFIRMED_BREAKOUT"
    assert entry2["breakout_date"] == "2026-01-15"  # unchanged, not overwritten


# ── Fully-resolved cycles start a fresh entry, not a reopened one ────

def test_fully_resolved_entry_is_not_reused_new_cycle_created():
    obs = []
    day1 = datetime(2026, 1, 5)
    entry = create_or_update_radar_entry("XYZ", 100.0, _minimal_features(), _score(),
                                         observations=obs, persist=False, as_of=day1)
    # Simulate full resolution — all 5 horizons frozen, exactly like the outcome engine would do.
    entry["outcomes"] = {f"{h}D": {"forward_return_%": 1.0} for h in HORIZONS}

    later = datetime(2026, 6, 1)
    create_or_update_radar_entry("XYZ", 130.0, _minimal_features(), _score(75.0),
                                 observations=obs, persist=False, as_of=later)
    assert len(obs) == 2  # a genuinely new cycle, old one left untouched
    assert obs[0]["outcomes"] != {}  # old entry's frozen outcomes preserved exactly
    assert obs[1]["first_radar_price"] == 130.0  # new cycle has its own fresh anchor


# ── Batch update ──────────────────────────────────────────────────────

def test_run_daily_radar_update_batches_create_and_update_counts():
    obs = []
    day1 = datetime(2026, 1, 5)
    create_or_update_radar_entry("EXIST", 50.0, _minimal_features(), _score(),
                                 observations=obs, persist=False, as_of=day1)

    candidates = [
        {"ticker": "EXIST", "current_price": 52.0, "features": _minimal_features(), "score_result": _score(70.0)},
        {"ticker": "NEWONE", "current_price": 20.0, "features": _minimal_features(), "score_result": _score(65.0)},
    ]
    result = run_daily_radar_update(candidates, observations=obs)
    assert result["created"] == 1
    assert result["updated"] == 1
    assert len(obs) == 2


# ── get_radar_table ────────────────────────────────────────────────────

def test_get_radar_table_sorted_by_score_descending():
    obs = []
    create_or_update_radar_entry("LOW", 10.0, _minimal_features(), _score(40.0), observations=obs, persist=False)
    create_or_update_radar_entry("HIGH", 20.0, _minimal_features(), _score(95.0), observations=obs, persist=False)
    table = get_radar_table(obs)
    assert [row["ticker"] for row in table] == ["HIGH", "LOW"]


def test_get_radar_table_excludes_discovery_type_rows():
    obs = [{"ticker": "DISC", "observation_type": "discovery", "current_score": 100}]
    create_or_update_radar_entry("RADAR", 10.0, _minimal_features(), _score(50.0), observations=obs, persist=False)
    table = get_radar_table(obs)
    assert len(table) == 1
    assert table[0]["ticker"] == "RADAR"


# ── Stage G: outcome-engine compatibility ─────────────────────────────

def test_radar_entry_shape_is_directly_compatible_with_outcome_engine():
    """The core Stage G claim: a radar entry needs NO adaptation to be
    processed by the existing, unmodified outcome engine — it already
    has entry_price, timestamp, and an outcomes dict in exactly the
    shape compute_outcomes_for_observation() expects."""
    obs = []
    entry = create_or_update_radar_entry("XYZ", 100.0, _minimal_features(), _score(),
                                         observations=obs, persist=False)
    assert "entry_price" in entry and entry["entry_price"] is not None
    assert "timestamp" in entry and entry["timestamp"]
    assert "outcomes" in entry and isinstance(entry["outcomes"], dict)
    # Calling the real outcome engine function must not raise, regardless
    # of whether it can actually fetch data for a fake ticker in this
    # environment — the point is the SHAPE is accepted, not the network call.
    try:
        compute_outcomes_for_observation(entry, bench_cache={})
    except Exception as e:
        assert False, f"outcome engine raised on a well-formed radar entry: {e}"


def test_get_observation_type_recognizes_radar_rows():
    obs_row = {"ticker": "XYZ", "observation_type": "apex10_radar"}
    assert get_observation_type(obs_row) == "apex10_radar"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
