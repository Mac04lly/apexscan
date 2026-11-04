"""Amendment 3: cohort split, three date concepts, dependence counts, independent-trade counting, month description."""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, pytest
from audit import cohorts as C, maturity as M

CAL = M.sessions_2026()


def test_cohort_rule_is_by_recorded_date_only():
    assert C.cohort_of("2026-08-06") == C.A1 and C.cohort_of("2026-08-06T10:51:00") == C.A1
    assert C.cohort_of("2026-08-07") == C.A2 and C.cohort_of("2026-08-05") == C.A2


def test_the_real_log_splits_605_and_290_and_nothing_is_pooled():
    disc = json.loads((REPO / "data" / "discoveries.json").read_text(encoding="utf-8"))
    t = C.population_a_tables(M.table(disc, pd.Timestamp("2026-10-30"), CAL), CAL, 63)
    assert list(t.index) == [C.A1, C.A2] and t.loc[C.A1, "records"] == len(disc) - t.loc[C.A2, "records"]
    assert t.loc[C.A1, "records"] == 605 and t.loc[C.A1, "distinct_recorded_dates"] == 1 and t.loc[C.A1, "distinct_entry_dates"] == 1
    assert t.loc[C.A1, "largest_single_date_share"] == 1.0 and t.loc[C.A1, "max_simultaneous_open_windows"] == 605
    assert t.loc[C.A2, "records"] == 290 and t.loc[C.A2, "distinct_recorded_dates"] == 40 and t.loc[C.A1, "matured_63"] == 0


def _tr(rows):
    return pd.DataFrame(rows, columns=["ticker", "entry_pos"])


def test_independent_trades_boundary_is_strictly_after_the_exit_session():
    H = 10
    assert len(C.independent_trades(_tr([("A", 0), ("A", 10)]), H)) == 1      # entry on the exit session itself: still overlapping
    assert len(C.independent_trades(_tr([("A", 0), ("A", 11)]), H)) == 2      # first session after exit: independent
    assert len(C.independent_trades(_tr([("A", 0), ("A", 5), ("A", 11)]), H)) == 2   # 5 skipped; 11 is after the exit of the trade at 0
    assert len(C.independent_trades(_tr([("A", 0), ("A", 5), ("A", 10), ("A", 21)]), H)) == 2   # 10 skipped (exit of 0 is 10); 21 > 10


def test_independent_trades_are_counted_per_ticker_never_across_tickers():
    assert len(C.independent_trades(_tr([("A", 3), ("B", 3), ("C", 3)]), 63)) == 3
    assert len(C.independent_trades(_tr([("B", 11), ("A", 0), ("B", 0), ("A", 5)]), 10)) == 3    # unsorted input; A: 0 kept 5 skipped, B: 0 and 11 kept


def test_a_later_overlapping_signal_does_not_extend_the_window():
    # trade at 0 exits at 10; the skipped signal at 8 must not push the window to 18, so the one at 11 is accepted
    assert len(C.independent_trades(_tr([("A", 0), ("A", 8), ("A", 11)]), 10)) == 2


def test_simultaneous_windows_and_overlap_pairs_by_hand():
    w = [(0, 10), (5, 15), (11, 20), (30, 40)]
    assert C.max_simultaneous(w) == 2 and C.overlapping_pairs(w) == 2          # (0,10)-(5,15) and (5,15)-(11,20)
    assert C.max_simultaneous([(0, 10), (10, 20)]) == 2 and C.overlapping_pairs([(0, 10), (11, 20)]) == 0


def test_dependence_table_counts_each_notion_separately():
    df = pd.DataFrame({"ticker": ["A", "B", "C", "A"], "discovered_at": ["d1", "d1", "d2", "d3"], "entry_pos": [1, 1, 2, 30]})
    t = C.dependence_table(df, 10)
    assert t["records"] == 4 and t["unique_tickers"] == 3 and t["distinct_recorded_dates"] == 3 and t["distinct_entry_dates"] == 3
    assert t["largest_single_date_share"] == 0.5 and t["max_simultaneous_open_windows"] == 3 and t["non_overlapping_trades"] == 4
    assert C.dependence_table(df.iloc[0:0], 10)["records"] == 0


def _replay(passes, start="2026-01-02", unev=(), elig=None):
    """passes: list of bool. unev: positions that are NOT evaluable (they carry no pass/fail)."""
    idx = pd.bdate_range(start, periods=len(passes))
    ev = [i not in set(unev) for i in range(len(passes))]
    return pd.DataFrame({"eligible_history": True if elig is None else elig, "evaluable": ev,
                         "passes_replayed_gates": [bool(p) and e for p, e in zip(passes, ev)]}, index=idx)


def test_four_gate_date_is_the_start_of_the_unbroken_run_containing_the_recorded_date():
    r = _replay([False, True, True, False, False, True, True, True, True])
    rec = r.index[7]
    o = C.four_gate_lower_bound(r, rec)
    assert o["reconstructed_four_gate_date"] == r.index[5].date().isoformat() and o["status"] == "ok"
    assert o["full_scanner_qualification_date"] == "not established" and o["recorded_discovery_date"] == rec.date().isoformat()


def test_an_unevaluable_session_cannot_be_bridged_into_one_continuous_run():
    # Monday pass, Tuesday NOT evaluable, Wednesday pass, Thursday pass: not one run from Monday
    r = _replay([True, True, True, True], start="2026-10-05", unev=(1,))          # 5 Oct 2026 is a Monday
    o = C.four_gate_lower_bound(r, r.index[3])
    assert o["reconstructed_four_gate_date"] == "2026-10-07" and o["reconstructed_four_gate_date"] != "2026-10-05"   # Wednesday, never Monday
    assert o["status"] == "start_uncertain_unevaluable_gap" and "may be earlier" in o["note"] and "verified unbroken run" in o["note"]
    assert o["status"] != "ok"


def test_a_session_with_no_replay_row_at_all_is_also_a_break_when_the_calendar_is_given():
    r = _replay([True, True, True, True], start="2026-10-05"); r = r.drop(r.index[1])         # Tuesday row absent
    cal = pd.bdate_range("2026-10-05", periods=4)
    assert C.four_gate_lower_bound(r, cal[3], calendar=cal)["reconstructed_four_gate_date"] == "2026-10-07"
    assert C.four_gate_lower_bound(r, cal[3], calendar=cal)["status"] == "start_uncertain_unevaluable_gap"


def test_the_recorded_date_itself_being_unevaluable_assigns_no_date():
    r = _replay([True, True, True], unev=(2,)); o = C.four_gate_lower_bound(r, r.index[2])
    assert o["status"] == "recorded_date_not_evaluable" and o["reconstructed_four_gate_date"] is None


def test_a_failure_before_the_run_gives_a_known_start_even_with_unevaluable_sessions_further_back():
    r = _replay([True, True, False, True, True], unev=(0,)); o = C.four_gate_lower_bound(r, r.index[4])
    assert o["status"] == "ok" and o["reconstructed_four_gate_date"] == r.index[3].date().isoformat()


def test_left_censoring_means_start_of_evaluable_history_not_start_of_downloaded_prices():
    # first 4 sessions have too little history (not eligible); the gates pass from the 5th session on
    elig = [False] * 4 + [True] * 4; r = _replay([True] * 8, elig=elig)
    o = C.four_gate_lower_bound(r, r.index[7])
    assert o["status"] == "left_censored" and o["reconstructed_four_gate_date"] == r.index[4].date().isoformat()
    assert o["first_replayed_date"] == r.index[0].date().isoformat() and o["first_evaluable_date"] == r.index[4].date().isoformat()
    assert o["first_evaluable_date"] != o["first_replayed_date"]                       # the two beginnings are different things
    assert o["note"] == "Left-censored at the beginning of available evaluable history; the true qualification start may be earlier."


def test_run_that_starts_on_the_first_evaluated_session_with_nothing_earlier_is_left_censored():
    r = _replay([True, True, True]); o = C.four_gate_lower_bound(r, r.index[2])
    assert o["status"] == "left_censored" and o["reconstructed_four_gate_date"] == r.index[0].date().isoformat()


def test_four_gate_run_ignores_later_data_and_flags_failure_and_missing_data():
    r = _replay([True, True, True, False, True])
    assert C.four_gate_lower_bound(r, r.index[2])["reconstructed_four_gate_date"] == r.index[0].date().isoformat()   # later failure is not used
    assert C.four_gate_lower_bound(r, r.index[3])["status"] == "gates_not_met_on_recorded_date"
    assert C.four_gate_lower_bound(r, r.index[3])["reconstructed_four_gate_date"] is None
    assert C.four_gate_lower_bound(r, "2025-01-01")["status"] == "no_evaluable_data"
    assert C.four_gate_lower_bound(r.iloc[0:0], "2026-02-01")["status"] == "no_evaluable_data"
    assert C.four_gate_lower_bound(_replay([True, True], unev=(0, 1)), "2026-01-30")["status"] == "no_evaluable_data"


def test_monthly_description_groups_by_entry_month_and_changes_nothing():
    t = pd.DataFrame({"entry": ["2026-08-07", "2026-08-20", "2026-09-01"], "net": [1.0, 3.0, -2.0]}); before = t.copy()
    m = C.monthly_descriptive(t, "entry", "net")
    assert list(m.month) == ["2026-08", "2026-09"] and list(m.trades) == [2, 1] and list(m["mean"]) == [2.0, -2.0] and list(m["median"]) == [2.0, -2.0]
    pd.testing.assert_frame_equal(t, before)


def test_the_module_never_describes_a_lower_bound_as_a_discovery_date():
    assert "lower bound" in C.LOWER_BOUND_NOTE and "not an ApexScan discovery date" in C.LOWER_BOUND_NOTE
    assert "cannot be established" in C.LIMITATION and C.A1.startswith("A1 - Initial persistence/seeding") and C.A2.startswith("A2 - Post-seeding first-recorded")


def test_the_window_of_an_accepted_trade_runs_a_full_horizon_from_its_own_entry():
    assert len(C.independent_trades(_tr([("A", 0), ("A", 11), ("A", 15)]), 10)) == 2    # 15 falls inside the window of the trade at 11 (ends 21)


def test_windows_sharing_exactly_one_session_count_as_an_overlapping_pair():
    assert C.overlapping_pairs([(0, 10), (10, 20)]) == 1 and C.overlapping_pairs([(0, 10), (11, 20)]) == 0
