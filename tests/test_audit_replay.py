import sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, pytest
from audit import replay as R

def make(n=330, seed=1):
    idx = pd.bdate_range("2025-07-01", periods=n)
    rng = np.random.default_rng(seed)
    # up, long decline, sharp late recovery (a CHDN-like shape); benchmark mildly rising
    c = np.concatenate([np.linspace(90, 110, 100), np.linspace(110, 74, 190), np.linspace(74, 95, n - 290)])
    c = c * (1 + rng.normal(0, 0.004, n))
    h = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": rng.integers(8e5, 2e6, n).astype(float)}, index=idx)
    b = pd.Series(4000 * np.cumprod(1 + rng.normal(0.0004, 0.005, n)), index=idx)
    return h, b

def oracle_first_open(h, t_end):
    """independent, loop-based: last unbroken run of close>SMA200 over the 365-day window ending at each date"""
    ok = {}
    for t in h.index:
        w = h[(h.index > t - pd.Timedelta(days=365)) & (h.index <= t)]["Close"].values
        ok[t] = len(w) >= 200 and w[-1] > w[-200:].mean()
    s = pd.Series(ok); s = s[s.index <= t_end]
    assert s.iloc[-1]
    f = s[~s]; return (s.index[0] if f.empty else s.index[s.index > f.index[-1]][0])

def test_replay_first_200day_date_matches_independent_oracle():
    h, b = make(); r = R.replay_ticker(h, b, dates=h.index[210:])
    got = R.first_pass_dates(r)["g1_above_200ma"]
    assert got == oracle_first_open(h, h.index[-1]).date().isoformat()

def test_replay_uses_only_data_up_to_each_date_no_lookahead():
    h, b = make(); t = h.index[260]
    full = R.replay_ticker(h, b, dates=[t]); trunc = R.replay_ticker(h[h.index <= t], b[b.index <= t], dates=[t])
    cols = ["close", "ma200", "perf_3m", "rs_3m", "vol_filter", "g1_above_200ma", "breakout", "days_since_200ma_cross"]
    pd.testing.assert_frame_equal(full[cols], trunc[cols])

def test_unavailable_benchmark_is_NOT_silently_zero_and_the_day_does_not_pass():
    h, b = make(); r = R.replay_ticker(h, b.iloc[0:0], dates=[h.index[-1]], calendar=h.index)
    assert pd.isna(r.rs_3m.iloc[0]) and r.bench_us_used.iloc[0] == "UNAVAILABLE"
    assert r.passes_replayed_gates.iloc[0] is None or pd.isna(r.passes_replayed_gates.iloc[0])

def test_scanner_silent_failure_can_be_reproduced_only_when_explicitly_requested_and_is_labelled():
    h, b = make(); r = R.replay_ticker(h, b.iloc[0:0], dates=[h.index[-1]], calendar=h.index, emulate_silent_failure=True)
    assert r.rs_3m.iloc[0] == 0.0 and bool(r.g3_rs_ok.iloc[0]) and "silently" in r.rs_note.iloc[0]

def test_volume_gate_uses_max_of_today_and_20d_average():
    h, b = make(); h = h.copy(); h["Volume"] = 50_000.0; h.iloc[-1, h.columns.get_loc("Volume")] = 400_000.0
    r = R.replay_ticker(h, b, dates=[h.index[-1]]); assert r.vol_filter.iloc[0] == 400_000 and bool(r.g4_volume_ok.iloc[0])

def test_short_history_is_not_eligible_not_zero_filled():
    h, b = make(); r = R.replay_ticker(h.iloc[:60], b, dates=[h.index[59]])
    assert not bool(r.eligible_history.iloc[0]) and "close" not in r.columns or pd.isna(r.get("close", pd.Series([np.nan])).iloc[0])

def test_under_200_bars_cannot_pass_the_200_day_gate():
    h, b = make(); r = R.replay_ticker(h.iloc[:150], b, dates=[h.index[149]])
    assert r.eligible_history.iloc[0] and not bool(r.g1_above_200ma.iloc[0])

def test_replay_uses_the_repos_own_scanner_functions():
    import scanner
    assert R.S is scanner and R.S.compute_rs is scanner.compute_rs and R.S.detect_base_breakout is scanner.detect_base_breakout


def test_volume_gate_passes_on_a_quiet_day_when_the_20d_average_is_high():
    h, b = make(); h = h.copy(); h["Volume"] = 500_000.0; h.iloc[-1, h.columns.get_loc("Volume")] = 50_000.0
    r = R.replay_ticker(h, b, dates=[h.index[-1]])
    assert r.vol_filter.iloc[0] >= 400_000 and bool(r.g4_volume_ok.iloc[0])


# ---------- Amendment 2: benchmark roles, same primary/fallback rule as ApexScan ----------
def test_primary_benchmark_is_used_and_recorded_when_it_has_enough_bars():
    h, b = make(); r = R.replay_ticker(h, b, dates=[h.index[-1]], calendar=h.index,
                                       bench_r2500=("^R25I", b, "SMMD", b * 0 + 1))
    assert r.bench_us_used.iloc[0] == "^GSPC" and r.bench_r2500_used.iloc[0] == "^R25I"

def test_fallback_is_used_only_when_primary_has_50_or_fewer_bars_and_is_labelled():
    h, b = make(); short = b.iloc[-50:]
    r = R.replay_ticker(h, b, dates=[h.index[-1]], calendar=h.index, bench_r2500=("^R25I", short, "SMMD", b))
    assert r.bench_r2500_used.iloc[0] == "SMMD" and "fallback" in r.bench_r2500_note.iloc[0]

def test_no_third_symbol_is_ever_substituted_when_both_are_short():
    h, b = make(); short = b.iloc[-30:]
    r = R.replay_ticker(h, b, dates=[h.index[-1]], calendar=h.index, bench_r3000g=("^RAG", short, "IWZ", short))
    assert r.bench_r3000g_used.iloc[0] == "UNAVAILABLE"

def test_gspc_has_no_fallback_a_short_gspc_is_unavailable_not_replaced():
    h, b = make(); r = R.replay_ticker(h, {"^GSPC": b.iloc[-30:], "SPY": b}, dates=[h.index[-1]], calendar=h.index)
    assert r.bench_us_used.iloc[0] == "UNAVAILABLE"

# ---------- Amendment 4: missing data is never filled; affected days are excluded with a reason ----------
def _row(h, b, **kw): return R.replay_ticker(h, b, dates=[h.index[-1]], calendar=b.index, **kw).iloc[0]

def test_missing_close_in_the_200_bar_lookback_excludes_the_day_with_a_reason():
    h, b = make(); h = h.copy(); h.iloc[-100, h.columns.get_loc("Close")] = np.nan
    r = _row(h, b); assert not r.evaluable and r.exclusion_reason == "missing_close_in_lookback" and pd.isna(r.get("close"))

def test_an_absent_session_in_the_lookback_excludes_the_day():
    h, b = make(); h = h.drop(h.index[-80])
    r = _row(h, b); assert not r.evaluable and r.exclusion_reason == "missing_session_in_lookback"

def test_missing_volume_in_the_last_20_bars_excludes_the_day():
    h, b = make(); h = h.copy(); h.iloc[-5, h.columns.get_loc("Volume")] = np.nan
    assert _row(h, b).exclusion_reason == "missing_volume_in_lookback"

def test_missing_high_or_low_only_blocks_the_breakout_signal_not_the_E1_gates():
    h, b = make(); h = h.copy(); h.iloc[-10, h.columns.get_loc("High")] = np.nan
    r = _row(h, b); assert r.evaluable and r.breakout is None and "not evaluable" in r.pattern and pd.notna(r.close)

def test_benchmark_session_missing_in_the_stocks_last_64_sessions_makes_the_day_not_evaluable():
    h, b = make(); b2 = b.drop(b.index[-30])
    r = R.replay_ticker(h, b2, dates=[h.index[-1]], calendar=h.index).iloc[0]
    assert not r.evaluable and r.exclusion_reason == "benchmark_missing_sessions"

def test_replay_does_not_modify_or_fill_its_inputs():
    h, b = make(); h = h.copy(); h.iloc[-3, h.columns.get_loc("Volume")] = np.nan; before = h.copy()
    _row(h, b); pd.testing.assert_frame_equal(h, before)


# ── continuity of passing runs (4 Nov 2026): fail / unevaluable / missing sessions break a run, nothing is bridged ──────────────
from audit import cohorts as _C

_GATES = ["g1_above_200ma", "g2_perf3m_ok", "g3_rs_ok", "g4_volume_ok", "passes_replayed_gates"]


def _grid(spec, start="2026-10-05", elig=None):
    """spec: one of 'P' pass, 'F' fail, 'U' unevaluable per session (5 Oct 2026 is a Monday)."""
    idx = pd.bdate_range(start, periods=len(spec)); rows = []
    for i, ch in enumerate(spec):
        e = True if elig is None else elig[i]
        row = {"eligible_history": e, "evaluable": e and ch != "U"}
        for g in _GATES: row[g] = {"P": True, "F": False}.get(ch) if row["evaluable"] else None
        rows.append(row)
    return pd.DataFrame(rows, index=idx)


def test_monday_pass_tuesday_unevaluable_wednesday_thursday_pass_is_not_one_run():
    r = _grid("PUPP")
    for g in _GATES:
        d = R.first_pass_detail(r)[g]
        assert d["date"] == "2026-10-07" and d["status"] == "start_uncertain_unevaluable_gap" and "may be earlier" in d["note"]
    assert R.first_pass_dates(r)["g1_above_200ma"] == "2026-10-07" != "2026-10-05"


def test_monday_pass_missing_tuesday_row_wednesday_thursday_pass_is_not_one_run_with_a_calendar():
    r = _grid("PPPP").drop(pd.Timestamp("2026-10-06")); cal = pd.bdate_range("2026-10-05", periods=4)
    d = R.first_pass_detail(r, calendar=cal)["passes_replayed_gates"]
    assert d["date"] == "2026-10-07" and d["status"] == "start_uncertain_unevaluable_gap"
    assert R.first_pass_dates(r, calendar=cal)["g1_above_200ma"] == "2026-10-07"
    assert R.first_pass_dates(r)["g1_above_200ma"] == "2026-10-05"           # without a calendar a missing row cannot be seen: documented


def test_the_last_session_not_evaluable_gives_no_date():
    r = _grid("PPU"); d = R.first_pass_detail(r)["g1_above_200ma"]
    assert d["date"] is None and d["status"] == "recorded_date_not_evaluable" and R.first_pass_dates(r)["g1_above_200ma"] == "last session not evaluable"


def test_a_failed_session_before_a_passing_run_makes_the_start_verified():
    r = _grid("UPFPPP"); d = R.first_pass_detail(r)["g1_above_200ma"]
    assert d["date"] == "2026-10-08" and d["status"] == "ok" and d["note"] == ""


def test_a_normal_uninterrupted_run_after_a_failure_and_a_final_failure():
    r = _grid("FFPPPP"); assert R.first_pass_detail(r)["g2_perf3m_ok"]["date"] == "2026-10-07" and R.first_pass_detail(r)["g2_perf3m_ok"]["status"] == "ok"
    r = _grid("PPPF"); d = R.first_pass_detail(r)["g1_above_200ma"]
    assert d["date"] is None and d["status"] == "gates_not_met_on_recorded_date" and R.first_pass_dates(r)["g1_above_200ma"] == "not passing on last date"


def test_left_censoring_when_nothing_earlier_was_evaluated_or_history_is_too_short():
    r = _grid("PPPP"); d = R.first_pass_detail(r)["g1_above_200ma"]
    assert d["status"] == "left_censored" and d["date"] == "2026-10-05" and d["note"] == _C.LEFT_CENSORED_NOTE
    r = _grid("UUPP", elig=[False, False, True, True]); d = R.first_pass_detail(r)["g1_above_200ma"]       # too little history at first
    assert d["status"] == "left_censored" and d["date"] == "2026-10-07"
    assert R.first_pass_detail(_grid("UU", elig=[False, False]))["g1_above_200ma"]["status"] == "no_evaluable_data"


def test_each_gate_is_judged_separately_and_since_trims_the_start():
    r = _grid("PPPPPP"); r.loc[r.index[2], "g3_rs_ok"] = False; r.loc[r.index[2], "passes_replayed_gates"] = False
    d = R.first_pass_detail(r)
    assert d["g1_above_200ma"]["date"] == "2026-10-05" and d["g3_rs_ok"]["date"] == "2026-10-08" and d["passes_replayed_gates"]["date"] == "2026-10-08"
    assert R.first_pass_detail(r, since="2026-10-07")["g1_above_200ma"]["date"] == "2026-10-07"


def test_replay_and_cohorts_cannot_disagree_on_the_same_table():
    for spec in ("PUPP", "FPPP", "PPPP", "UFPP", "PFUP", "PPFP"):
        r = _grid(spec)
        a = R.first_pass_detail(r)["passes_replayed_gates"]; b = _C.four_gate_lower_bound(r, r.index[-1])
        assert (a["date"], a["status"]) == (b["reconstructed_four_gate_date"], b["status"]), spec


def test_a_calendar_that_starts_before_the_table_does_not_create_false_breaks():
    r = _grid("PPPP"); cal = pd.bdate_range("2026-10-01", "2026-10-08")                 # calendar starts before the table, ends with it
    d = R.first_pass_detail(r, since="2026-10-05", calendar=cal)["g1_above_200ma"]
    assert d["date"] == "2026-10-05" and d["status"] == "left_censored"


def test_a_session_that_is_evaluable_but_has_no_gate_value_is_not_a_fail_and_not_a_pass():
    r = _grid("PPPP").astype({"g1_above_200ma": object, "passes_replayed_gates": object})
    r.loc[r.index[1], ["g1_above_200ma", "passes_replayed_gates"]] = [None, None]                  # evaluable day, value unavailable
    assert bool(r.loc[r.index[1], "evaluable"])
    d = R.first_pass_detail(r)["g1_above_200ma"]
    assert d["date"] == "2026-10-07" and d["status"] == "start_uncertain_unevaluable_gap"      # a gap, not a verified failure
