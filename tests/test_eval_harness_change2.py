"""
tests/test_eval_harness_change2.py — Change 2 (evaluation harness)

Covers: cluster-robust inference, interval withholding with too few independent windows,
independence labelling, trading-day/holiday arithmetic, the matched control (same quintile,
same window, leave-one-out, no control -> no result), end-to-end recovery of an injected effect
and of a null, read-only behaviour, the harness's self-validation simulations (acceptance
criteria: false-pass <= 10% on pure noise, detection >= 80% at the stated effect), and the UI.
No network.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from modules import eval_harness as H


# ── inference ─────────────────────────────────────────────────────────────
def test_t_table_known_values_and_conservative_between_keys():
    assert H.t_crit_95(1) == 12.706 and H.t_crit_95(5) == 2.571 and H.t_crit_95(11) == 2.201
    assert H.t_crit_95(30) == 2.042 and H.t_crit_95(10_000) == 1.960
    assert H.t_crit_95(35) == 2.042          # between keys 30 and 40 -> the wider (lower-df) value
    assert H.t_crit_95(0) == float("inf")


def test_cluster_robust_mean_matches_hand_calculation():
    # 3 clusters of 2. Residual sums per cluster: -4, 0, +4 ; CR1 variance = (3/2)*32/36
    r = H.cluster_robust_mean([1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2])
    assert r["n"] == 6 and r["k"] == 3 and r["mean"] == pytest.approx(3.5)
    assert r["se"] == pytest.approx(np.sqrt(1.5 * 32 / 36))
    assert r["ci"] is None                    # 3 < MIN_CLUSTERS_FOR_CI


def test_interval_is_withheld_below_six_clusters_and_present_at_six():
    vals = list(range(1, 11))
    five = H.cluster_robust_mean(vals, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    six = H.cluster_robust_mean(vals + [11, 12], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    assert five["ci"] is None and five["label"] == "Insufficient independent windows"
    assert six["ci"] is not None and six["k"] == 6 and six["label"] == "Exploratory"
    lo, hi = six["ci"]
    assert lo < six["mean"] < hi
    assert (hi - six["mean"]) == pytest.approx(H.t_crit_95(5) * six["se"])


def test_clustered_interval_is_wider_than_naive_when_rows_are_clustered():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(8), 25)
    vals = rng.normal(0, 1, 200) + rng.normal(0, 3, 8)[cl]       # strong within-cluster correlation
    robust = H.cluster_robust_mean(vals, cl)
    naive = H.naive_mean_ci(vals)
    assert (robust["ci"][1] - robust["ci"][0]) > (naive[1] - naive[0]) * 1.5


def test_labels_follow_the_documented_cluster_thresholds():
    assert H.MIN_CLUSTERS_FOR_CI == 6 and H.ADEQUATE_CLUSTERS == 12
    assert H.independence_label(5) == "Insufficient independent windows"
    assert H.independence_label(6) == "Exploratory" and H.independence_label(11) == "Exploratory"
    assert H.independence_label(12) == "Adequate"


def test_empty_and_single_cluster_inputs_do_not_crash():
    assert H.cluster_robust_mean([], [])["mean"] is None
    r = H.cluster_robust_mean([1.0, 2.0, 3.0], [0, 0, 0])
    assert r["k"] == 1 and r["se"] is None and r["ci"] is None
    assert H.naive_mean_ci([1.0]) is None


# ── dates / windows ───────────────────────────────────────────────────────
def test_trading_days_skip_weekends_and_the_labor_day_holiday():
    # Fri 9/4 -> Tue 9/8 is ONE trading-day step because Mon 9/7 is Labor Day (it would be 2 if holidays were ignored)
    assert H.trading_days_between("2026-09-04", "2026-09-08") == 1
    assert int(np.busday_count(np.datetime64("2026-09-04"), np.datetime64("2026-09-08"))) == 2
    assert H.trading_days_between("2026-09-08", "2026-09-04") == -1
    assert H.trading_days_between("2026-10-19", "2026-10-20") == 1


def test_first_trading_day_rolls_a_sunday_stamp_to_monday():
    assert str(H.first_trading_day("2026-08-23")) == "2026-08-24"      # the backfill cohort's stamp
    assert str(H.first_trading_day("2026-09-07")) == "2026-09-08"      # holiday rolls forward


def test_window_clusters_blocks_of_h_trading_days():
    dates = ["2026-10-05", "2026-10-06", "2026-10-09", "2026-10-12", "2026-10-13", "2026-10-30"]
    assert H.window_clusters(dates, 5) == [0, 0, 0, 1, 1, 3]
    assert H.window_clusters([], 5) == []
    assert H.window_clusters(["2026-10-05"], 20) == [0]


def _obs(date, ret=None, h="20D"):
    o = {"observation_id": f"o{date}{ret}", "timestamp": date + " 11:00:00", "outcomes": {}}
    if ret is not None:
        o["outcomes"][h] = {"forward_return_%": ret}
    return o


def test_independence_summary_one_date_is_one_window_and_says_so():
    s = H.independence_summary([_obs("2026-08-23", -5.0) for _ in range(300)], 20, "20D")
    assert s["n_rows"] == 300 and s["n_dates"] == 1 and s["n_windows"] == 1
    assert s["label"] == "Insufficient independent windows" and s["largest_date_share_%"] == 100.0
    assert "not 300 independent tests" in s["note"]


def test_independence_summary_counts_windows_not_rows_and_honours_the_horizon_filter():
    dates = [str(np.busday_offset(np.datetime64("2026-08-24"), 20 * i, holidays=H.NYSE_HOLIDAYS))
             for i in range(12)]                      # exactly 20 trading days apart -> 12 separate windows
    obs = [_obs(d, -1.0) for d in dates] + [_obs("2026-08-24")]          # last row unresolved
    s = H.independence_summary(obs, 20, "20D")
    assert s["n_rows"] == 12 and s["n_windows"] == 12 and s["label"] == "Adequate"
    assert H.independence_summary(obs, 20)["n_rows"] == 13               # no filter -> all rows counted
    assert H.independence_summary([], 20)["n_windows"] == 0


def test_many_dates_inside_one_horizon_window_are_still_one_independent_window():
    # 15 consecutive trading days: 15 distinct signal DATES, but all inside one 20-day window
    # (their forward windows overlap almost completely), so only ONE independent 20D window.
    days = [str(np.busday_offset(np.datetime64("2026-09-08"), i, holidays=H.NYSE_HOLIDAYS)) for i in range(15)]
    obs = [_obs(d, -3.0) for d in days for _ in range(10)]
    s20 = H.independence_summary(obs, 20, "20D")
    assert s20["n_dates"] == 15 and s20["n_windows"] == 1 and s20["label"] == "Insufficient independent windows"
    s5 = H.independence_summary(obs, 5, "20D")                      # the same dates form 3 independent 5-day windows
    assert s5["n_dates"] == 15 and s5["n_windows"] == 3


# ── matched control (synthetic universe with an exactly known truth) ──────
GRID = [1, 3, 5, 7, 10, 13, 15, 17, 20, 23, 27, 30, 35]
DRIFT = {0: -0.8, 1: -0.4, 2: -0.2, 3: 0.0, 4: 0.3}          # %/trading day, by distance quintile


def _universe(n_per_bucket=24, cohort="2026-08-23", skip_outcomes_after=None):
    """Members' cumulative return is exactly DRIFT[bucket] * t (%), so any window's control is known."""
    uni = []
    for b, d in DRIFT.items():
        for j in range(n_per_bucket):
            out = {f"{h}D": {"forward_return_%": d * h} for h in GRID
                   if skip_outcomes_after is None or h <= skip_outcomes_after}
            uni.append({"observation_id": f"b{b}_{j}", "ticker": f"T{b}{j}", "observation_type": "discovery",
                        "timestamp": cohort + " 11:00:00",
                        "technical_features": {"vs_200ma_%": b * 10 + j * 0.1},
                        "outcomes": out})
    return uni


def _expected_control(b, a, h):
    d = DRIFT[b]
    return ((1 + d * (a + h) / 100) / (1 + d * a / 100) - 1) * 100


def test_control_equals_the_same_bucket_window_return_exactly():
    uni = _universe()
    ctl = H.build_cohort_controls(uni)
    for b in (0, 2, 4):
        got = H.control_window_return(ctl, "2026-08-23", f"b{b}_3", a=5, h=10)
        assert got is not None
        assert got[0] == pytest.approx(_expected_control(b, 5, 10), abs=1e-9)
        assert got[1] == 23                                             # own stock excluded (24 - 1)


def test_control_is_bucket_specific_and_window_specific():
    ctl = H.build_cohort_controls(_universe())
    lo = H.control_window_return(ctl, "2026-08-23", "b0_0", 5, 10)[0]
    hi = H.control_window_return(ctl, "2026-08-23", "b4_0", 5, 10)[0]
    assert lo < -5 and hi > 2                                           # drift -0.8 vs +0.3 over 10 days
    early = H.control_window_return(ctl, "2026-08-23", "b0_0", 1, 5)[0]
    late = H.control_window_return(ctl, "2026-08-23", "b0_0", 20, 5)[0]
    assert early == pytest.approx(_expected_control(0, 1, 5), abs=1e-9)
    assert late == pytest.approx(_expected_control(0, 20, 5), abs=1e-9)


def test_control_excludes_the_events_own_stock():
    uni = _universe()
    for o in uni:
        if o["observation_id"] == "b1_5":
            # flat through 5D, then +500%: its window return over [5, 15] is +500% — a real outlier
            o["outcomes"] = {f"{h}D": {"forward_return_%": 0.0 if h <= 5 else 500.0} for h in GRID}
    ctl = H.build_cohort_controls(uni)
    own = H.control_window_return(ctl, "2026-08-23", "b1_5", 5, 10)[0]
    assert own == pytest.approx(_expected_control(1, 5, 10), abs=1e-9)  # outlier invisible to itself
    other = H.control_window_return(ctl, "2026-08-23", "b1_6", 5, 10)[0]
    assert other > _expected_control(1, 5, 10) + 15                     # but visible to its peers (500%/23 members)


def test_no_defensible_control_returns_none_rather_than_a_guess():
    ctl = H.build_cohort_controls(_universe())
    assert H.control_window_return(ctl, "2026-08-23", "b0_0", a=30, h=10) is None     # beyond the outcome grid
    assert H.control_window_return(ctl, "2026-08-23", "nope", 5, 10) is None          # unknown source
    assert H.control_window_return(ctl, "2030-01-01", "b0_0", 5, 10) is None          # unknown cohort
    assert H.control_window_return(ctl, "2026-08-23", "b0_0", -1, 10) is None         # event before entry
    small = H.build_cohort_controls(_universe(n_per_bucket=5))                        # buckets of 5 < 15 members
    assert H.control_window_return(small, "2026-08-23", "b0_0", 5, 10) is None
    tiny = H.build_cohort_controls(_universe(n_per_bucket=4))                         # cohort of 20 < 30
    assert "2026-08-23" not in tiny["cohorts"]


def test_members_without_resolved_outcomes_are_left_out_not_zeroed():
    uni = _universe()
    for o in uni[96:102]:                                                            # 6 of bucket 4's 24 members
        o["outcomes"] = {k: v for k, v in o["outcomes"].items() if int(k[:-1]) <= 10}  # unresolved beyond 10D
    ctl = H.build_cohort_controls(uni)
    got = H.control_window_return(ctl, "2026-08-23", "b4_20", a=10, h=10)
    assert got is not None and got[0] == pytest.approx(_expected_control(4, 10, 10), abs=1e-9)
    assert got[1] == 17                                                              # 24 - self - 6 unresolved


# ── end to end ────────────────────────────────────────────────────────────
def _events(universe, effect, a_values, h=5, bucket_of=lambda i: i % 5):
    ev, lookup = [], {o["observation_id"]: o for o in universe}
    row0 = H.first_trading_day("2026-08-23")
    for i, a in enumerate(a_values):
        b = bucket_of(i)
        src = f"b{b}_{i % 20}"
        date = str(np.busday_offset(row0, a, holidays=H.NYSE_HOLIDAYS))
        ev.append({"observation_id": f"e{i}", "observation_type": "invalidation_breach",
                   "source_observation_id": src, "timestamp": date + " 00:00:00",
                   "outcomes": {f"{h}D": {"forward_return_%": _expected_control(b, a, h) + effect}}})
    return ev, lookup


def test_injected_effect_is_recovered_exactly_and_flagged():
    uni = _universe()
    ev, lookup = _events(uni, effect=3.0, a_values=[1 + (i % 28) for i in range(120)])
    ctl = H.build_cohort_controls(uni)
    res = H.evaluate_events({"G": ev}, ctl, lookup, 5)["G"]
    assert res["n_matched"] == 120 and res["n_windows"] >= 6
    assert res["mean_diff_%"] == pytest.approx(3.0, abs=1e-9)
    assert res["ci"][0] > 0 and res["verdict"].startswith("Outperformed matched control")


def test_null_is_recovered_and_not_flagged():
    uni = _universe()
    ev, lookup = _events(uni, effect=0.0, a_values=[1 + (i % 28) for i in range(120)])
    res = H.evaluate_events({"G": ev}, H.build_cohort_controls(uni), lookup, 5)["G"]
    assert res["mean_diff_%"] == pytest.approx(0.0, abs=1e-9)
    assert res["verdict"].startswith("No detectable difference")


def test_underperformance_is_reported_as_such():
    uni = _universe()
    ev, lookup = _events(uni, effect=-4.0, a_values=[1 + (i % 28) for i in range(120)])
    res = H.evaluate_events({"G": ev}, H.build_cohort_controls(uni), lookup, 5)["G"]
    assert res["verdict"].startswith("Underperformed matched control")


def test_a_huge_raw_loss_that_the_control_also_suffered_is_not_an_effect():
    # Bucket 0 drifts -0.8%/day; events there lose ~8% over 10 days — exactly what the control lost.
    uni = _universe()
    ev, lookup = _events(uni, effect=0.0, a_values=[1 + (i % 20) for i in range(100)], h=10,
                         bucket_of=lambda i: 0)
    res = H.evaluate_events({"G": ev}, H.build_cohort_controls(uni), lookup, 10)["G"]
    assert res["mean_event_%"] < -7                      # a scary raw number
    assert abs(res["mean_diff_%"]) < 1e-9                # ...that is entirely the matched drift


def test_too_few_independent_windows_withholds_the_interval_even_for_a_huge_difference():
    uni = _universe()
    ev, lookup = _events(uni, effect=10.0, a_values=[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], h=5)
    res = H.evaluate_events({"G": ev}, H.build_cohort_controls(uni), lookup, 5)["G"]
    assert res["n_windows"] < 6 and res["ci"] is None
    assert res["verdict"].startswith("Insufficient independent windows")


def test_too_few_matched_events_gives_no_verdict():
    uni = _universe()
    ev, lookup = _events(uni, effect=5.0, a_values=[1, 8, 15, 22])
    res = H.evaluate_events({"G": ev}, H.build_cohort_controls(uni), lookup, 5)["G"]
    assert res["verdict"].startswith("Too few matched events")


def test_unmatched_events_are_counted_not_silently_dropped():
    uni = _universe()
    ev, lookup = _events(uni, effect=1.0, a_values=[1 + (i % 28) for i in range(40)])
    ev[0]["source_observation_id"] = "missing"
    ev[1]["outcomes"] = {}                                           # no frozen outcome
    res = H.evaluate_events({"G": ev}, H.build_cohort_controls(uni), lookup, 5)["G"]
    assert res["n_events"] == 40 and res["n_with_outcome"] == 39
    assert res["n_matched"] == 38 and res["n_unmatched"] == 1


def test_evaluation_never_mutates_its_inputs():
    uni = _universe()
    ev, lookup = _events(uni, effect=1.0, a_values=[1 + (i % 28) for i in range(60)])
    snap = copy.deepcopy((uni, ev))
    ctl = H.build_cohort_controls(uni)
    H.evaluate_events({"G": ev}, ctl, lookup, 5)
    H.independence_summary(uni, 20)
    assert (uni, ev) == snap


# ── self-validation: the harness measured on its own synthetic worlds ─────
def test_acceptance_false_pass_rate_on_pure_noise_is_at_most_10_percent():
    r = H.simulate_matched_test(n_sims=400, n_days=240, h=20, n_events=300, effect=0.0)
    assert r["harness"] <= 0.10
    r5 = H.simulate_matched_test(n_sims=400, n_days=60, h=5, n_events=300, effect=0.0)
    assert r5["harness"] <= 0.10


def test_naive_row_level_intervals_demonstrably_fail_the_same_noise_test():
    r = H.simulate_matched_test(n_sims=400, n_days=240, h=20, n_events=300, effect=0.0)
    assert r["naive"] >= 0.20, "the simulation must actually reproduce the failure the harness exists to prevent"


def test_acceptance_detects_an_injected_effect_at_least_80_percent_of_the_time():
    r = H.simulate_matched_test(n_sims=400, n_days=240, h=20, n_events=300, effect=3.0)
    assert r["harness"] >= 0.80
    r5 = H.simulate_matched_test(n_sims=400, n_days=60, h=5, n_events=300, effect=3.0)
    assert r5["harness"] >= 0.80


def test_power_rises_with_effect_size_and_reports_the_honest_minimum_detectable_effect():
    curve = H.power_curve(effects=(0.0, 1.0, 2.0, 3.0, 4.0), n_sims=300, n_days=240, h=20, n_events=300)
    vals = [curve[e] for e in (0.0, 1.0, 2.0, 3.0, 4.0)]
    assert vals == sorted(vals) and vals[0] <= 0.10 and vals[-1] >= 0.95
    assert curve[1.0] < 0.5          # a 1-point effect is NOT reliably detectable at this sample size


def test_harness_stays_robust_when_the_control_is_badly_matched():
    r = H.simulate_matched_test(n_sims=400, n_days=240, h=20, n_events=300, effect=0.0,
                                beta_sd=1.0, sigma_factor=2.0)
    assert r["harness"] <= 0.10


def test_with_the_current_sample_shape_the_interval_is_withheld_every_time():
    # ~45 trading days of events with 20-day windows -> about 3 windows -> never enough for an interval.
    r = H.simulate_matched_test(n_sims=200, n_days=45, h=20, n_events=300, effect=0.0)
    assert r["harness_withheld"] == 1.0 and r["harness"] == 0.0
    assert r["naive"] > 0.3          # whereas a row-level interval would 'find' an effect on pure noise


# ── real committed data (offline, read-only) ──────────────────────────────
@pytest.mark.skipif(not (REPO / "data" / "alpha_observations.json").exists(), reason="no committed store")
def test_real_data_evaluation_runs_is_consistent_and_read_only():
    obs = json.load(open(REPO / "data" / "alpha_observations.json"))
    snap = copy.deepcopy(obs)
    rows = H.evaluate_invalidation_events(obs, horizons=(5, 20))
    assert rows and obs == snap
    known = ("Too few matched", "Insufficient independent windows", "Outperformed matched control",
             "Underperformed matched control", "No detectable difference")
    for r in rows:
        assert r["n_matched"] <= r["n_with_outcome"] <= r["n_events"]
        assert r["verdict"].startswith(known)
        assert (r["ci"] is None) == (r["n_windows"] < H.MIN_CLUSTERS_FOR_CI or r["n_matched"] == 0)
        if r["ci"]:
            assert r["ci"][0] <= r["mean_diff_%"] <= r["ci"][1]
    disc = [o for o in obs if o.get("observation_type", "discovery") == "discovery"
            and o.get("model_version") == "APEX-9.0"]
    s = H.independence_summary(disc, 20, "20D")
    assert s["label"] == H.independence_label(s["n_windows"]) and s["n_windows"] <= s["n_dates"]


# ── UI ────────────────────────────────────────────────────────────────────
def _overview_script():
    import modules.eval_harness as _h
    from ui.alpha_lab import _render_overview
    _render_overview(_h._TEST_OBS, "20D")


def test_ui_overview_caption_no_longer_claims_a_robust_well_supported_sample(monkeypatch):
    from streamlit.testing.v1 import AppTest
    obs = []
    for i in range(300):
        obs.append({"observation_id": f"d{i}", "observation_type": "discovery", "model_version": "APEX-9.0",
                    "timestamp": "2026-08-23 11:49:29", "ticker": f"T{i}", "setup_id": "S2-HIGH-RS",
                    "stage": "2 ✅ Uptrend", "apex_score": 60, "apex_score_raw": 60.0,
                    "technical_features": {}, "outcomes": {"20D": {"forward_return_%": -5.0 + (i % 7),
                                                                   "excess_return_%": -6.0, "benchmark_return_%": 1.0,
                                                                   "mfe_%": 1.0, "mae_%": -8.0}}})
    monkeypatch.setattr(H, "_TEST_OBS", obs, raising=False)
    at = AppTest.from_function(_overview_script, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    caps = " ".join(c.value for c in at.caption)
    assert "Robust — a well-supported sample" not in caps
    assert "independent 20D window(s)" in caps and "Insufficient independent windows" in caps
    assert "300 resolved rows" in caps


def _inval_script():
    import modules.invalidation_alpha as _ia
    from ui.alpha_lab import _render_invalidation_alpha
    _render_invalidation_alpha(_ia._TEST_OBS, "20D")


@pytest.mark.skipif(not (REPO / "data" / "alpha_observations.json").exists(), reason="no committed store")
def test_ui_invalidation_tab_renders_the_matched_evaluation_on_real_data(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import modules.invalidation_alpha as ia
    obs = json.load(open(REPO / "data" / "alpha_observations.json"))
    monkeypatch.setattr(ia, "_TEST_OBS", obs, raising=False)
    monkeypatch.setattr(ia, "yf", object())
    monkeypatch.setattr(ia, "load_scan_status", lambda: {})
    monkeypatch.setattr(ia, "load_quarantine", lambda: [])
    at = AppTest.from_function(_inval_script, default_timeout=90)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [e.label for e in at.expander]
    assert any("Matched-control evaluation" in lb for lb in labels)
    frames = [d.value for d in at.dataframe]
    matched = [f for f in frames if "95% CI (cluster-robust)" in getattr(f, "columns", [])]
    assert matched, "the matched-control table must render"
    df = matched[0]
    assert set(df["Horizon"]) == {"5D", "10D", "20D"}
    assert {"Group", "Matched", "Indep. windows", "Difference", "Verdict"} <= set(df.columns)
    assert any("withheld" in str(v) for v in df["95% CI (cluster-robust)"]), \
        "with this sample at least one cell must be an honest 'withheld', never a made-up interval"
