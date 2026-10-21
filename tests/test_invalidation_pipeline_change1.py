"""
tests/test_invalidation_pipeline_change1.py — Change 1 (detector INVAL-2.0)

Covers the repair of the invalidation-event pipeline:
  * source rotation / starvation regression
  * scan-status recording (scanned != detected)
  * idempotence
  * first-event (causal) handling, incl. no look-ahead
  * event provenance (detector_version / event_basis)
  * mechanical (already-below-stop) breach flagging
  * safe, auditable cleanup (versioned events preserved, quarantine, no
    deletion by event type, outcomes never modified)
  * the 5% near-miss threshold is unchanged

No network: every price path is injected via the `fetch` parameter.
"""
import copy
import inspect
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modules.invalidation_alpha as ia
from modules.invalidation_alpha import (
    DETECTOR_VERSION, LEGACY_VERIFIED_VERSION, BASIS_CROSSED, BASIS_ALREADY_BELOW,
    BASIS_APPROACHED, BASIS_WITHIN_BAND, audit_unversioned_events,
    cleanup_buggy_near_miss_events, compute_invalidation_alpha_overview,
    compute_scan_coverage, derive_event_basis, event_basis_for,
    log_invalidation_events, _detect_first_event,
)

TODAY = date(2026, 10, 20)
DISC_DAY = "2026-08-23"


# ── helpers ────────────────────────────────────────────────────────────────
def _series(values, start="2026-08-24"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def _disc(i, entry=120.0, stop=100.0, ticker=None, ts=DISC_DAY + " 11:49:29"):
    return {
        "observation_id": f"src{i}", "ticker": ticker or f"T{i}", "market": "US",
        "timestamp": ts, "model_version": "APEX-9.0", "setup_id": "S2-HIGH-RS",
        "stage": "2 ✅ Uptrend", "entry_price": entry, "stop_price": stop,
        "market_regime": "Sideways", "market_features": {}, "outcomes": {},
    }


def _fetcher(paths, calls=None):
    """paths: ticker -> Series. Records every call so tests can see rotation."""
    def fetch(ticker, start_date):
        if calls is not None:
            calls.append(ticker)
        return paths.get(ticker)
    return fetch


FLAT = _series([150.0] * 10)  # never tests a stop of 100


# ── first-event / causality ───────────────────────────────────────────────
def test_first_approach_wins_not_global_minimum():
    # Approaches the band on day 3 (104 -> 4% above stop), then falls to 80.
    # The OLD bug picked the lowest price (80). Correct: first event only.
    path = _series([118, 112, 104, 95, 80, 90])
    etype, edate, eprice = _detect_first_event(path, 100.0, 5.0)
    assert etype == "invalidation_near_miss"
    assert edate == path.index[2] and eprice == 104.0


def test_breach_first_when_it_comes_first():
    path = _series([118, 99, 104, 80])
    etype, edate, eprice = _detect_first_event(path, 100.0, 5.0)
    assert etype == "invalidation_breach" and eprice == 99.0 and edate == path.index[1]


def test_no_lookahead_future_bars_do_not_change_event():
    full = _series([118, 112, 104, 95, 80, 60, 200])
    event_day = _detect_first_event(full, 100.0, 5.0)
    truncated = full.iloc[: full.index.get_loc(event_day[1]) + 1]
    assert _detect_first_event(truncated, 100.0, 5.0) == event_day
    # and appending very different future data never alters it
    altered = pd.concat([truncated, _series([1, 1, 1], start="2026-09-10")])
    assert _detect_first_event(altered, 100.0, 5.0) == event_day


def test_untested_level_returns_none():
    assert _detect_first_event(FLAT, 100.0, 5.0) is None


def test_near_miss_threshold_unchanged_at_5_percent():
    assert inspect.signature(log_invalidation_events).parameters["near_miss_pct"].default == 5.0
    assert inspect.signature(audit_unversioned_events).parameters["near_miss_pct"].default == 5.0
    # Probe either side of the 5% line (not exactly on it: the unchanged
    # float arithmetic puts a price of exactly 105.0 at 5.000000000000004%).
    assert _detect_first_event(_series([120, 104.99]), 100.0, 5.0)[0] == "invalidation_near_miss"
    assert _detect_first_event(_series([120, 105.01]), 100.0, 5.0) is None


# ── scan status: scanned != detected ──────────────────────────────────────
def test_scan_status_recorded_when_no_event_found():
    obs, status = [_disc(0, ticker="AAA")], {}
    res = log_invalidation_events(obs, scan_status=status, fetch=_fetcher({"AAA": FLAT}), today=TODAY)
    assert res["scanned"] == 1 and res["new_events"] == 0 and res["no_event"] == 1
    assert len(obs) == 1  # nothing appended
    rec = status["src0"]
    assert rec["result"] == "no_event" and rec["scanned_on"] == "2026-10-20"
    assert rec["detector_version"] == DETECTOR_VERSION and rec["bars"] == 10


def test_scan_status_recorded_for_no_data_and_for_events():
    obs = [_disc(0, ticker="AAA"), _disc(1, ticker="BBB"), _disc(2, ticker="CCC")]
    paths = {"BBB": _series([118, 99, 90]), "CCC": None}
    status = {}
    res = log_invalidation_events(obs, scan_status=status, fetch=_fetcher({**paths, "AAA": FLAT}), today=TODAY)
    assert res["scanned"] == 3 and res["new_events"] == 1 and res["breaches"] == 1
    assert res["no_event"] == 1 and res["no_data"] == 1 and res["skipped"] == 1
    assert {k: v["result"] for k, v in status.items()} == {
        "src0": "no_event", "src1": "breach", "src2": "no_data"}


def test_invalid_input_is_recorded_not_looped_forever():
    bad = _disc(0)
    bad["stop_price"] = 0  # falsy but not None -> reaches the scan, rejected as invalid
    status = {}
    res = log_invalidation_events([bad], scan_status=status, fetch=_fetcher({}), today=TODAY)
    assert res["skipped"] == 1 and status["src0"]["result"] == "invalid_input"


# ── rotation / starvation regression ──────────────────────────────────────
def test_rotation_sweeps_all_sources_when_none_produce_events():
    n = 500
    obs = [_disc(i, ticker=f"T{i}") for i in range(n)]
    paths = {f"T{i}": FLAT for i in range(n)}
    status, calls = {}, []
    fetch = _fetcher(paths, calls)

    r1 = log_invalidation_events(obs, max_tickers=200, scan_status=status, fetch=fetch, today=TODAY)
    assert r1["scanned"] == 200 and r1["remaining_unscanned"] == 300
    assert set(calls) == {f"T{i}" for i in range(200)}

    calls.clear()
    r2 = log_invalidation_events(obs, max_tickers=200, scan_status=status, fetch=fetch, today=TODAY)
    assert set(calls) == {f"T{i}" for i in range(200, 400)}, "run 2 must move on, not re-fetch run 1's sources"

    calls.clear()
    r3 = log_invalidation_events(obs, max_tickers=200, scan_status=status, fetch=fetch, today=TODAY)
    assert set(calls) == {f"T{i}" for i in range(400, 500)}
    assert r3["scanned"] == 100 and r3["remaining_unscanned"] == 0
    assert len(status) == n and all(s["result"] == "no_event" for s in status.values())


def test_old_fixed_prefix_behaviour_would_have_starved_late_sources():
    """Reproduces the baseline failure mode against the baseline code path,
    so the regression is demonstrably the thing the new rotation fixes."""
    import subprocess, types
    old_src = subprocess.run(
        ["git", "show", "baseline-pre-change1:modules/invalidation_alpha.py"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if old_src.returncode != 0:
        pytest.skip("baseline tag not available in this checkout")
    old = types.ModuleType("old_invalidation_alpha")
    exec(compile(old_src.stdout, "old_invalidation_alpha.py", "exec"), old.__dict__)
    n = 500
    obs = [_disc(i, ticker=f"T{i}") for i in range(n)]
    calls = []
    old._fetch_price_path = lambda t, d: (calls.append(t), FLAT)[1]
    old.yf = object()  # truthy, so the old guard passes
    for _ in range(3):
        old.log_invalidation_events(obs, max_tickers=200)
    assert set(calls) == {f"T{i}" for i in range(200)}, "old code never leaves the first 200"
    assert "T450" not in calls


def test_late_discovery_is_reached_and_detected_despite_many_no_event_sources():
    n = 450
    obs = [_disc(i, ticker=f"T{i}") for i in range(n)]
    late = _disc(900, ticker="LATE", ts="2026-10-06 12:59:25")
    obs.append(late)
    paths = {f"T{i}": FLAT for i in range(n)}
    paths["LATE"] = _series([118, 96, 90], start="2026-10-07")
    status = {}
    detected = 0
    for _ in range(3):
        res = log_invalidation_events(obs, max_tickers=200, scan_status=status,
                                      fetch=_fetcher(paths), today=TODAY)
        detected += res["new_events"]
    assert detected == 1
    ev = [o for o in obs if o.get("observation_type") == "invalidation_breach"]
    assert len(ev) == 1 and ev[0]["ticker"] == "LATE"


def test_next_day_rescans_least_recently_scanned_first():
    obs = [_disc(i, ticker=f"T{i}") for i in range(6)]
    paths = {f"T{i}": FLAT for i in range(6)}
    status = {}
    log_invalidation_events(obs, max_tickers=3, scan_status=status, fetch=_fetcher(paths), today=date(2026, 10, 18))
    log_invalidation_events(obs, max_tickers=3, scan_status=status, fetch=_fetcher(paths), today=date(2026, 10, 19))
    calls = []
    log_invalidation_events(obs, max_tickers=3, scan_status=status, fetch=_fetcher(paths, calls), today=TODAY)
    assert set(calls) == {"T0", "T1", "T2"}  # scanned 10-18, the oldest


# ── idempotence ───────────────────────────────────────────────────────────
def test_idempotent_same_day_and_event_sources_never_rescanned():
    obs = [_disc(0, ticker="AAA"), _disc(1, ticker="BBB")]
    paths = {"AAA": FLAT, "BBB": _series([118, 99, 90])}
    status, calls = {}, []
    fetch = _fetcher(paths, calls)
    log_invalidation_events(obs, scan_status=status, fetch=fetch, today=TODAY)
    snap_obs, snap_status = copy.deepcopy(obs), copy.deepcopy(status)

    calls.clear()
    res = log_invalidation_events(obs, scan_status=status, fetch=fetch, today=TODAY)
    assert res["scanned"] == 0 and res["new_events"] == 0 and calls == []
    assert obs == snap_obs and status == snap_status

    # next day: AAA is re-examined; BBB already has its event and is NOT re-fetched
    calls.clear()
    log_invalidation_events(obs, scan_status=status, fetch=fetch, today=date(2026, 10, 21))
    assert calls == ["AAA"]
    assert sum(1 for o in obs if o.get("source_observation_id") == "src1") == 1


def test_no_event_source_that_later_breaches_is_detected_on_rescan():
    obs = [_disc(0, ticker="AAA")]
    status = {}
    log_invalidation_events(obs, scan_status=status, fetch=_fetcher({"AAA": FLAT}), today=TODAY)
    assert len(obs) == 1
    later = _series([150, 140, 98])
    res = log_invalidation_events(obs, scan_status=status, fetch=_fetcher({"AAA": later}),
                                  today=date(2026, 10, 21))
    assert res["new_events"] == 1 and status["src0"]["result"] == "breach"


# ── provenance + mechanical flagging ──────────────────────────────────────
def test_new_events_carry_provenance():
    obs = [_disc(0, ticker="GEN", entry=120, stop=100), _disc(1, ticker="NM", entry=120, stop=100)]
    paths = {"GEN": _series([118, 99, 80]), "NM": _series([118, 104, 90])}
    log_invalidation_events(obs, scan_status={}, fetch=_fetcher(paths), today=TODAY)
    events = {o["ticker"]: o for o in obs if o.get("observation_type")}
    for e in events.values():
        assert e["detector_version"] == DETECTOR_VERSION
        assert e["logged_on"] == "2026-10-20"
        assert e["source_observation_id"].startswith("src")
    assert events["GEN"]["event_basis"] == BASIS_CROSSED and events["GEN"]["is_mechanical"] is False
    assert events["NM"]["event_basis"] == BASIS_APPROACHED and events["NM"]["is_mechanical"] is False


def test_mechanical_breach_flagged_when_stop_already_above_entry():
    # S3-TOP style: stock already below its 200MA when discovered.
    mech = _disc(0, ticker="MECH", entry=90.0, stop=100.0)
    gen = _disc(1, ticker="GEN", entry=120.0, stop=100.0)
    obs = [mech, gen]
    paths = {"MECH": _series([89, 85, 80]), "GEN": _series([118, 99, 90])}
    res = log_invalidation_events(obs, scan_status={}, fetch=_fetcher(paths), today=TODAY)
    assert res["breaches"] == 2 and res["mechanical_breaches"] == 1
    ev = {o["ticker"]: o for o in obs if o.get("observation_type")}
    assert ev["MECH"]["event_basis"] == BASIS_ALREADY_BELOW and ev["MECH"]["is_mechanical"] is True
    assert ev["GEN"]["event_basis"] == BASIS_CROSSED and ev["GEN"]["is_mechanical"] is False


def test_near_miss_within_band_at_discovery_is_labelled_not_re_detected():
    # Detection is unchanged; only the label differs.
    assert event_basis_for("invalidation_near_miss", 103.0, 100.0) == BASIS_WITHIN_BAND
    assert event_basis_for("invalidation_near_miss", 120.0, 100.0) == BASIS_APPROACHED
    assert event_basis_for("invalidation_breach", 100.0, 100.0) == BASIS_ALREADY_BELOW  # stop >= entry


def test_legacy_events_get_basis_derived_at_read_time_without_mutation():
    src = _disc(0, entry=90.0, stop=100.0)
    legacy = {"observation_id": "e0", "observation_type": "invalidation_breach",
              "source_observation_id": "src0", "ticker": "T0"}
    before = copy.deepcopy(legacy)
    assert derive_event_basis(legacy, src) == BASIS_ALREADY_BELOW
    assert derive_event_basis(legacy, None) == "unknown"
    assert legacy == before


def _with_outcome(ev, ret):
    ev["outcomes"] = {"20D": {"forward_return_%": ret, "excess_return_%": ret - 1,
                              "mfe_%": 0, "mae_%": ret, "benchmark_return_%": 1.0}}
    return ev


def test_overview_separates_mechanical_from_genuine_breaches():
    obs = [_disc(0, entry=90, stop=100), _disc(1, entry=120, stop=100), _disc(2, entry=130, stop=100)]
    obs += [
        _with_outcome({"observation_id": "e0", "observation_type": "invalidation_breach",
                       "source_observation_id": "src0", "ticker": "T0"}, -20.0),
        _with_outcome({"observation_id": "e1", "observation_type": "invalidation_breach",
                       "source_observation_id": "src1", "ticker": "T1"}, -4.0),
        _with_outcome({"observation_id": "e2", "observation_type": "invalidation_breach",
                       "source_observation_id": "src2", "ticker": "T2"}, -6.0),
    ]
    ov = compute_invalidation_alpha_overview(obs, "20D")
    assert ov["breach"]["n"] == 3                      # existing key unchanged
    assert ov["breach_mechanical_n"] == 1
    assert ov["breach_genuine"]["n"] == 2
    assert ov["breach_genuine"]["expectancy_%"] == pytest.approx(-5.0)


# ── coverage: scanned vs detected ─────────────────────────────────────────
def test_coverage_distinguishes_scanned_from_detected():
    obs = [_disc(i, ticker=f"T{i}") for i in range(4)]
    obs.append(_disc(9, ticker="LATE", ts="2026-10-06 12:59:25"))
    paths = {"T0": FLAT, "T1": FLAT, "T2": _series([118, 99]), "T3": None, "LATE": FLAT}
    status = {}
    log_invalidation_events(obs, max_tickers=3, scan_status=status, fetch=_fetcher(paths), today=TODAY)
    cov = compute_scan_coverage(obs, status, later_than="2026-08-23")
    assert cov["overall"]["eligible"] == 5
    assert cov["overall"]["scanned_with_status"] == 3
    assert cov["overall"]["detected"] == 1            # T2 only
    assert cov["overall"]["scanned_any"] == 3
    assert cov["later_discoveries"]["eligible"] == 1 and cov["later_discoveries"]["scanned_any"] == 0
    log_invalidation_events(obs, max_tickers=3, scan_status=status, fetch=_fetcher(paths), today=TODAY)
    cov = compute_scan_coverage(obs, status, later_than="2026-08-23")
    assert cov["overall"]["scanned_with_status"] == 5 and cov["overall"]["detected"] == 1
    assert cov["later_discoveries"]["scanned_with_status"] == 1 and cov["later_discoveries"]["detected"] == 0


def test_legacy_event_counts_as_scanned_but_not_as_status():
    obs = [_disc(0)]
    obs.append({"observation_id": "e0", "observation_type": "invalidation_near_miss",
                "source_observation_id": "src0", "ticker": "T0"})
    cov = compute_scan_coverage(obs, {})
    assert cov["overall"]["scanned_any"] == 1 and cov["overall"]["scanned_with_status"] == 0
    assert cov["overall"]["detected"] == 1


# ── safe, auditable cleanup ───────────────────────────────────────────────
def _legacy_near_miss(i, ts, price):
    return {"observation_id": f"leg{i}", "observation_type": "invalidation_near_miss",
            "source_observation_id": f"src{i}", "ticker": f"T{i}", "timestamp": ts + " 00:00:00",
            "entry_price": price,
            "outcomes": {"5D": {"forward_return_%": 3.3, "computed_at": "2026-09-30"}}}


def _cleanup_fixture():
    # src0: legacy near-miss that MATCHES the causal detector (104 on 2026-08-26)
    # src1: legacy near-miss logged at the series MINIMUM (the old bug) -> mismatch
    # src2: legacy, no price data -> unverifiable
    obs = [_disc(0, ticker="T0"), _disc(1, ticker="T1"), _disc(2, ticker="T2")]
    p0 = _series([118, 112, 104, 95, 80])             # first approach: day 3, 104
    p1 = _series([118, 104, 103, 102, 101.5, 106])    # first approach 104 on day 2; legacy says min
    paths = {"T0": p0, "T1": p1, "T2": None}
    d0 = p0.index[2].strftime("%Y-%m-%d")
    d1_min = p1.index[4].strftime("%Y-%m-%d")
    legacy = [_legacy_near_miss(0, d0, 104.0), _legacy_near_miss(1, d1_min, 101.5),
              _legacy_near_miss(2, "2026-09-01", 103.0)]
    versioned = {"observation_id": "v1", "observation_type": "invalidation_near_miss",
                 "source_observation_id": "srcV", "ticker": "TV", "timestamp": "2026-09-02 00:00:00",
                 "entry_price": 104.0, "detector_version": DETECTOR_VERSION,
                 "outcomes": {"5D": {"forward_return_%": -1.0}}}
    breach = {"observation_id": "b1", "observation_type": "invalidation_breach",
              "source_observation_id": "srcB", "ticker": "TB", "timestamp": "2026-09-03 00:00:00",
              "entry_price": 95.0, "outcomes": {"5D": {"forward_return_%": -2.0}}}
    return obs + legacy + [versioned, breach], paths


def test_audit_is_read_only_and_classifies_correctly():
    obs, paths = _cleanup_fixture()
    snap = copy.deepcopy(obs)
    res = {r["observation_id"]: r for r in audit_unversioned_events(obs, fetch=_fetcher(paths))}
    assert obs == snap, "audit must not mutate anything"
    assert res["leg0"]["status"] == "verified"
    assert res["leg1"]["status"] == "mismatch"
    assert res["leg2"]["status"] == "unverifiable"
    assert "v1" not in res and "b1" not in res, "versioned events and breaches are never audited"


def test_cleanup_quarantines_only_confirmed_mismatches_and_preserves_everything_else():
    obs, paths = _cleanup_fixture()
    audit = audit_unversioned_events(obs, fetch=_fetcher(paths))
    before = {o["observation_id"]: copy.deepcopy(o) for o in obs}
    quarantine, status = [], {"src1": {"result": "near_miss"}, "src0": {"result": "near_miss"}}

    summary = cleanup_buggy_near_miss_events(obs, audit, quarantine=quarantine, scan_status=status,
                                             now=pd.Timestamp("2026-10-20 10:00:00").to_pydatetime())
    assert summary == {"removed": 1, "verified_stamped": 1, "drift_stamped": 0,
                       "kept_unverified": 1, "versioned_untouched": 1}

    ids = {o["observation_id"] for o in obs}
    assert "leg1" not in ids                              # confirmed mismatch removed
    assert {"leg0", "leg2", "v1", "b1"} <= ids            # everything else kept

    # quarantined: full original record incl. outcomes, with reason + timestamp
    assert len(quarantine) == 1 and quarantine[0]["record"] == before["leg1"]
    assert quarantine[0]["reason"] and quarantine[0]["quarantined_at"] == "2026-10-20 10:00:00"
    assert "src1" not in status and "src0" in status      # only the removed event's source is freed

    byid = {o["observation_id"]: o for o in obs}
    assert byid["leg0"]["detector_version"] == LEGACY_VERIFIED_VERSION
    assert "detector_version" not in byid["leg2"]         # unverifiable: untouched
    for oid in ("v1", "b1", "leg2"):
        assert byid[oid] == before[oid]
    # stamping touched ONE field only; stored outcomes identical everywhere
    assert {k: v for k, v in byid["leg0"].items() if k != "detector_version"} == before["leg0"]
    for oid, o in byid.items():
        assert o.get("outcomes") == before[oid].get("outcomes")


def test_cleanup_refuses_to_delete_by_event_type_without_an_audit():
    obs, _ = _cleanup_fixture()
    snap = copy.deepcopy(obs)
    with pytest.raises(ValueError):
        cleanup_buggy_near_miss_events(obs, None, quarantine=[])
    assert obs == snap


def test_cleanup_with_empty_audit_removes_nothing():
    obs, _ = _cleanup_fixture()
    snap = copy.deepcopy(obs)
    summary = cleanup_buggy_near_miss_events(obs, [], quarantine=[])
    assert summary["removed"] == 0 and obs == snap


def test_cleanup_never_touches_versioned_events_even_if_audit_says_mismatch():
    obs, _ = _cleanup_fixture()
    forged = [{"observation_id": "v1", "status": "mismatch", "reason": "x"}]
    q = []
    summary = cleanup_buggy_near_miss_events(obs, forged, quarantine=q)
    assert summary["removed"] == 0 and q == []
    assert any(o["observation_id"] == "v1" for o in obs)


def test_cleanup_is_rerunnable():
    obs, paths = _cleanup_fixture()
    q, status = [], {}
    cleanup_buggy_near_miss_events(obs, audit_unversioned_events(obs, fetch=_fetcher(paths)), q, status)
    snap = copy.deepcopy(obs)
    audit2 = audit_unversioned_events(obs, fetch=_fetcher(paths))   # only leg2 (unverifiable) remains unversioned
    s2 = cleanup_buggy_near_miss_events(obs, audit2, q, status)
    assert s2["removed"] == 0 and obs == snap and len(q) == 1


def test_quarantined_source_becomes_eligible_for_corrected_rescan():
    obs, paths = _cleanup_fixture()
    q, status = [], {}
    cleanup_buggy_near_miss_events(obs, audit_unversioned_events(obs, fetch=_fetcher(paths)), q, status)
    res = log_invalidation_events(obs, scan_status=status, fetch=_fetcher(paths), today=TODAY)
    redetected = [o for o in obs if o.get("source_observation_id") == "src1"
                  and o.get("detector_version") == DETECTOR_VERSION]
    assert len(redetected) == 1
    assert redetected[0]["entry_price"] == 104.0          # first approach, not the later minimum


# ── data integrity on the REAL committed store (offline, in-memory only) ──
@pytest.mark.skipif(not (REPO_ROOT / "data" / "alpha_observations.json").exists(),
                    reason="no committed observation store")
def test_scan_never_modifies_existing_records_or_outcomes_on_real_data():
    real = json.load(open(REPO_ROOT / "data" / "alpha_observations.json"))
    before = copy.deepcopy(real)
    status = {}
    # synthetic no-event paths — each ticker's path sits at 10x its OWN stop, so it
    # can never breach or approach; exercises rotation + status over the real population
    stop_by_ticker = {o["ticker"]: o["stop_price"] for o in real if o.get("stop_price")}
    fetch = lambda t, d: _series([stop_by_ticker[t] * 10] * 10)
    for _ in range(4):
        log_invalidation_events(real, max_tickers=200, scan_status=status, fetch=fetch, today=TODAY)
    assert real == before, "a scan with no events must not change a single stored record"
    discoveries = [o for o in real if o.get("observation_type", "discovery") == "discovery"
                   and o.get("stop_price") is not None]
    event_srcs = {o.get("source_observation_id") for o in real
                  if o.get("observation_type", "") .startswith("invalidation_")}
    pending = [o for o in discoveries if o["observation_id"] not in event_srcs]
    assert len(status) == len(pending), "every source without a stored event now has a scan status"


# ── UI code path (Streamlit AppTest; storage + prices injected) ───────────
def _ui_script():
    import modules.invalidation_alpha as _ia
    from ui.alpha_lab import _render_invalidation_alpha
    _render_invalidation_alpha(_ia._TEST_OBS, "20D")


def _ui_setup(monkeypatch, obs, paths):
    from streamlit.testing.v1 import AppTest
    calls = {"save_obs": 0, "save_status": [], "save_q": [], "order": []}
    store = {"status": {}, "quarantine": []}

    monkeypatch.setattr(ia, "_TEST_OBS", obs, raising=False)
    monkeypatch.setattr(ia, "_fetch_price_path", lambda t, d: paths.get(t))
    monkeypatch.setattr(ia, "yf", object())  # pass the "yfinance available" guard
    monkeypatch.setattr(ia, "load_scan_status", lambda: store["status"])
    monkeypatch.setattr(ia, "load_quarantine", lambda: store["quarantine"])

    def _save_status(s):
        calls["save_status"].append(copy.deepcopy(s)); calls["order"].append("status"); return True
    def _save_q(q):
        calls["save_q"].append(copy.deepcopy(q)); calls["order"].append("quarantine"); return True
    monkeypatch.setattr(ia, "save_scan_status", _save_status)
    monkeypatch.setattr(ia, "save_quarantine", _save_q)

    import modules.alpha_validation as av
    def _save_obs(o):
        calls["save_obs"] += 1; calls["order"].append("observations")
    monkeypatch.setattr(av, "save_observations", _save_obs)
    return AppTest.from_function(_ui_script, default_timeout=30), calls, store


def _texts(at):
    return " ".join([w.value for w in at.warning] + [w.value for w in at.success] +
                    [w.value for w in at.info] + [c.value for c in at.caption])


def test_ui_scan_records_status_and_reports_scanned_vs_detected(monkeypatch):
    obs = [_disc(0, ticker="AAA"), _disc(1, ticker="BBB", entry=90.0, stop=100.0)]
    paths = {"AAA": FLAT, "BBB": _series([89, 85])}
    at, calls, store = _ui_setup(monkeypatch, obs, paths)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.button(key="inval_alpha_scan_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    msg = " ".join(s.value for s in at.success)
    assert "Scanned 2 source(s)" in msg and "detected 1 new event(s)" in msg
    assert "1 breach of which 1 mechanical" in msg and "1 scanned with no event" in msg
    assert calls["save_status"], "scan status must be saved even though one source had no event"
    assert set(calls["save_status"][-1]) == {"src0", "src1"}
    assert calls["save_obs"] == 1  # a new event existed
    assert "MECHANICAL" in _texts(at) or "mechanical" in _texts(at).lower()


def test_ui_scan_with_no_events_saves_status_but_not_observations(monkeypatch):
    obs = [_disc(0, ticker="AAA")]
    at, calls, _ = _ui_setup(monkeypatch, obs, {"AAA": FLAT})
    at.run(); at.button(key="inval_alpha_scan_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert calls["save_obs"] == 0 and len(calls["save_status"]) == 1
    assert len(obs) == 1


def test_ui_legacy_warning_offers_audit_not_blind_cleanup(monkeypatch):
    obs, paths = _cleanup_fixture()
    at, calls, _ = _ui_setup(monkeypatch, obs, paths)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "no detector provenance" in " ".join(w.value for w in at.warning)
    keys = {b.key for b in at.button}
    assert "inval_alpha_audit_btn" in keys and "inval_alpha_cleanup_btn" not in keys
    assert calls["save_obs"] == 0


def test_ui_audit_then_apply_quarantines_before_removing(monkeypatch):
    obs, paths = _cleanup_fixture()
    n_before = len(obs)
    at, calls, store = _ui_setup(monkeypatch, obs, paths)
    at.run()
    at.button(key="inval_alpha_audit_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(obs) == n_before and calls["save_obs"] == 0, "audit is read-only"
    assert "mismatch: 1" in " ".join(i.value for i in at.info)

    at.button(key="inval_alpha_apply_audit_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(obs) == n_before - 1 and all(o["observation_id"] != "leg1" for o in obs)
    assert calls["order"].index("quarantine") < calls["order"].index("observations")
    assert calls["save_q"][-1][0]["record"]["observation_id"] == "leg1"


def test_ui_apply_aborts_without_removal_if_quarantine_cannot_be_saved(monkeypatch):
    obs, paths = _cleanup_fixture()
    at, calls, _ = _ui_setup(monkeypatch, obs, paths)
    monkeypatch.setattr(ia, "save_quarantine", lambda q: False)
    at.run(); at.button(key="inval_alpha_audit_btn").click().run()
    at.button(key="inval_alpha_apply_audit_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("NOTHING was removed" in e.value for e in at.error)
    assert calls["save_obs"] == 0, "no removal may be persisted without a recoverable copy"
