"""
tests/test_invalidation_audit_change1b.py — Change 1b

The Change 1 audit compared stored legacy near-miss events to a re-derived
event by EXACT date and price. yfinance history is dividend/split-adjusted
retroactively, so a close stored at logging time can differ slightly from
today's adjusted close — nudging a first-approach date by a day or changing a
price at the cent level — and the audit would call a valid event a "mismatch".

1b: a four-way audit (verified / adjustment_drift / mismatch / unverifiable).
A fixed date tolerance was tried first and rejected: for a stock hovering just
above the band edge a ~1% adjustment shifts the first-approach date by many days,
while a tolerance wide enough to cover that would wave real argmin-bug events
through. Instead the audit reconstructs the as-logged path (today's path divided
by the factor implied by the stored event) and asks whether the causal detector
selects the stored event on it. Evidence on every row; drift events are kept
(stamped only), never quarantined; only positive evidence leads to quarantine.
No network: price paths are injected.
"""
import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.invalidation_alpha as ia
from modules.invalidation_alpha import (
    audit_unversioned_events, cleanup_buggy_near_miss_events, compute_scan_coverage,
    LEGACY_DRIFT_VERSION, LEGACY_VERIFIED_VERSION, AUDIT_PRICE_TOL,
    _detect_first_event,
)

STOP = 100.0


def _series(values, start="2026-08-24"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def _src(entry=120.0, ticker="TKR"):
    return {"observation_id": "src", "ticker": ticker, "timestamp": "2026-08-23 11:49:29",
            "model_version": "APEX-9.0", "entry_price": entry, "stop_price": STOP, "outcomes": {}}


def _stored(path, pos, price=None, oid="ev"):
    """A legacy near-miss stored at bar `pos` of `path` (as logged at the time)."""
    return {"observation_id": oid, "observation_type": "invalidation_near_miss",
            "source_observation_id": "src", "ticker": "TKR",
            "timestamp": path.index[pos].strftime("%Y-%m-%d") + " 00:00:00",
            "entry_price": round(float(path.iloc[pos]) if price is None else price, 2),
            "outcomes": {"5D": {"forward_return_%": 2.0}}}


def _audit(logged_path, today_path, pos, price=None):
    obs = [_src(), _stored(logged_path, pos, price)]
    return audit_unversioned_events(obs, fetch=lambda t, d: today_path)[0]


# ── the failure mode that motivated 1b ────────────────────────────────────
def test_exact_match_is_still_verified():
    path = _series([118, 112, 104, 95, 80])
    r = _audit(path, path, pos=2)
    assert r["status"] == "verified" and r["trading_day_gap"] == 0 and r["rescale_factor"] == 1.0


def test_dividend_adjusted_prices_same_date_are_drift_not_mismatch():
    logged = _series([118, 112, 104, 95, 80])
    r = _audit(logged, logged * 0.995, pos=2)           # 0.5% dividend adjustment since logging
    assert r["status"] == "adjustment_drift" and r["trading_day_gap"] == 0
    assert 0.994 < r["rescale_factor"] < 0.996


def test_adjustment_that_moves_first_approach_one_day_earlier_is_drift():
    # As logged, bar 2 (105.3) is just outside the 5% band, so the stored event is bar 3 (104.6).
    # After a 0.5% adjustment bar 2 becomes 104.77 -> inside the band -> causal event moves a day earlier.
    logged = _series([118, 112, 105.3, 104.6, 103, 104])
    assert _detect_first_event(logged, STOP, 5.0)[1] == logged.index[3]
    today = logged * 0.995
    assert _detect_first_event(today, STOP, 5.0)[1] == today.index[2]
    r = _audit(logged, today, pos=3)
    assert r["status"] == "adjustment_drift" and r["trading_day_gap"] == 1


def test_adjustment_that_moves_event_one_day_later_is_also_drift():
    logged = _series([118, 112, 104.9, 104.0, 100.5, 103])
    today = logged * 1.004
    assert _detect_first_event(today, STOP, 5.0)[1] == today.index[3]
    r = _audit(logged, today, pos=2)
    assert r["status"] == "adjustment_drift" and r["trading_day_gap"] == -1


def test_hovering_stock_multi_day_shift_is_drift_where_a_day_tolerance_would_fail():
    # Stock hovers 5.2-5.4% above the stop for six bars, then dips into the band.
    logged = _series([118, 112, 105.4, 105.3, 105.25, 105.3, 105.2, 105.35, 104.6, 103])
    assert _detect_first_event(logged, STOP, 5.0)[1] == logged.index[8]
    today = logged * 0.99                                # 1% adjustment pulls the whole hover into the band
    shifted = _detect_first_event(today, STOP, 5.0)[1]
    assert today.index.get_loc(shifted) <= 6             # causal date moved by 2+ trading days
    r = _audit(logged, today, pos=8)
    assert r["trading_day_gap"] >= 2 and r["status"] == "adjustment_drift"


# ── what must STILL be caught ─────────────────────────────────────────────
def test_old_argmin_signature_is_still_a_mismatch():
    path = _series([118, 112, 104, 103, 102, 101.2, 106])      # first approach bar 2; old bug stored the minimum, bar 5
    assert _detect_first_event(path, STOP, 5.0)[1] == path.index[2]
    r = _audit(path, path, pos=5)
    assert r["status"] == "mismatch" and r["trading_day_gap"] == 3


def test_one_day_argmin_gap_with_no_adjustment_is_a_mismatch():
    # The case a +/-1 day tolerance would have waved through as drift.
    path = _series([118, 112, 104, 103.5, 108])
    r = _audit(path, path, pos=3)                              # stored one bar after the first approach, prices unadjusted
    assert r["trading_day_gap"] == 1 and r["status"] == "mismatch"


def test_argmin_event_is_still_caught_when_history_has_ALSO_been_adjusted():
    logged = _series([118, 112, 104, 103, 102, 101.2, 106])
    r = _audit(logged, logged * 0.996, pos=5)                  # stored at the minimum AND history adjusted since
    assert r["status"] == "mismatch" and "rescaled" in r["reason"]


def test_type_difference_is_a_mismatch():
    # stored near-miss; on the as-logged path the series breaches first
    stored_path = _series([118, 112, 104, 103, 102])
    today = _series([118, 98, 102.96, 102, 101])                # adj. close at stored bar = 104 * 0.99
    r = _audit(stored_path, today, pos=2)
    assert r["status"] == "mismatch" and "invalidation_breach" in r["reason"]


def test_stored_price_outside_the_band_is_a_mismatch():
    # A stored 'near-miss' at 112 (12% above the stop) cannot be a near-miss under any adjustment.
    path = _series([118, 112, 110, 108])
    r = _audit(path, path, pos=1)
    assert r["status"] == "mismatch" and "no event" in r["reason"]


def test_price_basis_beyond_tolerance_is_unverifiable_not_mismatch():
    path = _series([118, 112, 104, 95, 80])
    r = _audit(path, path, pos=2, price=104.0 * 1.10)           # stored price 10% off today's adjusted close
    assert r["status"] == "unverifiable" and "beyond dividend-adjustment tolerance" in r["reason"]


def test_split_sized_difference_is_unverifiable_and_kept():
    logged = _series([118, 112, 104, 95, 80])
    r = _audit(logged, logged * 0.5, pos=2)                     # e.g. a 2:1 split adjusted history since
    assert r["status"] == "unverifiable"


def test_tolerance_boundary_is_the_documented_3_percent():
    assert AUDIT_PRICE_TOL == 0.03
    path = _series([118, 112, 104, 95, 80])
    assert _audit(path, path * (1 - 0.029), pos=2)["status"] == "adjustment_drift"
    assert _audit(path, path * (1 - 0.031), pos=2)["status"] == "unverifiable"


def test_unverifiable_unchanged_when_no_price_data():
    obs = [_src(), _stored(_series([118, 112, 104]), 2)]
    assert audit_unversioned_events(obs, fetch=lambda t, d: None)[0]["status"] == "unverifiable"


def test_stored_date_not_in_current_history_is_unverifiable():
    path = _series([118, 112, 104])
    ev = _stored(path, 2)
    ev["timestamp"] = "2026-12-31 00:00:00"
    r = audit_unversioned_events([_src(), ev], fetch=lambda t, d: path)[0]
    assert r["status"] == "unverifiable"


# ── evidence + read-only ──────────────────────────────────────────────────
def test_every_result_carries_the_evidence_needed_to_check_it_by_eye():
    logged = _series([118, 112, 104, 95, 80])
    r = _audit(logged, logged * 0.996, pos=2)
    for k in ("stored_date", "causal_date", "stored_price", "causal_price", "rescale_factor",
              "trading_day_gap", "reason", "status", "observation_id", "ticker"):
        assert k in r


def test_audit_remains_read_only():
    logged = _series([118, 112, 104, 95, 80])
    obs = [_src(), _stored(logged, 2)]
    snap = copy.deepcopy(obs)
    audit_unversioned_events(obs, fetch=lambda t, d: logged * 0.995)
    assert obs == snap


# ── cleanup handling of the new class ─────────────────────────────────────
def _mixed_fixture():
    drift_path = _series([118, 112, 104, 95, 80])
    bug_path = _series([118, 112, 104, 103, 102, 101.2, 106])
    obs = [
        {**_src(ticker="DRIFT"), "observation_id": "sD"},
        {**_src(ticker="BUG"), "observation_id": "sB"},
        {**_stored(drift_path, 2, oid="eD"), "source_observation_id": "sD", "ticker": "DRIFT"},
        {**_stored(bug_path, 5, oid="eB"), "source_observation_id": "sB", "ticker": "BUG"},
    ]
    paths = {"DRIFT": drift_path * 0.995, "BUG": bug_path}
    return obs, (lambda t, d: paths[t])


def test_cleanup_keeps_and_stamps_drift_and_quarantines_only_true_mismatches():
    obs, fetch = _mixed_fixture()
    audit = audit_unversioned_events(obs, fetch=fetch)
    assert {r["observation_id"]: r["status"] for r in audit} == {"eD": "adjustment_drift", "eB": "mismatch"}
    before = {o["observation_id"]: copy.deepcopy(o) for o in obs}
    q = []
    summary = cleanup_buggy_near_miss_events(obs, audit, quarantine=q)
    assert summary["removed"] == 1 and summary["drift_stamped"] == 1 and summary["verified_stamped"] == 0
    byid = {o["observation_id"]: o for o in obs}
    assert "eB" not in byid and "eD" in byid
    assert q[0]["record"] == before["eB"]
    assert byid["eD"]["detector_version"] == LEGACY_DRIFT_VERSION
    assert {k: v for k, v in byid["eD"].items() if k != "detector_version"} == before["eD"]
    assert byid["eD"]["outcomes"] == before["eD"]["outcomes"]


def test_drift_stamped_events_no_longer_trigger_the_provenance_warning():
    obs, fetch = _mixed_fixture()
    cleanup_buggy_near_miss_events(obs, audit_unversioned_events(obs, fetch=fetch), quarantine=[])
    legacy_left = [o for o in obs if o.get("observation_type") == "invalidation_near_miss"
                   and not o.get("detector_version")]
    assert legacy_left == []


def test_drift_events_are_not_rescanned_or_duplicated_after_cleanup():
    obs, fetch = _mixed_fixture()
    status = {}
    cleanup_buggy_near_miss_events(obs, audit_unversioned_events(obs, fetch=fetch), [], status)
    from datetime import date
    res = ia.log_invalidation_events(obs, scan_status=status, fetch=fetch, today=date(2026, 10, 21))
    assert sum(1 for o in obs if o.get("source_observation_id") == "sD") == 1   # drift event untouched
    assert sum(1 for o in obs if o.get("source_observation_id") == "sB") == 1   # re-detected causal event
    redone = [o for o in obs if o.get("source_observation_id") == "sB"][0]
    assert redone["detector_version"] == ia.DETECTOR_VERSION


def test_cleanup_never_removes_drift_even_if_forged_as_mismatch_for_a_versioned_event():
    obs, fetch = _mixed_fixture()
    audit = audit_unversioned_events(obs, fetch=fetch)
    cleanup_buggy_near_miss_events(obs, audit, quarantine=[])        # stamps eD
    forged = [{"observation_id": "eD", "status": "mismatch", "reason": "x"}]
    q = []
    s = cleanup_buggy_near_miss_events(obs, forged, quarantine=q)
    assert s["removed"] == 0 and q == [] and any(o["observation_id"] == "eD" for o in obs)


# ── measured specificity / sensitivity on the REAL population (synthetic prices) ─
def test_simulated_specificity_and_sensitivity_on_real_legacy_events():
    """For each real legacy near-miss: build a price path consistent with its stored event,
    then (a) apply a dividend-sized adjustment -> must NOT be quarantined, and
    (b) displace the stored event >= 2 trading days later (old argmin signature) -> MUST be caught."""
    import json, random
    real = json.load(open(Path(__file__).resolve().parent.parent / "data" / "alpha_observations.json"))
    byid = {o.get("observation_id"): o for o in real if o.get("observation_id")}
    legacy = [e for e in real if e.get("observation_type") == "invalidation_near_miss"
              and not e.get("detector_version")]   # unversioned = what the audit actually targets
    if len(legacy) < 20:
        pytest.skip("not enough committed legacy events")
    rng = random.Random(7)
    n = fp_new = fp_old = tp_new = 0
    for e in legacy:
        src = byid.get(e.get("source_observation_id"))
        if not src or not src.get("stop_price"):
            continue
        stop, pe = float(src["stop_price"]), float(e["entry_price"])
        k = max(2, round(e["days_since_discovery"] * 5 / 7))
        pre = [stop * 1.05 * (1 + rng.uniform(0.0005, 0.02)) + rng.uniform(0, stop * 0.05) for _ in range(k)]
        pre[-1] = stop * 1.05 * (1 + rng.uniform(0.0005, 0.012))      # bar just before the event sits close to the band edge
        post = [max(stop * 1.01, pe * (1 + rng.uniform(-0.01, 0.01))) for _ in range(8)]
        logged = _series(pre + [pe] + post)
        ev = {**e, "timestamp": logged.index[k].strftime("%Y-%m-%d") + " 00:00:00", "entry_price": round(pe, 2)}
        s2 = {**src, "observation_id": e["source_observation_id"], "timestamp": "2026-08-23 11:49:29"}
        d = rng.uniform(0.002, 0.015)                                  # realistic retroactive dividend adjustment
        res = audit_unversioned_events([s2, ev], fetch=lambda t, dd: logged * (1 - d))[0]
        fp_new += res["status"] == "mismatch"
        # old audit rule = exact date and exact price
        f = _detect_first_event(logged * (1 - d), stop, 5.0)
        fp_old += not (f and f[0] == "invalidation_near_miss" and f[1] == logged.index[k]
                       and round(f[2], 2) == ev["entry_price"])
        # displaced (buggy) event: stored later, at a lower price, still in band
        g = rng.randint(2, 6)
        kk = min(k + g, len(logged) - 1)
        bug_path = logged.copy()
        bug_ev = {**ev, "timestamp": bug_path.index[kk].strftime("%Y-%m-%d") + " 00:00:00",
                  "entry_price": round(float(bug_path.iloc[kk]), 2)}
        bug_path.iloc[k:kk + 1] = [max(stop * 1.012, stop * 1.049 - 0.01 * i * stop) for i in range(kk - k + 1)]
        bug_ev["entry_price"] = round(float(bug_path.iloc[kk]), 2)
        r2 = audit_unversioned_events([s2, bug_ev], fetch=lambda t, dd: bug_path)[0]
        tp_new += r2["status"] == "mismatch"
        n += 1
    assert n >= 20
    print(f"\nSIMULATION n={n}: drift false-positive rate old={fp_old/n:.0%} new={fp_new/n:.0%}; "
          f"argmin-bug detection new={tp_new/n:.0%}")
    assert fp_new / n <= 0.05, "new audit must almost never quarantine adjustment drift"
    assert fp_old / n >= 0.3, "the simulation must actually reproduce the old audit's false positives"
    assert tp_new / n >= 0.95, "the new audit must still catch the old argmin signature"


# ── UI path ───────────────────────────────────────────────────────────────
def _ui_script():
    import modules.invalidation_alpha as _ia
    from ui.alpha_lab import _render_invalidation_alpha
    _render_invalidation_alpha(_ia._TEST_OBS, "20D")


def test_ui_audit_shows_four_way_counts_and_apply_keeps_drift(monkeypatch):
    from streamlit.testing.v1 import AppTest
    obs, fetch = _mixed_fixture()
    saved = {"obs": 0, "q": []}
    monkeypatch.setattr(ia, "_TEST_OBS", obs, raising=False)
    monkeypatch.setattr(ia, "_fetch_price_path", fetch)
    monkeypatch.setattr(ia, "yf", object())
    monkeypatch.setattr(ia, "load_scan_status", lambda: {})
    monkeypatch.setattr(ia, "load_quarantine", lambda: [])
    monkeypatch.setattr(ia, "save_scan_status", lambda s: True)
    monkeypatch.setattr(ia, "save_quarantine", lambda q: (saved["q"].append(copy.deepcopy(q)) or True))
    import modules.alpha_validation as av
    monkeypatch.setattr(av, "save_observations", lambda o: saved.__setitem__("obs", saved["obs"] + 1))
    at = AppTest.from_function(_ui_script, default_timeout=30)
    at.run()
    at.button(key="inval_alpha_audit_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    info = " ".join(i.value for i in at.info)
    assert "verified: 0" in info and "adjustment drift (kept): 1" in info and "mismatch: 1" in info
    at.button(key="inval_alpha_apply_audit_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    ids = {o["observation_id"] for o in obs}
    assert "eD" in ids and "eB" not in ids
    assert saved["q"] and saved["q"][-1][0]["record"]["observation_id"] == "eB"
    assert "1 adjustment-drift stamped" in " ".join(s.value for s in at.success)
