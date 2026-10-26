"""
tests/test_research_card_log_change3s2.py — Change 3, Stage 2 (card snapshot log)

Records, append-only behaviour, resolution by the EXISTING outcome engine (unchanged, with a patched
price feed), touch flags, the two pre-registered hypotheses under the Change 2 harness's rules, and the
store: a failed read must NEVER cause an overwrite. No network.
"""
import copy
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_research_card_change3 as T            # shared hand-built series
from modules import research_card as R
from modules import research_card_log as RL
from modules import outcome_engine as OE
from modules import eval_harness as H


def _card(points=None, ticker="AAA", end="2026-10-16", **kw):
    h = T.make_hist(T.zigzag(points or (T.UP + [124.5])), end=end)
    return R.build_card(ticker, h, now=T.NOW, observed_at="2026-10-22 10:00:00 UTC", **kw)


# ── records ───────────────────────────────────────────────────────────────
def test_record_fields_and_level_arithmetic():
    c = _card()
    r = RL.make_record(c, logged_at="2026-10-22 10:00:00 UTC", new_id="id-1")
    assert r["observation_id"] == "id-1" and r["observation_type"] == "research_card"
    assert r["ticker"] == "AAA" and r["market"] == "US" and r["outcomes"] == {}
    assert r["timestamp"] == f"{c['observation_date']} 00:00:00" and r["entry_price"] == c["price"]["close"]
    assert r["setup_id"] == c["setup"]["label"] == "breakout_candidate" and r["status"] == "research_candidate"
    close = r["entry_price"]
    assert r["stop_pct_from_close"] == pytest.approx((r["stop_price"] / close - 1) * 100, abs=1e-3)
    assert r["target_pct_from_close"] == pytest.approx((r["target_price"] / close - 1) * 100, abs=1e-3)
    assert r["trigger_pct_from_close"] == pytest.approx((r["entry_ref"] / close - 1) * 100, abs=1e-3)
    assert r["stop_pct_from_close"] < 0 < r["target_pct_from_close"]
    assert r["logged_at"] == "2026-10-22 10:00:00 UTC" and r["card_version"] == R.CARD_VERSION


def test_records_exist_for_every_description_but_not_for_empty_cards():
    weak = RL.make_record(_card([140, 120, 132, 112, 124, 104, 112]))
    assert weak["setup_id"] == "weak_or_deteriorating" and weak["stop_price"] is None and weak["stop_pct_from_close"] is None
    pull = RL.make_record(_card(T.UP + [121]))
    assert pull["setup_id"] == "trend_pullback" and pull["target_price"] is None and pull["target_pct_from_close"] is None
    thin = R.build_card("X", T.make_hist(np.linspace(100, 120, 30)), now=T.NOW)
    assert RL.make_record(thin) is None
    assert RL.make_record(R.build_card("Z", T._daily(), market="NG", now=T.NOW)) is None


def test_record_is_strict_json_and_compact():
    r = RL.make_record(_card())
    blob = json.dumps(r, allow_nan=False)
    assert len(blob) < 9_000
    assert {"evidence_for", "evidence_against", "missing_data", "plan", "structure"} <= set(r["card"])


def test_compact_card_keeps_what_the_card_said():
    c = _card()
    k = RL.compact_card(c)
    assert k["plan"]["stop"] == c["plan"]["invalidation"]["stop"] and k["plan"]["target"] == c["plan"]["target"]["price"]
    assert k["setup"] == c["setup"]["label"] and k["observation_date"] == c["observation_date"]


# ── append-only ───────────────────────────────────────────────────────────
def test_append_adds_refuses_duplicates_and_never_mutates():
    a = RL.make_record(_card(ticker="AAA"), new_id="1")
    b = RL.make_record(_card(ticker="BBB"), new_id="2")
    log0 = [a]
    snap = copy.deepcopy(log0)
    log1, s1 = RL.append_card(log0, b)
    assert s1 == "added" and [r["ticker"] for r in log1] == ["AAA", "BBB"] and log1 is not log0
    assert log0 == snap, "the input list and its records are never altered"
    dup = RL.make_record(_card(ticker="AAA"), new_id="3")                       # same ticker, same date
    log2, s2 = RL.append_card(log1, dup)
    assert s2 == "duplicate" and log2 == log1 and all(r["observation_id"] != "3" for r in log2)
    later = RL.make_record(_card(ticker="AAA", end="2026-10-19"), new_id="4")   # same ticker, next day
    assert RL.append_card(log1, later)[1] == "added"


# ── engine compatibility (existing engine, patched price feed) ────────────
def _feed(stock_path, bench_path, start):
    idx = pd.bdate_range(start, periods=len(stock_path))
    def mk(p):
        p = np.asarray(p, float)
        return pd.DataFrame({"Open": p, "High": p * 1.01, "Low": p * 0.99, "Close": p, "Volume": 1e6}, index=idx)
    stock, bench = mk(stock_path), mk(bench_path)
    return lambda t, s, e: bench if t == OE._BENCH_TICKER else stock


def test_the_unchanged_engine_resolves_a_card_record_and_freezes_each_horizon(monkeypatch):
    start = (datetime.now().date() - timedelta(days=260))
    rec = RL.make_record(_card(), new_id="e1")
    rec["timestamp"] = f"{start} 00:00:00"
    rec["entry_price"] = 100.0
    rec["stop_pct_from_close"], rec["target_pct_from_close"], rec["trigger_pct_from_close"] = -6.0, 12.0, 2.0
    path = 100 + np.arange(70) * 0.5                                  # +0.5 per bar
    monkeypatch.setattr(OE, "_fetch_ticker_history", _feed(path, 100 + np.arange(70) * 0.1, str(start)))
    before = copy.deepcopy(rec)
    n = RL.resolve_card_outcomes([rec])
    assert n == 1
    o = rec["outcomes"]["5D"]
    assert o["forward_return_%"] == pytest.approx((path[5] / 100 - 1) * 100, abs=1e-2)
    assert o["mfe_%"] == pytest.approx((path[:6].max() * 1.01 / 100 - 1) * 100, abs=1e-2)
    assert o["mae_%"] == pytest.approx((path[:6].min() * 0.99 / 100 - 1) * 100, abs=1e-2)
    assert o["excess_return_%"] == pytest.approx(o["forward_return_%"] - o["benchmark_return_%"], abs=1e-3)
    assert {k: v for k, v in rec.items() if k != "outcomes"} == {k: v for k, v in before.items() if k != "outcomes"}
    frozen = copy.deepcopy(rec["outcomes"])
    monkeypatch.setattr(OE, "_fetch_ticker_history", _feed(path * 3, path * 3, str(start)))   # prices 'change'
    assert RL.resolve_card_outcomes([rec]) == 0
    assert rec["outcomes"] == frozen, "a frozen horizon is never rewritten"


def test_resolve_respects_max_records_and_injected_compute():
    recs = [{"ticker": f"T{i}"} for i in range(10)]
    seen = []
    n = RL.resolve_card_outcomes(recs, max_records=4, compute=lambda r, c: (seen.append(r["ticker"]), True)[1])
    assert n == 4 and seen == ["T0", "T1", "T2", "T3"]


# ── touch flags ───────────────────────────────────────────────────────────
def _rec_with(stop=-5.0, target=10.0, trigger=2.0, mae=-6.0, mfe=11.0):
    return {"stop_pct_from_close": stop, "target_pct_from_close": target, "trigger_pct_from_close": trigger,
            "outcomes": {"20D": {"mae_%": mae, "mfe_%": mfe}}}


def test_touch_flags_follow_the_engines_extremes():
    t = RL.touch_flags(_rec_with(), "20D")
    assert t == {"stop_touched": True, "target_touched": True, "trigger_touched": True}
    t = RL.touch_flags(_rec_with(mae=-4.9, mfe=9.9), "20D")
    assert t == {"stop_touched": False, "target_touched": False, "trigger_touched": True}
    assert RL.touch_flags(_rec_with(mfe=1.0), "20D")["trigger_touched"] is False
    assert RL.touch_flags(_rec_with(trigger=0.0, mfe=0.0), "20D")["trigger_touched"] is True      # trigger already met at the close
    assert RL.touch_flags(_rec_with(), "5D") == {"stop_touched": None, "target_touched": None, "trigger_touched": None}
    none_levels = _rec_with(stop=None, target=None, trigger=None)
    assert RL.touch_flags(none_levels, "20D") == {"stop_touched": None, "target_touched": None, "trigger_touched": None}


# ── pre-registered hypotheses ─────────────────────────────────────────────
def test_hypotheses_are_pre_registered_and_refer_to_real_descriptions():
    ids = [h["id"] for h in RL.HYPOTHESES]
    assert ids == ["H1", "H2"] and all(h["label"] in R.LABEL_TEXT for h in RL.HYPOTHESES)
    assert {h["label"] for h in RL.HYPOTHESES} == {"trend_pullback", "breakout_candidate"}
    assert all(h["horizon"] == "20D" and h["registered"] == "2026-10-22" for h in RL.HYPOTHESES)
    assert "not a random sample" in RL.SELECTION_NOTE and "SPY" in RL.CONTROL_NOTE


def _rows(label, n_windows, per_window, mean, noise, seed=0):
    rng = np.random.default_rng(seed)
    base = np.datetime64("2026-01-05")
    out = []
    for w in range(n_windows):
        d = str(np.busday_offset(base, 20 * w, holidays=H.NYSE_HOLIDAYS))     # exactly 20 TRADING days apart
        for k in range(per_window):
            out.append({"ticker": f"{label[:2]}{w}{k}", "setup_id": label, "timestamp": d + " 00:00:00",
                        "outcomes": {"20D": {"forward_return_%": 0.0, "excess_return_%": float(mean + rng.normal(0, noise))}}})
    return out


def _verdict(records, hid="H1"):
    return next(h for h in RL.evaluate_hypotheses(records) if h["id"] == hid)


def test_hypothesis_verdict_policy_follows_the_harness_rules():
    assert _verdict([])["verdict"].startswith("Too few resolved cards")
    assert _verdict(_rows("trend_pullback", 2, 3, 3.0, 1.0))["verdict"].startswith("Too few resolved cards (6)")
    few_windows = _verdict(_rows("trend_pullback", 3, 4, 5.0, 1.0))                # 12 cards, only 3 independent windows
    assert few_windows["windows"] == 3 and few_windows["ci"] is None
    assert few_windows["verdict"].startswith("Insufficient independent windows (3)")
    win = _verdict(_rows("trend_pullback", 12, 4, 4.0, 2.0))
    assert win["windows"] == 12 and win["ci"][0] > 0
    assert win["verdict"] == "Outperformed SPY (adequate, 12 windows)"
    lose = _verdict(_rows("trend_pullback", 12, 4, -4.0, 2.0))
    assert lose["verdict"] == "Underperformed SPY (adequate, 12 windows)"
    null = _verdict(_rows("trend_pullback", 12, 4, 0.0, 6.0, seed=3))
    assert null["verdict"].startswith("No detectable difference from SPY")
    exploratory = _verdict(_rows("trend_pullback", 7, 4, 6.0, 1.0))
    assert exploratory["verdict"] == "Outperformed SPY (exploratory, 7 windows)"


def test_hypotheses_only_use_their_own_label_and_resolved_horizon():
    recs = _rows("trend_pullback", 12, 4, 4.0, 1.0) + _rows("breakout_candidate", 12, 4, -4.0, 1.0)
    recs += [{"ticker": "U", "setup_id": "trend_pullback", "timestamp": "2026-03-02 00:00:00", "outcomes": {}}]
    h1, h2 = _verdict(recs, "H1"), _verdict(recs, "H2")
    assert h1["n"] == 48 and h1["mean_excess_%"] > 3 and h2["n"] == 48 and h2["mean_excess_%"] < -3
    assert h1["control"].startswith("SPY") and "not distance-matched" in h1["control"]


def test_summarize_log_counts_and_independence():
    recs = _rows("trend_pullback", 3, 4, 1.0, 1.0) + _rows("weak_or_deteriorating", 1, 2, 0.0, 1.0)
    for r in recs:
        r["status"] = "research_candidate"
    s = RL.summarize_log(recs)
    assert s["n_cards"] == 14 and s["by_setup"] == {"trend_pullback": 12, "weak_or_deteriorating": 2}
    assert s["resolved"]["20D"] == 14 and s["resolved"]["5D"] == 0
    assert s["independence_20D"]["label"] == "Insufficient independent windows"
    assert len(s["hypotheses"]) == 2 and "not a random sample" in s["selection_note"]


# ── the store: ok / missing / error ───────────────────────────────────────
class FakeStore:
    def __init__(self, monkeypatch, state="ok", data=None, push_ok=True, creds=("tok", "owner/repo")):
        self.state, self.data, self.push_ok, self.pushed = state, data, push_ok, []
        monkeypatch.setattr(RL, "_creds", lambda: creds)
        monkeypatch.setattr(RL, "_fetch_remote", lambda t, r, p: (self.state, copy.deepcopy(self.data)))
        monkeypatch.setattr(RL, "_push", lambda t, r, p, d, m: (self.pushed.append((p, copy.deepcopy(d), m)), self.push_ok)[1])


def test_append_and_save_appends_to_an_existing_log_and_preserves_it(monkeypatch):
    existing = [RL.make_record(_card(ticker=t), new_id=t) for t in ("AAA", "BBB", "CCC")]
    fs = FakeStore(monkeypatch, "ok", existing)
    res = RL.append_and_save(_card(ticker="DDD"), logged_at="x")
    assert res == {"status": "added", "n": 4}
    path, written, msg = fs.pushed[0]
    assert path == RL.CARD_LOG_PATH == "data/research_cards.json"
    assert written[:3] == existing and written[3]["ticker"] == "DDD" and len(written) == 4
    assert "DDD" in msg


def test_first_ever_write_creates_the_log_when_the_file_is_genuinely_missing(monkeypatch):
    fs = FakeStore(monkeypatch, "missing")
    assert RL.append_and_save(_card())["status"] == "added"
    assert len(fs.pushed) == 1 and len(fs.pushed[0][1]) == 1


def test_a_failed_read_NEVER_overwrites_the_log(monkeypatch):
    fs = FakeStore(monkeypatch, "error")
    assert RL.append_and_save(_card()) == {"status": "read_failed", "n": None}
    assert fs.pushed == [], "no write may happen when the read failed"
    assert RL.resolve_and_save()["status"] == "read_failed" and fs.pushed == []


def test_a_malformed_remote_file_is_never_replaced(monkeypatch):
    fs = FakeStore(monkeypatch, "ok", {"not": "a list"})
    assert RL.append_and_save(_card())["status"] == "read_failed"
    assert RL.resolve_and_save()["status"] == "read_failed"
    assert fs.pushed == []
    assert RL.load_log_state() == ("error", [])


def test_duplicate_save_and_unloggable_cards_do_not_write(monkeypatch):
    c = _card()
    existing = [RL.make_record(c, new_id="x")]
    fs = FakeStore(monkeypatch, "ok", existing)
    assert RL.append_and_save(c) == {"status": "duplicate", "n": 1}
    thin = R.build_card("X", T.make_hist(np.linspace(100, 120, 30)), now=T.NOW)
    assert RL.append_and_save(thin)["status"] == "not_loggable"
    assert fs.pushed == []


def test_missing_credentials_and_push_failure(monkeypatch):
    fs = FakeStore(monkeypatch, "ok", [], creds=("", ""))
    assert RL.append_and_save(_card())["status"] == "no_credentials" and fs.pushed == []
    assert RL.resolve_and_save()["status"] == "no_credentials" and RL.load_log_state() == ("no_credentials", [])
    fs2 = FakeStore(monkeypatch, "ok", [], push_ok=False)
    res = RL.append_and_save(_card())
    assert res["status"] == "save_failed" and res["n"] == 0 and len(fs2.pushed) == 1


def test_resolve_and_save_changes_only_outcomes_and_keeps_the_count(monkeypatch):
    recs = [RL.make_record(_card(ticker=t), new_id=t) for t in ("AAA", "BBB")]
    fs = FakeStore(monkeypatch, "ok", recs)

    def compute(rec, cache):
        rec["outcomes"]["5D"] = {"forward_return_%": 1.0}
        return True
    res = RL.resolve_and_save(compute=compute)
    assert res == {"status": "updated", "updated": 2}
    written = fs.pushed[0][1]
    assert len(written) == 2
    for w, o in zip(written, recs):
        assert {k: v for k, v in w.items() if k != "outcomes"} == {k: v for k, v in o.items() if k != "outcomes"}
        assert w["outcomes"] == {"5D": {"forward_return_%": 1.0}}


def test_resolve_and_save_with_nothing_due_or_empty_does_not_write(monkeypatch):
    fs = FakeStore(monkeypatch, "ok", [RL.make_record(_card(), new_id="a")])
    assert RL.resolve_and_save(compute=lambda r, c: False) == {"status": "nothing_due", "updated": 0}
    FakeStore(monkeypatch, "missing")
    assert RL.resolve_and_save()["status"] == "empty"
    assert fs.pushed == []


def test_load_log_state_maps_every_state(monkeypatch):
    recs = [RL.make_record(_card(), new_id="a")]
    FakeStore(monkeypatch, "ok", recs)
    assert RL.load_log_state() == ("ok", recs)
    FakeStore(monkeypatch, "missing")
    assert RL.load_log_state() == ("missing", [])
    FakeStore(monkeypatch, "error")
    assert RL.load_log_state() == ("error", [])


def test_the_log_never_touches_the_other_stores():
    import ast
    assert RL.CARD_LOG_PATH == "data/research_cards.json"
    tree = ast.parse(Path(RL.__file__).read_text())
    doc = ast.get_docstring(tree, clean=False)
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value != doc and not n.value.startswith("    ")]
    code_strings = " ".join(l for l in literals if len(l) < 200)
    for other in ("alpha_observations", "discoveries.json", "short_discoveries", "invalidation_scan_status"):
        assert other not in code_strings, f"the card log must not reference {other}"
