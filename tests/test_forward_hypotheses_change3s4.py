"""
tests/test_forward_hypotheses_change3s4.py — pre-registered forward hypothesis H3

A pre-registration is only worth something if it cannot leak: nothing logged on or before the registration date
may count, the definition must not drift, a verdict must not be given on thin evidence, and the descriptive
contrast must not pretend to be a test. Also checks the panel and that the module does no I/O of its own.
"""
import ast
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import test_research_card_change3 as T
from streamlit.testing.v1 import AppTest

from modules import eval_harness as H
from modules import forward_hypotheses as FH

REG = FH.REGISTERED
H3 = FH.HYPOTHESES[0]


def _date(w, per=20):
    """w-th date, exactly `per` trading days apart, starting after the registration date."""
    return str(np.busday_offset(np.datetime64("2026-10-27"), per * w, holidays=H.NYSE_HOLIDAYS))


def obs(date, excess=None, stage="1 ⏳ Base", score=75.0, model="APEX-9.0", otype=None, fwd=None, key="20D", tk="T"):
    o = {"observation_id": f"{tk}-{date}-{stage[:1]}-{score}", "ticker": tk, "timestamp": date + " 11:00:00",
         "model_version": model, "stage": stage, "apex_score": score, "outcomes": {}}
    if otype:
        o["observation_type"] = otype
    if excess is not None:
        o["outcomes"][key] = {"forward_return_%": excess + 1.0 if fwd is None else fwd, "excess_return_%": excess}
    return o


def windows_of(n_windows, per_window, mean, noise, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [obs(_date(w), float(mean + rng.normal(0, noise)), tk=f"G{w}_{k}", **kw)
            for w in range(n_windows) for k in range(per_window)]


# ── the registration itself ───────────────────────────────────────────────
def test_the_registered_definition_is_fixed():
    assert [h["id"] for h in FH.HYPOTHESES] == ["H3"]
    assert (H3["registered"], H3["horizon"], H3["stage_prefix"], H3["min_score"]) == ("2026-10-26", "20D", "1", 70.0)
    assert "2026-10-26" in H3["text"] and "SPY" in H3["text"] and "20-day" in H3["text"]
    assert "in-sample" in H3["motivation"] and "excluded from the test" in H3["motivation"]
    assert "scepticism" in FH.NOTE and "12" in FH.NOTE and "descriptive only" in FH.NOTE


# ── scope: no leakage from before the registration ────────────────────────
def test_nothing_on_or_before_the_registration_date_counts():
    before = [obs("2026-08-06", 40.0, tk="DOCU"), obs("2026-10-20", 30.0, tk="X"), obs(REG, 25.0, tk="TODAY")]
    after = [obs("2026-10-27", 1.0, tk="NEW")]
    assert not any(FH.in_scope(o, H3) for o in before)
    assert FH.in_scope(after[0], H3)
    r = FH.evaluate_forward(before + after, H3)
    assert r["n_in_scope"] == 1 and r["n_resolved"] == 1 and r["mean_excess_%"] == 1.0
    assert r["first_date"] == "2026-10-27"


def test_a_spectacular_pre_registration_winner_cannot_move_the_result():
    base = windows_of(7, 4, 0.0, 1.0)
    leak = [obs("2026-08-06", 300.0, tk=f"L{i}") for i in range(30)]
    assert FH.evaluate_forward(base + leak, H3) == FH.evaluate_forward(base, H3)


# ── membership ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kw,expected", [
    (dict(score=70.0), True), (dict(score=69.99), False), (dict(score=70.01), True), (dict(score=None), False),
    (dict(stage="2 ✅ Uptrend"), False), (dict(stage="3 ⚠️ Top/Decline"), False), (dict(stage="? Unknown"), False),
    (dict(stage="1 ⏳ Base", score=100.0), True),
])
def test_membership_rule(kw, expected):
    o = obs("2026-11-02", 1.0, **kw)
    assert FH.matches(o, H3) is expected


@pytest.mark.parametrize("kw", [dict(otype="invalidation_breach"), dict(otype="invalidation_near_miss"),
                                dict(otype="apex10_radar"), dict(otype="research_card"), dict(model="APEX-9.1"),
                                dict(model=None)])
def test_only_apex_9_discoveries_are_in_scope(kw):
    assert not FH.in_scope(obs("2026-11-02", 1.0, **kw), H3)
    assert FH.evaluate_forward([obs("2026-11-02", 1.0, **kw)], H3)["n_in_scope"] == 0


def test_a_discovery_without_an_explicit_type_counts_as_a_discovery():
    assert FH.in_scope(obs("2026-11-02", 1.0), H3) and "observation_type" not in obs("2026-11-02", 1.0)


# ── resolved outcomes only, 20-day excess only ────────────────────────────
def test_unresolved_and_other_horizons_are_ignored_but_counted_as_logged():
    rows = [obs("2026-11-02", None, tk="U"), obs("2026-11-03", 5.0, key="5D", tk="F"), obs("2026-11-04", 2.0, tk="R")]
    r = FH.evaluate_forward(rows, H3)
    assert r["n_in_scope"] == 3 and r["n_resolved"] == 1 and r["mean_excess_%"] == 2.0


def test_it_uses_excess_return_not_raw_return():
    r = FH.evaluate_forward([obs("2026-11-02", 2.0, fwd=50.0)], H3)
    assert r["mean_excess_%"] == 2.0


# ── the verdict policy ────────────────────────────────────────────────────
def _v(rows):
    return FH.evaluate_forward(rows, H3)


def test_verdict_policy_follows_the_harness_rules():
    assert _v([]) ["verdict"].startswith("Too few resolved observations (0)")
    assert _v(windows_of(2, 3, 5.0, 1.0))["verdict"].startswith("Too few resolved observations (6)")
    few = _v(windows_of(3, 4, 6.0, 1.0))                                   # 12 observations, 3 independent windows
    assert few["windows"] == 3 and few["ci"] is None and few["verdict"].startswith("Insufficient independent windows (3)")
    win = _v(windows_of(12, 4, 4.0, 2.0))
    assert win["windows"] == 12 and win["ci"][0] > 0 and win["verdict"] == "Outperformed SPY (adequate, 12 windows)"
    assert _v(windows_of(12, 4, -4.0, 2.0))["verdict"] == "Underperformed SPY (adequate, 12 windows)"
    assert _v(windows_of(12, 4, 0.0, 6.0, seed=3))["verdict"].startswith("No detectable difference from SPY")
    assert _v(windows_of(7, 4, 6.0, 1.0))["verdict"] == "Outperformed SPY (exploratory, 7 windows)"


def test_windows_count_independent_periods_not_dates():
    same_window = [obs(str(np.busday_offset(np.datetime64("2026-10-27"), i, holidays=H.NYSE_HOLIDAYS)), 3.0, tk=f"S{i}")
                   for i in range(15)]                                      # 15 dates inside one 20-day window
    r = _v(same_window)
    assert r["n_resolved"] == 15 and r["windows"] == 1
    assert r["ci"] is None and r["verdict"].startswith("Insufficient independent windows (1)")


# ── the descriptive contrast ──────────────────────────────────────────────
def test_contrast_uses_other_discoveries_from_the_same_period_only_and_has_no_interval():
    grp = windows_of(8, 3, 5.0, 0.5, seed=1)
    others = windows_of(8, 5, -1.0, 0.5, seed=2, stage="2 ✅ Uptrend")
    old_others = [obs("2026-09-01", -90.0, stage="2 ✅ Uptrend", tk=f"O{i}") for i in range(20)]   # pre-registration
    low_score = [obs(_date(1), -3.0, score=50.0, tk=f"LS{i}") for i in range(4)]                       # not a member
    r = _v(grp + others + old_others + low_score)
    o_expected = np.mean([o["outcomes"]["20D"]["excess_return_%"] for o in others + low_score])
    assert r["others_n"] == len(others) + len(low_score)
    assert r["others_mean_excess_%"] == pytest.approx(o_expected, abs=0.006)
    assert r["difference_%"] == pytest.approx(r["mean_excess_%"] - r["others_mean_excess_%"], abs=0.011)
    assert not any(k.startswith("others_ci") or k == "difference_ci" for k in r), "the contrast is descriptive: no interval"


def test_no_difference_is_reported_without_both_sides():
    assert _v(windows_of(7, 3, 2.0, 1.0))["difference_%"] is None            # no 'others' at all
    assert _v([obs("2026-11-02", 1.0, stage="2 ✅ Uptrend")])["difference_%"] is None


# ── hygiene ───────────────────────────────────────────────────────────────
def test_evaluation_never_mutates_its_input():
    rows = windows_of(7, 3, 2.0, 1.0) + windows_of(2, 2, 0.0, 1.0, stage="2 ✅ Uptrend")
    snap = copy.deepcopy(rows)
    FH.evaluate_all(rows)
    assert rows == snap


def test_the_module_does_no_io_of_its_own():
    tree = ast.parse(Path(FH.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not (imported & {"requests", "os", "pathlib", "json", "subprocess", "urllib", "streamlit", "yfinance"}), imported
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not (names & {"open", "write", "write_text", "read_text", "urlopen", "post", "request"}), names


# ── real committed data ───────────────────────────────────────────────────
@pytest.mark.skipif(not (REPO / "data" / "alpha_observations.json").exists(), reason="no committed store")
def test_real_data_counts_only_forward_matching_discoveries():
    real = json.load(open(REPO / "data" / "alpha_observations.json"))
    snap = copy.deepcopy(real)
    r = FH.evaluate_all(real)[0]
    expect = [o for o in real if o.get("observation_type", "discovery") == "discovery" and o.get("model_version") == "APEX-9.0"
              and str(o["timestamp"])[:10] > REG and str(o.get("stage", "")).startswith("1")
              and o.get("apex_score") is not None and o["apex_score"] >= 70]
    assert r["n_in_scope"] == len(expect)
    assert r["first_date"] is None or r["first_date"] > REG
    in_sample = [o for o in real if o.get("model_version") == "APEX-9.0" and str(o["timestamp"])[:10] == "2026-08-06"]
    assert in_sample == [] or r["n_in_scope"] < len(in_sample), "the in-sample scan can never be what is being tested"
    assert real == snap
    assert r["verdict"].split(" (")[0] in ("Too few resolved observations", "Insufficient independent windows",
                                          "Outperformed SPY", "Underperformed SPY", "No detectable difference from SPY")


# ── panel ─────────────────────────────────────────────────────────────────
def _full(date, excess, stage="1 ⏳ Base", score=75.0, tk="T"):
    return {"observation_id": f"{tk}{date}", "observation_type": "discovery", "model_version": "APEX-9.0",
            "timestamp": date + " 11:00:00", "ticker": tk, "setup_id": "S1-BASE", "stage": stage, "apex_score": score,
            "apex_score_raw": score, "technical_features": {},
            "outcomes": {"20D": {"forward_return_%": excess + 1.0, "excess_return_%": excess, "benchmark_return_%": 1.0,
                                 "mfe_%": 2.0, "mae_%": -3.0}}}


def _script():
    import modules.forward_hypotheses as _fh
    from ui.alpha_lab import _render_overview
    _render_overview(_fh._TEST_OBS, "20D")


def _run(monkeypatch, rows):
    monkeypatch.setattr(FH, "_TEST_OBS", rows, raising=False)
    at = AppTest.from_function(_script, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _texts(at):
    out = [e.value for coll in (at.markdown, at.caption, at.info, at.warning, at.success) for e in coll]
    out += [d.value.to_string() for d in at.dataframe]
    return out


def test_panel_shows_the_registration_and_withholds_the_interval_with_few_windows(monkeypatch):
    rows = [_full("2026-08-06", 40.0, tk="OLD")] + [_full(_date(w), 3.0, tk=f"N{w}") for w in range(3) for _ in range(1)]
    rows += [_full(_date(w), 3.0 + k * 0.1, tk=f"M{w}{k}") for w in range(3) for k in range(3)]
    at = _run(monkeypatch, rows)
    assert any("Pre-registered forward hypotheses" in e.label for e in at.expander)
    t = " ".join(_texts(at))
    assert "H3" in t and "Registered 2026-10-26" in t and "excluded from the test" in t
    assert "withheld" in t and "Insufficient independent windows" in t and "descriptive only" in t
    assert "Robust — a well-supported sample" not in t


def test_panel_reports_an_interval_only_when_the_windows_exist(monkeypatch):
    rows = [_full(_date(w), 4.0 + np.sin(w + k), tk=f"G{w}{k}") for w in range(12) for k in range(4)]
    at = _run(monkeypatch, rows)
    df = next(d.value for d in at.dataframe if "Mean excess vs SPY" in d.value.columns)
    assert "withheld" not in df["95% CI (cluster-robust)"].iloc[0]
    assert df["Independent windows"].iloc[0] == 12 and df["Verdict"].iloc[0].startswith("Outperformed SPY (adequate")


def test_nothing_in_the_panel_uses_recommendation_wording(monkeypatch):
    rows = [_full(_date(w), 3.0, tk=f"G{w}{k}") for w in range(8) for k in range(3)]
    for s in _texts(_run(monkeypatch, rows)):
        assert not T.BANNED.search(s), s[:150]


def test_the_overview_calls_the_panel_last():
    tree = ast.parse((REPO / "ui" / "alpha_lab.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_render_overview")
    last = fn.body[-1]
    assert isinstance(last, ast.Expr) and ast.unparse(last.value) == "_render_forward_hypotheses(observations)"
