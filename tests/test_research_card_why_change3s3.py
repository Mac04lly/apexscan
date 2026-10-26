"""
tests/test_research_card_why_change3s3.py — Change 3, follow-up: "Why this description"

TEXT ONLY. The description rules must be exactly what they were; the card just now says which rule fired and with
what numbers. So the most important tests compare against the PREVIOUS version of the module (loaded from git):
every card, level, plan and status must be identical except for the new `why` field, on hand-built series, on
hundreds of fuzzed series, and across 5,000 random direct calls to the rule function. The remaining tests check
that each explanation is true to the card's own numbers, that nothing rendered uses recommendation wording, and
that the UI shows it.
"""
import copy
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import test_research_card_change3 as T
import test_research_card_ui_change3s2 as TU
from modules import research_card as R
from modules import research_card_log as RL


def _old_module():
    r = subprocess.run(["git", "show", "baseline-pre-why:modules/research_card.py"], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("baseline tag not available in this checkout")
    mod = types.ModuleType("old_research_card")
    exec(compile(r.stdout, "old_research_card.py", "exec"), mod.__dict__)
    return mod


def _strip_why(card):
    c = copy.deepcopy(card)
    if c.get("setup"):
        c["setup"].pop("why", None)
    return c


def _olli_like():
    """Higher highs and higher lows, but the close sits below the 200-day SMA and the 50-day EMA below it too."""
    decline = np.linspace(150, 70, 260)
    legs = T.zigzag([70, 78, 73, 82, 77, 85], lead=0)
    return np.concatenate([decline, legs])


# ── nothing else changed ──────────────────────────────────────────────────
def test_every_card_is_identical_to_the_previous_version_except_for_the_new_field():
    old = _old_module()
    series = [T.make_hist(T.zigzag(p), end="2026-10-16") for p in
              (T.UP + [121], T.UP + [124.5], [140, 120, 132, 112, 124, 104, 112], T.UP + [133, 134, 135, 136, 137, 138, 139, 140])]
    series += [T._breakout_series(True), T._breakout_series(False), T.make_hist(_olli_like(), end="2026-10-16"),
               T.make_hist(T.zigzag(T.UP)[:50])]
    rng = np.random.default_rng(77)
    series += [T._fuzz_series(rng) for _ in range(400)]
    n_setup = 0
    for h in series:
        new_card = R.build_card("X", h, now=T.NOW, observed_at="t")
        old_card = old.build_card("X", h, now=T.NOW, observed_at="t")
        assert _strip_why(new_card) == _strip_why(old_card)
        n_setup += bool(new_card["setup"])
    assert n_setup > 150, "the comparison must cover many real descriptions, not just refusals"


def test_the_rule_function_returns_the_same_label_for_5000_random_inputs():
    old = _old_module()
    rng = np.random.default_rng(5)
    hits = {}
    for _ in range(5000):
        st = rng.choice(["uptrend", "downtrend", "mixed", "insufficient"])
        sh = {"price": float(rng.uniform(60, 160)), "date": "2026-01-02"} if rng.random() < 0.85 else None
        sl = {"price": float(rng.uniform(50, 140)), "date": "2026-01-02"} if rng.random() < 0.85 else None
        structure = {"structure": st, "last_swing_high": sh, "last_swing_low": sl}
        close, atr = float(rng.uniform(50, 160)), float(rng.uniform(0.3, 6))
        zones = [{"low": (lo := float(rng.uniform(50, 170))), "high": lo + float(rng.uniform(0, 3)), "touches": int(rng.integers(1, 4))}
                 for _ in range(int(rng.integers(0, 4)))]
        sma200 = float(rng.uniform(60, 160)) if rng.random() < 0.7 else None
        ema50 = float(rng.uniform(60, 160)) if rng.random() < 0.7 else None
        bo = {"zone_high": float(rng.uniform(50, 160)), "date": "d", "bars_ago": 2} if rng.random() < 0.15 else None
        a = old.interpret_setup(structure, close, atr, zones, sma200, ema50, bo)
        b = R.interpret_setup(structure, close, atr, zones, sma200, ema50, bo)
        assert a["label"] == b["label"]
        assert {k: v for k, v in b.items() if k != "why"} == a
        hits[b["label"]] = hits.get(b["label"], 0) + 1
    assert set(hits) == set(R.LABEL_TEXT), "every description must be exercised"
    assert min(hits.values()) >= 30, hits


# ── the explanation is true to the card's own numbers ─────────────────────
def test_the_olli_case_explains_why_hh_hl_swings_still_give_a_weak_description():
    c = T.card(T.make_hist(_olli_like(), end="2026-10-16"))
    assert c["structure"]["structure"] == "uptrend"                          # fixture really has HH/HL
    close, sma200, ema50 = c["price"]["close"], c["trend"]["sma200"]["value"], c["trend"]["ema50"]["value"]
    assert close < sma200 and ema50 < sma200
    assert c["setup"]["label"] == "weak_or_deteriorating"
    why = c["setup"]["why"]
    assert why.startswith("Rule: below the long-term trend")
    assert f"{close:.2f}" in why and f"{sma200:.2f}" in why and f"{ema50:.2f}" in why
    assert "HH/HL" in why and "whatever the swing pattern is" in why and "bright-line" in why
    assert f"{abs(close / sma200 - 1) * 100:.1f}%" in why
    assert any("HH/HL" in e for e in c["evidence_for"]), "the card still lists the supporting evidence the rule overrides"


def test_downtrend_explanation_cites_the_swing_pattern():
    c = T.card(T.zigzag([140, 120, 132, 112, 124, 104, 112]))
    assert c["setup"]["label"] == "weak_or_deteriorating" and "LH/LL" in c["setup"]["why"]
    assert c["setup"]["why"].startswith("Rule: downtrend structure")


def test_pullback_explanation_matches_the_cards_swings_and_close():
    c = T.card(T.zigzag(T.UP + [121]))
    why, close = c["setup"]["why"], c["price"]["close"]
    m = re.search(r"close is ([\d.]+) \(([\d.]+) ATR\) below the last swing high \(([\d.]+)\), at least 1 ATR, and still above "
                  r"the last swing low \(([\d.]+)\)", why)
    assert m, why
    d, atr_mult, sh, sl = (float(x) for x in m.groups())
    swings = c["structure"]["swings"]
    last_high = [s for s in swings if s["type"] == "high"][-1]["price"]
    last_low = [s for s in swings if s["type"] == "low"][-1]["price"]
    assert sh == pytest.approx(last_high, abs=0.011) and sl == pytest.approx(last_low, abs=0.011)
    assert d == pytest.approx(sh - close, abs=0.02) and atr_mult >= 1.0 and close > sl


def test_breakout_candidate_explanation_matches_the_zone_and_the_atr_window():
    c = T.card(T.zigzag(T.UP + [124.5]))
    m = re.search(r"starts at ([\d.]+), ([\d.]+) \(([\d.]+) ATR\) above the close, inside the 1.5-ATR window", c["setup"]["why"])
    assert m, c["setup"]["why"]
    zone_low, d, k = (float(x) for x in m.groups())
    assert d == pytest.approx(zone_low - c["price"]["close"], abs=0.02) and k <= 1.5
    assert zone_low == pytest.approx(c["levels"]["resistances"][0]["low"], abs=0.011)


def test_breakout_occurred_explanation_matches_the_detected_breakout():
    c = T.card(T._breakout_series(True))
    b = c["levels"]["recent_breakout"]
    why = c["setup"]["why"]
    assert why.startswith("Rule: recent breakout") and f"{b['zone_high']:.2f}" in why and b["date"] in why
    assert f"({b['bars_ago']} bar(s) ago)" in why


def test_trend_extended_says_which_reason_applies():
    st = {"structure": "uptrend", "last_swing_high": {"price": 130.0}, "last_swing_low": {"price": 110.0}}
    far = R.interpret_setup(st, 129.5, 2, [{"low": 140, "high": 141, "touches": 2}], None, None, None)
    assert far["label"] == "trend_extended" and "nearest resistance is" in far["why"] and "needs 1.5 or less" in far["why"]
    none_ = R.interpret_setup(st, 129.5, 2, [], None, None, None)
    assert "No resistance zone lies above the close" in none_["why"]
    broken = R.interpret_setup(st, 105, 2, [{"low": 140, "high": 141, "touches": 2}], None, None, None)
    assert "at or below the last swing low (110.00)" in broken["why"]


def test_unclassified_explanation_names_the_pattern():
    mixed = R.interpret_setup({"structure": "mixed", "last_swing_high": None, "last_swing_low": None}, 100, 2, [], None, None, None)
    assert mixed["label"] == "unclassified" and "do not move together" in mixed["why"]
    few = R.interpret_setup({"structure": "insufficient", "last_swing_high": None, "last_swing_low": None}, 100, 2, [], None, None, None)
    assert "fewer than two confirmed swing highs and lows" in few["why"]


def test_every_card_with_a_description_explains_it_consistently():
    rng = np.random.default_rng(31)
    seen = set()
    for _ in range(500):
        c = R.build_card("FZ", T._fuzz_series(rng), now=T.NOW)
        if not c["setup"]:
            continue
        why, label, structure = c["setup"]["why"], c["setup"]["label"], c["structure"]["structure"]
        seen.add(label)
        assert why.startswith("Rule: ") and len(why) < 420 and why.endswith(".")
        if label == "weak_or_deteriorating":
            assert ("LH/LL" in why) if structure == "downtrend" else ("200-day SMA" in why and "50-day EMA" in why)
        if label == "breakout_candidate":
            assert "ATR" in why and "resistance zone" in why
        if label == "trend_pullback":
            assert "last swing high" in why and "last swing low" in why
        if label == "breakout_occurred":
            assert "breakout" in why
        json.dumps(c, allow_nan=False)
        assert not T.BANNED.search(why), why
    assert {"weak_or_deteriorating", "trend_extended", "unclassified"} <= seen


def test_the_explanation_cannot_crash_on_a_partial_breakout_record():
    st = {"structure": "uptrend", "last_swing_high": None, "last_swing_low": None}
    out = R.interpret_setup(st, 100, 2, [], None, None, {"zone_high": 99.0})
    assert out["label"] == "breakout_occurred" and "99.00" in out["why"]


# ── log and UI ────────────────────────────────────────────────────────────
def test_the_log_keeps_the_explanation_with_the_card():
    c = T.card(T.make_hist(_olli_like(), end="2026-10-16"))
    rec = RL.make_record(c)
    assert rec["card"]["setup_why"] == c["setup"]["why"] and rec["setup_id"] == "weak_or_deteriorating"
    json.dumps(rec, allow_nan=False)


def test_the_panel_shows_the_why_line_for_the_olli_case(monkeypatch):
    at = TU._analyse(TU._app(monkeypatch, T.make_hist(_olli_like(), end=TU.END)))
    TU._no_exc(at)
    texts = TU._texts(at)
    joined = " ".join(texts)
    assert "NO LONG SETUP DEFINED" in joined and "Weak or deteriorating structure" in joined
    why = next(t for t in texts if t.startswith("**Why:**"))
    assert "200-day SMA" in why and "bright-line" in why and "HH/HL" in why
    assert not T.BANNED.search(why)
    assert any("HH/HL" in t for t in texts), "the supporting evidence is still listed alongside the explanation"


def test_every_description_shows_a_why_line_in_the_panel(monkeypatch):
    for pts, tk in ((T.UP + [124.5], "AAA"), (T.UP + [121], "BBB"), ([140, 120, 132, 112, 124, 104, 112], "CCC")):
        at = TU._analyse(TU._app(monkeypatch, TU._hist(pts)), tk)
        TU._no_exc(at)
        assert any(t.startswith("**Why:** Rule: ") for t in TU._texts(at)), tk
