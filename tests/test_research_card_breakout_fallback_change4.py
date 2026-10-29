"""
tests/test_research_card_breakout_fallback_change4.py — Change 4: "price is back below the broken level"

TEXT ONLY. A breakout stays "recent" for BREAKOUT_LOOKBACK_BARS bars after the bar that closed above a zone, but price
can fall back under that level meanwhile (DOCU: breakout above 84.36 on 10-22, close 82.04 on 10-27) and the card said
nothing. The card now adds ONE sentence to `why` (breakout_occurred only) and ONE line to `evidence_against`.

What must not change: any label, status, level, stop, target, reward-to-risk, sizing figure, volume or hypothesis
bucket. The invariance tests below do not need a git tag (the older baseline-tag tests skip in a checkout without
one): they switch the single helper `below_breakout_level` off and require that the ONLY differences between that
card and the real card are the two additions, and that they appear exactly when the close is at or below the level.
Switching the helper off reproduces the pre-change card; that was verified once against the production module over
3,006 series (identical apart from the two additions).
"""
import copy
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import test_research_card_change3 as T
import test_research_card_ui_change3s2 as TU
from modules import research_card as R
from modules import research_card_log as RL
from ui import research_card as U


@pytest.fixture(autouse=True)
def _fresh_cache():
    """st.cache_data survives across AppTest instances: without this a panel test reads the previous test's history."""
    U._load_history.clear()
    yield
    U._load_history.clear()


FELL = np.array([116, 118, 120, 122, 124, 126, 128, 129, 127, 125, 123], float)   # crosses 126.378 on 128, ends at 123
HELD = np.array([116, 118, 120, 122, 124, 126, 128, 129, 130, 131, 132], float)   # crosses and stays above


def _series(tail, vol_spike=False, end=None):
    close = np.concatenate([T.zigzag(T.UP[:-1]), tail])
    vol = np.full(len(close), 1e6)
    if vol_spike:
        vol[-5:] = 2.5e6
    return T.make_hist(close, vol=vol, end=end) if end else T.make_hist(close, vol=vol)


MULTI_TOUCH = [100, 120, 108, 126, 114, 126.6, 115.5]            # two swing highs (126.38 and 126.98) form ONE resistance zone
MT_FELL = np.array([117, 119, 121, 123, 125, 127.5, 129, 128, 126, 124, 122], float)
MT_HELD = np.array([117, 119, 121, 123, 125, 127.5, 129, 130, 131, 132], float)


def _multi_touch(tail):
    return T.make_hist(np.concatenate([T.zigzag(MULTI_TOUCH), tail]))


def _tail_series(rng):
    """UP-like head, then a random 5-11 bar tail: many end above the broken level, many end back under it."""
    scale = rng.uniform(0.3, 3.0)
    head = T.zigzag(T.UP[:-1]) * scale
    tail = np.maximum(114 * scale + np.cumsum(rng.normal(0.9, 2.2, int(rng.integers(5, 12))) * scale), 1.0)
    close = np.concatenate([head, tail])
    return T.make_hist(close, vol=np.where(rng.random(len(close)) < 0.15, 2.5e6, 1e6), end="2026-10-16")


# ── the helper ────────────────────────────────────────────────────────────
def test_helper_returns_nothing_unless_the_close_is_at_or_below_the_level():
    f = R.below_breakout_level
    assert f(None, 100.0) is None and f({}, 100.0) is None
    assert f({"zone_high": 100.0, "date": "d"}, 100.01) is None
    b = f({"zone_high": 100.0, "date": "2026-10-22"}, 97.3)
    assert b["word"] == "back below" and b["level"] == 100.0 and b["close"] == 97.3
    assert b["pct_under"] == pytest.approx(2.7) and b["tail"] == ", 2.7% under it" and b["date"] == "2026-10-22"
    at = f({"zone_high": 100.0}, 100.0)
    assert at["word"] == "back at" and at["tail"] == ", no longer above it" and at["pct_under"] == 0
    assert f({"zone_high": 100.0}, 99.97)["tail"] == ", less than 0.1% under it"


@pytest.mark.parametrize("bad", [None, "x", float("nan"), float("inf"), -5.0, 0.0])
def test_helper_survives_junk_instead_of_raising(bad):
    assert R.below_breakout_level({"zone_high": bad}, 100.0) is None
    assert R.below_breakout_level({"zone_high": 100.0}, bad) is None


def test_helper_compares_at_the_cards_own_four_decimal_precision():
    # the card's price field is rounded to 4 dp; why and evidence must never disagree about "above"
    raw_close, level = 126.37800004, 126.378
    assert R.below_breakout_level({"zone_high": level}, raw_close) is not None
    assert R.below_breakout_level({"zone_high": level}, round(raw_close, 4)) is not None


# ── the reported case ─────────────────────────────────────────────────────
@pytest.mark.parametrize("spike", [False, True])
def test_a_breakout_that_fell_back_says_so_in_why_and_in_the_evidence_against(spike):
    c = T.card(_series(FELL, spike))
    b, close = c["levels"]["recent_breakout"], c["price"]["close"]
    assert c["setup"]["label"] == "breakout_occurred" and close < b["zone_high"]
    why = c["setup"]["why"]
    assert why.startswith("Rule: recent breakout in an uptrend structure.")
    pct = (1 - close / round(b["zone_high"], 4)) * 100
    assert f"The latest close ({close:.2f}) is back below that level ({b['zone_high']:.2f}), {pct:.1f}% under it." in why
    assert why.endswith(".") and len(why) < 420
    line = [e for e in c["evidence_against"] if "breakout level" in e]
    assert line == [f"The latest close ({close:.2f}) is back below the breakout level ({b['zone_high']:.2f}), "
                    f"{pct:.1f}% under it; the breakout bar was {b['date']}."]


def test_the_level_is_the_TOP_of_the_zone_the_close_crossed_not_its_low():
    """On a one-swing zone low == high, so the cases above cannot tell them apart. This zone has two touches."""
    assert R.below_breakout_level({"zone_low": 120.0, "zone_high": 126.0}, 123.0)["level"] == 126.0
    c = T.card(_multi_touch(MT_FELL))
    b, close = c["levels"]["recent_breakout"], c["price"]["close"]
    assert c["setup"]["label"] == "breakout_occurred" and b["zone_low"] < b["zone_high"] and b["zone_touches"] == 2
    assert close < b["zone_low"] < b["zone_high"]                        # fell below the whole zone, so low vs high is visible
    assert f"is back below that level ({b['zone_high']:.2f})" in c["setup"]["why"]
    assert f"({b['zone_low']:.2f})" not in c["setup"]["why"]
    line = next(e for e in c["evidence_against"] if "breakout level" in e)
    assert f"the breakout level ({b['zone_high']:.2f})" in line and f"({b['zone_low']:.2f})" not in line
    held = T.card(_multi_touch(MT_HELD))
    assert held["setup"]["label"] == "breakout_occurred" and "latest close" not in held["setup"]["why"]


def test_a_breakout_that_holds_gets_no_new_text():
    c = T.card(_series(HELD))
    assert c["setup"]["label"] == "breakout_occurred" and c["price"]["close"] > c["levels"]["recent_breakout"]["zone_high"]
    assert "latest close" not in c["setup"]["why"]
    assert not [e for e in c["evidence_against"] if "breakout level" in e]
    held_on = T.card(T._breakout_series(True))                       # the original fixture: still above the level
    assert "latest close" not in held_on["setup"]["why"]


def test_cards_with_no_breakout_or_a_breakout_still_above_its_level_are_untouched():
    quiet = 0
    for pts in (T.UP + [121], T.UP + [124.5], [140, 120, 132, 112, 124, 104, 112]):
        c = T.card(T.zigzag(pts))
        b = c["levels"]["recent_breakout"]
        assert b is None or c["price"]["close"] > b["zone_high"]         # the third fixture: a breakout that is still above
        assert "latest close" not in c["setup"]["why"] and not [e for e in c["evidence_against"] if "breakout level" in e]
        quiet += b is None
    assert quiet >= 2                                                    # both kinds are really exercised


# ── what must NOT change ──────────────────────────────────────────────────
@pytest.mark.parametrize("spike,status", [(False, "research_candidate"), (True, "trigger_met")])
def test_label_status_levels_and_plan_are_exactly_as_before(spike, status):
    """Pins current behaviour for the fallen-back breakout. The trigger_met wording ("on the last complete bar") with a
    close back under the level is a separate, reported item; if it is ever changed, this pin changes with it."""
    c = T.card(_series(FELL, spike))
    p = c["plan"]
    assert c["setup"]["label"] == "breakout_occurred" and p["status"] == status
    assert p["entry_ref"] == c["price"]["close"] == 123.0 and p["entry_ref_basis"] == "last complete close"
    assert p["entry_condition"].startswith("Breakout above 126.38 closed on ")
    assert p["invalidation"] is not None and p["sizing"] is not None
    assert c["levels"]["recent_breakout"]["bars_ago"] == 4 and c["levels"]["recent_breakout"]["volume_confirmed"] is spike


def _flatten_without_the_additions(card):
    c = copy.deepcopy(card)
    if c.get("setup"):
        c["setup"].pop("why", None)
    c["evidence_against"] = []
    return c


def test_with_the_helper_off_the_only_differences_are_the_two_additions(monkeypatch):
    rng = np.random.default_rng(9090)
    series = [_series(FELL, False), _series(FELL, True), _series(HELD), T._breakout_series(True), T._breakout_series(False),
              _multi_touch(MT_FELL), _multi_touch(MT_HELD),
              T.make_hist(T.zigzag(T.UP + [121])), T.make_hist(T.zigzag(T.UP)[:50])]
    series += [T._fuzz_series(rng) for _ in range(500)] + [_tail_series(rng) for _ in range(600)]
    real, off = [], []
    for h in series:
        real.append(R.build_card("X", h, now=T.NOW, observed_at="t"))
    with monkeypatch.context() as m:
        m.setattr(R, "below_breakout_level", lambda breakout, close: None)
        for h in series:
            off.append(R.build_card("X", h, now=T.NOW, observed_at="t"))
    n_why = n_ev = 0
    labels_hit = set()
    for new, old in zip(real, off):
        assert _flatten_without_the_additions(new) == _flatten_without_the_additions(old)        # labels, status, plan, levels...
        b = (new["levels"] or {}).get("recent_breakout")
        under = bool(b and new["price"]["close"] <= round(b["zone_high"], 4))
        extra = [e for e in new["evidence_against"] if e not in old["evidence_against"]]
        assert [e for e in new["evidence_against"] if e in old["evidence_against"]] == old["evidence_against"]
        assert len(extra) == (1 if under else 0), (under, extra)
        n_ev += len(extra)
        if extra:
            assert "breakout level" in extra[0] and not T.BANNED.search(extra[0])
            labels_hit.add(new["setup"]["label"])
        wn, wo = (new["setup"] or {}).get("why"), (old["setup"] or {}).get("why")
        if wn != wo:
            n_why += 1
            assert new["setup"]["label"] == "breakout_occurred" and under
            assert wn.startswith(wo + " The latest close (") and wn.endswith(".") and len(wn) < 420
            assert not T.BANNED.search(wn)
        elif new["setup"] and new["setup"]["label"] == "breakout_occurred":
            assert not under                                                    # every fallen-back breakout got its sentence
        json.dumps(new, allow_nan=False)
    assert n_why >= 15 and n_ev >= n_why, (n_why, n_ev)
    assert "breakout_occurred" in labels_hit


def test_the_why_sentence_and_the_evidence_line_always_appear_together_for_a_breakout_card():
    rng = np.random.default_rng(5151)
    seen = 0
    for _ in range(600):
        c = R.build_card("X", _tail_series(rng), now=T.NOW)
        if not c["setup"] or c["setup"]["label"] != "breakout_occurred":
            continue
        has_why = "The latest close (" in c["setup"]["why"]
        has_line = any("breakout level" in e for e in c["evidence_against"])
        assert has_why == has_line
        seen += has_why
    assert seen >= 10


def test_new_wording_respects_the_wording_policy_everywhere_it_can_appear():
    rng = np.random.default_rng(6262)
    cards = [T.card(_series(FELL, s)) for s in (False, True)] + [R.build_card("X", _tail_series(rng), now=T.NOW) for _ in range(300)]
    for c in cards:
        if not c["setup"]:
            continue
        for s in [c["setup"]["why"]] + c["evidence_against"]:
            assert not T.BANNED.search(s), s
        for path, v in T._walk(c):
            assert path.split("/")[-1].split("[")[0] not in T.BANNED_KEYS


# ── the log and the panel ─────────────────────────────────────────────────
def test_the_log_record_keeps_the_text_and_the_label_bucket_is_unchanged():
    c = T.card(_series(FELL, True))
    rec = RL.make_record(c)
    assert rec["setup_id"] == "breakout_occurred" and rec["status"] == "trigger_met" and rec["card_version"] == "CARD-1.0"
    assert "The latest close (" in rec["card"]["setup_why"]
    assert any("breakout level" in e for e in rec["card"]["evidence_against"])
    json.dumps(rec, allow_nan=False)


def test_the_panel_shows_the_sentence_under_why_and_the_line_under_contradicting(monkeypatch):
    at = TU._analyse(TU._app(monkeypatch, _series(FELL, False, end=TU.END)))
    TU._no_exc(at)
    texts = TU._texts(at)
    why = next(t for t in texts if t.startswith("**Why:**"))
    assert "back below that level (126.38), 2.7% under it." in why
    assert any(t.startswith("- The latest close (123.00) is back below the breakout level (126.38)") for t in texts)
    joined = " ".join(texts)
    assert "Recent breakout" in joined and not T.BANNED.search(why)


def test_the_panel_stays_quiet_when_the_breakout_holds(monkeypatch):
    at = TU._analyse(TU._app(monkeypatch, _series(HELD, False, end=TU.END)))
    TU._no_exc(at)
    assert not [t for t in TU._texts(at) if "latest close" in t or "breakout level" in t]
