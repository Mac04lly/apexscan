"""
tests/test_research_card_change3.py — Change 3, Stage 1 (research card module)

Hand-built price series with known swings, so every expectation is derived independently of the
code under test. Covers: indicators against independent references, swing detection (confirmation,
plateau, reaction filter, PREFIX-STABILITY), structure, zones, structural invalidation, targets
("no defensible target"), reward-to-risk, sizing, setup descriptions (positive AND negative),
status, partial-session handling, point-in-time (asof) invariance against garbage future data,
a fuzz run for fabricated levels, the wording policy (no recommendations, no score) and strict
JSON. No network, no UI, nothing written.
"""
import copy
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import research_card as R

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)      # far in the future: every bar is a complete session


# ── builders ──────────────────────────────────────────────────────────────
def make_hist(close, high=None, low=None, vol=None, start="2025-01-02", end=None):
    close = np.asarray(close, float)
    n = len(close)
    high = close * 1.003 if high is None else np.asarray(high, float)
    low = close * 0.997 if low is None else np.asarray(low, float)
    vol = np.full(n, 1e6) if vol is None else np.asarray(vol, float)
    idx = pd.bdate_range(end=end, periods=n) if end else pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def zigzag(points, steps=10, lead=60):
    """Flat lead-in at points[0], then linear legs of `steps` bars between the turning points."""
    c = [np.full(lead, float(points[0]))]
    prev = points[0]
    for p in points[1:]:
        c.append(np.linspace(prev, p, steps + 1)[1:])
        prev = p
    return np.concatenate(c)


def card(close_or_hist, **kw):
    h = close_or_hist if isinstance(close_or_hist, pd.DataFrame) else make_hist(close_or_hist)
    kw.setdefault("now", NOW)
    return R.build_card("TEST", h, **kw)


UP = [100, 120, 108, 126, 114, 132]                 # HH 120,126,132 / HL 100(base),108,114


# ── indicators ────────────────────────────────────────────────────────────
def test_sma_and_ema_known_values():
    assert R.sma([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)[-1] == pytest.approx(9.0)
    e = R.ema([1, 2, 3, 4, 5], 3)                    # seed = mean(1,2,3) = 2 ; alpha = 0.5
    assert np.isnan(e[:2]).all() and list(e[2:]) == [2.0, 3.0, 4.0]
    assert np.isnan(R.ema([1, 2], 3)).all() and np.isnan(R.sma([1, 2], 3)).all()


def test_ema_obeys_its_recursion_everywhere():
    v = np.random.default_rng(1).normal(100, 3, 120)
    e, a = R.ema(v, 20), 2 / 21
    assert e[19] == pytest.approx(v[:20].mean())
    for i in range(20, 120):
        assert e[i] == pytest.approx(a * v[i] + (1 - a) * e[i - 1])


def test_atr_constant_range_and_a_gap():
    n = 40
    c = np.full(n, 100.0)
    atr = R.atr_wilder(c + 1, c - 1, c)
    assert np.isnan(atr[:13]).all() and atr[13] == pytest.approx(2.0) and atr[-1] == pytest.approx(2.0)
    h, l = c + 1, c - 1
    h[20], l[20] = 110.0, 108.0                     # gap: TR = max(2, 10, 8) = 10
    atr2 = R.atr_wilder(h, l, c)
    assert atr2[20] == pytest.approx((2.0 * 13 + 10.0) / 14)


def _rsi_reference(close, period=14):
    close = list(close)
    gains = [max(close[i] - close[i - 1], 0) for i in range(1, len(close))]
    losses = [max(close[i - 1] - close[i], 0) for i in range(1, len(close))]
    ag, al = sum(gains[:period]) / period, sum(losses[:period]) / period
    out = {period: 100 - 100 / (1 + ag / al)}
    for k in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[k]) / period
        al = (al * (period - 1) + losses[k]) / period
        out[k + 1] = 100 - 100 / (1 + ag / al)
    return out


def test_rsi_matches_an_independent_wilder_implementation():
    c = 100 + np.cumsum(np.random.default_rng(2).normal(0, 1, 200))
    got, ref = R.rsi_wilder(c), _rsi_reference(c)
    for i, v in ref.items():
        assert got[i] == pytest.approx(v, abs=1e-9)
    assert np.isnan(got[:14]).all()


def test_rsi_extremes_and_undefined_flat():
    assert R.rsi_wilder(np.arange(1, 40, dtype=float))[-1] == 100.0
    assert R.rsi_wilder(np.arange(40, 1, -1, dtype=float))[-1] == pytest.approx(0.0)
    assert np.isnan(R.rsi_wilder(np.full(40, 50.0))).all()           # undefined, never a made-up 50
    assert np.isnan(R.rsi_wilder([1, 2, 3])).all()


# ── swings ────────────────────────────────────────────────────────────────
def test_zigzag_swings_are_found_dated_and_confirmed_five_bars_later():
    h = make_hist(zigzag(UP + [120]))
    sw = R.find_swings(h)
    assert [s["type"] for s in sw] == ["low", "high", "low", "high", "low", "high"]
    expect = [("low", 100), ("high", 120), ("low", 108), ("high", 126), ("low", 114), ("high", 132)]
    for s, (t, p) in zip(sw, expect):
        assert s["type"] == t and s["price"] == pytest.approx(p * (1.003 if t == "high" else 0.997), rel=0.003)
        assert s["confirmed_idx"] == s["idx"] + 5
        assert s["date"] == h.index[s["idx"]].strftime("%Y-%m-%d")
        assert s["confirmed_date"] == h.index[s["confirmed_idx"]].strftime("%Y-%m-%d")


def test_the_unconfirmed_tail_is_never_reported_as_a_swing():
    c = zigzag(UP + [120])
    peak = 59 + 50                                                   # lead-in is indices 0-59; turning point k sits at 59+10k
    assert c[peak] == 132.0
    early = R.find_swings(make_hist(c[:peak + 5]))                   # ends at peak+4: only 4 bars after the peak
    assert not [s for s in early if s["idx"] == peak]
    assert all(s["confirmed_idx"] <= peak + 4 for s in early)
    exact = R.find_swings(make_hist(c[:peak + 6]))                   # ends at peak+5: the 5th confirming bar exists
    got = [s for s in exact if s["idx"] == peak]
    assert len(got) == 1 and got[0]["confirmed_idx"] == peak + 5 == len(c[:peak + 6]) - 1
    assert all(s["confirmed_idx"] <= len(c[:peak + 6]) - 1 for s in exact)


def test_a_flat_top_yields_exactly_one_swing_at_its_last_bar():
    c = np.concatenate([np.linspace(100, 120, 21), np.full(4, 120.0), np.linspace(118, 100, 19)])
    h = make_hist(np.concatenate([np.full(40, 100.0), c]))
    highs = [s for s in R.find_swings(h) if s["type"] == "high"]
    assert len(highs) == 1
    assert highs[0]["idx"] == 40 + 21 + 3                              # the last of the four equal bars


def _noisy_flat(n=60):
    close = np.full(n, 100.0)
    high, low = np.full(n, 102.0), np.full(n, 98.0)                    # ATR ~ 4
    return close, high, low


def test_reaction_filter_rejects_a_pivot_whose_price_barely_reacted():
    close, high, low = _noisy_flat()
    high[30] = 103.0                                                   # a one-bar high ...
    low[31:36] = 101.5                                                 # ... after which lows stay high: reaction 1.5 < 1 ATR
    h = make_hist(close, high, low)
    assert not [s for s in R.find_swings(h) if s["idx"] == 30 and s["type"] == "high"]
    assert [s for s in R.find_swings(h, reaction_atr=0.25) if s["idx"] == 30 and s["type"] == "high"]


def test_reaction_filter_accepts_a_pivot_that_did_react():
    close, high, low = _noisy_flat()
    high[30] = 103.0
    low[31:36] = 94.0                                                  # reaction 9 > 1 ATR
    assert [s for s in R.find_swings(make_hist(close, high, low)) if s["idx"] == 30 and s["type"] == "high"]


@pytest.mark.parametrize("seed", range(6))
def test_swings_are_prefix_stable_a_confirmed_swing_never_changes(seed):
    rng = np.random.default_rng(100 + seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.016, 320)))
    rngh = np.abs(rng.normal(0, 0.008, 320))
    full = make_hist(close, close * (1 + rngh), close * (1 - np.abs(rng.normal(0, 0.008, 320))))
    S = R.find_swings(full)
    assert len(S) > 8
    key = lambda s: (s["type"], s["idx"], round(s["price"], 9), s["date"], s["confirmed_idx"])
    for t in range(40, 320, 9):
        assert [key(s) for s in R.find_swings(full.iloc[:t])] == [key(s) for s in S if s["confirmed_idx"] <= t - 1]


def test_indicators_are_prefix_stable():
    rng = np.random.default_rng(7)
    c = 100 + np.cumsum(rng.normal(0, 1, 300))
    h, l = c + np.abs(rng.normal(0, 1, 300)), c - np.abs(rng.normal(0, 1, 300))
    full = {"ema": R.ema(c, 20), "sma": R.sma(c, 50), "atr": R.atr_wilder(h, l, c), "rsi": R.rsi_wilder(c)}
    for t in (60, 111, 200, 299):
        part = {"ema": R.ema(c[:t], 20), "sma": R.sma(c[:t], 50), "atr": R.atr_wilder(h[:t], l[:t], c[:t]),
                "rsi": R.rsi_wilder(c[:t])}
        for k in full:
            np.testing.assert_allclose(part[k], full[k][:t], equal_nan=True)


# ── structure ─────────────────────────────────────────────────────────────
def _sw(kind, price, i):
    return {"type": kind, "price": price, "idx": i, "date": f"2026-01-{i:02d}", "confirmed_idx": i + 5,
            "confirmed_date": f"2026-01-{i + 5:02d}"}


def test_structure_classification_and_breaks():
    up = [_sw("low", 90, 1), _sw("high", 100, 2), _sw("low", 95, 3), _sw("high", 110, 4)]
    assert R.classify_structure(up, 105)["structure"] == "uptrend"
    assert R.classify_structure(up, 105)["break_of_structure"] is None
    assert R.classify_structure(up, 94)["break_of_structure"].startswith("bearish")
    down = [_sw("high", 110, 1), _sw("low", 100, 2), _sw("high", 105, 3), _sw("low", 95, 4)]
    assert R.classify_structure(down, 100)["structure"] == "downtrend"
    assert R.classify_structure(down, 106)["break_of_structure"].startswith("bullish")
    mixed = [_sw("low", 90, 1), _sw("high", 100, 2), _sw("low", 85, 3), _sw("high", 110, 4)]   # HH + LL
    assert R.classify_structure(mixed, 100)["structure"] == "mixed"
    few = [_sw("low", 90, 1), _sw("high", 100, 2)]
    assert R.classify_structure(few, 95)["structure"] == "insufficient"


# ── zones ─────────────────────────────────────────────────────────────────
def test_zones_cluster_nearby_swing_points_and_count_touches():
    pts = [_sw("high", 100.0, 1), _sw("low", 100.3, 2), _sw("high", 100.4, 3), _sw("high", 110.0, 4)]
    z = R.cluster_zones(pts, atr=1.0)
    assert len(z) == 2
    assert z[0]["touches"] == 3 and z[0]["kind"] == "tested" and z[0]["low"] == 100.0 and z[0]["high"] == 100.4
    assert z[0]["n_highs"] == 2 and z[0]["n_lows"] == 1
    assert z[1]["touches"] == 1 and z[1]["kind"] == "single swing"
    assert R.cluster_zones(pts, atr=float("nan")) == [] and R.cluster_zones([], atr=1.0) == []


# ── invalidation, target, risk, sizing ────────────────────────────────────
def test_structural_invalidation_uses_the_most_recent_swing_low_below_entry():
    sw = [_sw("low", 90, 1), _sw("high", 110, 2), _sw("low", 95, 3), _sw("high", 120, 4)]
    inv = R.structural_invalidation(sw, entry_ref=118.0, atr_last=2.0)
    assert inv["swing_low"] == 95 and inv["stop"] == pytest.approx(95 - 0.25 * 2.0)
    assert inv["risk_per_share"] == pytest.approx(118 - 94.5) and inv["risk_pct"] == pytest.approx(23.5 / 118 * 100)
    assert inv["swing_date"] == "2026-01-03"


def test_if_the_latest_swing_low_is_above_entry_an_older_one_below_is_used():
    sw = [_sw("low", 90, 1), _sw("high", 110, 2), _sw("low", 105, 3)]
    inv = R.structural_invalidation(sw, entry_ref=100.0, atr_last=2.0)
    assert inv["swing_low"] == 90


def test_no_swing_low_below_entry_means_no_invalidation_level():
    assert R.structural_invalidation([_sw("low", 105, 1)], 100.0, 2.0) is None
    assert R.structural_invalidation([], 100.0, 2.0) is None
    assert R.structural_invalidation([_sw("low", 95, 1)], 100.0, float("nan")) is None


def test_target_is_the_nearest_zone_above_both_entry_and_the_broken_zone():
    zones = [{"low": 98, "high": 99, "touches": 2, "last_date": "d"}, {"low": 105, "high": 106, "touches": 1, "last_date": "d"},
             {"low": 110, "high": 111, "touches": 3, "last_date": "d"}]
    assert R.pick_target(zones, 100.0, 100.0)["price"] == 105
    assert R.pick_target(zones, 100.0, 106.0)["price"] == 110
    assert R.pick_target(zones, 100.0, 200.0) is None


def test_sizing_math_cap_and_participation():
    s = R.sizing_illustration(50.0, 48.0, 10_000, 1.0, adv20_shares=1000, adv20_dollar=50_000)
    assert s["risk_dollars"] == 100 and s["shares"] == 50 and not s["capped_by_account"]
    assert s["position_value"] == 2500 and s["actual_risk_dollars"] == 100
    assert s["pct_of_adv20_shares"] == pytest.approx(5.0) and s["pct_of_adv20_dollar_volume"] == pytest.approx(5.0)
    assert R.sizing_illustration(100.0, 97.0, 10_000, 1.0, None, None)["shares"] == 33        # floor(100/3)
    capped = R.sizing_illustration(500.0, 499.0, 10_000, 1.0, None, None)
    assert capped["capped_by_account"] and capped["shares"] == 20 and capped["position_value"] == 10_000
    assert capped["pct_of_adv20_shares"] is None


# ── setup descriptions (positive AND negative) ────────────────────────────
ST_UP = {"structure": "uptrend", "last_swing_high": _sw("high", 130, 4), "last_swing_low": _sw("low", 110, 3)}
Z = lambda lo, hi: {"low": lo, "high": hi, "touches": 2}


def test_setup_labels_each_have_a_positive_and_a_negative_case():
    I = R.interpret_setup
    assert I({"structure": "downtrend"}, 100, 2, [], None, None, None)["label"] == "weak_or_deteriorating"
    assert I(ST_UP, 100, 2, [], 120.0, 110.0, None)["label"] == "weak_or_deteriorating"        # below 200 SMA, 50 EMA < 200 SMA
    assert I(ST_UP, 100, 2, [], 90.0, 110.0, None)["label"] != "weak_or_deteriorating"         # 50 EMA above the 200 SMA: not weak
    assert I(ST_UP, 125, 2, [Z(127, 128)], None, None, None)["label"] == "breakout_candidate"
    assert I(ST_UP, 120, 2, [Z(127, 128)], None, None, None)["label"] == "trend_pullback"      # resistance 7 away > 1.5 ATR
    assert I(ST_UP, 129.5, 2, [], None, None, None)["label"] == "trend_extended"               # pullback only 0.5 ATR deep
    assert I(ST_UP, 125, 2, [Z(127, 128)], None, None, {"zone_high": 124})["label"] == "breakout_occurred"
    assert I({"structure": "mixed", "last_swing_high": None, "last_swing_low": None}, 100, 2, [], None, None, None)["label"] == "unclassified"
    far_below_low = I(ST_UP, 105, 2, [Z(140, 141)], None, None, None)                          # below the last swing low: structure broken
    assert far_below_low["label"] == "trend_extended"


# ── whole-card behaviour on constructed series ────────────────────────────
def test_downtrend_card_is_weak_with_no_plan():
    c = card(zigzag([140, 120, 132, 112, 124, 104, 112]))
    assert c["structure"]["structure"] == "downtrend" and c["setup"]["label"] == "weak_or_deteriorating"
    p = c["plan"]
    assert p["status"] == "no_long_setup" and p["entry_condition"] is None
    assert p["invalidation"] is None and p["target"] is None and p["reward_risk"] is None and p["sizing"] is None
    assert any("LH/LL" in e for e in c["evidence_against"])


def test_pullback_card_has_a_structural_stop_and_an_honest_no_target_reason():
    c = card(zigzag(UP + [121]))
    assert c["setup"]["label"] == "trend_pullback" and c["plan"]["status"] == "research_candidate"
    p = c["plan"]
    assert p["entry_ref_basis"] == "last swing high" and "Close above" in p["entry_condition"]
    inv = p["invalidation"]
    assert inv["stop"] < p["entry_ref"] and inv["risk_pct"] > 0
    assert inv["swing_confirmed"] >= inv["swing_date"]
    assert p["target"] is None and "No defensible target" in p["target_reason"] and p["reward_risk"] is None
    assert any("HH/HL" in e for e in c["evidence_for"])


def test_breakout_candidate_card_gets_a_target_and_exact_reward_risk():
    c = card(zigzag(UP + [124.5]))
    assert c["setup"]["label"] == "breakout_candidate"
    p = c["plan"]
    assert "volume at least 1.5x" in p["entry_condition"]
    assert p["target"] is not None and p["target"]["price"] > p["entry_ref"]
    risk = p["entry_ref"] - p["invalidation"]["stop"]
    assert p["reward_risk"] == pytest.approx((p["target"]["price"] - p["entry_ref"]) / risk, abs=0.01)
    assert p["target_reason"] is None


def _breakout_series(vol_spike):
    close = zigzag(UP[:-1] + [134])                       # last leg crosses the 126 zone ~3 bars before the end
    n = len(close)
    vol = np.full(n, 1e6)
    if vol_spike:
        vol[-4:] = 2.2e6
    return make_hist(close, vol=vol)


def test_volume_confirmed_breakout_is_trigger_met_and_unconfirmed_is_not():
    on = card(_breakout_series(True))
    assert on["setup"]["label"] == "breakout_occurred"
    assert on["levels"]["recent_breakout"]["volume_confirmed"] is True
    assert on["plan"]["status"] == "trigger_met" and "confirmed" in on["plan"]["entry_condition"]
    assert on["plan"]["entry_ref_basis"] == "last complete close"
    off = card(_breakout_series(False))
    assert off["setup"]["label"] == "breakout_occurred"
    assert off["levels"]["recent_breakout"]["volume_confirmed"] is False
    assert off["plan"]["status"] == "research_candidate" and "NOT confirmed" in off["plan"]["entry_condition"]


def test_wide_structural_stop_is_flagged_and_stated():
    c = card(zigzag([100, 150, 80, 160, 90, 170, 150]))
    assert c["plan"]["invalidation"] is not None
    assert c["plan"]["invalidation"]["risk_pct"] > R.WIDE_STOP_PCT and c["risk_context"]["wide_stop"] is True
    assert any("wide" in e for e in c["evidence_against"])


def test_sizing_and_reward_risk_in_the_card_follow_the_stated_inputs():
    c = card(zigzag(UP + [124.5]), account_size=25_000, risk_pct=0.5)
    s, p = c["plan"]["sizing"], c["plan"]
    assert s["account_size_illustration"] == 25_000 and s["risk_pct"] == 0.5 and s["risk_dollars"] == 125
    assert s["shares"] == math.floor(125 / p["invalidation"]["risk_per_share"])
    d = card(zigzag(UP + [124.5]))
    assert d["plan"]["sizing"]["account_size_illustration"] == 10_000 and "Illustration only" in d["plan"]["sizing"]["note"]


def test_ma200_comparison_appears_only_when_200_bars_exist():
    short = card(zigzag(UP + [124.5]))
    assert short["plan"]["invalidation"]["ma200_distance_pct_for_comparison"] is None
    assert any("200-day SMA unavailable" in m for m in short["missing_data"])
    long_ = card(zigzag(UP + [124.5], lead=260))
    assert long_["plan"]["invalidation"]["ma200_distance_pct_for_comparison"] is not None
    assert long_["trend"]["sma200"]["value"] is not None


def test_trend_context_labels_distances_and_alignment():
    t = card(zigzag(UP + [124.5], lead=260))["trend"]
    for k, lab in (("ema20", "20-day EMA"), ("ema50", "50-day EMA"), ("sma200", "200-day SMA")):
        assert t[k]["label"] == lab and t[k]["value"] is not None and t[k]["slope_20bar_pct"] is not None
    assert t["alignment"] in ("bullish (20 > 50 > 200)", "bearish (20 < 50 < 200)", "mixed")
    assert "lag" in t["note"]


def test_weekly_context_uses_completed_weeks_only():
    close = np.linspace(100, 160, 300)
    wed = make_hist(close, end="2026-10-21")                                # last bar is a Wednesday
    w = R.weekly_context(wed)
    assert w["available"] and w["close"] == pytest.approx(wed.loc["2026-10-16", "Close"], abs=1e-3)
    fri = make_hist(close, end="2026-10-23")                                # last bar is a Friday: that week is complete
    assert R.weekly_context(fri)["close"] == pytest.approx(fri.loc["2026-10-23", "Close"], abs=1e-3)
    assert R.weekly_context(make_hist(close[:80]))["available"] is False


def test_volume_context_baseline_excludes_the_bar_being_measured():
    vol = np.full(80, 1e6)
    vol[-1] = 3e6
    v = R.volume_context(make_hist(np.full(80, 100.0), vol=vol))
    assert v["rvol_last"] == pytest.approx(3.0)                              # 3e6 / mean of the 50 PRIOR bars (1e6)
    assert R.volume_context(make_hist(np.full(30, 100.0)))["rvol_last"] is None


def test_up_down_volume_ratio():
    close = np.array([100.0 + (1 if i % 2 else -1) * 0.5 + i * 0.0 for i in range(60)])
    vol = np.where(np.arange(60) % 2 == 1, 2e6, 1e6)                         # up days have 2x the volume
    v = R.volume_context(make_hist(close, vol=vol))
    assert v["up_down_volume_ratio_20"] == pytest.approx(2.0)


def test_rsi_divergence_is_reported_only_when_the_swings_support_it():
    # Two rising highs; momentum into the second is much weaker (shallower, slower rise).
    close = np.concatenate([np.full(60, 100.0), np.linspace(100, 130, 11)[1:], np.linspace(130, 112, 9)[1:],
                            np.linspace(112, 133, 31)[1:], np.linspace(133, 118, 9)[1:]])
    m = R.momentum_context(make_hist(close), R.find_swings(make_hist(close)))
    assert m["divergence"]["bearish"].startswith("candidate")
    assert "not a trade signal" in m["note"]
    flat = make_hist(np.full(100, 100.0))
    assert R.momentum_context(flat, R.find_swings(flat))["divergence"]["bearish"] == "not assessable"


def test_earnings_proximity_is_flagged_only_when_close():
    h = make_hist(zigzag(UP + [124.5]))
    last = h.index[-1]
    near = card(h, earnings_date=str((last + pd.Timedelta(days=3)).date()))
    assert near["risk_context"]["earnings"]["flag"] is True and near["risk_context"]["earnings"]["days_until"] == 3
    assert any("Earnings in 3 day(s)" in e for e in near["evidence_against"])
    far = card(h, earnings_date=str((last + pd.Timedelta(days=30)).date()))
    assert far["risk_context"]["earnings"]["flag"] is False
    assert any("Earnings date unavailable" in m for m in card(h)["missing_data"])


def test_card_states_what_is_missing_and_what_it_cannot_know():
    c = card(zigzag(UP + [124.5]))
    assert any("spread" in m.lower() for m in c["missing_data"])
    assert c["risk_context"]["liquidity"]["spread"].startswith("unavailable")
    assert any("dividend" in x for x in c["confidence_limits"])
    assert "Setup types have not been shown to predict returns" in c["disclaimer"]


# ── data hygiene: partial sessions, markets, thin data ────────────────────
def _daily(n=120, end="2026-10-20", last_vol=None):
    h = make_hist(np.linspace(100, 130, n), end=end)
    if last_vol is not None:
        h.iloc[-1, h.columns.get_loc("Volume")] = last_vol
    return h


def test_todays_bar_is_dropped_while_the_us_session_may_still_be_forming():
    h = _daily(end="2026-10-20")
    morning = datetime(2026, 10, 20, 14, 0, tzinfo=timezone.utc)             # 10:00 ET
    clean, notes = R.prepare_history(h, now=morning)
    assert clean.index[-1] == pd.Timestamp("2026-10-19") and any("excluded" in n for n in notes)
    settled = datetime(2026, 10, 20, 20, 30, tzinfo=timezone.utc)            # 16:30 ET
    assert R.prepare_history(h, now=settled)[0].index[-1] == pd.Timestamp("2026-10-20")
    just_closed = datetime(2026, 10, 20, 20, 5, tzinfo=timezone.utc)         # 16:05 ET: inside the settle buffer
    assert R.prepare_history(h, now=just_closed)[0].index[-1] == pd.Timestamp("2026-10-19")
    next_day = datetime(2026, 10, 21, 13, 0, tzinfo=timezone.utc)
    assert R.prepare_history(h, now=next_day)[0].index[-1] == pd.Timestamp("2026-10-20")
    assert R.prepare_history(h, now=morning, drop_incomplete=False)[0].index[-1] == pd.Timestamp("2026-10-20")


def test_a_partial_last_bar_never_reaches_relative_volume():
    h = _daily(end="2026-10-20", last_vol=1e4)                               # a partial day's tiny volume
    c = R.build_card("TEST", h, now=datetime(2026, 10, 20, 14, 0, tzinfo=timezone.utc))
    assert c["observation_date"] == "2026-10-19" and c["volume"]["rvol_last"] == pytest.approx(1.0)
    assert any("excluded" in n for n in c["data"]["notes"])
    leaked = R.build_card("TEST", h, now=NOW)                                # session complete: that volume is real
    assert leaked["volume"]["rvol_last"] < 0.05


def test_without_a_timezone_database_the_conservative_rule_applies(monkeypatch):
    monkeypatch.setitem(sys.modules, "zoneinfo", None)
    h = _daily(end="2026-10-20")
    assert R.prepare_history(h, now=datetime(2026, 10, 20, 14, 0, tzinfo=timezone.utc))[0].index[-1] == pd.Timestamp("2026-10-19")
    assert R.prepare_history(h, now=datetime(2026, 10, 20, 22, 0, tzinfo=timezone.utc))[0].index[-1] == pd.Timestamp("2026-10-20")


def test_prepare_history_cleans_without_mutating_and_strips_timezones():
    h = make_hist(np.linspace(100, 120, 90))
    h.index = h.index.tz_localize("America/New_York")
    dirty = pd.concat([h, h.iloc[[10]]])
    dirty.iloc[5, dirty.columns.get_loc("Close")] = np.nan
    dirty.iloc[7, dirty.columns.get_loc("High")] = dirty.iloc[7]["Low"] - 1          # high below low: unusable
    snap = dirty.copy()
    clean, notes = R.prepare_history(dirty, now=NOW)
    pd.testing.assert_frame_equal(dirty, snap)
    assert clean.index.tz is None and clean.index.is_monotonic_increasing and not clean.index.duplicated().any()
    assert len(clean) == 88 and any("unusable" in n for n in notes)
    with pytest.raises(ValueError):
        R.prepare_history(h.drop(columns=["Volume"]))


def test_non_us_markets_get_no_card_and_say_why():
    c = R.build_card("ZENITHBANK", _daily(), market="NG", now=NOW)
    assert c["plan"]["status"] == "unvalidated_market" and c["setup"] is None and c["plan"]["invalidation"] is None
    assert "do not transfer" in c["plan"]["status_text"]


def test_too_little_history_returns_insufficient_data_with_no_levels():
    c = card(zigzag(UP)[:50])
    assert c["plan"]["status"] == "insufficient_data" and c["structure"] is None and c["levels"] is None
    assert c["plan"]["invalidation"] is None and c["plan"]["target"] is None and "Insufficient data" in c["plan"]["target_reason"]
    assert card(make_hist([100.0]))["plan"]["status"] == "insufficient_data"
    assert card(make_hist(np.array([])))["plan"]["status"] == "insufficient_data"


# ── point-in-time ─────────────────────────────────────────────────────────
def _random_hist(seed, n=300):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.016, n)))
    return make_hist(close, close * (1 + np.abs(rng.normal(0, 0.008, n))), close * (1 - np.abs(rng.normal(0, 0.008, n))),
                     rng.integers(5e5, 3e6, n).astype(float), end="2026-10-16")


@pytest.mark.parametrize("seed", range(5))
def test_asof_cuts_the_future_off_even_when_the_future_is_garbage(seed):
    h = _random_hist(seed)
    for t in (90, 133, 187, 241, 299):
        garbage = h.copy()                                                    # fresh copy: the past must stay genuine
        garbage.iloc[t:] = garbage.iloc[t:] * np.random.default_rng(t).uniform(0.05, 40.0, (len(garbage) - t, 5))
        garbage.iloc[t:, garbage.columns.get_loc("High")] = garbage.iloc[t:][["High", "Low", "Close"]].max(axis=1)
        garbage.iloc[t:, garbage.columns.get_loc("Low")] = garbage.iloc[t:][["High", "Low", "Close"]].min(axis=1)
        asof = h.index[t - 1]
        reference = R.build_card("T", h.iloc[:t], now=NOW)
        assert R.build_card("T", garbage, asof=asof, now=NOW) == reference
        assert R.build_card("T", h, asof=asof, now=NOW) == reference


@pytest.mark.parametrize("seed", range(5))
def test_no_swing_in_a_card_is_dated_after_the_observation(seed):
    h = _random_hist(seed + 20)
    for t in range(80, 300, 31):
        c = R.build_card("T", h.iloc[:t], now=NOW)
        for s in c["structure"]["swings"]:
            assert s["confirmed_date"] <= c["observation_date"] and s["date"] <= s["confirmed_date"]
        inv = c["plan"]["invalidation"]
        if inv:
            assert inv["swing_confirmed"] <= c["observation_date"]
        for z in c["levels"]["supports"] + c["levels"]["resistances"]:
            assert z["last_date"] <= c["observation_date"]


# ── fabricated levels: fuzz ───────────────────────────────────────────────
def _fuzz_series(rng):
    n = int(rng.integers(0, 400))
    kind = int(rng.integers(0, 7))
    if kind == 0:   close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    elif kind == 1: close = np.full(n, float(rng.uniform(1, 500)))
    elif kind == 2: close = np.linspace(50, float(rng.uniform(10, 300)), n) if n else np.array([])
    elif kind == 3: close = np.where(np.arange(n) < n // 2, 100.0, float(rng.uniform(20, 200)))
    elif kind == 4: close = 20 + np.abs(np.cumsum(rng.normal(0, 3, n)))
    elif kind == 5: close = 100 + 10 * np.sin(np.arange(n) / 7.0) + rng.normal(0, 0.5, n)
    else:           close = np.maximum(0.5, 5 + np.cumsum(rng.normal(0, 0.6, n)))
    close = np.maximum(close, 0.01)
    spread = np.abs(rng.normal(0, 0.01, n)) * (kind != 1)
    h = make_hist(close, close * (1 + spread), close * (1 - spread),
                  np.where(rng.random(n) < 0.1, 0.0, rng.integers(1e4, 5e6, n).astype(float)), end="2026-10-16")
    if n > 20 and rng.random() < 0.5:
        h.iloc[int(rng.integers(0, n)), h.columns.get_loc("Close")] = np.nan
    return h


def _walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, path + "/" + str(k))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, o


def test_fuzz_no_crash_no_nan_and_every_level_traces_to_a_confirmed_swing():
    rng = np.random.default_rng(2026)
    checked_levels = 0
    for _ in range(600):
        h = _fuzz_series(rng)
        c = R.build_card("FZ", h, now=NOW)
        json.dumps(c, allow_nan=False)                                        # strict JSON: no NaN / inf anywhere
        for path, v in _walk(c):
            assert not (isinstance(v, float) and not math.isfinite(v)), path
        p = c["plan"]
        if c["levels"] is None:
            assert p["invalidation"] is None and p["target"] is None and p["sizing"] is None and p["reward_risk"] is None
            continue
        clean, _ = R.prepare_history(h, now=NOW)
        sw = R.find_swings(clean)
        prices = {round(s["price"], 4) for s in sw}
        atr_last = R.atr_wilder(clean["High"], clean["Low"], clean["Close"])[-1]
        for z in c["levels"]["supports"] + c["levels"]["resistances"]:
            assert round(z["low"], 4) in prices and round(z["high"], 4) in prices
            checked_levels += 1
        if p["invalidation"]:
            inv = p["invalidation"]
            assert round(inv["swing_low"], 4) in prices
            assert inv["stop"] == pytest.approx(inv["swing_low"] - 0.25 * atr_last, abs=1e-3)
            assert 0 < inv["stop"] < p["entry_ref"] and inv["risk_pct"] > 0 and p["entry_condition"]
            checked_levels += 1
        if p["target"]:
            assert round(p["target"]["price"], 4) in prices and p["target"]["price"] > p["entry_ref"]
            checked_levels += 1
        if p["reward_risk"] is not None:
            assert p["reward_risk"] >= 0 and p["target"] and p["invalidation"]
        if p["sizing"]:
            assert isinstance(p["sizing"]["shares"], int) and p["sizing"]["shares"] >= 0
            assert p["sizing"]["position_value"] <= p["sizing"]["account_size_illustration"] + 0.01
        if p["entry_condition"] is None:
            assert p["invalidation"] is None and p["sizing"] is None
    assert checked_levels > 300, "the fuzz must actually exercise levels, not just refuse everything"


# ── wording policy, no score, determinism, read-only ──────────────────────
BANNED = re.compile(r"\b(buy|sell|recommend\w*|guarantee\w*|profitab\w*|will (rise|fall|go up|go down)|strong buy)\b", re.I)
BANNED_KEYS = {"score", "rating", "probability", "recommendation", "signal", "confidence_score"}


def _every_kind_of_card():
    cards = [card(zigzag(UP + [121])), card(zigzag(UP + [124.5])), card(_breakout_series(True)),
             card(_breakout_series(False)), card(zigzag([140, 120, 132, 112, 124, 104, 112])),
             card(zigzag(UP + [133, 134, 135, 136, 137, 138, 139, 140, 141, 142])),
             card(zigzag(UP)[:50]), R.build_card("X", _daily(), market="NG", now=NOW)]
    rng = np.random.default_rng(3)
    cards += [R.build_card("FZ", _fuzz_series(rng), now=NOW) for _ in range(40)]
    return cards


def test_wording_policy_no_recommendations_and_no_score_anywhere():
    labels = set()
    for c in _every_kind_of_card():
        for path, v in _walk(c):
            key = path.split("/")[-1].split("[")[0]
            assert key not in BANNED_KEYS, f"forbidden field {path}"
            if isinstance(v, str):
                assert not BANNED.search(v), f"forbidden wording at {path}: {v!r}"
        assert "Setup types have not been shown to predict returns" in c["disclaimer"]
        if c["setup"]:
            labels.add(c["setup"]["label"])
    assert {"trend_pullback", "breakout_candidate", "breakout_occurred", "weak_or_deteriorating"} <= labels


def test_card_is_deterministic_and_never_mutates_its_input():
    h = _random_hist(11)
    snap = h.copy()
    a, b = R.build_card("T", h, now=NOW), R.build_card("T", h, now=NOW)
    assert a == b
    pd.testing.assert_frame_equal(h, snap)
    json.dumps(a, allow_nan=False)


def test_status_vocabulary_is_closed():
    allowed = {"research_candidate", "trigger_met", "no_long_setup", "insufficient_data", "unvalidated_market"}
    assert {c["plan"]["status"] for c in _every_kind_of_card()} <= allowed
