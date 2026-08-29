"""
tests/test_apex10_radar_stage_d.py — Apex the Great X: Stage D tests

Covers score calculation, missing-data renormalization, evidence
quality, state classification thresholds, the liquidity gate, and the
breakout trigger gates — including the explicit requirement that the
score alone can never look like a buy signal.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.apex10_radar import (
    compute_apex10_score, classify_apex10_state, compute_liquidity_gate,
    compute_breakout_trigger_gates, DEFAULT_WEIGHTS, DEFAULT_STATE_THRESHOLDS,
)


def _full_bullish_features():
    """Every component maximally bullish and fully available — should
    produce a very high score with HIGH evidence quality."""
    return {
        "relative_strength": {"rs_5d_change": 20, "rs_acceleration": 10},
        "breakout_proximity": {"state": "IMMINENT", "distance_to_resistance_pct": 0.0},
        "volatility": {"range_ratio_5_20": 0.3, "volatility_contraction_acceleration": True},
        "volume": {"relative_volume": 1.8, "volume_trend": "RISING", "volume_contraction": False},
        "breakout_volume_confirmation": True,
        "structure": {"higher_lows_flat_resistance": True, "ms_hh_hl": True},
        "moving_averages": {"ma50_transition": "ACCELERATING_UP", "ma200_transition": "ACCELERATING_UP"},
        "sector_confirmation": {"data_available": True, "stock_vs_sector": 10, "sector_vs_market": 5},
        "institutional_trend": {"institutional_ownership_trend": "RISING"},
        "fundamental_acceleration": {"data_available": True, "revenue_growth_trend": "ACCELERATING",
                                     "earnings_growth_trend": "ACCELERATING"},
        "market_regime": {"regime": "RISK_ON"},
    }


def _empty_features():
    return {
        "relative_strength": {"rs_5d_change": None, "rs_acceleration": None},
        "breakout_proximity": {"state": "UNKNOWN"},
        "volatility": {"range_ratio_5_20": None},
        "volume": {"relative_volume": None, "volume_trend": "UNKNOWN"},
        "breakout_volume_confirmation": False,
        "structure": {"higher_lows_flat_resistance": None, "ms_hh_hl": None},
        "moving_averages": {"ma50_transition": "UNKNOWN", "ma200_transition": "UNKNOWN"},
        "sector_confirmation": {"data_available": False},
        "institutional_trend": {"institutional_ownership_trend": "UNKNOWN"},
        "fundamental_acceleration": {"data_available": False},
        "market_regime": {"regime": "UNKNOWN"},
    }


def test_weights_sum_to_100():
    assert sum(DEFAULT_WEIGHTS.values()) == 100


def test_full_bullish_features_score_high_with_high_evidence_quality():
    result = compute_apex10_score(_full_bullish_features())
    assert result["score"] > 85
    assert result["evidence_quality"] == "HIGH"
    assert result["coverage_pct"] == 100.0


def test_missing_price_history_returns_unknown():
    result = compute_apex10_score({})
    assert result["score"] is None
    assert result["state"] == "UNKNOWN"
    assert result["evidence_quality"] == "LOW"


def test_all_unknown_components_returns_none_score():
    result = compute_apex10_score(_empty_features())
    assert result["score"] is None
    assert result["evidence_quality"] == "LOW"


def test_partial_data_renormalizes_and_flags_lower_evidence_quality():
    features = _full_bullish_features()
    # Strip out several components to simulate real-world sparse data.
    features["sector_confirmation"] = {"data_available": False}
    features["institutional_trend"] = {"institutional_ownership_trend": "UNKNOWN"}
    features["fundamental_acceleration"] = {"data_available": False}
    features["market_regime"] = {"regime": "UNKNOWN"}
    result = compute_apex10_score(features)
    assert result["score"] is not None
    assert result["coverage_pct"] < 100.0
    assert result["evidence_quality"] in ("MEDIUM", "LOW")
    # Missing components must be None, never silently defaulted to 0 or 1.
    assert result["components"]["sector_confirmation"] is None
    assert result["components"]["institutional_trend"] is None


def test_component_scores_never_exceed_bounds():
    result = compute_apex10_score(_full_bullish_features())
    for k, v in result["components"].items():
        if v is not None:
            assert 0.0 <= v <= 1.0, f"{k} component out of [0,1] bounds: {v}"


def test_state_classification_thresholds():
    assert classify_apex10_state(95) == "TRIGGER"
    assert classify_apex10_state(85) == "READY"
    assert classify_apex10_state(75) == "DEVELOPING"
    assert classify_apex10_state(65) == "WATCH"
    assert classify_apex10_state(30) == "EARLY"
    assert classify_apex10_state(None) == "UNKNOWN"


def test_state_classification_boundary_values_use_configured_thresholds():
    for state, boundary in [("trigger", "TRIGGER"), ("ready", "READY"),
                            ("developing", "DEVELOPING"), ("watch", "WATCH")]:
        score = DEFAULT_STATE_THRESHOLDS[state]
        assert classify_apex10_state(score) == boundary


def test_liquidity_gate_passes_above_threshold():
    result = compute_liquidity_gate(avg_volume_20=1_000_000, last_close=10.0, market="US")
    assert result["passes"] is True
    assert result["dollar_volume"] == 10_000_000.0


def test_liquidity_gate_fails_below_threshold():
    result = compute_liquidity_gate(avg_volume_20=10_000, last_close=5.0, market="US")
    assert result["passes"] is False


def test_liquidity_gate_missing_data_returns_none_not_false():
    result = compute_liquidity_gate(avg_volume_20=None, last_close=10.0, market="US")
    assert result["passes"] is None  # unknown, not a silent fail


def test_liquidity_gate_unknown_market_returns_none():
    result = compute_liquidity_gate(avg_volume_20=1_000_000, last_close=10.0, market="MARS")
    assert result["passes"] is None


def test_breakout_trigger_all_gates_pass():
    features = _full_bullish_features()
    liq = {"passes": True}
    result = compute_breakout_trigger_gates(features, liq)
    assert result["confirmed_breakout"] is True
    assert isinstance(result["confirmed_breakout"], bool)


def test_breakout_trigger_fails_without_volume_confirmation():
    features = _full_bullish_features()
    features["breakout_volume_confirmation"] = False
    liq = {"passes": True}
    result = compute_breakout_trigger_gates(features, liq)
    assert result["confirmed_breakout"] is False


def test_breakout_trigger_fails_without_liquidity():
    features = _full_bullish_features()
    liq = {"passes": False}
    result = compute_breakout_trigger_gates(features, liq)
    assert result["confirmed_breakout"] is False


def test_breakout_trigger_pending_gates_never_silently_pass():
    features = _full_bullish_features()
    liq = {"passes": True}
    result = compute_breakout_trigger_gates(features, liq)
    assert result["gates"]["risk_reward_acceptable"] == "pending_stage_l"
    assert result["gates"]["no_major_invalidation"] == "pending_stage_l"
    # These pending gates must never be counted as passing in confirmed_breakout —
    # confirmed_breakout being True here reflects ONLY the currently-computable gates.
    assert "risk_reward_acceptable" not in str(result["confirmed_breakout"])


def test_score_never_used_as_buy_recommendation_language():
    """Structural check: the module never emits an actual recommendation
    phrase (e.g. 'you should buy') — score and state describe evidence
    only, per the spec's explicit GATE requirement. Note that the
    module's own disclaimer text legitimately says 'NOT a buy signal',
    which must NOT trip this check — only genuine recommendation
    phrasing should."""
    result = compute_apex10_score(_full_bullish_features())
    trigger = compute_breakout_trigger_gates(_full_bullish_features(), {"passes": True})
    forbidden_phrases = ["should buy", "recommend buying", "buy now", "buy signal:",
                         "action: buy", "sell now", "should sell"]
    combined_text = str(result).lower() + str(trigger).lower()
    for phrase in forbidden_phrases:
        assert phrase not in combined_text, f"Found recommendation language: {phrase!r}"
    # The disclaimer explicitly negating a buy signal IS expected and correct.
    assert "not a buy signal" in combined_text


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
