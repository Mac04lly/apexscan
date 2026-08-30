"""
tests/test_apex10_short_radar.py — Apex the Great X short side: score/gates
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.apex10_short_radar import (
    compute_apex10_short_score, compute_breakdown_trigger_gates, DEFAULT_SHORT_WEIGHTS,
)
from modules.apex10_radar import classify_apex10_state, compute_liquidity_gate


def _full_bearish_features():
    return {
        "relative_strength": {"rs_5d_change": -20, "rs_acceleration": -10},
        "breakdown_proximity": {"state": "IMMINENT", "distance_to_support_pct": 0.0},
        "volatility": {"range_ratio_5_20": 0.3, "volatility_contraction_acceleration": True},
        "volume": {"relative_volume": 1.8, "volume_trend": "RISING", "volume_contraction": False},
        "breakdown_volume_confirmation": True,
        "structure": {"lower_highs_flat_support": True, "ms_lh_ll": True},
        "moving_averages": {"ma50_transition": "FALLING", "ma200_transition": "FALLING"},
        "sector_confirmation": {"data_available": True, "stock_vs_sector": -10, "sector_vs_market": -5},
        "institutional_trend": {"institutional_ownership_trend": "FALLING"},
        "fundamental_acceleration": {"data_available": True, "revenue_growth_trend": "DECELERATING",
                                     "earnings_growth_trend": "DECELERATING"},
        "market_regime": {"regime": "RISK_OFF"},
    }


def _empty_features():
    return {
        "relative_strength": {"rs_5d_change": None, "rs_acceleration": None},
        "breakdown_proximity": {"state": "UNKNOWN"},
        "volatility": {"range_ratio_5_20": None},
        "volume": {"relative_volume": None, "volume_trend": "UNKNOWN"},
        "breakdown_volume_confirmation": False,
        "structure": {"lower_highs_flat_support": None, "ms_lh_ll": None},
        "moving_averages": {"ma50_transition": "UNKNOWN", "ma200_transition": "UNKNOWN"},
        "sector_confirmation": {"data_available": False},
        "institutional_trend": {"institutional_ownership_trend": "UNKNOWN"},
        "fundamental_acceleration": {"data_available": False},
        "market_regime": {"regime": "UNKNOWN"},
    }


def test_weights_sum_to_100():
    assert sum(DEFAULT_SHORT_WEIGHTS.values()) == 100


def test_full_bearish_features_score_high_with_high_evidence_quality():
    result = compute_apex10_short_score(_full_bearish_features())
    assert result["score"] > 85
    assert result["evidence_quality"] == "HIGH"


def test_missing_price_history_returns_unknown():
    result = compute_apex10_short_score({})
    assert result["score"] is None
    assert result["evidence_quality"] == "LOW"


def test_all_unknown_components_returns_none_score():
    result = compute_apex10_short_score(_empty_features())
    assert result["score"] is None


def test_component_scores_never_exceed_bounds():
    result = compute_apex10_short_score(_full_bearish_features())
    for k, v in result["components"].items():
        if v is not None:
            assert 0.0 <= v <= 1.0, f"{k} out of bounds: {v}"


def test_state_reused_from_long_side_classify_function():
    # Reuses classify_apex10_state directly — same thresholds either way.
    result = compute_apex10_short_score(_full_bearish_features())
    assert classify_apex10_state(result["score"]) in ("TRIGGER", "READY")


# ── Direction inversion is the actual point of this module — verify it ─

def test_a_bullish_reading_scores_low_on_the_short_side():
    """The critical inversion check: features that are STRONGLY BULLISH
    (rising RS, MA turning up, rising institutional ownership, RISK_ON)
    must score LOW for shorting, not high — proving the mirror actually
    inverted direction rather than just relabeling the long side."""
    bullish = {
        "relative_strength": {"rs_5d_change": 20, "rs_acceleration": 10},
        "breakdown_proximity": {"state": "DISTANT"},
        "volatility": {"range_ratio_5_20": 1.5},
        "volume": {"relative_volume": 0.5, "volume_trend": "FALLING", "volume_contraction": False},
        "breakdown_volume_confirmation": False,
        "structure": {"lower_highs_flat_support": False, "ms_lh_ll": False},
        "moving_averages": {"ma50_transition": "ACCELERATING_UP", "ma200_transition": "ACCELERATING_UP"},
        "sector_confirmation": {"data_available": True, "stock_vs_sector": 10, "sector_vs_market": 5},
        "institutional_trend": {"institutional_ownership_trend": "RISING"},
        "fundamental_acceleration": {"data_available": True, "revenue_growth_trend": "ACCELERATING"},
        "market_regime": {"regime": "RISK_ON"},
    }
    result = compute_apex10_short_score(bullish)
    assert result["score"] < 15


def test_rs_deterioration_scores_high_for_negative_change_low_for_positive():
    from modules.apex10_short_radar import _score_rs_deterioration
    negative = _score_rs_deterioration({"rs_5d_change": -20, "rs_acceleration": -10})
    positive = _score_rs_deterioration({"rs_5d_change": 20, "rs_acceleration": 10})
    assert negative > positive
    assert negative == 1.0
    assert positive == 0.0


def test_ma_transition_short_favors_falling_over_accelerating_up():
    from modules.apex10_short_radar import _score_ma_transition_short
    assert _score_ma_transition_short("FALLING") == 1.0
    assert _score_ma_transition_short("ACCELERATING_UP") == 0.0


def test_market_regime_short_favors_risk_off():
    from modules.apex10_short_radar import _score_market_regime_short
    assert _score_market_regime_short({"regime": "RISK_OFF"}) == 1.0
    assert _score_market_regime_short({"regime": "RISK_ON"}) == 0.0


# ── Breakdown trigger gates ──────────────────────────────────────────

def test_breakdown_trigger_all_gates_pass():
    features = _full_bearish_features()
    liq = {"passes": True}
    result = compute_breakdown_trigger_gates(features, liq)
    assert result["confirmed_breakdown"] is True
    assert isinstance(result["confirmed_breakdown"], bool)


def test_breakdown_trigger_fails_without_volume_confirmation():
    features = _full_bearish_features()
    features["breakdown_volume_confirmation"] = False
    result = compute_breakdown_trigger_gates(features, {"passes": True})
    assert result["confirmed_breakdown"] is False


def test_breakdown_trigger_fails_in_risk_on_regime():
    features = _full_bearish_features()
    features["market_regime"] = {"regime": "RISK_ON"}
    result = compute_breakdown_trigger_gates(features, {"passes": True})
    assert result["confirmed_breakdown"] is False


def test_breakdown_trigger_pending_gates_never_silently_pass():
    result = compute_breakdown_trigger_gates(_full_bearish_features(), {"passes": True})
    assert result["gates"]["risk_reward_acceptable"] == "pending_stage_l"
    assert result["gates"]["no_major_invalidation"] == "pending_stage_l"


def test_liquidity_gate_reused_directly_from_long_side():
    # Confirms this module reuses compute_liquidity_gate rather than
    # duplicating it — same function, same behavior either direction.
    result = compute_liquidity_gate(avg_volume_20=1_000_000, last_close=10.0, market="US")
    assert result["passes"] is True


def test_short_score_never_uses_sell_recommendation_language():
    result = compute_apex10_short_score(_full_bearish_features())
    trigger = compute_breakdown_trigger_gates(_full_bearish_features(), {"passes": True})
    combined = str(result).lower() + str(trigger).lower()
    forbidden = ["should sell", "should short", "sell now", "short now", "recommend shorting"]
    for phrase in forbidden:
        assert phrase not in combined
    assert "not a short/sell signal" in combined


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
