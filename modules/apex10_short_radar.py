"""
modules/apex10_short_radar.py — Apex the Great X, short side: score & state

Bearish mirror of modules/apex10_radar.py. Reuses that module's
direction-neutral pieces directly (_clip01, classify_apex10_state,
compute_liquidity_gate — none of those care about direction) and
mirrors the direction-SENSITIVE per-component scoring functions with
inverted interpretations, per component:

  Long favors  →  Short favors
  RS rising    →  RS falling
  Near a high  →  Near a low
  MA turning up→  MA falling
  Rising inst. →  Falling inst. ownership
  RISK_ON      →  RISK_OFF

Same discipline as the long side: weights are HYPOTHESES, not tuned
against outcomes; nothing here ever emits a sell/short recommendation,
only a score, a state, and separately-gated trigger conditions.

The state vocabulary (EARLY/WATCH/DEVELOPING/READY/TRIGGER) and its
thresholds are reused as-is via classify_apex10_state() — a short
"TRIGGER" means the same thing structurally a long "TRIGGER" does
(confirmed-computable gates all passed), just for the opposite
direction. These will be stored under a different observation_type
("apex10_short_radar"), so there's no risk of the two being confused in
the data even though the labels are shared.
"""
from __future__ import annotations
from typing import Optional

from modules.apex10_radar import (
    _clip01, classify_apex10_state, compute_liquidity_gate, DEFAULT_LIQUIDITY_MIN_DOLLAR_VOLUME,
)

DEFAULT_SHORT_WEIGHTS = {
    "rs_deterioration": 15,
    "breakdown_proximity": 10,
    "structure_compression": 10,
    "volume_behavior": 10,
    "lower_high_structure": 10,
    "ma50_transition": 10,
    "ma200_transition": 5,
    "sector_confirmation": 10,
    "institutional_trend": 5,
    "fundamental_deceleration": 10,
    "market_regime": 5,
}
assert sum(DEFAULT_SHORT_WEIGHTS.values()) == 100, "DEFAULT_SHORT_WEIGHTS must sum to 100"


# ══════════════════════════════════════════════════════════════════════
# PER-COMPONENT SCORING — mirrors modules/apex10_radar.py's private
# functions, inverted where direction matters.
# ══════════════════════════════════════════════════════════════════════

def _score_rs_deterioration(rs: dict) -> Optional[float]:
    change, accel = rs.get("rs_5d_change"), rs.get("rs_acceleration")
    if change is None and accel is None:
        return None
    parts = []
    if change is not None:
        parts.append(_clip01((-change + 20) / 40))    # a NEGATIVE change scores high here
    if accel is not None:
        parts.append(_clip01((-accel + 10) / 20))
    return round(sum(parts) / len(parts), 3)


def _score_breakdown_proximity(bp: dict) -> Optional[float]:
    mapping = {"IMMINENT": 1.0, "VERY_CLOSE": 0.8, "DEVELOPING": 0.5, "EARLY": 0.25, "DISTANT": 0.0}
    return mapping.get(bp.get("state"))


def _score_structure_compression(vol: dict) -> Optional[float]:
    """Compression favors an imminent move in EITHER direction — same
    interpretation as the long side, reused conceptually (not imported
    directly since apex10_radar's version is a private underscore
    function not meant as a public import point; duplicating this one
    specific mapping is cheap and keeps this module's public surface
    self-contained)."""
    ratio = vol.get("range_ratio_5_20")
    if ratio is None:
        return None
    score = _clip01(1.0 - ratio)
    if vol.get("volatility_contraction_acceleration"):
        score = _clip01(score + 0.15)
    return round(score, 3)


def _score_volume_behavior(volume: dict, breakdown_volume_confirmation: bool) -> Optional[float]:
    if volume.get("relative_volume") is None and volume.get("volume_trend") == "UNKNOWN":
        return None
    if breakdown_volume_confirmation:
        return 1.0
    if volume.get("volume_contraction"):
        return 0.6
    if volume.get("volume_trend") == "FLAT":
        return 0.3
    return 0.0


def _score_lower_high_structure(structure: dict) -> Optional[float]:
    lhfs = structure.get("lower_highs_flat_support")
    ms_lh_ll = structure.get("ms_lh_ll")
    if lhfs is None and ms_lh_ll is None:
        return None
    if lhfs is True:
        return 1.0
    if ms_lh_ll is True:
        return 0.5
    return 0.0


def _score_ma_transition_short(label: str) -> Optional[float]:
    # apex10_features._classify_ma_transition() doesn't distinguish an
    # "accelerating down" state from plain FALLING — FALLING is treated
    # as the strongest available short-favorable signal.
    mapping = {"FALLING": 1.0, "FLATTENING": 0.3, "TURNING_UP": 0.0, "ACCELERATING_UP": 0.0}
    return mapping.get(label)


def _score_sector_confirmation_short(sector: dict) -> Optional[float]:
    if not sector.get("data_available"):
        return None
    svs, secvm = sector.get("stock_vs_sector", 0), sector.get("sector_vs_market", 0)
    if svs < 0 and secvm < 0:
        return 1.0  # stock weak within an already-weak sector
    if svs < 0 or secvm < 0:
        return 0.6
    return 0.0


def _score_institutional_trend_short(inst: dict) -> Optional[float]:
    mapping = {"FALLING": 1.0, "FLAT": 0.5, "RISING": 0.0}
    return mapping.get(inst.get("institutional_ownership_trend"))


def _score_fundamental_deceleration(fund: dict) -> Optional[float]:
    if not fund.get("data_available"):
        return None
    mapping = {"DECELERATING": 1.0, "FLAT": 0.5, "ACCELERATING": 0.0}
    scores = [mapping[v] for k, v in fund.items() if k.endswith("_trend") and v in mapping]
    return round(sum(scores) / len(scores), 3) if scores else None


def _score_market_regime_short(regime: dict) -> Optional[float]:
    mapping = {"RISK_OFF": 1.0, "NEUTRAL": 0.5, "RISK_ON": 0.0}
    return mapping.get(regime.get("regime"))


# ══════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE — mirrors compute_apex10_score() exactly in structure
# ══════════════════════════════════════════════════════════════════════

def compute_apex10_short_score(features: dict, weights: dict = None) -> dict:
    weights = weights or DEFAULT_SHORT_WEIGHTS
    if "relative_strength" not in features:
        return {"score": None, "state": "UNKNOWN", "evidence_quality": "LOW",
               "components": {}, "reason": "Insufficient price history to compute any component."}

    component_scores = {
        "rs_deterioration": _score_rs_deterioration(features["relative_strength"]),
        "breakdown_proximity": _score_breakdown_proximity(features["breakdown_proximity"]),
        "structure_compression": _score_structure_compression(features["volatility"]),
        "volume_behavior": _score_volume_behavior(
            features["volume"], features.get("breakdown_volume_confirmation", False)),
        "lower_high_structure": _score_lower_high_structure(features["structure"]),
        "ma50_transition": _score_ma_transition_short(features["moving_averages"].get("ma50_transition")),
        "ma200_transition": _score_ma_transition_short(features["moving_averages"].get("ma200_transition")),
        "sector_confirmation": _score_sector_confirmation_short(features["sector_confirmation"]),
        "institutional_trend": _score_institutional_trend_short(features["institutional_trend"]),
        "fundamental_deceleration": _score_fundamental_deceleration(features["fundamental_acceleration"]),
        "market_regime": _score_market_regime_short(features["market_regime"]),
    }

    available = {k: v for k, v in component_scores.items() if v is not None}
    available_weight = sum(weights[k] for k in available)
    total_possible_weight = sum(weights.values())

    if available_weight == 0:
        return {"score": None, "state": "UNKNOWN", "evidence_quality": "LOW",
               "components": component_scores, "reason": "No components had usable data."}

    raw = sum(available[k] * weights[k] for k in available)
    score = round(raw / available_weight * 100, 1)
    coverage_pct = round(available_weight / total_possible_weight * 100, 1)
    evidence_quality = "HIGH" if coverage_pct >= 90 else ("MEDIUM" if coverage_pct >= 60 else "LOW")

    return {
        "score": score, "evidence_quality": evidence_quality, "coverage_pct": coverage_pct,
        "components": component_scores, "component_weights": dict(weights),
    }


# ══════════════════════════════════════════════════════════════════════
# BREAKDOWN TRIGGER — mirrors compute_breakout_trigger_gates() exactly
# ══════════════════════════════════════════════════════════════════════

def compute_breakdown_trigger_gates(features: dict, liquidity_gate: dict) -> dict:
    """
    Same structure as the long side's compute_breakout_trigger_gates():
    checks currently-computable gates only. risk_reward_acceptable and
    no_major_invalidation remain "pending_stage_l". Additionally never
    claims to account for short-squeeze risk, borrow cost, or borrow
    availability — see module docstring's honesty note on this.
    """
    bp = features.get("breakdown_proximity", {})
    price_breaks_support = (bp.get("state") == "IMMINENT"
                           and bp.get("distance_to_support_pct") == 0.0)
    volume_confirmed = bool(features.get("breakdown_volume_confirmation"))
    liquidity_ok = liquidity_gate.get("passes") is True
    regime = features.get("market_regime", {}).get("regime")
    regime_compatible = None if regime in (None, "UNKNOWN") else regime != "RISK_ON"

    gates = {
        "price_breaks_support": price_breaks_support,
        "volume_confirmation": volume_confirmed,
        "minimum_liquidity": liquidity_ok,
        "market_regime_compatible": regime_compatible,
        "risk_reward_acceptable": "pending_stage_l",
        "no_major_invalidation": "pending_stage_l",
    }

    hard_gates_pass = price_breaks_support and volume_confirmed and liquidity_ok
    confirmed_breakdown = bool(hard_gates_pass and regime_compatible is not False)

    return {
        "gates": gates, "confirmed_breakdown": confirmed_breakdown,
        "note": ("confirmed_breakdown means the currently-computable gates all passed — "
                "it is NOT a short/sell signal. risk_reward_acceptable and "
                "no_major_invalidation cannot be evaluated until Stage L exists. Short-squeeze "
                "risk, borrow cost, and borrow availability are not modeled at all. Human "
                "review and the Gate workflow remain required before any trade."),
    }
