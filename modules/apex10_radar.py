"""
modules/apex10_radar.py — Apex the Great X: Stage D, Pre-Breakout Score & State

Combines Stage C's individual feature families
(modules/apex10_features.py) into ONE composite 0-100 score plus a
state label — and, separately, a set of breakout-trigger gates.

Per the spec's explicit instruction, the weights below are HYPOTHESES,
not tuned or optimized against any historical outcome — that question
belongs to modules/walk_forward.py, once real data exists to ask it of.
"Do not tune them using the same historical data used to judge
performance" applies here exactly as it does everywhere else in this
project.

Nothing in this module ever emits a buy/sell recommendation. The score
and state describe evidence; converting evidence into a trade decision
is Stage L (Trade Intelligence) plus a human — see the spec's explicit
GATE section: "DO NOT automatically execute trades... ADEX is initially
RESEARCH + SIGNAL + RISK + DECISION SUPPORT."

No Streamlit import — pure computation, same discipline as Stage C.
Nothing here persists anything (that's Stage E): every function is a
stateless transform — features in, score/state/gates out.
"""
from __future__ import annotations
from typing import Optional

# ── Score weights — HYPOTHESES, not optimized. See module docstring.
# Mirrors the spec's own suggested initial weights exactly. ────────────
DEFAULT_WEIGHTS = {
    "rs_improvement": 15,
    "breakout_proximity": 10,
    "structure_compression": 10,
    "volume_behavior": 10,
    "higher_low_structure": 10,
    "ma50_transition": 10,
    "ma200_transition": 5,
    "sector_confirmation": 10,
    "institutional_trend": 5,
    "fundamental_acceleration": 10,
    "market_regime": 5,
}
assert sum(DEFAULT_WEIGHTS.values()) == 100, "DEFAULT_WEIGHTS must sum to 100"

# ── State classification thresholds — configurable, per spec. ─────────
DEFAULT_STATE_THRESHOLDS = {"watch": 60, "developing": 70, "ready": 80, "trigger": 90}

# ── Liquidity gates — market-specific, configurable. Placeholders
# pending your real calibration, flagged explicitly rather than
# silently assumed correct (spec's LIQUIDITY PROTECTION section is
# explicit that Apex the Great X "must not become a microcap/
# illiquidity detector"). ──────────────────────────────────────────────
DEFAULT_LIQUIDITY_MIN_DOLLAR_VOLUME = {"US": 5_000_000.0, "NGX": 50_000_000.0}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ══════════════════════════════════════════════════════════════════════
# PER-COMPONENT SCORING
#
# Each function maps ONE Stage C feature family's output onto a 0.0-1.0
# "how pre-breakout does this look" score, or None when the underlying
# data is UNKNOWN. Every mapping is a simple, documented, monotonic
# function — deliberately not "smart": a clever undocumented formula
# would itself be an unvalidated model hiding inside what's supposed to
# be a transparent, inspectable hypothesis.
# ══════════════════════════════════════════════════════════════════════

def _score_rs_improvement(rs: dict) -> Optional[float]:
    change, accel = rs.get("rs_5d_change"), rs.get("rs_acceleration")
    if change is None and accel is None:
        return None
    parts = []
    if change is not None:
        parts.append(_clip01((change + 20) / 40))    # -20..+20 -> 0..1
    if accel is not None:
        parts.append(_clip01((accel + 10) / 20))      # -10..+10 -> 0..1
    return round(sum(parts) / len(parts), 3)


def _score_breakout_proximity(bp: dict) -> Optional[float]:
    mapping = {"IMMINENT": 1.0, "VERY_CLOSE": 0.8, "DEVELOPING": 0.5, "EARLY": 0.25, "DISTANT": 0.0}
    return mapping.get(bp.get("state"))  # None passes through for UNKNOWN


def _score_structure_compression(vol: dict) -> Optional[float]:
    ratio = vol.get("range_ratio_5_20")
    if ratio is None:
        return None
    score = _clip01(1.0 - ratio)
    if vol.get("volatility_contraction_acceleration"):
        score = _clip01(score + 0.15)
    return round(score, 3)


def _score_volume_behavior(volume: dict, breakout_volume_confirmation: bool) -> Optional[float]:
    if volume.get("relative_volume") is None and volume.get("volume_trend") == "UNKNOWN":
        return None
    if breakout_volume_confirmation:
        return 1.0
    if volume.get("volume_contraction"):
        return 0.6
    if volume.get("volume_trend") == "FLAT":
        return 0.3
    return 0.0


def _score_higher_low_structure(structure: dict) -> Optional[float]:
    hlfr = structure.get("higher_lows_flat_resistance")
    hh_hl = structure.get("ms_hh_hl")
    if hlfr is None and hh_hl is None:
        return None
    if hlfr is True:
        return 1.0
    if hh_hl is True:
        return 0.5
    return 0.0


def _score_ma_transition(label: str) -> Optional[float]:
    mapping = {"ACCELERATING_UP": 1.0, "TURNING_UP": 0.7, "FLATTENING": 0.3, "FALLING": 0.0}
    return mapping.get(label)  # None for "UNKNOWN"


def _score_sector_confirmation(sector: dict) -> Optional[float]:
    if not sector.get("data_available"):
        return None
    svs, secvm = sector.get("stock_vs_sector", 0), sector.get("sector_vs_market", 0)
    if svs > 0 and secvm > 0:
        return 1.0
    if svs > 0 or secvm > 0:
        return 0.6
    return 0.0


def _score_institutional_trend(inst: dict) -> Optional[float]:
    mapping = {"RISING": 1.0, "FLAT": 0.5, "FALLING": 0.0}
    return mapping.get(inst.get("institutional_ownership_trend"))


def _score_fundamental_acceleration(fund: dict) -> Optional[float]:
    if not fund.get("data_available"):
        return None
    mapping = {"ACCELERATING": 1.0, "FLAT": 0.5, "DECELERATING": 0.0}
    scores = [mapping[v] for k, v in fund.items() if k.endswith("_trend") and v in mapping]
    return round(sum(scores) / len(scores), 3) if scores else None


def _score_market_regime(regime: dict) -> Optional[float]:
    mapping = {"RISK_ON": 1.0, "NEUTRAL": 0.5, "RISK_OFF": 0.0}
    return mapping.get(regime.get("regime"))


# ══════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE
# ══════════════════════════════════════════════════════════════════════

def compute_apex10_score(features: dict, weights: dict = None) -> dict:
    """
    Takes the FULL output of
    modules.apex10_features.compute_precursor_features() and combines
    it into one 0-100 score. Renormalizes over only the components that
    had real (non-UNKNOWN) data, so a stock isn't unfairly punished
    purely for missing data in one family — but see `evidence_quality`
    below: this is exactly why it exists and must always be read
    ALONGSIDE the score, never dropped or hidden. A 95 with LOW evidence
    quality is a different thing from a 95 with HIGH evidence quality,
    even though the number looks identical.
    """
    weights = weights or DEFAULT_WEIGHTS
    if "relative_strength" not in features:
        return {"score": None, "state": "UNKNOWN", "evidence_quality": "LOW",
               "components": {}, "reason": "Insufficient price history to compute any component."}

    component_scores = {
        "rs_improvement": _score_rs_improvement(features["relative_strength"]),
        "breakout_proximity": _score_breakout_proximity(features["breakout_proximity"]),
        "structure_compression": _score_structure_compression(features["volatility"]),
        "volume_behavior": _score_volume_behavior(
            features["volume"], features.get("breakout_volume_confirmation", False)),
        "higher_low_structure": _score_higher_low_structure(features["structure"]),
        "ma50_transition": _score_ma_transition(features["moving_averages"].get("ma50_transition")),
        "ma200_transition": _score_ma_transition(features["moving_averages"].get("ma200_transition")),
        "sector_confirmation": _score_sector_confirmation(features["sector_confirmation"]),
        "institutional_trend": _score_institutional_trend(features["institutional_trend"]),
        "fundamental_acceleration": _score_fundamental_acceleration(features["fundamental_acceleration"]),
        "market_regime": _score_market_regime(features["market_regime"]),
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


def classify_apex10_state(score: Optional[float], thresholds: dict = None) -> str:
    if score is None:
        return "UNKNOWN"
    thresholds = thresholds or DEFAULT_STATE_THRESHOLDS
    if score >= thresholds["trigger"]:
        return "TRIGGER"
    if score >= thresholds["ready"]:
        return "READY"
    if score >= thresholds["developing"]:
        return "DEVELOPING"
    if score >= thresholds["watch"]:
        return "WATCH"
    return "EARLY"


# ══════════════════════════════════════════════════════════════════════
# LIQUIDITY GATE & BREAKOUT TRIGGER — deliberately separate from the
# score. Per spec: "The score alone must NEVER automatically mean BUY."
# ══════════════════════════════════════════════════════════════════════

def compute_liquidity_gate(avg_volume_20: Optional[float], last_close: Optional[float],
                           market: str = "US", min_dollar_volume: dict = None) -> dict:
    """A hard safety gate, independent of the score, so a low-volume
    stock can never buy its way to a high score just because its
    volatility happens to be low. Thresholds are placeholders pending
    your real calibration."""
    min_dv = (min_dollar_volume or DEFAULT_LIQUIDITY_MIN_DOLLAR_VOLUME).get(market)
    if min_dv is None:
        return {"passes": None, "reason": f"No liquidity threshold configured for market={market!r}."}
    if avg_volume_20 is None or last_close is None:
        return {"passes": None, "reason": "Insufficient data to compute dollar volume."}
    dollar_volume = avg_volume_20 * last_close
    return {
        "passes": bool(dollar_volume >= min_dv), "dollar_volume": round(dollar_volume, 0),
        "min_required": min_dv, "market": market,
    }


def compute_breakout_trigger_gates(features: dict, liquidity_gate: dict) -> dict:
    """
    Checks the trigger gates that ARE computable at this stage. Two
    gates the spec also asks for — acceptable risk/reward, no major
    invalidation — require an entry/stop/target, which don't exist
    until Stage L (Trade Intelligence). Those are explicitly reported as
    "pending_stage_l", never faked as passing, so a caller can't
    mistake "not yet checkable" for "checked and fine."
    """
    bp = features.get("breakout_proximity", {})
    price_breaks_resistance = (bp.get("state") == "IMMINENT"
                              and bp.get("distance_to_resistance_pct") == 0.0)
    volume_confirmed = bool(features.get("breakout_volume_confirmation"))
    liquidity_ok = liquidity_gate.get("passes") is True
    regime = features.get("market_regime", {}).get("regime")
    regime_compatible = None if regime in (None, "UNKNOWN") else regime != "RISK_OFF"

    gates = {
        "price_breaks_resistance": price_breaks_resistance,
        "volume_confirmation": volume_confirmed,
        "minimum_liquidity": liquidity_ok,
        "market_regime_compatible": regime_compatible,
        "risk_reward_acceptable": "pending_stage_l",
        "no_major_invalidation": "pending_stage_l",
    }

    hard_gates_pass = price_breaks_resistance and volume_confirmed and liquidity_ok
    confirmed_breakout = bool(hard_gates_pass and regime_compatible is not False)

    return {
        "gates": gates, "confirmed_breakout": confirmed_breakout,
        "note": ("confirmed_breakout means the currently-computable gates all passed — "
                "it is NOT a buy signal. risk_reward_acceptable and no_major_invalidation "
                "cannot be evaluated until Stage L exists. Human review and the Gate "
                "workflow remain required before any trade, per the spec."),
    }
