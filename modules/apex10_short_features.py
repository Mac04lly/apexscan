"""
modules/apex10_short_features.py — Apex the Great X, short side: features

Bearish mirror of modules/apex10_features.py. Deliberately reuses that
module's functions directly wherever the underlying computation is
direction-neutral — RS change, ATR/volatility, volume ratios, MA slope
classification, 52-week position, fundamentals/institutional/sector/
regime are all just numbers or labels; only the SCORING interpretation
of them differs by direction (built in apex10_short_radar.py, not
here). Only two things are genuinely new:
  - Breakdown proximity (distance to nearest SUPPORT below price,
    mirroring breakout proximity's distance to nearest resistance).
  - Lower-high structure detection (mirroring higher-low structure).

Reuses scanner.detect_market_structure() exactly as Stage C does — that
function already returns `ms_lh_ll` (confirmed by reading
analyze_stock_short(), which already consumes it) — nothing new needed
there either.

════════════════════════════════════════════════════════════════════════
NO-LOOK-AHEAD — reused, not reimplemented
════════════════════════════════════════════════════════════════════════
Uses modules.apex10_features._slice_as_of() — the same tested boundary
(6 dedicated tests including future-price/future-volume spike
injections) — rather than writing new slicing logic that could get the
guarantee subtly wrong a second time.

════════════════════════════════════════════════════════════════════════
SHORTING-SPECIFIC RISK THIS MODULE DOES NOT AND CANNOT MODEL
════════════════════════════════════════════════════════════════════════
Short-squeeze risk (a heavily-shorted stock can spike violently against
a short position) depends on real-time short-interest-% and days-to-
cover data this app doesn't have a reliable source for — `short_pct_float`
is already `None` in scanner.py's own short-side output ("filled in by
dashboard.py's fundamentals enrichment" — i.e., not always available).
Borrow cost and borrow availability aren't modeled at all; neither is
the mechanical fact that a short position's loss is theoretically
unbounded while a long position's is capped at 100%. None of this is
new information — it's the same honesty this project has applied to
every other data gap — but it matters more here, since shorting is
mechanically riskier than the long side this whole system started from.
"""
from __future__ import annotations
from typing import Optional

import pandas as pd

from modules.apex10_features import (
    _slice_as_of, _find_swings, SWING_LOOKBACK, SWING_HIGH_FLATNESS_TOLERANCE_PCT,
    compute_relative_strength_features, compute_volatility_features, compute_volume_features,
    compute_moving_average_features, compute_52w_position_features,
    compute_fundamental_acceleration_features, compute_institutional_trend_features,
    compute_sector_confirmation_features, compute_market_regime_feature,
)
from scanner import detect_market_structure

SUPPORT_THRESHOLDS_PCT = {"imminent": 1.0, "very_close": 3.0, "developing": 7.0, "early": 15.0}
BREAKDOWN_VOLUME_MULT = 1.4  # matches BREAKOUT_VOLUME_MULT in apex10_features.py


def compute_breakdown_proximity_features(hist: pd.DataFrame, thresholds: dict = None) -> dict:
    """Mirror of compute_breakout_proximity_features: distance to the
    NEAREST support level below current price — the highest of the
    computed rolling lows that's still at or below current price (the
    closest floor underneath where price sits right now)."""
    thresholds = thresholds or SUPPORT_THRESHOLDS_PCT
    if hist is None or len(hist) < 21:
        return {"distance_to_support_pct": None, "support_price": None,
               "support_source": None, "state": "UNKNOWN"}

    current = float(hist["Close"].iloc[-1])
    windows = {
        "20d_low": min(20, len(hist)), "50d_low": min(50, len(hist)),
        "3mo_low": min(63, len(hist)), "6mo_low": min(126, len(hist)),
        "52w_low": min(252, len(hist)),
    }
    lows = {name: float(hist["Low"].iloc[-bars:].min()) for name, bars in windows.items()}

    below = {name: l for name, l in lows.items() if l <= current}
    if below:
        support_source = max(below, key=below.get)  # nearest = highest of the qualifying lows
        support_price = below[support_source]
        distance_pct = round((current - support_price) / current * 100, 2)
    else:
        support_source = "at_or_below_all_computed_lows"
        support_price = current
        distance_pct = 0.0

    if distance_pct <= thresholds["imminent"]:
        state = "IMMINENT"
    elif distance_pct <= thresholds["very_close"]:
        state = "VERY_CLOSE"
    elif distance_pct <= thresholds["developing"]:
        state = "DEVELOPING"
    elif distance_pct <= thresholds["early"]:
        state = "EARLY"
    else:
        state = "DISTANT"

    return {
        "distance_to_support_pct": distance_pct, "support_price": round(support_price, 2),
        "support_source": support_source, "state": state,
        "all_lows": {k: round(v, 2) for k, v in lows.items()},
    }


def compute_short_structure_features(hist: pd.DataFrame) -> dict:
    """Mirror of compute_structure_features. Reuses
    scanner.detect_market_structure() exactly as the long side does —
    same single call, just reading `ms_lh_ll` instead of `ms_hh_hl` —
    and reuses apex10_features._find_swings() for the lower-highs +
    roughly-flat-support pattern (mirroring higher_lows_flat_resistance)."""
    if hist is None or len(hist) < 21:
        return {"ms_structure": "UNKNOWN", "ms_lh_ll": None,
               "lower_highs_flat_support": None}

    ms = detect_market_structure(hist)  # reused verbatim, not duplicated

    swing_highs, swing_lows = _find_swings(hist, SWING_LOOKBACK)
    lower_highs_flat_support = None
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        sh = [v for _, v in swing_highs[-2:]]
        sl = [v for _, v in swing_lows[-2:]]
        falling_highs = sh[-1] < sh[-2]
        flat_lows = (abs(sl[-1] - sl[-2]) / sl[-2] * 100 <= SWING_HIGH_FLATNESS_TOLERANCE_PCT
                    if sl[-2] else False)
        lower_highs_flat_support = bool(falling_highs and flat_lows)

    return {
        "ms_structure": ms["ms_structure"], "ms_lh_ll": bool(ms.get("ms_lh_ll", False)),
        "ms_last_swing_high": ms["ms_last_swing_high"], "ms_last_swing_low": ms["ms_last_swing_low"],
        "lower_highs_flat_support": lower_highs_flat_support,
    }


def compute_short_precursor_features(hist: pd.DataFrame, benchmark_close: pd.Series,
                                     as_of_date=None, sector_avg_rs: Optional[float] = None,
                                     fundamental_snapshots: Optional[list] = None,
                                     institutional_snapshots: Optional[list] = None,
                                     results_df=None) -> dict:
    """
    Single entry point, mirroring compute_precursor_features(). Slices
    to as_of_date FIRST via the same tested _slice_as_of() boundary,
    then computes every feature family from that truncated view —
    reusing the long-side functions directly for everything
    direction-neutral, and the two new functions above for what isn't.
    """
    hist_sliced = _slice_as_of(hist, as_of_date)
    bench_sliced = _slice_as_of(benchmark_close, as_of_date)

    bars_available = len(hist_sliced) if hist_sliced is not None else 0
    if bars_available < 21:
        return {
            "as_of_date": str(as_of_date) if as_of_date else None,
            "data_quality": {"bars_available": bars_available, "sufficient_history": False,
                            "note": "Fewer than 21 bars available as of this date."},
        }

    rs_features = compute_relative_strength_features(hist_sliced["Close"], bench_sliced)
    breakdown = compute_breakdown_proximity_features(hist_sliced)
    volatility = compute_volatility_features(hist_sliced)          # reused, direction-neutral
    volume = compute_volume_features(hist_sliced)                  # reused, direction-neutral
    structure = compute_short_structure_features(hist_sliced)
    ma = compute_moving_average_features(hist_sliced)              # reused, direction-neutral
    pos_52w = compute_52w_position_features(hist_sliced)           # reused, direction-neutral
    fundamentals = compute_fundamental_acceleration_features(fundamental_snapshots)  # reused
    institutional = compute_institutional_trend_features(institutional_snapshots)    # reused
    sector = compute_sector_confirmation_features(rs_features.get("rs_current"), sector_avg_rs)  # reused
    regime = compute_market_regime_feature(results_df)             # reused

    near_support = breakdown["state"] in ("IMMINENT", "VERY_CLOSE")
    breakdown_volume_confirmation = bool(
        volume.get("relative_volume") is not None
        and volume["relative_volume"] >= BREAKDOWN_VOLUME_MULT
        and breakdown["state"] == "IMMINENT"
    )

    return {
        "as_of_date": str(as_of_date) if as_of_date else None,
        "data_quality": {"bars_available": bars_available, "sufficient_history": bars_available >= 252},
        "relative_strength": rs_features,
        "breakdown_proximity": breakdown,
        "volatility": volatility,
        "volume": volume,
        "structure": structure,
        "moving_averages": ma,
        "position_52w": pos_52w,
        "fundamental_acceleration": fundamentals,
        "institutional_trend": institutional,
        "sector_confirmation": sector,
        "market_regime": regime,
        "breakdown_volume_confirmation": breakdown_volume_confirmation,
        "_near_support": near_support,
    }
