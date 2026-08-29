"""
modules/apex10_features.py — Apex the Great X: Stage C, Pre-Breakout Feature Engine

Deterministic, testable, no Streamlit import, operates on historical
OHLCV + a benchmark close series. Handles missing/insufficient data by
returning None/"UNKNOWN" for that specific field — never a fabricated
number, never a silent zero standing in for "don't know."

════════════════════════════════════════════════════════════════════════
THE SINGLE MOST IMPORTANT RULE IN THIS FILE — NO LOOK-AHEAD
════════════════════════════════════════════════════════════════════════
Every public compute_* function in this module is written to operate
ONLY on trailing windows relative to the LAST ROW of whatever `hist`
Series/DataFrame it's given — the same convention scanner.py's own
compute_rs(), adr_pct(), volume_surge_ratio(), and detect_base_breakout()
already use (confirmed by reading them before writing this file; none
of them peek ahead of their input's last row).

That means look-ahead safety is enforced ONE PLACE ONLY:
_slice_as_of(), called first, before ANY feature is computed. If the
caller passes an `as_of_date`, hist/benchmark_close/sector_close are all
truncated to that date before anything downstream ever sees them. Every
reused scanner.py function then operates on trailing windows of an
already-truncated frame, and can never see a bar beyond as_of_date
because there isn't one — it was never given the data, not merely told
not to look at it.

This is verified, not just asserted: see
tests/test_apex10_no_lookahead.py — the spec's mandatory test appends
future rows AFTER computing at date T and asserts the result at T is
byte-for-byte identical.

════════════════════════════════════════════════════════════════════════
FEATURE FAMILIES 8-11 — HONEST SCOPE LIMIT
════════════════════════════════════════════════════════════════════════
Fundamental Acceleration (8), Institutional Accumulation (9), Sector
Confirmation (10), and Market Regime (11) all require data this module
cannot produce from a single ticker's OHLCV history alone:
  - 8/9 need genuine POINT-IN-TIME fundamental/ownership snapshots.
    This app has no such source today (confirmed in the Stage-A audit
    and already documented as a gap in modules/alpha_validation.py's
    own FIELD_COVERAGE) — using TODAY's fundamentals to score a
    historical date would be exactly the look-ahead bias the spec
    explicitly warns against, so these return "UNKNOWN" unless the
    caller supplies real historical snapshots.
  - 10/11 need cross-sectional (multi-ticker, scan-wide) context — a
    single ticker's price history cannot tell you what its sector or
    the broader market is doing. These accept optional pre-computed
    inputs (sector_avg_rs, results_df) and return "UNKNOWN" without
    them, rather than silently defaulting to a plausible-looking guess.
These four are wired up for real at Stage D, once radar orchestration
has scan-wide context available to hand them.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from scanner import compute_rs, adr_pct, volume_surge_ratio, detect_market_structure

# ── Configurable defaults (mirrors the future config.yaml `apex10:`
# block — every threshold here is a named constant, never a bare number
# buried in an if-statement, per the spec's explicit instruction). ────
RESISTANCE_THRESHOLDS_PCT = {"imminent": 1.0, "very_close": 3.0, "developing": 7.0, "early": 15.0}
VOLATILITY_CONTRACTION_RATIO = 0.85      # atr5/atr20 below this = contracting
VOLUME_CONTRACTION_RATIO = 0.80          # avg_vol5/avg_vol20 below this = contracting
BREAKOUT_VOLUME_MULT = 1.4               # matches the 1.4x threshold used elsewhere in this app
MA_FLAT_SLOPE_THRESHOLD_PCT_PER_DAY = 0.05
SWING_LOOKBACK = 5                       # matches detect_market_structure's own default
SWING_HIGH_FLATNESS_TOLERANCE_PCT = 3.0  # "roughly flat resistance" tolerance


def _slice_as_of(series_or_df, as_of_date) -> "pd.Series | pd.DataFrame":
    """THE no-look-ahead boundary. Everything downstream only ever sees
    what this function lets through. as_of_date=None means "use
    everything given" (the live/current-day case, where the caller is
    already only holding data up to today)."""
    if as_of_date is None or series_or_df is None or len(series_or_df) == 0:
        return series_or_df
    idx = series_or_df.index
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        normalized_idx = pd.to_datetime(idx).normalize()
        cutoff = pd.to_datetime(as_of_date).normalize()
        mask = normalized_idx <= cutoff
        return series_or_df[mask]
    except Exception:
        # If the index isn't date-like, there's nothing safe to slice —
        # fail closed by returning nothing rather than risk leaking data.
        return series_or_df.iloc[0:0]


def _safe_slope_pct_per_day(ma_series: pd.Series, window: int = 10) -> Optional[float]:
    """Linear-regression slope of a moving-average series over its last
    `window` non-NaN bars, normalized to %/day of the series' own level
    so it's comparable across differently-priced stocks."""
    vals = ma_series.dropna()
    if len(vals) < window:
        return None
    recent = vals.iloc[-window:]
    x = np.arange(len(recent))
    try:
        slope, _ = np.polyfit(x, recent.values, 1)
    except Exception:
        return None
    level = recent.iloc[-1]
    if not level:
        return None
    return round((slope / level) * 100, 4)


def _classify_ma_transition(slope: Optional[float], acceleration: Optional[float]) -> str:
    if slope is None:
        return "UNKNOWN"
    if slope < -MA_FLAT_SLOPE_THRESHOLD_PCT_PER_DAY:
        return "FALLING"
    if abs(slope) <= MA_FLAT_SLOPE_THRESHOLD_PCT_PER_DAY:
        return "FLATTENING"
    if acceleration is not None and acceleration > 0:
        return "ACCELERATING_UP"
    return "TURNING_UP"


# ══════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 1 — RELATIVE STRENGTH
# ══════════════════════════════════════════════════════════════════════

def compute_relative_strength_features(close: pd.Series, bench_close: pd.Series) -> dict:
    def _rs_bars_back(bars_back: int) -> Optional[float]:
        if bars_back <= 0:
            c, b = close, bench_close
        elif len(close) <= bars_back or len(bench_close) <= bars_back:
            return None
        else:
            c, b = close.iloc[:-bars_back], bench_close.iloc[:-bars_back]
        if len(c) < 2 or len(b) < 2:
            return None
        return compute_rs(c, b)

    rs_now = _rs_bars_back(0)
    rs_5 = _rs_bars_back(5)
    rs_10 = _rs_bars_back(10)
    rs_20 = _rs_bars_back(20)
    rs_40 = _rs_bars_back(40)

    def _delta(a, b):
        return round(a - b, 2) if (a is not None and b is not None) else None

    change_5d = _delta(rs_now, rs_5)
    change_10d = _delta(rs_now, rs_10)
    change_5d_prior = _delta(rs_5, rs_10)  # the 5d change as of 5 days ago
    rs_acceleration = _delta(change_5d, change_5d_prior)

    return {
        "rs_current": rs_now,
        "rs_5d_change": change_5d,
        "rs_10d_change": change_10d,
        "rs_20d_change": _delta(rs_now, rs_20),
        "rs_40d_change": _delta(rs_now, rs_40),
        "rs_acceleration": rs_acceleration,
        "rs_percentile": None,  # requires scan-wide context — see module docstring
        "rs_percentile_note": "UNKNOWN — requires same-scan percentile ranking, not computable "
                              "from single-ticker history alone.",
    }


# ══════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 2 — BREAKOUT PROXIMITY
# ══════════════════════════════════════════════════════════════════════

def compute_breakout_proximity_features(hist: pd.DataFrame,
                                        thresholds: dict = None) -> dict:
    thresholds = thresholds or RESISTANCE_THRESHOLDS_PCT
    if hist is None or len(hist) < 21:
        return {"distance_to_resistance_pct": None, "resistance_price": None,
               "resistance_source": None, "state": "UNKNOWN"}

    current = float(hist["Close"].iloc[-1])
    windows = {
        "20d_high": min(20, len(hist)), "50d_high": min(50, len(hist)),
        "3mo_high": min(63, len(hist)), "6mo_high": min(126, len(hist)),
        "52w_high": min(252, len(hist)),
    }
    highs = {name: float(hist["High"].iloc[-bars:].max()) for name, bars in windows.items()}

    # Nearest overhead resistance: the SMALLEST high that's still at or
    # above current price (the closest ceiling above where price sits
    # right now). If price is already above every computed high, it's
    # making fresh highs — no overhead resistance in this window set.
    above = {name: h for name, h in highs.items() if h >= current}
    if above:
        resistance_source = min(above, key=above.get)
        resistance_price = above[resistance_source]
        distance_pct = round((resistance_price - current) / resistance_price * 100, 2)
    else:
        resistance_source = "at_or_above_all_computed_highs"
        resistance_price = current
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
        "distance_to_resistance_pct": distance_pct, "resistance_price": round(resistance_price, 2),
        "resistance_source": resistance_source, "state": state,
        "all_highs": {k: round(v, 2) for k, v in highs.items()},
    }


# ══════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 3 — VOLATILITY COMPRESSION
# ══════════════════════════════════════════════════════════════════════

def _true_range(hist: pd.DataFrame) -> pd.Series:
    prev_close = hist["Close"].shift(1)
    tr = pd.concat([
        hist["High"] - hist["Low"],
        (hist["High"] - prev_close).abs(),
        (hist["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def compute_volatility_features(hist: pd.DataFrame) -> dict:
    if hist is None or len(hist) < 21:
        return {"atr_pct_5": None, "atr_pct_10": None, "atr_pct_20": None,
               "atr_ratio_5_20": None, "bb_width_pct": None,
               "range_ratio_5_20": None, "volatility_percentile": None,
               "volatility_contraction": None, "volatility_contraction_acceleration": None}

    tr = _true_range(hist)
    close = hist["Close"]

    def _atr_pct(n):
        if len(tr.dropna()) < n:
            return None
        atr = tr.rolling(n).mean().iloc[-1]
        c = close.iloc[-1]
        return round(atr / c * 100, 3) if c else None

    atr5, atr10, atr20 = _atr_pct(5), _atr_pct(10), _atr_pct(20)
    atr_ratio = round(atr5 / atr20, 3) if (atr5 is not None and atr20) else None

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_width_pct = None
    if not pd.isna(sma20.iloc[-1]) and not pd.isna(std20.iloc[-1]) and sma20.iloc[-1]:
        upper, lower = sma20.iloc[-1] + 2 * std20.iloc[-1], sma20.iloc[-1] - 2 * std20.iloc[-1]
        bb_width_pct = round((upper - lower) / sma20.iloc[-1] * 100, 3)

    range_ratio = None
    if len(hist) >= 20:
        r5 = (hist["High"].iloc[-5:].max() - hist["Low"].iloc[-5:].min()) / close.iloc[-1] * 100
        r20 = (hist["High"].iloc[-20:].max() - hist["Low"].iloc[-20:].min()) / close.iloc[-1] * 100
        range_ratio = round(r5 / r20, 3) if r20 else None

    # Volatility percentile: rank of the CURRENT 20-bar ATR% within its
    # own trailing history (up to 252 bars) — computed entirely from tr,
    # which only ever contains bars already in `hist`, so this cannot
    # see beyond as_of_date any more than anything else here can.
    volatility_percentile = None
    atr_pct_series = (tr.rolling(20).mean() / close * 100).dropna()
    if len(atr_pct_series) >= 20:
        window = atr_pct_series.iloc[-252:]
        volatility_percentile = round((window < window.iloc[-1]).mean() * 100, 1)

    contraction = bool(atr_ratio is not None and atr_ratio < VOLATILITY_CONTRACTION_RATIO)
    contraction_accel = bool(atr5 is not None and atr10 is not None and atr20 is not None
                            and atr5 < atr10 < atr20)

    return {
        "atr_pct_5": atr5, "atr_pct_10": atr10, "atr_pct_20": atr20,
        "atr_ratio_5_20": atr_ratio, "bb_width_pct": bb_width_pct,
        "range_ratio_5_20": range_ratio, "volatility_percentile": volatility_percentile,
        "volatility_contraction": contraction,
        "volatility_contraction_acceleration": contraction_accel,
    }


# ══════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 4 — VOLUME BEHAVIOR
# ══════════════════════════════════════════════════════════════════════

def compute_volume_features(hist: pd.DataFrame) -> dict:
    if hist is None or len(hist) < 21:
        return {"avg_volume_5": None, "avg_volume_10": None, "avg_volume_20": None,
               "avg_volume_50": None, "relative_volume": None, "up_down_volume_ratio": None,
               "volume_trend": "UNKNOWN", "volume_contraction": None}

    vol = hist["Volume"]
    avg5 = vol.iloc[-5:].mean()
    avg10 = vol.iloc[-10:].mean()
    avg20 = vol.iloc[-20:].mean()
    avg50 = vol.iloc[-50:].mean() if len(hist) >= 50 else None

    relative_volume = round(vol.iloc[-1] / avg50, 3) if (avg50 and avg50 > 0) else None

    window = hist.iloc[-20:]
    up_days = window[window["Close"] > window["Close"].shift(1)]
    down_days = window[window["Close"] < window["Close"].shift(1)]
    up_vol, down_vol = up_days["Volume"].sum(), down_days["Volume"].sum()
    up_down_ratio = round(up_vol / down_vol, 3) if down_vol > 0 else None

    if avg10 > avg20 * 1.05:
        trend = "RISING"
    elif avg10 < avg20 * 0.95:
        trend = "FALLING"
    else:
        trend = "FLAT"

    volume_contraction = bool(avg5 < avg20 * VOLUME_CONTRACTION_RATIO)

    return {
        "avg_volume_5": round(avg5, 0), "avg_volume_10": round(avg10, 0),
        "avg_volume_20": round(avg20, 0), "avg_volume_50": round(avg50, 0) if avg50 else None,
        "relative_volume": relative_volume, "up_down_volume_ratio": up_down_ratio,
        "volume_trend": trend, "volume_contraction": volume_contraction,
    }


# ══════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 5 — MARKET STRUCTURE
# ══════════════════════════════════════════════════════════════════════

def _find_swings(hist: pd.DataFrame, n: int = SWING_LOOKBACK):
    """Same swing-point algorithm as scanner.detect_market_structure()
    (same default n), duplicated ONLY because that function doesn't
    expose the raw swing-point list — it returns just the last swing
    high/low and boolean flags. Kept intentionally identical so results
    stay consistent with the existing scanner rather than drifting."""
    if len(hist) < n * 4 + 10:
        return [], []
    highs, lows = hist["High"].values, hist["Low"].values
    swing_highs, swing_lows = [], []
    for i in range(n, len(hist) - n):
        if all(highs[i] >= highs[i - j] for j in range(1, n + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, n + 1)):
            swing_highs.append((i, highs[i]))
        if all(lows[i] <= lows[i - j] for j in range(1, n + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, n + 1)):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def compute_structure_features(hist: pd.DataFrame) -> dict:
    if hist is None or len(hist) < 21:
        return {"ms_structure": "UNKNOWN", "ms_hh_hl": None, "base_depth_pct": None,
               "base_duration_bars": None, "higher_lows_flat_resistance": None}

    ms = detect_market_structure(hist)  # reused verbatim, not duplicated

    bars = min(40, len(hist) - 1)
    window = hist.iloc[-bars:]
    base_high, base_low = float(window["High"].max()), float(window["Low"].min())
    base_depth_pct = round((base_high - base_low) / base_high * 100, 2) if base_high else None
    base_duration_bars = int(bars - 1 - window["High"].values.argmax())

    swing_highs, swing_lows = _find_swings(hist)
    higher_lows_flat_resistance = None
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        sh = [v for _, v in swing_highs[-2:]]
        sl = [v for _, v in swing_lows[-2:]]
        rising_lows = sl[-1] > sl[-2]
        flat_highs = abs(sh[-1] - sh[-2]) / sh[-2] * 100 <= SWING_HIGH_FLATNESS_TOLERANCE_PCT if sh[-2] else False
        higher_lows_flat_resistance = bool(rising_lows and flat_highs)

    return {
        "ms_structure": ms["ms_structure"], "ms_hh_hl": bool(ms["ms_hh_hl"]),
        "ms_last_swing_high": ms["ms_last_swing_high"], "ms_last_swing_low": ms["ms_last_swing_low"],
        "base_depth_pct": base_depth_pct, "base_duration_bars": base_duration_bars,
        "higher_lows_flat_resistance": higher_lows_flat_resistance,
    }


# ══════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 6 — MOVING AVERAGES
# ══════════════════════════════════════════════════════════════════════

def compute_moving_average_features(hist: pd.DataFrame) -> dict:
    if hist is None or len(hist) < 21:
        return {"price_vs_20ma_pct": None, "price_vs_50ma_pct": None, "price_vs_200ma_pct": None,
               "ma50_slope": None, "ma200_slope": None, "ma50_transition": "UNKNOWN",
               "ma200_transition": "UNKNOWN", "ma_compression_pct": None}

    close = hist["Close"]
    current = float(close.iloc[-1])
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    def _pct_vs(ma_series):
        v = ma_series.iloc[-1]
        return round((current / v - 1) * 100, 2) if (not pd.isna(v) and v) else None

    ma50_slope = _safe_slope_pct_per_day(ma50, 10)
    ma50_slope_prior = _safe_slope_pct_per_day(ma50.iloc[:-10], 10) if len(ma50) > 20 else None
    ma50_accel = (round(ma50_slope - ma50_slope_prior, 4)
                 if (ma50_slope is not None and ma50_slope_prior is not None) else None)

    ma200_slope = _safe_slope_pct_per_day(ma200, 10)
    ma200_slope_prior = _safe_slope_pct_per_day(ma200.iloc[:-10], 10) if len(ma200) > 20 else None
    ma200_accel = (round(ma200_slope - ma200_slope_prior, 4)
                  if (ma200_slope is not None and ma200_slope_prior is not None) else None)

    ma_compression_pct = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]) and current:
        ma_compression_pct = round(abs(ma50.iloc[-1] - ma200.iloc[-1]) / current * 100, 2)

    return {
        "price_vs_20ma_pct": _pct_vs(ma20), "price_vs_50ma_pct": _pct_vs(ma50),
        "price_vs_200ma_pct": _pct_vs(ma200),
        "ma50_slope": ma50_slope, "ma50_acceleration": ma50_accel,
        "ma200_slope": ma200_slope, "ma200_acceleration": ma200_accel,
        "ma50_transition": _classify_ma_transition(ma50_slope, ma50_accel),
        "ma200_transition": _classify_ma_transition(ma200_slope, ma200_accel),
        "ma_compression_pct": ma_compression_pct,
    }


# ══════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 7 — 52-WEEK POSITION
# ══════════════════════════════════════════════════════════════════════

def compute_52w_position_features(hist: pd.DataFrame) -> dict:
    if hist is None or len(hist) < 21:
        return {"distance_from_52w_high_pct": None, "distance_from_52w_low_pct": None}
    bars = min(252, len(hist))
    window = hist.iloc[-bars:]
    current = float(hist["Close"].iloc[-1])
    high_252 = float(window["High"].max())
    low_252 = float(window["Low"].min())
    return {
        "distance_from_52w_high_pct": round((current / high_252 - 1) * 100, 2) if high_252 else None,
        "distance_from_52w_low_pct": round((current / low_252 - 1) * 100, 2) if low_252 else None,
        "bars_used": bars, "full_52w_available": bars >= 252,
    }


# ══════════════════════════════════════════════════════════════════════
# FEATURE FAMILIES 8-11 — see module docstring for why these are
# honestly scoped to UNKNOWN pending Stage D's cross-sectional inputs.
# ══════════════════════════════════════════════════════════════════════

def compute_fundamental_acceleration_features(fundamental_snapshots: Optional[list] = None) -> dict:
    """fundamental_snapshots: optional chronological list of dicts, each
    a GENUINE point-in-time snapshot (e.g. [{"date": ..., "revenue_growth":
    ..., "earnings_growth": ..., "eps_growth_%": ..., "roe": ...,
    "debt_to_equity": ..., "free_cash_flow": ...}, ...]). Without real
    historical snapshots, every trend is UNKNOWN — never backfilled from
    today's fundamentals, which would be look-ahead bias."""
    fields = ["revenue_growth", "earnings_growth", "eps_growth_%", "roe",
             "debt_to_equity", "free_cash_flow", "gross_margin"]
    if not fundamental_snapshots or len(fundamental_snapshots) < 2:
        result = {f"{f}_trend": "UNKNOWN" for f in fields}
        result["data_available"] = False
        return result
    latest, prior = fundamental_snapshots[-1], fundamental_snapshots[-2]
    result = {}
    for f in fields:
        lv, pv = latest.get(f), prior.get(f)
        if lv is None or pv is None:
            result[f"{f}_trend"] = "UNKNOWN"
        else:
            result[f"{f}_trend"] = "ACCELERATING" if lv > pv else ("DECELERATING" if lv < pv else "FLAT")
    result["data_available"] = True
    return result


def compute_institutional_trend_features(institutional_snapshots: Optional[list] = None) -> dict:
    """Same point-in-time requirement as Feature Family 8 — see there."""
    if not institutional_snapshots or len(institutional_snapshots) < 2:
        return {"institutional_ownership_trend": "UNKNOWN", "data_available": False}
    latest = institutional_snapshots[-1].get("institutional_ownership")
    prior = institutional_snapshots[-2].get("institutional_ownership")
    if latest is None or prior is None:
        return {"institutional_ownership_trend": "UNKNOWN", "data_available": False}
    trend = "RISING" if latest > prior else ("FALLING" if latest < prior else "FLAT")
    return {"institutional_ownership_trend": trend, "data_available": True,
           "ownership_change_pct_points": round((latest - prior) * 100, 2)}


def compute_sector_confirmation_features(stock_rs: Optional[float] = None,
                                         sector_avg_rs: Optional[float] = None,
                                         market_rs: float = 100.0) -> dict:
    """Requires the caller to supply sector_avg_rs from a scan-wide
    context — deliberately not computed here. See module docstring."""
    if stock_rs is None or sector_avg_rs is None:
        return {"stock_vs_sector": "UNKNOWN", "sector_vs_market": "UNKNOWN", "data_available": False}
    return {
        "stock_vs_sector": round(stock_rs - sector_avg_rs, 2),
        "sector_vs_market": round(sector_avg_rs - market_rs, 2),
        "data_available": True,
    }


def compute_market_regime_feature(results_df=None) -> dict:
    """Reuses ai.engine.classify_market_regime() rather than duplicating
    regime logic — maps its existing Bull/Correction/Sideways/Bear/Unknown
    labels onto this spec's RISK_ON/NEUTRAL/RISK_OFF scheme."""
    if results_df is None:
        return {"regime": "UNKNOWN", "data_available": False}
    from ai.engine import classify_market_regime
    raw = classify_market_regime(results_df)
    mapped = {"Bull": "RISK_ON", "Correction": "NEUTRAL", "Sideways": "NEUTRAL",
             "Bear": "RISK_OFF"}.get(raw, "UNKNOWN")
    return {"regime": mapped, "raw_regime_label": raw, "data_available": mapped != "UNKNOWN"}


# ══════════════════════════════════════════════════════════════════════
# TOP-LEVEL ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def compute_precursor_features(hist: pd.DataFrame, benchmark_close: pd.Series,
                               as_of_date=None, sector_avg_rs: Optional[float] = None,
                               fundamental_snapshots: Optional[list] = None,
                               institutional_snapshots: Optional[list] = None,
                               results_df=None) -> dict:
    """
    Single entry point: slices everything to as_of_date FIRST (see
    _slice_as_of and the module docstring), then computes every feature
    family from that — and only that — truncated view.
    """
    hist_sliced = _slice_as_of(hist, as_of_date)
    bench_sliced = _slice_as_of(benchmark_close, as_of_date)

    bars_available = len(hist_sliced) if hist_sliced is not None else 0
    if bars_available < 21:
        return {
            "as_of_date": str(as_of_date) if as_of_date else None,
            "data_quality": {"bars_available": bars_available, "sufficient_history": False,
                            "note": "Fewer than 21 bars available as of this date — most "
                                   "features cannot be computed reliably yet."},
        }

    rs_features = compute_relative_strength_features(hist_sliced["Close"], bench_sliced)
    breakout = compute_breakout_proximity_features(hist_sliced)
    volatility = compute_volatility_features(hist_sliced)
    volume = compute_volume_features(hist_sliced)
    structure = compute_structure_features(hist_sliced)
    ma = compute_moving_average_features(hist_sliced)
    pos_52w = compute_52w_position_features(hist_sliced)
    fundamentals = compute_fundamental_acceleration_features(fundamental_snapshots)
    institutional = compute_institutional_trend_features(institutional_snapshots)
    sector = compute_sector_confirmation_features(rs_features.get("rs_current"), sector_avg_rs)
    regime = compute_market_regime_feature(results_df)

    # Cross-family judgments the spec explicitly frames as combinations,
    # not single-family flags — computed here rather than inside any one
    # family function to avoid one family silently depending on another.
    near_resistance = breakout["state"] in ("IMMINENT", "VERY_CLOSE")
    constructive_volume_dryup = bool(
        volume.get("volume_contraction") and volatility.get("volatility_contraction") and near_resistance
    )
    breakout_volume_confirmation = bool(
        volume.get("relative_volume") is not None
        and volume["relative_volume"] >= BREAKOUT_VOLUME_MULT
        and breakout["state"] == "IMMINENT"
    )

    return {
        "as_of_date": str(as_of_date) if as_of_date else None,
        "data_quality": {"bars_available": bars_available, "sufficient_history": bars_available >= 252},
        "relative_strength": rs_features,
        "breakout_proximity": breakout,
        "volatility": volatility,
        "volume": volume,
        "structure": structure,
        "moving_averages": ma,
        "position_52w": pos_52w,
        "fundamental_acceleration": fundamentals,
        "institutional_trend": institutional,
        "sector_confirmation": sector,
        "market_regime": regime,
        "constructive_volume_dryup": constructive_volume_dryup,
        "breakout_volume_confirmation": breakout_volume_confirmation,
    }
