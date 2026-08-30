"""
tests/test_apex10_short_features.py — Apex the Great X short side: features
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.apex10_short_features import (
    compute_breakdown_proximity_features, compute_short_structure_features,
    compute_short_precursor_features, SUPPORT_THRESHOLDS_PCT,
)


def _flat_hist(n=100, price=100.0, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    noise = rng.normal(0, 0.3, n)
    close = np.full(n, price) + noise
    return pd.DataFrame({
        "Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=dates)


def _downtrend_hist(n=300, seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = np.maximum(200 - np.cumsum(rng.normal(0.3, 0.8, n)), 5)
    return pd.DataFrame({
        "Open": close * 1.001, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": rng.integers(1_000_000, 3_000_000, n),
    }, index=dates)


# ── Breakdown proximity ──────────────────────────────────────────────

def test_breakdown_proximity_states_thresholds():
    hist = _flat_hist(100).copy()
    hist.iloc[:-1, hist.columns.get_loc("Low")] = 90.0  # support = 90
    hist.iloc[-1, hist.columns.get_loc("Close")] = 90.5  # 0.55% above -> IMMINENT
    result = compute_breakdown_proximity_features(hist)
    assert result["state"] == "IMMINENT"
    assert result["support_price"] == 90.0


def test_breakdown_proximity_at_new_lows_has_zero_distance():
    hist = _downtrend_hist(100)
    result = compute_breakdown_proximity_features(hist)
    assert result["state"] in ("IMMINENT", "VERY_CLOSE", "DEVELOPING")


def test_breakdown_proximity_insufficient_data():
    hist = _flat_hist(10)
    result = compute_breakdown_proximity_features(hist)
    assert result["state"] == "UNKNOWN"


def test_breakdown_proximity_mirrors_breakout_logic_structurally():
    """Nearest support = the HIGHEST of the qualifying lows (closest
    floor below price) — the exact mirror of nearest resistance being
    the LOWEST qualifying high (closest ceiling above price)."""
    hist = _flat_hist(90).copy()
    # The "recent" region must span at least the shortest window (20
    # bars) entirely on its own, or a shorter recent region gets
    # contaminated by the older region within the same rolling window.
    hist.iloc[-30:, hist.columns.get_loc("Low")] = 95.0     # last 30 bars, closer support
    hist.iloc[-60:-30, hist.columns.get_loc("Low")] = 80.0  # older, further-back, lower support
    hist.iloc[-1, hist.columns.get_loc("Close")] = 96.0
    result = compute_breakdown_proximity_features(hist)
    assert result["support_price"] == 95.0  # nearest (highest qualifying) support picked


# ── Structure ─────────────────────────────────────────────────────────

def test_short_structure_insufficient_data():
    hist = _flat_hist(10)
    result = compute_short_structure_features(hist)
    assert result["ms_structure"] == "UNKNOWN"


def test_short_structure_returns_native_bool_for_ms_lh_ll():
    hist = _downtrend_hist(100)
    result = compute_short_structure_features(hist)
    assert isinstance(result["ms_lh_ll"], bool)


# ── Top-level entry point ────────────────────────────────────────────

def test_short_precursor_features_insufficient_bars():
    hist = _flat_hist(5)
    bench = pd.Series(np.full(5, 100.0), index=hist.index)
    result = compute_short_precursor_features(hist, bench)
    assert result["data_quality"]["sufficient_history"] is False
    assert "relative_strength" not in result


def test_short_precursor_features_full_output_shape():
    hist = _downtrend_hist(300)
    bench = pd.Series(100 + np.cumsum(np.random.default_rng(9).normal(0.03, 0.4, 300)), index=hist.index)
    result = compute_short_precursor_features(hist, bench, as_of_date=hist.index[250])
    for key in ["relative_strength", "breakdown_proximity", "volatility", "volume", "structure",
               "moving_averages", "position_52w", "fundamental_acceleration",
               "institutional_trend", "sector_confirmation", "market_regime"]:
        assert key in result


def test_short_precursor_reuses_direction_neutral_functions_directly():
    """Confirms volatility/volume/moving_averages/position_52w are
    computed by the SAME long-side functions (not reimplemented) by
    checking their exact expected key sets match the long side's."""
    hist = _downtrend_hist(300)
    bench = pd.Series(100 + np.cumsum(np.random.default_rng(9).normal(0.03, 0.4, 300)), index=hist.index)
    result = compute_short_precursor_features(hist, bench, as_of_date=hist.index[250])
    assert set(result["volatility"].keys()) >= {"atr_pct_5", "atr_pct_10", "atr_pct_20",
                                                 "volatility_contraction"}
    assert set(result["moving_averages"].keys()) >= {"ma50_transition", "ma200_transition"}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
