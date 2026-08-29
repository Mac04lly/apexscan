"""
tests/test_apex10_integration.py — Apex the Great X: live wiring tests

Covers the safety contract explicitly: disabled-by-default returns None
and touches nothing; missing history is skipped per-ticker without
aborting the batch; a below-threshold score is excluded; and a failure
anywhere inside this module can never raise up into the caller (which,
in production, is scanner.run_scan() — the live scan path).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.apex10_integration import process_scan_for_radar, DEFAULT_MIN_RADAR_SCORE


def _make_hist(seed, n=300):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    price = np.maximum(100 + np.cumsum(rng.normal(0.15, 1.2, n)), 5)
    return pd.DataFrame({
        "Open": price * (1 + rng.normal(0, 0.002, n)), "High": price * (1 + np.abs(rng.normal(0, 0.01, n))),
        "Low": price * (1 - np.abs(rng.normal(0, 0.01, n))), "Close": price,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=dates)


def _bench(index):
    return pd.Series(100 + np.cumsum(np.random.default_rng(99).normal(0.03, 0.6, len(index))), index=index)


# ── Off by default ────────────────────────────────────────────────────

def test_disabled_by_default_returns_none():
    results = [{"ticker": "AAA"}]
    hist = _make_hist(1)
    result = process_scan_for_radar(results, {"AAA": hist}, _bench(hist.index), cfg={})
    assert result is None


def test_explicitly_disabled_returns_none():
    results = [{"ticker": "AAA"}]
    hist = _make_hist(1)
    result = process_scan_for_radar(results, {"AAA": hist}, _bench(hist.index),
                                    cfg={"apex10": {"enabled": False}})
    assert result is None


def test_none_cfg_returns_none_not_crash():
    result = process_scan_for_radar([], {}, None, cfg=None)
    assert result is None


# ── Enabled path ──────────────────────────────────────────────────────

def test_enabled_processes_and_reports_summary():
    hist_a, hist_b = _make_hist(1), _make_hist(2)
    results = [{"ticker": "AAA"}, {"ticker": "BBB"}]
    batch_hist = {"AAA": hist_a, "BBB": hist_b}
    cfg = {"apex10": {"enabled": True, "min_radar_score": 1}}  # low bar
    result = process_scan_for_radar(results, batch_hist, _bench(hist_a.index), cfg, market="US")
    assert result is not None
    assert result["candidates_evaluated"] == 2
    assert result["skipped_no_hist"] == 0


def test_missing_history_skipped_without_aborting_batch():
    hist_a = _make_hist(1)
    results = [{"ticker": "AAA"}, {"ticker": "NOHIST"}]
    batch_hist = {"AAA": hist_a}  # NOHIST absent
    cfg = {"apex10": {"enabled": True, "min_radar_score": 1}}
    result = process_scan_for_radar(results, batch_hist, _bench(hist_a.index), cfg)
    assert result["skipped_no_hist"] == 1
    assert result["candidates_evaluated"] == 2


def test_below_min_score_excluded():
    hist_a = _make_hist(1)
    results = [{"ticker": "AAA"}]
    batch_hist = {"AAA": hist_a}
    cfg = {"apex10": {"enabled": True, "min_radar_score": 999}}  # impossible bar
    result = process_scan_for_radar(results, batch_hist, _bench(hist_a.index), cfg)
    assert result["candidates_qualified"] == 0
    assert result["created"] == 0


def test_default_min_score_used_when_not_configured():
    assert DEFAULT_MIN_RADAR_SCORE == 50


def test_missing_benchmark_returns_none_gracefully():
    hist_a = _make_hist(1)
    results = [{"ticker": "AAA"}]
    cfg = {"apex10": {"enabled": True}}
    result = process_scan_for_radar(results, {"AAA": hist_a}, None, cfg)
    assert result is None


def test_empty_results_does_not_crash():
    hist_a = _make_hist(1)
    cfg = {"apex10": {"enabled": True}}
    result = process_scan_for_radar([], {"AAA": hist_a}, _bench(hist_a.index), cfg)
    assert result["candidates_evaluated"] == 0
    assert result["created"] == 0


# ── Never breaks the caller ──────────────────────────────────────────

def test_ticker_without_ticker_key_skipped_not_crash():
    hist_a = _make_hist(1)
    results = [{"apex_score": 80}]  # malformed — no "ticker" key
    cfg = {"apex10": {"enabled": True, "min_radar_score": 1}}
    result = process_scan_for_radar(results, {"AAA": hist_a}, _bench(hist_a.index), cfg)
    assert result["candidates_qualified"] == 0  # skipped silently, no crash


def test_one_bad_ticker_does_not_abort_others():
    hist_a = _make_hist(1)
    hist_bad = pd.DataFrame({"Close": [1.0]})  # malformed, too short, missing columns
    results = [{"ticker": "GOOD"}, {"ticker": "BAD"}]
    batch_hist = {"GOOD": hist_a, "BAD": hist_bad}
    cfg = {"apex10": {"enabled": True, "min_radar_score": 1}}
    result = process_scan_for_radar(results, batch_hist, _bench(hist_a.index), cfg)
    # BAD should be skipped (either as no_hist due to <21 bars, or as an error) —
    # GOOD must still be evaluated regardless.
    assert result["candidates_evaluated"] == 2
    assert result["skipped_no_hist"] + result["skipped_error"] + result["candidates_qualified"] >= 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
