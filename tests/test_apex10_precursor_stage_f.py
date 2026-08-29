"""
tests/test_apex10_precursor_stage_f.py — Apex the Great X: Stage F tests

Covers candidate selection (bounded, resolved-outcomes-only), the
hard candidate cap, trading-day offset resolution, full-pipeline
trajectory reconstruction, no-look-ahead preservation through this new
caller, and the aggregate-findings denominator discipline. No real
network calls anywhere — every test supplies `histories` directly, per
the module's own test-injection design.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.apex10_precursor import (
    select_precursor_candidates, run_precursor_study, aggregate_precursor_findings,
    reconstruct_precursor_trajectory, _trading_day_offset_date,
    MAX_CANDIDATES_HARD_CAP, BENCHMARK_TICKER,
)


def _discovery_obs(ticker, ts, excess_return, horizon="20D", resolved=True, setup_id="S2-HIGH-RS"):
    outcomes = {horizon: {"excess_return_%": excess_return}} if resolved else {}
    return {"ticker": ticker, "observation_type": "discovery", "timestamp": ts,
           "entry_price": 50.0, "setup_id": setup_id, "outcomes": outcomes}


def _make_hist(seed, end, n=400):
    dates = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    price = np.maximum(50 + np.cumsum(rng.normal(0.1, 1.0, len(dates))), 5)
    return pd.DataFrame({
        "Open": price, "High": price * 1.01, "Low": price * 0.99, "Close": price,
        "Volume": rng.integers(1_000_000, 5_000_000, len(dates)),
    }, index=dates)


# ── Candidate selection ───────────────────────────────────────────────

def test_selects_only_winners_above_threshold():
    obs = [
        _discovery_obs("WIN", "2025-06-01 09:30:00", 35.0),
        _discovery_obs("LOSER", "2025-06-01 09:30:00", -5.0),
        _discovery_obs("MARGINAL", "2025-06-01 09:30:00", 10.0),
    ]
    candidates = select_precursor_candidates(obs, min_excess_return_pct=20.0)
    assert [c["ticker"] for c in candidates] == ["WIN"]


def test_excludes_unresolved_observations():
    obs = [_discovery_obs("PENDING", "2025-08-01 09:30:00", 50.0, resolved=False)]
    candidates = select_precursor_candidates(obs, min_excess_return_pct=20.0)
    assert candidates == []


def test_excludes_non_discovery_observation_types():
    obs = [{"ticker": "RADAR", "observation_type": "apex10_radar", "timestamp": "2025-06-01 09:30:00",
           "outcomes": {"20D": {"excess_return_%": 50.0}}}]
    candidates = select_precursor_candidates(obs, min_excess_return_pct=20.0)
    assert candidates == []


def test_candidates_sorted_by_excess_return_descending():
    obs = [
        _discovery_obs("A", "2025-06-01 09:30:00", 25.0),
        _discovery_obs("B", "2025-06-01 09:30:00", 60.0),
        _discovery_obs("C", "2025-06-01 09:30:00", 40.0),
    ]
    candidates = select_precursor_candidates(obs, min_excess_return_pct=20.0)
    assert [c["ticker"] for c in candidates] == ["B", "C", "A"]


def test_max_candidates_hard_cap_enforced_even_if_caller_asks_for_more():
    obs = [_discovery_obs(f"T{i}", "2025-06-01 09:30:00", 100.0) for i in range(200)]
    candidates = select_precursor_candidates(obs, min_excess_return_pct=20.0,
                                             max_candidates=10_000)  # absurdly high request
    assert len(candidates) == MAX_CANDIDATES_HARD_CAP  # capped regardless


def test_max_candidates_respects_smaller_explicit_request():
    obs = [_discovery_obs(f"T{i}", "2025-06-01 09:30:00", 100.0) for i in range(30)]
    candidates = select_precursor_candidates(obs, min_excess_return_pct=20.0, max_candidates=5)
    assert len(candidates) == 5


# ── Trading-day offset resolution ─────────────────────────────────────

def test_trading_day_offset_returns_none_when_insufficient_history():
    hist = _make_hist(1, "2025-06-05", n=10)
    result = _trading_day_offset_date(hist.index, pd.Timestamp("2025-06-05").date(), offset_bars=60)
    assert result is None


def test_trading_day_offset_resolves_correctly():
    hist = _make_hist(1, "2025-06-05", n=100)
    result = _trading_day_offset_date(hist.index, pd.Timestamp("2025-06-05").date(), offset_bars=0)
    assert result == hist.index[-1]


# ── Full pipeline ──────────────────────────────────────────────────────

def test_run_precursor_study_end_to_end_with_injected_histories():
    obs = [_discovery_obs("WIN1", "2025-06-01 09:30:00", 35.0)]
    histories = {"WIN1": _make_hist(1, "2025-06-05"), BENCHMARK_TICKER: _make_hist(99, "2025-08-05")}
    study = run_precursor_study(obs, min_excess_return_pct=20.0, histories=histories)
    assert study["status"] == "ok"
    assert study["trajectories_computed"] == 1
    traj = study["trajectories"][0]["trajectory"]
    assert len(traj) == len(study["offsets_trading_days"])
    for point in traj:
        if point["status"] == "ok":
            assert point["score"] is None or 0 <= point["score"] <= 100
            assert "key_features" in point


def test_run_precursor_study_no_candidates_returns_empty_not_crash():
    study = run_precursor_study([], min_excess_return_pct=20.0)
    assert study["status"] == "no_candidates"
    assert study["trajectories"] == []


def test_run_precursor_study_skips_ticker_with_no_data_from_batch():
    obs = [_discovery_obs("WIN1", "2025-06-01 09:30:00", 35.0),
          _discovery_obs("NODATA", "2025-06-01 09:30:00", 40.0)]
    histories = {"WIN1": _make_hist(1, "2025-06-05"), BENCHMARK_TICKER: _make_hist(99, "2025-08-05")}
    # NODATA is simply absent from histories, simulating batch_fetch_history's
    # own documented behavior of silently omitting tickers it couldn't fetch.
    study = run_precursor_study(obs, min_excess_return_pct=20.0, histories=histories)
    assert study["trajectories_computed"] == 1
    assert len(study["skipped"]) == 1
    assert study["skipped"][0]["ticker"] == "NODATA"


def test_run_precursor_study_missing_benchmark_skips_everything_gracefully():
    obs = [_discovery_obs("WIN1", "2025-06-01 09:30:00", 35.0)]
    histories = {"WIN1": _make_hist(1, "2025-06-05")}  # no benchmark at all
    study = run_precursor_study(obs, min_excess_return_pct=20.0, histories=histories)
    assert study["trajectories_computed"] == 0
    assert len(study["skipped"]) == 1


# ── No-look-ahead preservation through this new caller ────────────────

def test_no_lookahead_preserved_through_precursor_pipeline():
    obs = [_discovery_obs("WIN1", "2025-06-01 09:30:00", 35.0)]
    hist_short = _make_hist(7, "2025-06-05")
    bench = _make_hist(99, "2025-08-05")

    future_dates = pd.bdate_range(start=hist_short.index[-1] + pd.Timedelta(days=1), periods=60)
    future_rows = pd.DataFrame({"Open": [999.0] * 60, "High": [999.0] * 60, "Low": [999.0] * 60,
                                "Close": [999.0] * 60, "Volume": [999] * 60}, index=future_dates)
    hist_long = pd.concat([hist_short, future_rows])

    study_short = run_precursor_study(obs, min_excess_return_pct=20.0,
                                      histories={"WIN1": hist_short, BENCHMARK_TICKER: bench})
    study_long = run_precursor_study(obs, min_excess_return_pct=20.0,
                                     histories={"WIN1": hist_long, BENCHMARK_TICKER: bench})

    assert study_short["trajectories"][0]["trajectory"] == study_long["trajectories"][0]["trajectory"], (
        "Appending future data after the discovery date changed the reconstructed "
        "trajectory — look-ahead leak in the precursor pipeline."
    )


# ── Aggregate findings — denominator discipline ────────────────────────

def test_aggregate_findings_reports_denominator_and_percentages():
    obs = [_discovery_obs("WIN1", "2025-06-01 09:30:00", 35.0),
          _discovery_obs("WIN2", "2025-07-01 09:30:00", 22.0)]
    histories = {"WIN1": _make_hist(1, "2025-06-05"), "WIN2": _make_hist(2, "2025-07-05"),
                BENCHMARK_TICKER: _make_hist(99, "2025-08-05")}
    study = run_precursor_study(obs, min_excess_return_pct=20.0, histories=histories)
    result = aggregate_precursor_findings(study, offset_trading_days=20, min_n=1)
    assert result["status"] == "ok"
    assert result["n"] == 2
    for cond_name, cond_result in result["conditions"].items():
        assert cond_result["n"] == 2
        assert 0.0 <= cond_result["pct"] <= 100.0


def test_aggregate_findings_refuses_below_min_n():
    obs = [_discovery_obs("WIN1", "2025-06-01 09:30:00", 35.0)]
    histories = {"WIN1": _make_hist(1, "2025-06-05"), BENCHMARK_TICKER: _make_hist(99, "2025-08-05")}
    study = run_precursor_study(obs, min_excess_return_pct=20.0, histories=histories)
    result = aggregate_precursor_findings(study, offset_trading_days=20, min_n=10)
    assert result["status"] == "insufficient_sample"
    assert result["n"] == 1
    assert "conditions" not in result


def test_aggregate_findings_empty_study_does_not_crash():
    result = aggregate_precursor_findings({"trajectories": []}, offset_trading_days=20, min_n=1)
    assert result["status"] == "insufficient_sample"
    assert result["n"] == 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
