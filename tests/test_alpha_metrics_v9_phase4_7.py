"""
tests/test_alpha_metrics_v9_phase4_7.py — V9 Phases 4-7 regression tests

Uses hand-built synthetic AlphaObservations (not live data) so every
expected number is known in advance — this tests the STATISTICS, not
whatever happens to be in data/alpha_observations.json on a given day.
Run with: pytest tests/test_alpha_metrics_v9_phase4_7.py -v
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.alpha_metrics import (
    compute_alpha_metrics_by_setup,
    compute_alpha_metrics_by_feature,
    rank_feature_alpha,
    compute_conditional_alpha,
    compute_combination_alpha,
    get_combination_alpha_table,
    compute_alpha_lab_overview,
    generate_research_findings,
    get_feature_value,
    FEATURE_REGISTRY,
    SETUP_LABELS,
)


def _obs(ticker, setup_id, rs_3m=None, rs_percentile=None, stage="2 ✅ Uptrend",
        volume_ratio=None, revenue_growth=None, institutional_ownership=None,
        market_regime="Sideways", theme="Information Technology", forward_return_20d=None,
        excess_return_20d=None):
    outcomes = {}
    if forward_return_20d is not None:
        outcomes["20D"] = {
            "forward_return_%": forward_return_20d,
            "excess_return_%": excess_return_20d,
            "mfe_%": abs(forward_return_20d) + 1, "mae_%": -abs(forward_return_20d) - 1,
            "benchmark_return_%": 1.0,
        }
    return {
        "ticker": ticker, "setup_id": setup_id, "stage": stage, "market_regime": market_regime,
        "outcomes": outcomes,
        "technical_features": {"rs_3m": rs_3m, "rs_percentile": rs_percentile, "volume_ratio": volume_ratio},
        "fundamental_features": {"revenue_growth": revenue_growth,
                                 "institutional_ownership": institutional_ownership},
        "valuation_features": {},
        "market_features": {"theme": theme, "market_regime": market_regime},
    }


# ── Phase 4: Setup Alpha ────────────────────────────────────────────────

def test_setup_alpha_groups_correctly_and_computes_expectancy():
    obs = [
        _obs("A", "S2-PULLBACK-50", forward_return_20d=10, excess_return_20d=8),
        _obs("B", "S2-PULLBACK-50", forward_return_20d=-4, excess_return_20d=-6),
        _obs("C", "S2-HIGH-RS", forward_return_20d=2, excess_return_20d=1),
    ]
    rows = compute_alpha_metrics_by_setup(obs, "20D")
    by_id = {r["setup_id"]: r for r in rows}
    assert by_id["S2-PULLBACK-50"]["n"] == 2
    assert by_id["S2-PULLBACK-50"]["expectancy_%"] == 3.0  # (10 + -4) / 2
    assert by_id["S2-PULLBACK-50"]["win_rate_%"] == 50.0
    assert by_id["S2-HIGH-RS"]["n"] == 1
    assert by_id["S2-PULLBACK-50"]["setup_label"] == SETUP_LABELS["S2-PULLBACK-50"]


def test_setup_alpha_includes_zero_sample_setups():
    obs = [_obs("A", "S1-BASE", forward_return_20d=None)]  # no outcome yet
    rows = compute_alpha_metrics_by_setup(obs, "20D")
    assert rows[0]["n"] == 0
    assert rows[0]["sample_classification"].startswith("Too Small")


def test_setup_alpha_empty_input():
    assert compute_alpha_metrics_by_setup([], "20D") == []


# ── Phase 5: Feature Alpha ──────────────────────────────────────────────

def test_feature_value_dotted_path():
    obs = _obs("A", "S1-BASE", rs_3m=42.0)
    assert get_feature_value(obs, "technical_features.rs_3m") == 42.0
    assert get_feature_value(obs, "technical_features.missing_field") is None
    assert get_feature_value(obs, "not_a_dict.x") is None


def test_feature_bucketing_categorical():
    obs = [
        _obs("A", "S1-BASE", stage="2 ✅ Uptrend", forward_return_20d=10, excess_return_20d=5),
        _obs("B", "S1-BASE", stage="2 ✅ Uptrend", forward_return_20d=-2, excess_return_20d=-3),
        _obs("C", "S1-BASE", stage="4 🔴 Downtrend", forward_return_20d=-8, excess_return_20d=-9),
    ]
    rows = compute_alpha_metrics_by_feature(obs, "20D", "stage")
    labels = {r["bucket_label"]: r for r in rows}
    assert labels["2 ✅ Uptrend"]["n"] == 2
    assert labels["4 🔴 Downtrend"]["n"] == 1


def test_feature_bucketing_continuous_quantiles_and_ordering():
    obs = [_obs(f"T{i}", "S1-BASE", rs_3m=float(i), forward_return_20d=float(i) / 10)
           for i in range(20)]
    rows = compute_alpha_metrics_by_feature(obs, "20D", "rs_3m", n_buckets=4)
    assert len(rows) <= 4 and len(rows) >= 1
    total_n = sum(r["n"] for r in rows)
    assert total_n == 20
    # Buckets should be ordered ascending by their lower edge.
    los = [r["bucket_lo"] for r in rows]
    assert los == sorted(los)


def test_feature_bucketing_excludes_observations_without_outcome():
    obs = [
        _obs("A", "S1-BASE", rs_3m=10, forward_return_20d=5, excess_return_20d=3),
        _obs("B", "S1-BASE", rs_3m=20, forward_return_20d=None),  # pending, excluded
    ]
    rows = compute_alpha_metrics_by_feature(obs, "20D", "rs_3m")
    assert sum(r["n"] for r in rows) == 1


def test_feature_bucketing_unknown_feature_key_raises():
    try:
        compute_alpha_metrics_by_feature([], "20D", "not_a_real_feature")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rank_feature_alpha_marks_thin_data_inconclusive():
    obs = [_obs("A", "S1-BASE", rs_3m=10, forward_return_20d=5, excess_return_20d=3)]
    ranked = rank_feature_alpha(obs, "20D", feature_keys=["rs_3m"], min_n=10)
    assert ranked[0]["is_conclusive"] is False
    assert ranked[0]["spread_%"] is None


def test_rank_feature_alpha_computes_spread_when_data_sufficient():
    # Large enough on each side that even after a 5-way rank split (the
    # default n_buckets), the pure-low and pure-high buckets each still
    # clear min_n=10 — the boundary bucket where low/high values mix is
    # allowed to fall below threshold; the extremes are what matter here.
    low = [_obs(f"L{i}", "S1-BASE", rs_3m=1.0, forward_return_20d=-5.0, excess_return_20d=-5.0)
           for i in range(40)]
    high = [_obs(f"H{i}", "S1-BASE", rs_3m=100.0, forward_return_20d=8.0, excess_return_20d=8.0)
           for i in range(40)]
    ranked = rank_feature_alpha(low + high, "20D", feature_keys=["rs_3m"], min_n=10)
    r = ranked[0]
    assert r["is_conclusive"] is True
    assert r["spread_%"] > 0  # high bucket outperformed low bucket


def test_every_feature_registry_entry_is_resolvable():
    obs = _obs("A", "S1-BASE")
    for key, spec in FEATURE_REGISTRY.items():
        # Should not raise even when the value is None.
        get_feature_value(obs, spec["path"])


# ── Phase 6: Conditional & Combination Research ─────────────────────────

def test_conditional_alpha_breaks_down_by_sector():
    obs = [
        _obs("A", "S1-BASE", rs_percentile=95, theme="Financials",
             forward_return_20d=6, excess_return_20d=4),
        _obs("B", "S1-BASE", rs_percentile=90, theme="Healthcare",
             forward_return_20d=2, excess_return_20d=1),
        _obs("C", "S1-BASE", rs_percentile=10, theme="Financials",  # excluded, low RS
             forward_return_20d=-9, excess_return_20d=-9),
    ]
    # Explicit threshold, matching the V9 spec's own "RS>80" literal-
    # threshold example — avoids relying on how the top-bucket-by-quantile
    # fallback happens to split a 3-observation sample.
    result = compute_conditional_alpha(obs, "20D", "rs_percentile", "sector", threshold=80)
    values = {r["condition_value"]: r for r in result["rows"]}
    assert "Financials" in values and values["Financials"]["n"] == 1
    assert "Healthcare" in values and values["Healthcare"]["n"] == 1


def test_conditional_alpha_unknown_condition_type_raises():
    try:
        compute_conditional_alpha([], "20D", "rs_percentile", "not_a_real_condition")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_combination_alpha_matches_all_conditions():
    obs = [
        _obs("A", "S1-BASE", rs_percentile=90, stage="2 ✅ Uptrend",
             forward_return_20d=7, excess_return_20d=5),   # matches RS+Stage2
        _obs("B", "S1-BASE", rs_percentile=90, stage="4 🔴 Downtrend",
             forward_return_20d=-5, excess_return_20d=-5),  # fails stage condition
    ]
    conditions = [("technical_features.rs_percentile", ">=", 80), ("stage", "startswith", "2")]
    result = compute_combination_alpha(obs, "20D", conditions)
    assert result["n"] == 1
    assert result["expectancy_%"] == 7.0


def test_combination_alpha_table_returns_all_presets():
    rows = get_combination_alpha_table([], "20D")
    assert len(rows) == 6  # PRESET_COMBINATIONS has 6 entries per the V9 §16 list
    for r in rows:
        assert r["n"] == 0  # empty input -> every preset resolves to zero matches


# ── Phase 7: Overview & Findings ────────────────────────────────────────

def test_overview_counts_resolved_vs_unresolved():
    obs = [
        _obs("A", "S1-BASE", forward_return_20d=5, excess_return_20d=3),
        _obs("B", "S1-BASE", forward_return_20d=None),
    ]
    # Give B some outcomes dict presence but not this horizon, to test "resolved" == "has ANY outcome"
    obs[1]["outcomes"] = {"5D": {"forward_return_%": 1.0}}
    ov = compute_alpha_lab_overview(obs, "20D")
    assert ov["total_observations"] == 2
    assert ov["resolved_observations"] == 2  # both have an outcomes dict with something in it
    assert ov["unresolved_observations"] == 0


def test_overview_empty_input_does_not_raise():
    ov = compute_alpha_lab_overview([], "20D")
    assert ov["total_observations"] == 0
    assert ov["best_setup"] is None
    assert ov["best_feature"] is None


def test_generate_research_findings_respects_min_n():
    obs = [_obs(f"A{i}", "S2-HIGH-RS", rs_percentile=90, forward_return_20d=6.0, excess_return_20d=4.0)
          for i in range(12)]
    findings_strict = generate_research_findings(obs, "20D", min_n=20)
    findings_loose = generate_research_findings(obs, "20D", min_n=10)
    assert findings_strict == []
    assert len(findings_loose) > 0
    for f in findings_loose:
        assert f["n"] >= 10
        assert "source" in f and "type" in f["source"]


def test_generate_research_findings_empty_input():
    assert generate_research_findings([], "20D") == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
