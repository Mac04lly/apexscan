"""
tests/test_apex10_short_no_lookahead.py — Apex the Great X short side:
MANDATORY no-look-ahead test

Same structure and same standard as
tests/test_apex10_no_lookahead.py (the long side's mandatory suite),
applied to the short side's new entry point,
compute_short_precursor_features(). Confirms _slice_as_of() reuse
actually carries the guarantee through unchanged, rather than assuming
it does.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.apex10_short_features import compute_short_precursor_features


def _make_synthetic_downtrend(n: int, seed: int = 17, start: str = "2023-01-02"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    price = np.maximum(200 - np.cumsum(rng.normal(0.1, 1.3, n)), 5.0)
    hist = pd.DataFrame({
        "Open": price * (1 + rng.normal(0, 0.002, n)),
        "High": price * (1 + np.abs(rng.normal(0, 0.01, n))),
        "Low": price * (1 - np.abs(rng.normal(0, 0.01, n))),
        "Close": price,
        "Volume": rng.integers(500_000, 6_000_000, n),
    }, index=dates)
    bench = pd.Series(100 + np.cumsum(rng.normal(0.03, 0.6, n)), index=dates)
    return hist, bench


def test_no_lookahead_full_short_feature_set_identical_before_and_after_future_data():
    hist_full, bench_full = _make_synthetic_downtrend(400)
    as_of = hist_full.index[300]

    hist_at_t = hist_full.loc[:as_of]
    bench_at_t = bench_full.loc[:as_of]
    result_at_t = compute_short_precursor_features(hist_at_t, bench_at_t, as_of_date=as_of)

    result_with_future_data = compute_short_precursor_features(hist_full, bench_full, as_of_date=as_of)

    assert result_at_t == result_with_future_data, (
        "LOOK-AHEAD LEAK in the short-side feature engine: the snapshot at T changed "
        "after future data was appended."
    )


def test_no_lookahead_per_family_explicit_assertions_short_side():
    hist_full, bench_full = _make_synthetic_downtrend(400, seed=21)
    as_of = hist_full.index[280]

    before = compute_short_precursor_features(hist_full.loc[:as_of], bench_full.loc[:as_of], as_of_date=as_of)
    after = compute_short_precursor_features(hist_full, bench_full, as_of_date=as_of)

    for family in ["relative_strength", "breakdown_proximity", "volatility", "volume",
                  "structure", "moving_averages", "position_52w"]:
        assert before[family] == after[family], f"Look-ahead leak in short-side family: {family}"


def test_no_lookahead_support_uses_only_past_lows():
    """Targeted test mirroring the long side's resistance-spike test:
    plants a future price CRASH (an extreme new low) after T and
    confirms support/distance-to-support at T is unaffected."""
    hist_full, bench_full = _make_synthetic_downtrend(300, seed=31)
    as_of = hist_full.index[200]

    before = compute_short_precursor_features(hist_full.loc[:as_of], bench_full.loc[:as_of], as_of_date=as_of)

    crashed = hist_full.copy()
    crashed.loc[crashed.index[-1], "Low"] = crashed["Low"].min() / 100  # extreme future low

    after = compute_short_precursor_features(crashed, bench_full, as_of_date=as_of)

    assert before["breakdown_proximity"] == after["breakdown_proximity"], (
        "Future price crash affected historical support calculation — look-ahead leak."
    )


def test_no_lookahead_holds_at_multiple_cutoff_points_short_side():
    hist_full, bench_full = _make_synthetic_downtrend(450, seed=41)
    cutoffs = [hist_full.index[i] for i in (60, 150, 260, 350, 400)]

    for as_of in cutoffs:
        before = compute_short_precursor_features(hist_full.loc[:as_of], bench_full.loc[:as_of], as_of_date=as_of)
        after = compute_short_precursor_features(hist_full, bench_full, as_of_date=as_of)
        assert before == after, f"Look-ahead leak detected at cutoff {as_of}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
