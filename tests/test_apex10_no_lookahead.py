"""
tests/test_apex10_no_lookahead.py — Apex the Great X: MANDATORY no-look-ahead test

Per the spec: "Given data through date T, calculate ADEX score(T). Then
append future data after T. The score at T MUST remain identical. If
the score changes: FAIL. Investigate the leakage."

This is not a generic regression test — it's a structural guarantee
about the feature engine's design. Every feature family gets its own
explicit assertion, not just the top-level dict, so a leak in one
specific family can't hide behind an aggregate pass.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.apex10_features import compute_precursor_features


def _make_synthetic_history(n: int, seed: int = 7, start: str = "2023-01-02"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    price = np.maximum(100 + np.cumsum(rng.normal(0.08, 1.4, n)), 5.0)
    hist = pd.DataFrame({
        "Open": price * (1 + rng.normal(0, 0.002, n)),
        "High": price * (1 + np.abs(rng.normal(0, 0.01, n))),
        "Low": price * (1 - np.abs(rng.normal(0, 0.01, n))),
        "Close": price,
        "Volume": rng.integers(500_000, 6_000_000, n),
    }, index=dates)
    bench = pd.Series(100 + np.cumsum(rng.normal(0.04, 0.7, n)), index=dates)
    return hist, bench


def test_no_lookahead_full_feature_set_identical_before_and_after_future_data():
    """The core mandated test: compute at T with data ending at T, then
    again with 60 MORE future bars appended after T (but as_of_date
    still T) — results must be byte-for-byte identical."""
    hist_full, bench_full = _make_synthetic_history(400)
    as_of = hist_full.index[300]  # T — well within the data, with room to extend

    # Data available "at the time" — nothing past T.
    hist_at_t = hist_full.loc[:as_of]
    bench_at_t = bench_full.loc[:as_of]
    result_at_t = compute_precursor_features(hist_at_t, bench_at_t, as_of_date=as_of)

    # Now simulate time passing: the SAME T, but hist/bench now contain
    # 60 additional bars beyond T that didn't exist when result_at_t was
    # first computed.
    result_with_future_data = compute_precursor_features(hist_full, bench_full, as_of_date=as_of)

    assert result_at_t == result_with_future_data, (
        "LOOK-AHEAD LEAK DETECTED: the feature snapshot at date T changed after "
        "future data was appended. This must never happen — investigate which "
        "feature family used unbounded data instead of a slice ending at T."
    )


def test_no_lookahead_per_family_explicit_assertions():
    """Same test, but asserting each feature family individually so a
    leak in one family can't be masked by the others' correctness."""
    hist_full, bench_full = _make_synthetic_history(400, seed=11)
    as_of = hist_full.index[280]

    hist_at_t = hist_full.loc[:as_of]
    bench_at_t = bench_full.loc[:as_of]

    before = compute_precursor_features(hist_at_t, bench_at_t, as_of_date=as_of)
    after = compute_precursor_features(hist_full, bench_full, as_of_date=as_of)

    for family in ["relative_strength", "breakout_proximity", "volatility", "volume",
                  "structure", "moving_averages", "position_52w"]:
        assert before[family] == after[family], f"Look-ahead leak in feature family: {family}"


def test_no_lookahead_holds_at_multiple_cutoff_points():
    """Sweeps several different T values across the same series — a
    leak that only shows up near certain boundary conditions (e.g. the
    end of a rolling window) shouldn't be able to hide from a single
    fixed test date."""
    hist_full, bench_full = _make_synthetic_history(450, seed=23)
    cutoffs = [hist_full.index[i] for i in (60, 150, 260, 350, 400)]

    for as_of in cutoffs:
        hist_at_t = hist_full.loc[:as_of]
        bench_at_t = bench_full.loc[:as_of]
        before = compute_precursor_features(hist_at_t, bench_at_t, as_of_date=as_of)
        after = compute_precursor_features(hist_full, bench_full, as_of_date=as_of)
        assert before == after, f"Look-ahead leak detected at cutoff {as_of}"


def test_no_lookahead_resistance_uses_only_past_highs():
    """Targeted test for Feature Family 2's explicit warning: 'Future
    price cannot influence resistance.' Plants an enormous spike AFTER
    T and confirms resistance/distance-to-resistance at T is unaffected."""
    hist_full, bench_full = _make_synthetic_history(300, seed=31)
    as_of = hist_full.index[200]

    before = compute_precursor_features(hist_full.loc[:as_of], bench_full.loc[:as_of], as_of_date=as_of)

    # Plant a massive future spike far above anything in the real data —
    # if this leaks in, resistance/distance_to_resistance will change.
    spiked = hist_full.copy()
    spiked.loc[spiked.index[-1], "High"] = spiked["High"].max() * 100

    after = compute_precursor_features(spiked, bench_full, as_of_date=as_of)

    assert before["breakout_proximity"] == after["breakout_proximity"], (
        "Future price spike affected historical resistance calculation — look-ahead leak."
    )


def test_no_lookahead_volume_uses_only_past_volume():
    """Targeted test for Feature Family 4: 'Future volume cannot
    influence current volume features.' Plants a huge future volume
    spike and confirms volume features at T are unaffected."""
    hist_full, bench_full = _make_synthetic_history(300, seed=41)
    as_of = hist_full.index[200]

    before = compute_precursor_features(hist_full.loc[:as_of], bench_full.loc[:as_of], as_of_date=as_of)

    spiked = hist_full.copy()
    spiked.loc[spiked.index[-1], "Volume"] = spiked["Volume"].max() * 1000

    after = compute_precursor_features(spiked, bench_full, as_of_date=as_of)

    assert before["volume"] == after["volume"], (
        "Future volume spike affected historical volume features — look-ahead leak."
    )


def test_as_of_date_none_uses_all_provided_data():
    """When as_of_date is None (the live/current-day case), the function
    should use everything it's given — this is the one case where NOT
    truncating is correct, since the caller is expected to already be
    holding only up-to-today data."""
    hist_full, bench_full = _make_synthetic_history(300, seed=51)
    result_none = compute_precursor_features(hist_full, bench_full, as_of_date=None)
    result_explicit_last = compute_precursor_features(hist_full, bench_full,
                                                       as_of_date=hist_full.index[-1])
    # Compare everything except the as_of_date label itself, which is
    # expected to differ (None vs. an explicit date string) — the actual
    # computed features must be identical either way.
    result_none_no_label = {k: v for k, v in result_none.items() if k != "as_of_date"}
    result_explicit_no_label = {k: v for k, v in result_explicit_last.items() if k != "as_of_date"}
    assert result_none_no_label == result_explicit_no_label


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
