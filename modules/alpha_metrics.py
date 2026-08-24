"""
modules/alpha_metrics.py — Alpha Metrics (V9 Phase 3)

Given a set of observations that already have frozen Phase 2 outcomes,
computes the statistics needed to objectively answer "has this score
bucket / setup / feature historically produced superior outcomes":
expectancy, win rate, median return, profit factor, a 95% confidence
interval, and a plain-language sample-size classification.

This module only ever reads observations — it never writes to
alpha_observations.json, and it has no dependency on scanner.py,
dashboard.py, or any live scan. It's a pure function of whatever
observation list (with outcomes) you give it, which makes it safe to
test in isolation, and safe to reuse later for setup-based (Phase 4) or
feature-based (Phase 5) grouping without any changes to this file.
"""
from __future__ import annotations
import math
from typing import Optional

# Same n=30 threshold already used elsewhere in ApexScan's own honesty
# discipline (Discovery Tracker's Sample Readiness indicator) — kept
# consistent rather than inventing a different bar here.
def classify_sample_size(n: int) -> str:
    if n < 10:
        return "Too Small — not enough data to draw any conclusion"
    if n < 30:
        return "Emerging — directional only, not yet statistically reliable"
    if n < 100:
        return "Meaningful — a real pattern, worth attention"
    return "Robust — a well-supported sample"


def _confidence_interval_95(returns: list) -> Optional[tuple]:
    """Standard normal-approximation 95% CI on the mean. Returns None if
    the sample is too small (<2) for a standard deviation to exist at all."""
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)
    margin = 1.96 * se
    return (round(mean - margin, 3), round(mean + margin, 3))


def compute_alpha_metrics(observations: list, horizon: str = "20D") -> dict:
    """
    Computes Alpha Metrics for one horizon across a list of observations.
    Only observations with a FROZEN outcome for this exact horizon are
    included — an observation still waiting on that horizon contributes
    nothing (not a zero, not an estimate; genuinely excluded).

    Returns a dict with n, expectancy_%, win_rate_%, median_return_%,
    profit_factor, confidence_interval_95, sample_classification, and
    avg_excess_return_% (mean outperformance vs. the S&P 500 over the
    same window, when benchmark data was available for those trades).
    """
    returns = []
    excess_returns = []
    for obs in observations or []:
        outcome = (obs.get("outcomes") or {}).get(horizon)
        if not outcome or outcome.get("forward_return_%") is None:
            continue
        returns.append(float(outcome["forward_return_%"]))
        if outcome.get("excess_return_%") is not None:
            excess_returns.append(float(outcome["excess_return_%"]))

    n = len(returns)
    if n == 0:
        return {
            "horizon": horizon, "n": 0, "expectancy_%": None, "win_rate_%": None,
            "median_return_%": None, "profit_factor": None,
            "confidence_interval_95": None, "avg_excess_return_%": None,
            "sample_classification": classify_sample_size(0),
        }

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    win_rate = round(len(wins) / n * 100, 1)
    expectancy = round(sum(returns) / n, 3)  # mean return per trade, this horizon

    sorted_returns = sorted(returns)
    mid = n // 2
    median = sorted_returns[mid] if n % 2 == 1 else round((sorted_returns[mid - 1] + sorted_returns[mid]) / 2, 3)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_profit > 0 else None)

    return {
        "horizon": horizon,
        "n": n,
        "expectancy_%": expectancy,
        "win_rate_%": win_rate,
        "median_return_%": median,
        "profit_factor": profit_factor,
        "confidence_interval_95": _confidence_interval_95(returns),
        "avg_excess_return_%": round(sum(excess_returns) / len(excess_returns), 3) if excess_returns else None,
        "sample_classification": classify_sample_size(n),
    }


def compute_alpha_metrics_by_score_bucket(observations: list, horizon: str = "20D",
                                          buckets: Optional[list] = None) -> list:
    """
    Same computation as compute_alpha_metrics(), but grouped by
    apex_score_raw bucket — the direct evolution of Discovery Tracker's
    existing 'Does Apex Score Predict Returns?' table, now using fixed-
    horizon outcomes with MFE/MAE/benchmark-excess and real confidence
    intervals instead of a single ad-hoc 'whatever day you happened to
    check' price comparison.
    """
    if buckets is None:
        buckets = [("150+", 150, 10_000), ("125-149", 125, 150), ("100-124", 100, 125),
                   ("80-99", 80, 100), ("65-79", 65, 80), ("50-64", 50, 65), ("<50", 0, 50)]

    results = []
    for label, lo, hi in buckets:
        bucket_obs = [
            o for o in observations
            if o.get("apex_score_raw") is not None and lo <= float(o["apex_score_raw"]) < hi
        ]
        metrics = compute_alpha_metrics(bucket_obs, horizon)
        metrics["score_bucket"] = label
        results.append(metrics)
    return results
