"""
modules/walk_forward.py — Walk-Forward Validation (V9 Phase 9)

Per the V9 spec (§26, §44): "Do not tune the model using the same data
used to evaluate it." This module answers exactly one question — does a
setup/feature edge that showed up in EARLIER observations still hold up
in LATER, chronologically-subsequent observations it could not possibly
have influenced?

Deliberately NOT random k-fold cross-validation: a random split would
let a later observation land in a "training" fold used to evaluate an
earlier "test" fold, which leaks future information backwards in time —
exactly wrong for anything sequential. Every split here respects
chronological order: TEST always comes after TRAIN.

Per the spec's explicit phase gate ("Only now begin serious model
optimization" — Phase 9 comes AFTER Phase 8's governance layer), this
module also refuses to produce a verdict on too little history — see
walk_forward_readiness() — rather than running a statistically hollow
analysis just because it technically can.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from modules.alpha_metrics import compute_alpha_metrics_by_setup

# Matches the sample-size bar used elsewhere in this app (Discovery
# Tracker's "Sample Readiness" indicator, Alpha Lab's min_n defaults) —
# not a new, arbitrary threshold invented for this module alone.
MIN_OBSERVATIONS_PER_FOLD = 30
DEFAULT_N_FOLDS = 3


def _parse_ts(observation: dict) -> Optional[datetime]:
    try:
        return datetime.strptime(observation.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def get_observation_date_range(observations: list):
    """Returns (earliest, latest) datetime across observations that have
    a parseable timestamp, or (None, None) if none do."""
    dates = [d for d in (_parse_ts(o) for o in observations or []) if d is not None]
    if not dates:
        return None, None
    return min(dates), max(dates)


def split_train_test(observations: list, split_date: datetime) -> tuple:
    """Simplest possible walk-forward split: everything discovered
    strictly before split_date is TRAIN, everything on/after is TEST.
    Nothing in TEST could have influenced anything computed from TRAIN."""
    train, test = [], []
    for o in observations or []:
        d = _parse_ts(o)
        if d is None:
            continue
        (train if d < split_date else test).append(o)
    return train, test


def rolling_folds(observations: list, n_folds: int = DEFAULT_N_FOLDS) -> list:
    """Splits the observed date range into n_folds+1 chronological
    chunks and yields (train, test) pairs where TEST is always the chunk
    immediately following TRAIN. Returns [] if there's no dated data at
    all — never raises on empty input."""
    dated = sorted(
        [(o, d) for o in (observations or []) if (d := _parse_ts(o)) is not None],
        key=lambda pair: pair[1],
    )
    if not dated:
        return []
    n = len(dated)
    chunk_size = max(1, n // (n_folds + 1))
    folds = []
    for i in range(n_folds):
        train_end = (i + 1) * chunk_size
        test_end = min(n, (i + 2) * chunk_size)
        if train_end >= n or test_end <= train_end:
            break
        train = [o for o, _ in dated[:train_end]]
        test = [o for o, _ in dated[train_end:test_end]]
        folds.append((train, test))
    return folds


def evaluate_setup_stability(observations: list, horizon: str = "20D",
                             n_folds: int = DEFAULT_N_FOLDS,
                             min_n: int = MIN_OBSERVATIONS_PER_FOLD) -> dict:
    """
    The core Phase 9 test: in each fold, find the best-expectancy setup
    on TRAIN (with n >= min_n), then check whether that SAME setup's
    edge held up (n >= min_n and expectancy > 0) on the chronologically
    later TEST chunk. Deliberately simple rank-stability, not a fitted
    model — per the spec's own §44/§45, descriptive statistics and
    out-of-sample validation come BEFORE any model-fitting, not after.
    """
    folds = rolling_folds(observations, n_folds)
    if not folds:
        return {
            "status": "insufficient_history", "folds_run": 0, "fold_results": [],
            "pass_rate": None,
            "detail": "Not enough dated observations yet to form even one train/test split.",
        }

    fold_results = []
    for i, (train, test) in enumerate(folds):
        train_setups = [s for s in compute_alpha_metrics_by_setup(train, horizon) if s["n"] >= min_n]
        if not train_setups:
            fold_results.append({
                "fold": i, "status": "insufficient_train_data",
                "train_n": len(train), "test_n": len(test),
            })
            continue
        best_train_setup = max(train_setups, key=lambda s: s["expectancy_%"])
        test_setups_by_id = {s["setup_id"]: s for s in compute_alpha_metrics_by_setup(test, horizon)}
        test_result = test_setups_by_id.get(best_train_setup["setup_id"])

        held_up = None
        if test_result is not None and test_result["n"] >= min_n and test_result["expectancy_%"] is not None:
            held_up = test_result["expectancy_%"] > 0

        fold_results.append({
            "fold": i, "status": "ok",
            "train_range_n": len(train), "test_range_n": len(test),
            "best_train_setup": best_train_setup["setup_id"],
            "train_expectancy_%": best_train_setup["expectancy_%"],
            "train_n": best_train_setup["n"],
            "test_n": test_result["n"] if test_result else 0,
            "test_expectancy_%": test_result["expectancy_%"] if test_result else None,
            "held_up": held_up,
        })

    usable = [r for r in fold_results if r.get("held_up") is not None]
    pass_rate = (sum(1 for r in usable if r["held_up"]) / len(usable)) if usable else None

    return {
        "status": "ok" if usable else "insufficient_test_data",
        "folds_run": len(fold_results), "folds_usable": len(usable),
        "fold_results": fold_results, "pass_rate": pass_rate,
    }


def walk_forward_readiness(observations: list, horizon: str = "20D",
                           min_n: int = MIN_OBSERVATIONS_PER_FOLD,
                           n_folds: int = DEFAULT_N_FOLDS) -> dict:
    """
    Entry point: first checks whether there's genuinely enough
    chronological, resolved history to run walk-forward validation
    meaningfully AT ALL. If not, says exactly what's missing (current
    count vs. estimated need) instead of running a hollow analysis on
    too little data. If ready, runs evaluate_setup_stability() and adds
    a `passed` flag (pass_rate >= 0.5 across usable folds, with at least
    2 usable folds) that modules/model_registry.record_walk_forward_result()
    consumes as the promotion gate.
    """
    resolved = [o for o in (observations or []) if (o.get("outcomes") or {}).get(horizon)]
    lo, hi = get_observation_date_range(resolved)
    needed = min_n * (n_folds + 1)

    if len(resolved) < needed:
        return {
            "ready": False, "passed": False,
            "resolved_observations": len(resolved), "needed_estimate": needed,
            "date_range": (lo.isoformat() if lo else None, hi.isoformat() if hi else None),
            "message": (f"{len(resolved)} of an estimated {needed}+ resolved observations "
                       f"needed before a {n_folds}-fold walk-forward split is meaningful. "
                       f"Let ApexScan keep scanning and resolving outcomes."),
        }

    result = evaluate_setup_stability(observations, horizon, n_folds, min_n)
    passed = (result["status"] == "ok" and result["folds_usable"] >= 2
             and result["pass_rate"] is not None and result["pass_rate"] >= 0.5)
    result.update({
        "ready": True, "passed": passed,
        "resolved_observations": len(resolved),
        "date_range": (lo.isoformat() if lo else None, hi.isoformat() if hi else None),
    })
    return result
