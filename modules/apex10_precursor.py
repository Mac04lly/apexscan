"""
modules/apex10_precursor.py — Apex the Great X: Stage F, Historical
Precursor Engine

Walks backward from known winners (existing "discovery"-type
observations with a strong resolved excess return) and reconstructs
what Apex the Great X's own Stage C/D feature engine and score would
have said about them 5/10/20/30/40/60 trading days BEFORE their
discovery — using Stage C and D completely unmodified.

════════════════════════════════════════════════════════════════════════
API-LOAD SAFETY — this is the stage the Stage-A audit specifically
flagged as the highest throttling risk in the whole spec. Three
concrete, code-enforced limits, not just documented intentions:
════════════════════════════════════════════════════════════════════════
1. MAX_CANDIDATES_HARD_CAP — select_precursor_candidates() cannot
   return more than this many tickers, no matter what max_candidates
   argument is passed. A study is bounded by construction, not by the
   caller remembering to be careful.
2. ONE bulk fetch for an entire study run — fetch_precursor_histories()
   calls scanner.batch_fetch_history() exactly once, with every
   candidate ticker AND the benchmark bundled into the same request.
   This is scanner.py's own existing, already-proven
   throttling-mitigation function (confirmed by reading it before
   writing this file) — not a new fetcher. Reusing it here means this
   module introduces ZERO new network-call code.
3. All 6 backward offsets for a ticker come from that ONE fetch — no
   per-offset, per-ticker refetching. 20 candidates x 6 offsets is one
   batched network call, not 120.

════════════════════════════════════════════════════════════════════════
NO-LOOK-AHEAD — reused, not reimplemented
════════════════════════════════════════════════════════════════════════
Every feature computation here goes through
modules.apex10_features.compute_precursor_features(as_of_date=...)
UNCHANGED — the same function whose no-look-ahead guarantee already has
6 dedicated tests (tests/test_apex10_no_lookahead.py), including
targeted future-price and future-volume spike injections. This module
adds no new slicing logic of its own to get that guarantee wrong.

════════════════════════════════════════════════════════════════════════
"WHY OUR WINNERS WON" — denominator discipline
════════════════════════════════════════════════════════════════════════
aggregate_precursor_findings() always reports n (the denominator)
alongside every percentage, and refuses to report a percentage at all
below min_n — per spec: "The denominator must always be visible. Avoid
cherry-picking winners." A percentage without its sample size is
exactly the kind of manufactured-looking evidence this whole project
has been built to avoid.

════════════════════════════════════════════════════════════════════════
KNOWN, DOCUMENTED LIMITATIONS (not solved by this module — flagged in
the Stage-A audit, repeated here at the point they actually matter)
════════════════════════════════════════════════════════════════════════
- Survivorship bias: candidates come from ApexScan's OWN discovery
  history, which only exists for tickers that were scanned and passed
  Stage 2 gating at the time — this is not, and cannot be with this
  app's current data sources, a bias-free sample of "all historical
  winners." It answers "what did OUR winners look like", not "what do
  ALL market winners look like."
- No point-in-time fundamentals/institutional data: Feature Families
  8/9 will show UNKNOWN at every offset for every candidate, exactly as
  they do everywhere else in this project, for the same reason.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

import pandas as pd

from modules.apex10_baseline import get_observation_type
from modules.apex10_features import compute_precursor_features
from modules.apex10_radar import compute_apex10_score, classify_apex10_state
from modules.gh_storage import load_json_from_github, save_json_to_github

BENCHMARK_TICKER = "^GSPC"
MAX_CANDIDATES_HARD_CAP = 50
DEFAULT_OFFSETS_TRADING_DAYS = [60, 40, 30, 20, 10, 5, 0]
PRECURSOR_STUDY_PATH = "data/apex10_feature_history.json"


def _get_gh_creds():
    try:
        import streamlit as st
        return st.secrets.get("github_token", ""), st.secrets.get("github_repo", "")
    except Exception:
        return "", ""


def _parse_discovery_date(timestamp: str):
    if not timestamp:
        return None
    try:
        return datetime.strptime(timestamp.split(" ")[0], "%Y-%m-%d").date()
    except Exception:
        try:
            return pd.to_datetime(timestamp).date()
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════
# CANDIDATE SELECTION — bounded, reusing already-resolved outcomes
# ══════════════════════════════════════════════════════════════════════

def select_precursor_candidates(observations: list, min_excess_return_pct: float = 20.0,
                                horizon: str = "20D", max_candidates: int = 20) -> list:
    """
    Selects known winners from EXISTING 'discovery'-type observations —
    only ones with an already-FROZEN outcome at `horizon` are eligible,
    reusing Phase 2's completed work rather than guessing at "winners"
    from still-pending observations. Hard-capped at
    MAX_CANDIDATES_HARD_CAP regardless of what's requested.
    """
    max_candidates = min(max_candidates, MAX_CANDIDATES_HARD_CAP)
    candidates = []
    for o in observations or []:
        if get_observation_type(o) != "discovery":
            continue
        outcome = (o.get("outcomes") or {}).get(horizon)
        if not outcome or outcome.get("excess_return_%") is None:
            continue
        if outcome["excess_return_%"] < min_excess_return_pct:
            continue
        if not o.get("ticker") or not o.get("timestamp"):
            continue
        candidates.append({
            "ticker": o["ticker"], "discovery_timestamp": o["timestamp"],
            "entry_price": o.get("entry_price"), "excess_return_%": outcome["excess_return_%"],
            "setup_id": o.get("setup_id"),
        })
    candidates.sort(key=lambda c: c["excess_return_%"], reverse=True)
    return candidates[:max_candidates]


# ══════════════════════════════════════════════════════════════════════
# ONE BULK FETCH — reuses scanner.batch_fetch_history() verbatim
# ══════════════════════════════════════════════════════════════════════

def fetch_precursor_histories(tickers: list, period: str = "2y") -> dict:
    """ONE batched call for every candidate ticker PLUS the benchmark.
    No new fetch logic — see module docstring."""
    from scanner import batch_fetch_history
    all_tickers = list(dict.fromkeys(list(tickers) + [BENCHMARK_TICKER]))  # de-dupe, preserve order
    return batch_fetch_history(all_tickers, period=period)


# ══════════════════════════════════════════════════════════════════════
# TRADING-DAY OFFSET RESOLUTION — real calendar rows, not a calendar-day
# approximation (same principle outcome_engine.py uses for forward
# returns, applied backward here).
# ══════════════════════════════════════════════════════════════════════

def _trading_day_offset_date(hist_index, discovery_date, offset_bars: int):
    """Walks back `offset_bars` REAL trading days (actual rows already
    in hist_index) from discovery_date. Returns None if there isn't
    enough history before discovery_date for this offset — never
    extrapolates or approximates a date that wasn't actually a trading
    day in the fetched data."""
    idx = pd.to_datetime(hist_index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    idx = idx.normalize()
    cutoff = pd.to_datetime(discovery_date).normalize()
    on_or_before = idx[idx <= cutoff]
    if len(on_or_before) <= offset_bars:
        return None
    return on_or_before[-(offset_bars + 1)]


# ══════════════════════════════════════════════════════════════════════
# TRAJECTORY RECONSTRUCTION — Stage C/D reused unchanged
# ══════════════════════════════════════════════════════════════════════

_KEY_FEATURE_PATHS = {
    "rs_5d_change": ("relative_strength", "rs_5d_change"),
    "distance_to_resistance_pct": ("breakout_proximity", "distance_to_resistance_pct"),
    "volatility_contraction": ("volatility", "volatility_contraction"),
    "volume_contraction": ("volume", "volume_contraction"),
    "higher_lows_flat_resistance": ("structure", "higher_lows_flat_resistance"),
    "ma50_transition": ("moving_averages", "ma50_transition"),
}


def _extract_key_features(features: dict) -> dict:
    out = {}
    for name, (family, field) in _KEY_FEATURE_PATHS.items():
        out[name] = features.get(family, {}).get(field)
    return out


def reconstruct_precursor_trajectory(ticker: str, discovery_timestamp: str, hist: pd.DataFrame,
                                     bench_hist: pd.DataFrame,
                                     offsets_trading_days: list = None) -> dict:
    """
    For ONE candidate: computes the apex10 feature snapshot and score at
    each backward offset from its discovery date, via Stage C/D
    unchanged. Every as_of_date passed is <= discovery_date by
    construction (_trading_day_offset_date only ever returns dates from
    on_or_before). Offsets without enough history are reported as such,
    never fabricated.
    """
    offsets_trading_days = offsets_trading_days or DEFAULT_OFFSETS_TRADING_DAYS
    discovery_date = _parse_discovery_date(discovery_timestamp)
    trajectory = []

    if discovery_date is None or hist is None or bench_hist is None or "Close" not in hist:
        return {"ticker": ticker, "discovery_timestamp": discovery_timestamp,
               "status": "no_usable_data", "trajectory": []}

    bench_close = bench_hist["Close"]

    for offset in offsets_trading_days:
        as_of = _trading_day_offset_date(hist.index, discovery_date, offset)
        if as_of is None:
            trajectory.append({"offset_trading_days": offset, "as_of_date": None,
                              "status": "insufficient_history"})
            continue

        features = compute_precursor_features(hist, bench_close, as_of_date=as_of)
        if "relative_strength" not in features:
            trajectory.append({"offset_trading_days": offset, "as_of_date": str(as_of.date()),
                              "status": "insufficient_history"})
            continue

        score_result = compute_apex10_score(features)
        trajectory.append({
            "offset_trading_days": offset, "as_of_date": str(as_of.date()), "status": "ok",
            "score": score_result["score"],
            "state": classify_apex10_state(score_result["score"]),
            "evidence_quality": score_result["evidence_quality"],
            "key_features": _extract_key_features(features),
        })

    return {"ticker": ticker, "discovery_timestamp": discovery_timestamp,
           "status": "ok", "trajectory": trajectory}


# ══════════════════════════════════════════════════════════════════════
# FULL PIPELINE — single entry point
# ══════════════════════════════════════════════════════════════════════

def run_precursor_study(observations: list, min_excess_return_pct: float = 20.0,
                        horizon: str = "20D", max_candidates: int = 20,
                        offsets_trading_days: list = None, period: str = "2y",
                        histories: Optional[dict] = None) -> dict:
    """
    Full Stage F pipeline. ONE batch fetch for the entire study (unless
    `histories` is pre-supplied, e.g. by a test or a caller reusing an
    already-fetched batch). Returns a study record ready to persist via
    save_precursor_study().
    """
    offsets_trading_days = offsets_trading_days or DEFAULT_OFFSETS_TRADING_DAYS
    candidates = select_precursor_candidates(observations, min_excess_return_pct, horizon, max_candidates)
    if not candidates:
        return {"status": "no_candidates", "computed_at": datetime.now().isoformat(),
               "candidates_considered": 0, "trajectories_computed": 0,
               "skipped": [], "trajectories": []}

    tickers = [c["ticker"] for c in candidates]
    if histories is None:
        histories = fetch_precursor_histories(tickers, period=period)
    bench_hist = histories.get(BENCHMARK_TICKER)

    trajectories, skipped = [], []
    for c in candidates:
        hist = histories.get(c["ticker"])
        if hist is None or len(hist) < 30 or bench_hist is None or len(bench_hist) < 30:
            skipped.append({"ticker": c["ticker"], "reason": "no_usable_data_from_batch_fetch"})
            continue
        traj = reconstruct_precursor_trajectory(c["ticker"], c["discovery_timestamp"],
                                                hist, bench_hist, offsets_trading_days)
        traj["excess_return_%"] = c["excess_return_%"]
        traj["setup_id"] = c["setup_id"]
        trajectories.append(traj)

    return {
        "status": "ok", "computed_at": datetime.now().isoformat(),
        "min_excess_return_pct_filter": min_excess_return_pct, "horizon": horizon,
        "offsets_trading_days": offsets_trading_days,
        "candidates_considered": len(candidates), "trajectories_computed": len(trajectories),
        "skipped": skipped, "trajectories": trajectories,
    }


# ══════════════════════════════════════════════════════════════════════
# PERSISTENCE — append-only study history, same pattern as
# modules/apex10_baseline.py's snapshot history.
# ══════════════════════════════════════════════════════════════════════

def save_precursor_study(study: dict) -> bool:
    token, repo = _get_gh_creds()
    if not (token and repo):
        return False
    history, _ = load_json_from_github(token, repo, PRECURSOR_STUDY_PATH)
    history = history if isinstance(history, list) else []
    history.append(study)
    return save_json_to_github(token, repo, PRECURSOR_STUDY_PATH, history,
                               message=f"Apex the Great X precursor study "
                                      f"({study.get('trajectories_computed', 0)} trajectories)")


def load_precursor_studies() -> list:
    token, repo = _get_gh_creds()
    if token and repo:
        data, _ = load_json_from_github(token, repo, PRECURSOR_STUDY_PATH)
        if isinstance(data, list):
            return data
    return []


def get_latest_precursor_study() -> Optional[dict]:
    studies = load_precursor_studies()
    return studies[-1] if studies else None


# ══════════════════════════════════════════════════════════════════════
# "WHY OUR WINNERS WON" — aggregation with mandatory denominator
# ══════════════════════════════════════════════════════════════════════

_AGGREGATE_CONDITIONS = {
    "rs_accelerating": lambda kf: kf.get("rs_5d_change") is not None and kf["rs_5d_change"] > 0,
    "volatility_contracting": lambda kf: kf.get("volatility_contraction") is True,
    "volume_contracting": lambda kf: kf.get("volume_contraction") is True,
    "higher_low_structure": lambda kf: kf.get("higher_lows_flat_resistance") is True,
    "ma50_turning_up_or_better": lambda kf: kf.get("ma50_transition") in ("TURNING_UP", "ACCELERATING_UP"),
}


def aggregate_precursor_findings(study: dict, offset_trading_days: int = 20, min_n: int = 5) -> dict:
    """
    Aggregates trajectory conditions at ONE specific backward offset,
    across every successfully-computed trajectory in a study. Always
    reports n; refuses to report a percentage at all below min_n.
    """
    points = []
    for traj in study.get("trajectories", []):
        match = next((p for p in traj.get("trajectory", [])
                    if p.get("offset_trading_days") == offset_trading_days and p.get("status") == "ok"),
                    None)
        if match:
            points.append(match)

    n = len(points)
    if n < min_n:
        return {
            "offset_trading_days": offset_trading_days, "n": n, "status": "insufficient_sample",
            "message": (f"Only {n} winner(s) have usable data at this offset — below the "
                       f"minimum of {min_n} needed to report a percentage without it being "
                       f"misleading."),
        }

    conditions_result = {}
    for name, cond_fn in _AGGREGATE_CONDITIONS.items():
        count = sum(1 for p in points if cond_fn(p.get("key_features", {})))
        conditions_result[name] = {"count": count, "n": n, "pct": round(count / n * 100, 1)}

    return {"offset_trading_days": offset_trading_days, "n": n, "status": "ok",
           "conditions": conditions_result}
