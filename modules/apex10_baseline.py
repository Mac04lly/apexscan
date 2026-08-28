"""
modules/apex10_baseline.py — Apex the Great X: Stage B, Baseline Snapshot

Per the spec: "This becomes the control model against which Apex the
Great X is compared. DO NOT optimize anything yet." This module does
exactly one thing — it formalizes the EXISTING Apex Score system's
already-observed track record as APEX_V1_BASELINE, using data and
functions that already exist (modules/alpha_metrics.py's
compute_alpha_metrics), and persists a snapshot of it. It does not
change scoring, does not touch alpha_observations.json's schema, and
does not introduce any new feature or signal.

Why this is a separate small store (data/apex10_baseline.json) rather
than living inside alpha_observations.json: a baseline snapshot is a
SUMMARY artifact (one row per horizon, computed FROM the observations),
not a raw observation itself — it belongs next to the other apex10_*
stores, not mixed into the observation stream those stores measure.

Schema convention established here for the rest of Apex the Great X:
every future row appended to data/alpha_observations.json will carry
observation_type — "discovery" for the existing Phase-1-style rows,
"apex10_radar" for the new radar entries built in a later stage.
get_observation_type() below treats a MISSING observation_type as
"discovery" for backward compatibility with the 599 existing rows,
which predate this field and are never retroactively rewritten to add
it — consistent with this project's immutability rule.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from modules.alpha_metrics import compute_alpha_metrics
from modules.alpha_validation import load_observations
from modules.gh_storage import load_json_from_github, save_json_to_github

BASELINE_HORIZONS = ["5D", "10D", "20D", "40D", "60D"]
BASELINE_MODEL_VERSION = "APEX_V1_BASELINE"
BASELINE_PATH = "data/apex10_baseline.json"


def get_observation_type(observation: dict) -> str:
    """Backward-compatible accessor — see module docstring. Never
    mutates the observation; existing rows keep their exact original
    content forever."""
    return observation.get("observation_type") or "discovery"


def _get_gh_creds():
    try:
        import streamlit as st
        return st.secrets.get("github_token", ""), st.secrets.get("github_repo", "")
    except Exception:
        return "", ""


def compute_baseline_snapshot(observations: Optional[list] = None,
                              horizons: list = None) -> dict:
    """
    Computes the current Apex Score system's track record across every
    horizon, using ONLY existing "discovery"-type observations (Apex the
    Great X's own future apex10_radar rows are explicitly excluded here
    — the whole point of a baseline is that it measures APEX V1 alone,
    uncontaminated by anything Apex the Great X adds later).

    Every number here comes straight from modules.alpha_metrics — no new
    statistics are invented for this snapshot. Sample sizes will
    honestly read 0 for as long as they actually are 0.
    """
    obs = observations if observations is not None else load_observations()
    baseline_obs = [o for o in (obs or []) if get_observation_type(o) == "discovery"]
    horizons = horizons or BASELINE_HORIZONS

    per_horizon = {}
    for h in horizons:
        per_horizon[h] = compute_alpha_metrics(baseline_obs, h)

    return {
        "model_version": BASELINE_MODEL_VERSION,
        "computed_at": datetime.now().isoformat(),
        "total_discovery_observations": len(baseline_obs),
        "horizons": per_horizon,
        "note": ("Control group for Apex the Great X comparisons. Computed from "
                "existing 'discovery'-type observations only. Not optimized, not "
                "tuned — a direct read of the existing Apex Score system's "
                "already-observed track record as of computed_at."),
    }


def save_baseline_snapshot(snapshot: dict) -> bool:
    token, repo = _get_gh_creds()
    if not (token and repo):
        return False
    history, _ = load_json_from_github(token, repo, BASELINE_PATH)
    history = history if isinstance(history, list) else []
    history.append(snapshot)
    return save_json_to_github(token, repo, BASELINE_PATH, history,
                               message=f"Apex the Great X baseline snapshot "
                                      f"({snapshot['total_discovery_observations']} observations)")


def load_baseline_history() -> list:
    """Every snapshot ever taken, oldest first — never overwritten, so
    the baseline's own track record over time stays visible (e.g. to
    see how the sample size and resulting metrics grew as more
    discoveries resolved)."""
    token, repo = _get_gh_creds()
    if token and repo:
        data, _ = load_json_from_github(token, repo, BASELINE_PATH)
        if isinstance(data, list):
            return data
    return []


def get_latest_baseline() -> Optional[dict]:
    history = load_baseline_history()
    return history[-1] if history else None
