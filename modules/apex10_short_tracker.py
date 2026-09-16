"""
modules/apex10_short_tracker.py — Apex the Great X, short side: Radar
Persistence + Forward Outcome Labeling

Bearish mirror of modules/apex10_tracker.py. Until this module existed,
modules/apex10_short_radar.py computed real technical features and a
short_score for every short candidate, but scanner.py only ever kept
the final score + status in that day's report — the features that
produced the score were computed and discarded on every single scan.
Confirmed empirically: data/alpha_observations.json had zero entries
of type "apex10_short_radar" despite 543 long-side "apex10_radar"
entries existing over the same window. There was no way to ever learn
whether compression, breakdown proximity, or any other short-side
feature actually predicts anything, because the first half of that
question — what did the feature look like at the moment of the call —
was never saved anywhere.

This module fixes that by reusing the exact same persistence shape and
the exact same outcome engine as the long side, for the reason
documented in apex10_tracker.py: modules/outcome_engine.py's
compute_all_pending_outcomes() keys only on entry_price + timestamp +
outcomes, with no concept of observation_type or direction. Give a
short radar entry that same shape and it gets its 5/10/20/40/60D
forward returns frozen automatically — no new outcome-computation code
needed, and no risk of the two directions being confused, since
observation_type ("apex10_short_radar" vs "apex10_radar") always
distinguishes them.

One direction-specific point worth being explicit about, mirroring the
long side's own note: a short radar entry's forward returns measure
what happens to the RAW PRICE from first_radar_date, not the P&L of an
actual short position. A -8% forward return means the stock fell 8%,
which would be favorable for a short thesis — sign interpretation is a
consumer-side decision (e.g. Feature Alpha), not baked into storage
here. This mirrors the project's existing discipline of storing
observable facts and deferring interpretation.

Same duplicate-entry and immutability rules as the long side apply
unchanged — see modules/apex10_tracker.py's docstring for the full
reasoning; not re-derived here to avoid the two modules drifting out
of sync on rules that have nothing to do with direction.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from modules.alpha_validation import load_observations, save_observations
from modules.apex10_baseline import get_observation_type
from modules.apex10_radar import classify_apex10_state
from modules.outcome_engine import HORIZONS as OUTCOME_HORIZONS

SHORT_RADAR_OBSERVATION_TYPE = "apex10_short_radar"
FEATURE_VERSION = "APEX10-SHORT-FEATURES-C1"
SCORE_MODEL_VERSION = "APEX10-SHORT-SCORE-D1"


def _is_fully_resolved(observation: dict) -> bool:
    return len(observation.get("outcomes", {})) >= len(OUTCOME_HORIZONS)


def _find_active_short_radar_entry(observations: list, ticker: str) -> Optional[dict]:
    matches = [o for o in observations
              if o.get("ticker") == ticker
              and get_observation_type(o) == SHORT_RADAR_OBSERVATION_TYPE]
    active = [o for o in matches if not _is_fully_resolved(o)]
    if not active:
        return None
    return max(active, key=lambda o: o.get("timestamp", ""))


def create_or_update_short_radar_entry(ticker: str, current_price: float, features: dict,
                                       score_result: dict, market: str = "US",
                                       liquidity_gate: Optional[dict] = None,
                                       trigger_gates: Optional[dict] = None,
                                       observations: Optional[list] = None,
                                       persist: bool = True,
                                       as_of: Optional[datetime] = None) -> dict:
    """
    Creates a new short radar cycle for `ticker`, or updates its
    existing active one — same no-duplicates rule as the long side.
    See modules/apex10_tracker.create_or_update_radar_entry for the
    reasoning behind `observations`/`persist`/`as_of`; identical here.
    """
    owns_list = observations is None
    if owns_list:
        observations = load_observations()

    existing = _find_active_short_radar_entry(observations, ticker)
    now = as_of or datetime.now()
    today_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today_date_str = today_str[:10]

    score = score_result.get("score")
    state = classify_apex10_state(score)
    score_entry = {"date": today_date_str, "score": score, "state": state,
                  "evidence_quality": score_result.get("evidence_quality")}

    bp = features.get("breakdown_proximity", {})

    if existing is None:
        entry = {
            "ticker": ticker, "observation_type": SHORT_RADAR_OBSERVATION_TYPE,
            "timestamp": today_str,          # IMMUTABLE anchor — outcome engine reads this
            "entry_price": current_price,    # IMMUTABLE anchor — outcome engine reads this
            "market": market,
            "first_radar_date": today_date_str, "first_radar_price": current_price,
            "current_price": current_price, "current_score": score, "current_state": state,
            "evidence_quality": score_result.get("evidence_quality"),
            "days_on_radar": 0,
            "score_history": [score_entry],
            "score_components": score_result.get("components"),
            "support_price": bp.get("support_price"),
            "distance_to_support_pct": bp.get("distance_to_support_pct"),
            "rs": features.get("relative_strength", {}).get("rs_current"),
            "rs_trend": features.get("relative_strength", {}).get("rs_5d_change"),
            "structure": features.get("structure"),
            "volatility": features.get("volatility"),
            "volume": features.get("volume"),
            "moving_averages": features.get("moving_averages"),
            "sector_confirmation": features.get("sector_confirmation"),
            "market_regime": features.get("market_regime"),
            "liquidity_gate": liquidity_gate,
            "breakdown_status": "PRE_BREAKDOWN",
            "breakdown_date": None, "breakdown_price": None,
            "feature_version": FEATURE_VERSION, "model_version": SCORE_MODEL_VERSION,
            "outcomes": {},
        }
        observations.append(entry)
    else:
        entry = existing
        entry["current_price"] = current_price
        entry["current_score"] = score
        entry["current_state"] = state
        entry["evidence_quality"] = score_result.get("evidence_quality")
        entry["score_components"] = score_result.get("components")
        entry["support_price"] = bp.get("support_price")
        entry["distance_to_support_pct"] = bp.get("distance_to_support_pct")
        entry["rs"] = features.get("relative_strength", {}).get("rs_current")
        entry["rs_trend"] = features.get("relative_strength", {}).get("rs_5d_change")
        entry["structure"] = features.get("structure")
        entry["volatility"] = features.get("volatility")
        entry["volume"] = features.get("volume")
        entry["moving_averages"] = features.get("moving_averages")
        entry["sector_confirmation"] = features.get("sector_confirmation")
        entry["market_regime"] = features.get("market_regime")
        if liquidity_gate is not None:
            entry["liquidity_gate"] = liquidity_gate

        try:
            first_date = datetime.strptime(entry["first_radar_date"], "%Y-%m-%d").date()
            entry["days_on_radar"] = (now.date() - first_date).days
        except Exception:
            pass

        matching_idx = next((i for i, e in enumerate(entry["score_history"])
                           if e.get("date") == today_date_str), None)
        if matching_idx is not None:
            entry["score_history"][matching_idx] = score_entry
        else:
            entry["score_history"].append(score_entry)  # append-only across days

        if (trigger_gates and trigger_gates.get("confirmed_breakdown")
                and entry.get("breakdown_status") == "PRE_BREAKDOWN"):
            entry["breakdown_status"] = "CONFIRMED_BREAKDOWN"
            entry["breakdown_date"] = today_date_str
            entry["breakdown_price"] = current_price

    if persist and owns_list:
        save_observations(observations)

    return entry


def run_daily_short_radar_update(candidates: list, market: str = "US",
                                 observations: Optional[list] = None) -> dict:
    """
    Batch entry point — mirrors run_daily_radar_update() exactly.
    `candidates`: list of {"ticker", "current_price", "features",
    "score_result", "liquidity_gate", "trigger_gates"} dicts already
    computed by the caller (scanner.py's short-side scan loop). Does
    NOT fetch price data or compute features itself — that stays at
    the call site, same as the long side.
    """
    owns_list = observations is None
    if owns_list:
        observations = load_observations()

    created, updated = 0, 0
    for c in candidates:
        existing_before = _find_active_short_radar_entry(observations, c["ticker"]) is not None
        create_or_update_short_radar_entry(
            c["ticker"], c["current_price"], c["features"], c["score_result"],
            market=market, liquidity_gate=c.get("liquidity_gate"),
            trigger_gates=c.get("trigger_gates"), observations=observations, persist=False,
        )
        if existing_before:
            updated += 1
        else:
            created += 1

    if owns_list and (created or updated):
        save_observations(observations)

    return {"created": created, "updated": updated, "total_candidates": len(candidates)}


def get_short_radar_table(observations: Optional[list] = None) -> list:
    """Flat, display-ready list of every apex10_short_radar entry, most
    recently scored first. Includes fully-resolved (closed) cycles too —
    filtering those out, if wanted, is a display-layer decision."""
    observations = observations if observations is not None else load_observations()
    radar_rows = [o for o in observations
                  if get_observation_type(o) == SHORT_RADAR_OBSERVATION_TYPE]
    radar_rows.sort(key=lambda o: o.get("current_score") or 0, reverse=True)
    return radar_rows
