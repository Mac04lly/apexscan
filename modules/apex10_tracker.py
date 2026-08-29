"""
modules/apex10_tracker.py — Apex the Great X: Stage E, Radar Persistence
(combined with Stage G, Forward Outcome Labeling — see below for why)

Persists radar entries as observation_type="apex10_radar" rows inside
the EXISTING data/alpha_observations.json store (your Stage-A decision),
using the existing modules/alpha_validation.load_observations() /
save_observations() — no new JSON file, no new GitHub path.

════════════════════════════════════════════════════════════════════════
THE MUTABLE/IMMUTABLE SPLIT — resolving the tension flagged before Stage
B was built
════════════════════════════════════════════════════════════════════════
A radar entry has two kinds of fields:
  - IMMUTABLE, set once at creation and never touched again:
    `timestamp` (= first_radar_date) and `entry_price` (= first_radar_price).
    These are the anchor the outcome engine measures FROM — exactly the
    same two fields a "discovery" observation uses for the same purpose.
  - MUTABLE, updated in place as the radar entry evolves:
    current_price, current_score, current_state, days_on_radar,
    breakout_status/date/price.
  - APPEND-ONLY, inside the mutable record: `score_history`. Once a
    day's entry is written to this list it is never edited — "Day 5:
    68" stays "Day 5: 68" forever, even after Day 18 says 92. This is
    what keeps the "never retroactively edit a historical fact" rule
    intact for the one part of a radar entry that genuinely needs to
    change over time.

════════════════════════════════════════════════════════════════════════
WHY STAGE G NEEDED ALMOST NO NEW CODE
════════════════════════════════════════════════════════════════════════
modules/outcome_engine.py's compute_all_pending_outcomes() was read
before writing this file: it iterates EVERY observation in the store,
keying only on `entry_price` + `timestamp` + `outcomes` — it has no
concept of observation_type at all. Give a radar entry that exact same
shape (done above) and the existing, already-tested outcome engine
freezes its 5/10/20/40/60D returns automatically, with zero new code.
This is real reuse, not just architectural tidiness — duplicating that
logic here would be exactly the kind of unnecessary duplication the
Stage-A audit was asked to flag against.

One consequence worth being explicit about: a radar entry's forward
returns are measured from FIRST_RADAR_DATE (when it was first flagged
as developing), not from breakout_date. That's a deliberate choice —
it answers "did flagging this early actually matter", which is the
whole point of Apex the Great X, rather than only "did the eventual
breakout work" (which the existing Discovery Tracker already answers
for confirmed setups).

════════════════════════════════════════════════════════════════════════
DUPLICATE-ENTRY PREVENTION
════════════════════════════════════════════════════════════════════════
Per spec: "If a stock is already on the radar: DO NOT create duplicate
entries. Update the existing record." An "active" cycle is one whose
outcomes dict doesn't yet have all 5 horizons frozen. Once a ticker's
radar cycle is fully resolved (all 5 horizons frozen, same completeness
check the outcome engine itself uses), any further sighting starts a
genuinely NEW cycle rather than reopening the old one — the old,
now-complete entry is a permanent historical fact, consistent with the
immutability rule everywhere else in this project.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from modules.alpha_validation import load_observations, save_observations
from modules.apex10_baseline import get_observation_type
from modules.apex10_radar import classify_apex10_state
from modules.outcome_engine import HORIZONS as OUTCOME_HORIZONS

RADAR_OBSERVATION_TYPE = "apex10_radar"
FEATURE_VERSION = "APEX10-FEATURES-C1"   # Stage C
SCORE_MODEL_VERSION = "APEX10-SCORE-D1"  # Stage D


def _is_fully_resolved(observation: dict) -> bool:
    return len(observation.get("outcomes", {})) >= len(OUTCOME_HORIZONS)


def _find_active_radar_entry(observations: list, ticker: str) -> Optional[dict]:
    matches = [o for o in observations
              if o.get("ticker") == ticker and get_observation_type(o) == RADAR_OBSERVATION_TYPE]
    active = [o for o in matches if not _is_fully_resolved(o)]
    if not active:
        return None
    return max(active, key=lambda o: o.get("timestamp", ""))


def create_or_update_radar_entry(ticker: str, current_price: float, features: dict,
                                 score_result: dict, market: str = "US",
                                 liquidity_gate: Optional[dict] = None,
                                 trigger_gates: Optional[dict] = None,
                                 observations: Optional[list] = None,
                                 persist: bool = True,
                                 as_of: Optional[datetime] = None) -> dict:
    """
    Creates a new radar cycle for `ticker`, or updates its existing
    active one — never both, per the spec's explicit no-duplicates rule.

    `observations` / `persist` exist so callers (and tests) can operate
    on an in-memory list without hitting GitHub storage every call —
    the default (persist=True, observations=None) is the normal
    load -> mutate -> save flow used everywhere else in this app.

    `as_of` exists purely for deterministic testing of day-over-day
    progression (e.g. simulating "Day 1" then "Day 5" without waiting
    for real wall-clock time to pass); production callers never pass it
    and get the real datetime.now(), same as everywhere else in this app.
    """
    owns_list = observations is None
    if owns_list:
        observations = load_observations()

    existing = _find_active_radar_entry(observations, ticker)
    now = as_of or datetime.now()
    today_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today_date_str = today_str[:10]

    score = score_result.get("score")
    state = classify_apex10_state(score)
    score_entry = {"date": today_date_str, "score": score, "state": state,
                  "evidence_quality": score_result.get("evidence_quality")}

    bp = features.get("breakout_proximity", {})

    if existing is None:
        entry = {
            "ticker": ticker, "observation_type": RADAR_OBSERVATION_TYPE,
            "timestamp": today_str,          # IMMUTABLE anchor — outcome engine reads this
            "entry_price": current_price,    # IMMUTABLE anchor — outcome engine reads this
            "market": market,
            "first_radar_date": today_date_str, "first_radar_price": current_price,
            "current_price": current_price, "current_score": score, "current_state": state,
            "evidence_quality": score_result.get("evidence_quality"),
            "days_on_radar": 0,
            "score_history": [score_entry],
            "score_components": score_result.get("components"),
            "resistance_price": bp.get("resistance_price"),
            "distance_to_resistance_pct": bp.get("distance_to_resistance_pct"),
            "rs": features.get("relative_strength", {}).get("rs_current"),
            "rs_trend": features.get("relative_strength", {}).get("rs_5d_change"),
            "structure": features.get("structure"),
            "volatility": features.get("volatility"),
            "volume": features.get("volume"),
            "moving_averages": features.get("moving_averages"),
            "sector_confirmation": features.get("sector_confirmation"),
            "market_regime": features.get("market_regime"),
            "liquidity_gate": liquidity_gate,
            "breakout_status": "PRE_BREAKOUT",
            "breakout_date": None, "breakout_price": None,
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
        entry["resistance_price"] = bp.get("resistance_price")
        entry["distance_to_resistance_pct"] = bp.get("distance_to_resistance_pct")
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
            # Same calendar day, later reading: refine today's own entry
            # in place rather than append a second one. This does NOT
            # violate the append-only rule — that rule protects PAST
            # days' facts from being rewritten once the day has ended;
            # today's own entry isn't history yet until tomorrow.
            entry["score_history"][matching_idx] = score_entry
        else:
            entry["score_history"].append(score_entry)  # append-only across days — see module docstring

        if (trigger_gates and trigger_gates.get("confirmed_breakout")
                and entry.get("breakout_status") == "PRE_BREAKOUT"):
            entry["breakout_status"] = "CONFIRMED_BREAKOUT"
            entry["breakout_date"] = today_date_str
            entry["breakout_price"] = current_price

    if persist and owns_list:
        save_observations(observations)

    return entry


def run_daily_radar_update(candidates: list, market: str = "US",
                           observations: Optional[list] = None) -> dict:
    """
    Batch entry point: `candidates` is a list of dicts already computed
    by the caller — {"ticker", "current_price", "features", "score_result",
    "liquidity_gate", "trigger_gates"} — one per ticker being tracked
    today. Deliberately does NOT fetch price data or compute features
    itself; that's scan-orchestration (batch_fetch_history, Stage C/D),
    which belongs at the call site, not duplicated in here. Single
    load/save round-trip for the whole batch rather than one per ticker.
    """
    owns_list = observations is None
    if owns_list:
        observations = load_observations()

    created, updated = 0, 0
    for c in candidates:
        existing_before = _find_active_radar_entry(observations, c["ticker"]) is not None
        create_or_update_radar_entry(
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


def get_radar_table(observations: Optional[list] = None) -> list:
    """Flat, display-ready list of every apex10_radar entry, most
    recently scored first. Includes fully-resolved (closed) cycles too —
    filtering those out, if wanted, is a display-layer decision."""
    observations = observations if observations is not None else load_observations()
    radar_rows = [o for o in observations if get_observation_type(o) == RADAR_OBSERVATION_TYPE]
    radar_rows.sort(key=lambda o: o.get("current_score") or 0, reverse=True)
    return radar_rows
