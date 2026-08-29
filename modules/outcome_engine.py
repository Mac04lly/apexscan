"""
modules/outcome_engine.py — Outcome Engine (V9 Phase 2)

For every AlphaObservation old enough to have reached a given horizon
(5/10/20/40/60 trading days), computes and PERMANENTLY FREEZES:
  - forward_return_% at that horizon
  - MFE (Maximum Favorable Excursion — best price reached during the hold)
  - MAE (Maximum Adverse Excursion — worst price reached during the hold)
  - benchmark_return_% (S&P 500 over the identical date range)
  - excess_return_% (forward_return minus benchmark_return)

Golden rule, enforced directly in code: once a horizon's outcome is
written for an observation, it is NEVER recomputed or overwritten, even
if this runs again later. An outcome is a fact about what happened in a
fixed historical window — it doesn't change with time, and recomputing
it "fresher" would be a subtle form of the same hindsight contamination
the whole Alpha Observation System exists to prevent.

Type-agnostic by design: every function here keys ONLY on `entry_price`,
`timestamp`, and `outcomes` — there is no branch anywhere on
observation_type. This was confirmed (not assumed) before
modules/apex10_tracker.py (Apex the Great X, Stage E/G) was built on
top of it: as long as a new observation kind provides those same three
fields with the same meaning, this engine freezes its forward returns
automatically, with no new code required here. Keep it that way — don't
add an observation_type branch to this file; if a future observation
kind needs genuinely different outcome logic, that belongs in a new
module that calls into (or wraps) this one, not a conditional inside it.

Trading days are counted from REAL fetched price data (actual rows
returned by yfinance for that ticker), not a calendar-day approximation —
more accurate given holidays/exchange-specific closures.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

from modules.alpha_validation import load_observations, save_observations

HORIZONS = [5, 10, 20, 40, 60]
_BENCH_TICKER = "^GSPC"

# Calendar-day buffer per horizon, generous enough to guarantee that many
# trading days have elapsed even across holidays — used only to decide
# whether an observation is "old enough to attempt," not for the outcome
# math itself (that uses real fetched rows).
_HORIZON_CALENDAR_BUFFER = {5: 10, 10: 18, 20: 32, 40: 62, 60: 92}


def _parse_discovery_date(observation: dict):
    ts = observation.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.strptime(ts.split(" ")[0], "%Y-%m-%d").date()
    except Exception:
        try:
            return pd.to_datetime(ts).date()
        except Exception:
            return None


def _fetch_ticker_history(ticker: str, start_date, end_date):
    if yf is None:
        return None
    try:
        hist = yf.Ticker(ticker).history(
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if hist is None or hist.empty:
            return None
        if hasattr(hist.index, "tz") and hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)
        hist.index = pd.to_datetime(hist.index).normalize()
        return hist
    except Exception:
        return None


def compute_outcomes_for_observation(observation: dict, bench_cache: Optional[dict] = None) -> bool:
    """
    Attempts to fill in any not-yet-frozen horizon outcomes for one
    observation. Returns True if anything new was written. Never
    overwrites an already-frozen horizon — checked explicitly before
    every write.
    """
    ticker = observation.get("ticker")
    entry_price = observation.get("entry_price")
    discovery_date = _parse_discovery_date(observation)
    if not ticker or not entry_price or not discovery_date:
        return False

    outcomes = observation.setdefault("outcomes", {})
    today = datetime.now().date()

    pending_horizons = [
        h for h in HORIZONS
        if f"{h}D" not in outcomes  # ── frozen check: never recompute a written horizon ──
        and (today - discovery_date).days >= _HORIZON_CALENDAR_BUFFER[h]
    ]
    if not pending_horizons:
        return False

    max_horizon = max(pending_horizons)
    fetch_end = min(today, discovery_date + timedelta(days=_HORIZON_CALENDAR_BUFFER[max_horizon] + 10))
    hist = _fetch_ticker_history(ticker, discovery_date, fetch_end)
    if hist is None or len(hist) < 2:
        return False

    bench_hist = None
    if bench_cache is not None:
        cache_key = (discovery_date, fetch_end)
        bench_hist = bench_cache.get(cache_key)
        if bench_hist is None:
            bench_hist = _fetch_ticker_history(_BENCH_TICKER, discovery_date, fetch_end)
            bench_cache[cache_key] = bench_hist if bench_hist is not None else pd.DataFrame()
        if bench_hist is not None and bench_hist.empty:
            bench_hist = None
    else:
        bench_hist = _fetch_ticker_history(_BENCH_TICKER, discovery_date, fetch_end)

    entry_price = float(entry_price)
    wrote_anything = False

    for h in pending_horizons:
        if len(hist) <= h:
            continue  # not enough trading days fetched yet for this horizon

        window = hist.iloc[0:h + 1]
        price_at_h = float(window["Close"].iloc[h])
        forward_return = round((price_at_h / entry_price - 1) * 100, 3)
        mfe = round((float(window["High"].max()) / entry_price - 1) * 100, 3)
        mae = round((float(window["Low"].min()) / entry_price - 1) * 100, 3)

        benchmark_return = None
        excess_return = None
        if bench_hist is not None:
            try:
                bench_window = bench_hist.reindex(window.index, method="ffill")
                bench_entry = float(bench_window["Close"].iloc[0])
                bench_at_h = float(bench_window["Close"].iloc[h])
                if bench_entry:
                    benchmark_return = round((bench_at_h / bench_entry - 1) * 100, 3)
                    excess_return = round(forward_return - benchmark_return, 3)
            except Exception:
                pass

        outcomes[f"{h}D"] = {
            "forward_return_%": forward_return,
            "mfe_%": mfe,
            "mae_%": mae,
            "benchmark_return_%": benchmark_return,
            "excess_return_%": excess_return,
            "computed_at": datetime.now().strftime("%Y-%m-%d"),
        }
        wrote_anything = True

    return wrote_anything


def compute_all_pending_outcomes(max_observations: int = 200) -> int:
    """
    Runs compute_outcomes_for_observation() across every tracked
    observation that has at least one not-yet-frozen, now-computable
    horizon. Bounded by max_observations per run to keep this fast and
    within reasonable API call volume on a free-tier deployment — any
    remainder is picked up on the next run (nothing is lost, since
    'pending' status is just the absence of a key, checked fresh every
    time this runs).
    """
    observations = load_observations()
    if not observations:
        return 0

    bench_cache: dict = {}
    updated_count = 0
    processed = 0

    for obs in observations:
        if processed >= max_observations:
            break
        outcomes = obs.get("outcomes", {})
        if len(outcomes) >= len(HORIZONS):
            continue  # fully frozen already — nothing left to ever compute
        processed += 1
        if compute_outcomes_for_observation(obs, bench_cache):
            updated_count += 1

    if updated_count > 0:
        save_observations(observations)

    return updated_count


def auto_compute_outcomes_if_due(state_key: str = "_outcome_engine_last_run") -> int:
    """
    Once-per-day trigger, same pattern as the existing Discovery Tracker /
    Short Tracker auto-refresh — checked on page load, not a real
    background job (Streamlit Cloud has no persistent process for one).
    Uses Streamlit session_state as a same-session throttle; the
    frozen-horizon check above is what actually prevents redundant work
    across different sessions/days, so this throttle is a performance
    nicety, not the correctness mechanism.
    """
    try:
        import streamlit as st
        today = datetime.now().strftime("%Y-%m-%d")
        if st.session_state.get(state_key) == today:
            return 0
        st.session_state[state_key] = today
    except Exception:
        pass

    try:
        return compute_all_pending_outcomes()
    except Exception:
        return 0
