"""
modules/invalidation_alpha.py — Invalidation Alpha

Answers a different question than every other Alpha Lab tab: not "does
a feature known AT DISCOVERY predict returns N days later," but "does
what happens to the ORIGINAL THESIS after discovery — testing or
breaking its invalidation level — itself predict direction?"

Two event types, discovered by walking each ticker's real daily price
history since discovery CHRONOLOGICALLY — whichever happens first wins,
using only information available on that day (never information from
later days, which is what caused an earlier version of this file to
mislabel results — see note in log_invalidation_events):
  - invalidation_breach:    first close below stop_price. Tests whether
                             that should mean getting out immediately
                             (negative forward returns) or is usually a
                             shakeout right before a bounce (positive).
  - invalidation_near_miss: first close that comes within near_miss_pct
                             of stop_price WITHOUT having breached it
                             first. Tests whether a genuinely first-time
                             approach to the level is itself a usable
                             entry signal.

This deliberately does NOT invent a new statistics engine. Per
modules/outcome_engine.py's own docstring, that engine is type-agnostic
— it only needs entry_price + timestamp on an observation, with no
branch anywhere on observation_type, and this was confirmed (not
assumed) before apex10_tracker.py was built the same way. So an event
logged here is just another observation appended to the same
alpha_observations.json store, anchored at the event date instead of
the discovery date. compute_all_pending_outcomes() freezes its forward
returns automatically — nothing in outcome_engine.py or alpha_metrics.py
needs to change for this to work.

Safe to re-run: an already-logged source discovery is skipped every
time after (tracked via source_observation_id), so running the scan
again only picks up discoveries that didn't have an event yet, or that
had none last time but do now.
"""
from __future__ import annotations
import logging
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

log = logging.getLogger("apexscan.modules.invalidation_alpha")

_EVENT_TYPES = ("invalidation_breach", "invalidation_near_miss")


def cleanup_buggy_near_miss_events(observations: list) -> int:
    """
    One-time repair for invalidation_near_miss observations logged by
    the pre-fix version of log_invalidation_events(), which picked the
    single lowest price across a ticker's ENTIRE history-to-date (via
    argmin) as "the event," rather than the first genuine approach to
    the line. That's a look-ahead bug — the lowest point of any series
    is almost always followed by higher prices, which is why it
    produced a meaningless ~100% win rate instead of a real signal (see
    the module docstring above for the full explanation).

    invalidation_breach observations were NOT affected — breach
    detection always used "first close below the line," which was
    already causal — so this only ever removes invalidation_near_miss
    entries, never breach entries.

    Removing them also frees their source discoveries to be rescanned:
    log_invalidation_events() skips any source whose observation_id
    already appears in an existing event's source_observation_id, and
    under the old code each source could only ever produce one event
    (breach OR near-miss, never both), so deleting the bad near-miss
    entries makes those sources eligible again next time the scan runs.

    Mutates `observations` in place via slice assignment, so the
    caller's existing list reference stays valid — caller persists via
    save_observations(). Returns the count removed. Safe to run more
    than once; a second run simply finds nothing left to remove.
    """
    before = len(observations)
    observations[:] = [o for o in observations if o.get("observation_type") != "invalidation_near_miss"]
    return before - len(observations)


def backfill_stop_prices(observations: list) -> int:
    """
    Populates stop_price on discovery observations logged before that
    field existed (build_observation() used to always set it to None).
    Needs NO network call: entry_price and vs_200ma_% were already
    frozen on the observation at discovery time, so this only
    recomputes a value from data that was always there, using the
    identical formula build_observation() now applies going forward.
    Mutates `observations` in place; caller persists via
    save_observations(). Returns the count updated.
    """
    from modules.alpha_validation import _compute_stop_price

    count = 0
    for obs in observations:
        if obs.get("stop_price") is not None:
            continue
        if obs.get("observation_type", "discovery") != "discovery":
            continue  # only discovery observations carry a thesis to invalidate
        row = {
            "price": obs.get("entry_price"),
            "vs_200ma_%": (obs.get("technical_features") or {}).get("vs_200ma_%"),
        }
        sp = _compute_stop_price(row)
        if sp is not None:
            obs["stop_price"] = sp
            count += 1
    return count


def _fetch_price_path(ticker: str, start_date) -> Optional[pd.Series]:
    if yf is None:
        return None
    try:
        hist = yf.Ticker(ticker).history(start=start_date.strftime("%Y-%m-%d"))
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        if hasattr(hist.index, "tz") and hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)
        hist.index = pd.to_datetime(hist.index).normalize()
        close = hist["Close"].dropna()
        return close if not close.empty else None
    except Exception:
        return None


def log_invalidation_events(observations: list, near_miss_pct: float = 5.0,
                             max_tickers: int = 200) -> dict:
    """
    Walks real daily price history since discovery for every discovery
    observation that has a stop_price and hasn't been scanned yet
    (tracked via source_observation_id — safe to re-run). Logs at most
    one new event observation per source discovery: a breach if price
    ever closed below stop_price, else a near-miss if it came within
    near_miss_pct without breaching, else nothing (still too early or
    never tested the level — not an error, just not yet decidable).

    New event observations are appended directly to `observations`
    (mutated in place, same pattern compute_outcomes_for_observation
    uses) — caller persists via save_observations(). Each one carries
    only entry_price + timestamp as REQUIRED by the outcome engine's
    type-agnostic contract, plus stage/setup_id/market_features carried
    over from the source so Setup/Feature/Conditional Alpha can group
    on this new population later without any changes to those functions.

    Bounded by max_tickers per run, mirroring
    compute_all_pending_outcomes()'s own per-run cap for the same
    free-tier API budget reason — any remainder is picked up next run.
    """
    if yf is None:
        return {"new_events": 0, "breaches": 0, "near_misses": 0, "skipped": 0,
                "error": "yfinance not available this session"}

    discovery_obs = [o for o in observations if o.get("observation_type", "discovery") == "discovery"]
    already_scanned = {
        o.get("source_observation_id") for o in observations
        if o.get("observation_type") in _EVENT_TYPES
    }
    candidates = [
        o for o in discovery_obs
        if o.get("stop_price") is not None and o.get("observation_id") not in already_scanned
    ][:max_tickers]

    new_events, breaches, near_misses, skipped = [], 0, 0, 0

    for obs in candidates:
        ticker = obs.get("ticker")
        stop_price = obs.get("stop_price")
        ts = obs.get("timestamp")
        if not ticker or not ts or not stop_price:
            skipped += 1
            continue
        try:
            disc_date = pd.to_datetime(str(ts).split(" ")[0]).date()
            close = _fetch_price_path(ticker, disc_date)
            if close is None or len(close) < 2:
                skipped += 1
                continue

            # Walk forward chronologically and take whichever happens
            # FIRST, using only information available on that day — never
            # the whole series in hindsight. The earlier version picked
            # the single lowest price across the entire history-to-date
            # for near misses, which is a look-ahead bug: the lowest
            # point of any series is mechanically followed by higher
            # prices most of the time, which is why that produced a
            # meaningless 100% win rate rather than a real signal.
            event_type, event_date, event_price = None, None, None
            for dt, price in close.items():
                if price < stop_price:
                    event_type, event_date, event_price = "invalidation_breach", dt, float(price)
                    break
                dist_pct = (float(price) / float(stop_price) - 1) * 100
                if dist_pct <= near_miss_pct:
                    event_type, event_date, event_price = "invalidation_near_miss", dt, float(price)
                    break

            if event_type is None:
                continue  # hasn't tested the level yet either way — nothing to log

            new_events.append({
                "observation_id": str(uuid.uuid4()),
                "ticker": ticker,
                "market": obs.get("market", "US"),
                "timestamp": event_date.strftime("%Y-%m-%d") + " 00:00:00",
                "entry_price": round(event_price, 2),
                "observation_type": event_type,
                "source_observation_id": obs.get("observation_id"),
                "stage": obs.get("stage"),
                "setup_id": obs.get("setup_id"),
                "market_regime": obs.get("market_regime"),
                "market_features": obs.get("market_features"),
                "distance_to_stop_pct": round((event_price / float(stop_price) - 1) * 100, 2),
                "days_since_discovery": (event_date.date() - disc_date).days,
            })
            if event_type == "invalidation_breach":
                breaches += 1
            else:
                near_misses += 1

        except Exception as e:
            log.warning(f"Invalidation event scan skipped for {ticker}: {e}")
            skipped += 1

    if new_events:
        observations.extend(new_events)

    return {"new_events": len(new_events), "breaches": breaches,
            "near_misses": near_misses, "skipped": skipped}


def compute_invalidation_alpha_overview(observations: list, horizon: str = "20D") -> dict:
    """
    Pure computation for ui/alpha_lab.py's Invalidation Alpha tab —
    keeps that file's stated design of being pure display logic, same
    as every other Phase 3+ metric it shows. Returns breach/near-miss
    Alpha Metrics at `horizon` (via the existing, unmodified
    compute_alpha_metrics()), plus coverage counts so the UI can show
    how much of the discovery population has a stop_price at all and
    how much of that hasn't been through an event scan yet.
    """
    from modules.alpha_metrics import compute_alpha_metrics

    discovery_obs = [o for o in observations if o.get("observation_type", "discovery") == "discovery"]
    with_stop = [o for o in discovery_obs if o.get("stop_price") is not None]
    breach_obs = [o for o in observations if o.get("observation_type") == "invalidation_breach"]
    near_miss_obs = [o for o in observations if o.get("observation_type") == "invalidation_near_miss"]
    logged_source_ids = {o.get("source_observation_id") for o in breach_obs + near_miss_obs}
    not_yet_scanned = [o for o in with_stop if o.get("observation_id") not in logged_source_ids]

    return {
        "total_discovery_observations": len(discovery_obs),
        "with_stop_price": len(with_stop),
        "not_yet_scanned": len(not_yet_scanned),
        "breach": compute_alpha_metrics(breach_obs, horizon),
        "near_miss": compute_alpha_metrics(near_miss_obs, horizon),
    }
