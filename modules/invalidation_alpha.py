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

CHANGE 1 (detector INVAL-2.0) — pipeline repair, detection rules unchanged:
  * SCANNED is not DETECTED. Every source the scan examines gets a status
    record (data/invalidation_scan_status.json, keyed by source
    observation_id) even when no event is found. Detected events remain
    ordinary observations in alpha_observations.json. Keeping status out
    of the discovery records means historical discoveries are never edited.
  * Rotation, not a fixed prefix: never-scanned sources go first, then the
    least recently scanned, so a source at position 450 of 652 is reached
    even when the first 200 never produce an event (the old scan re-fetched
    the same first-200 forever).
  * Provenance: new events carry detector_version and event_basis.
    "already_below_at_discovery" breaches are MECHANICAL (the stop was at or
    above the entry price when the discovery was logged) — flagged, never
    silently mixed with genuine post-discovery crossings.
  * Cleanup no longer deletes by event type. Versioned events are never
    touched; unversioned legacy near-miss events are audited against the
    causal detector, only confirmed mismatches are removed, and they go to
    a quarantine list (full record, outcomes intact) rather than vanishing.
The 5% near-miss threshold and first-event-wins semantics are unchanged.
"""
from __future__ import annotations
import copy
import logging
import uuid
from datetime import datetime, date
from typing import Callable, Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

log = logging.getLogger("apexscan.modules.invalidation_alpha")

_EVENT_TYPES = ("invalidation_breach", "invalidation_near_miss")

# Version of the DETECTION + PROVENANCE rules. Bump only if detection
# semantics change; stamped on every event this module creates.
DETECTOR_VERSION = "INVAL-2.0"
# Stamped on legacy (unversioned) near-miss events that an audit re-derived
# from price history and found identical to what the causal detector gives.
LEGACY_VERIFIED_VERSION = "INVAL-2.0-legacy-verified"

SCAN_STATUS_PATH = "data/invalidation_scan_status.json"
QUARANTINE_PATH = "data/invalidation_quarantine.json"

# event_basis values
BASIS_CROSSED = "crossed_after_discovery"            # breach: genuine post-discovery close below stop
BASIS_ALREADY_BELOW = "already_below_at_discovery"   # breach: MECHANICAL — stop >= entry at discovery
BASIS_APPROACHED = "approached_after_discovery"      # near-miss: price came down into the band later
BASIS_WITHIN_BAND = "within_band_at_discovery"       # near-miss: entry was already inside the band
BASIS_UNKNOWN = "unknown"


def _detect_first_event(close: pd.Series, stop_price: float, near_miss_pct: float):
    """Causal first-event detector, shared by the scan and the audit so both
    apply exactly the same rule. Walks CHRONOLOGICALLY and returns the first
    of (breach | near-miss) using only information available on that day:
    (event_type, event_timestamp, event_price), or None if the level has not
    been tested. Logic is unchanged from the version fixed for the original
    look-ahead bug."""
    for dt, price in close.items():
        if price < stop_price:
            return "invalidation_breach", dt, float(price)
        dist_pct = (float(price) / float(stop_price) - 1) * 100
        if dist_pct <= near_miss_pct:
            return "invalidation_near_miss", dt, float(price)
    return None


def event_basis_for(event_type: str, entry_price, stop_price, near_miss_pct: float = 5.0) -> str:
    """Classifies an event from DISCOVERY-TIME data only (entry vs stop).
    Mechanical breach = the stop was already at/above the entry price when
    the discovery was logged (stock already below its 200MA)."""
    try:
        entry_price, stop_price = float(entry_price), float(stop_price)
    except (TypeError, ValueError):
        return BASIS_UNKNOWN
    if event_type == "invalidation_breach":
        return BASIS_ALREADY_BELOW if stop_price >= entry_price else BASIS_CROSSED
    if event_type == "invalidation_near_miss":
        within = (entry_price / stop_price - 1) * 100 <= near_miss_pct
        return BASIS_WITHIN_BAND if within else BASIS_APPROACHED
    return BASIS_UNKNOWN


def derive_event_basis(event: dict, source: Optional[dict], near_miss_pct: float = 5.0) -> str:
    """Read-time basis for any event. Uses the stored value when present;
    for legacy events derives it from the source discovery WITHOUT writing
    anything back to the stored record."""
    if event.get("event_basis"):
        return event["event_basis"]
    if not source:
        return BASIS_UNKNOWN
    return event_basis_for(event.get("observation_type"), source.get("entry_price"),
                           source.get("stop_price"), near_miss_pct)


def audit_unversioned_events(observations: list, near_miss_pct: float = 5.0,
                             fetch: Optional[Callable] = None) -> list:
    """READ-ONLY audit of legacy near-miss events (those with no
    detector_version). Re-runs the causal detector on each source's real
    price history and compares with the stored event:
      verified     — same type, same date, same price as the causal detector
      mismatch     — the causal detector gives something else (the signature
                     of the old argmin look-ahead bug)
      unverifiable — source missing, no stop, or no price data (kept as-is)
    Mutates nothing. Breach events are not audited: breach detection was
    always causal."""
    fetch_fn = fetch or _fetch_price_path
    by_id = {o.get("observation_id"): o for o in observations if o.get("observation_id")}
    results = []
    for ev in observations:
        if ev.get("observation_type") != "invalidation_near_miss" or ev.get("detector_version"):
            continue
        rec = {"observation_id": ev.get("observation_id"), "ticker": ev.get("ticker")}
        src = by_id.get(ev.get("source_observation_id"))
        if not src or not src.get("stop_price") or not src.get("timestamp"):
            results.append({**rec, "status": "unverifiable", "reason": "source discovery or stop_price missing"})
            continue
        try:
            disc_date = pd.to_datetime(str(src["timestamp"]).split(" ")[0]).date()
            close = fetch_fn(src["ticker"], disc_date)
        except Exception as e:  # network/provider failure — never treat as a mismatch
            results.append({**rec, "status": "unverifiable", "reason": f"price fetch failed: {e}"})
            continue
        if close is None or len(close) < 2:
            results.append({**rec, "status": "unverifiable", "reason": "no price history available"})
            continue
        found = _detect_first_event(close, float(src["stop_price"]), near_miss_pct)
        if found is None:
            results.append({**rec, "status": "mismatch",
                            "reason": "causal detector finds no event on this path"})
            continue
        etype, edate, eprice = found
        same = (etype == "invalidation_near_miss"
                and edate.strftime("%Y-%m-%d") == str(ev.get("timestamp", ""))[:10]
                and round(eprice, 2) == ev.get("entry_price"))
        if same:
            results.append({**rec, "status": "verified", "reason": "matches causal detector"})
        else:
            results.append({**rec, "status": "mismatch",
                            "reason": f"causal detector gives {etype} on {edate.strftime('%Y-%m-%d')}"})
    return results


def cleanup_buggy_near_miss_events(observations: list, audit_results: Optional[list],
                                   quarantine: Optional[list] = None,
                                   scan_status: Optional[dict] = None,
                                   now: Optional[datetime] = None) -> dict:
    """
    Safe, auditable replacement for the old delete-every-near-miss cleanup.

    Requires the output of audit_unversioned_events(); without it nothing
    can be justified, so it raises rather than guessing. Then:
      * events that carry a detector_version are NEVER touched;
      * legacy near-miss events audited as `verified` are kept and only
        stamped with detector_version=LEGACY_VERIFIED_VERSION (no other
        field, and never `outcomes`, is modified);
      * legacy events audited as `mismatch` are removed from `observations`
        and appended — complete, outcomes included — to `quarantine`
        together with the reason and timestamp, so they stay recoverable;
      * `unverifiable` / un-audited events are kept untouched;
      * a quarantined event's source gets its scan-status entry dropped so
        the corrected scan re-examines it.
    Mutates `observations` (slice assignment), `quarantine` and
    `scan_status` in place; the caller persists all three. Returns counts.
    """
    if audit_results is None:
        raise ValueError("cleanup_buggy_near_miss_events requires audit_results from "
                         "audit_unversioned_events(); it will not delete by event type.")
    if quarantine is None:
        quarantine = []
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    verdict = {r["observation_id"]: r for r in audit_results}

    kept, removed, stamped, unverifiable, versioned = [], 0, 0, 0, 0
    for o in observations:
        if o.get("observation_type") != "invalidation_near_miss":
            kept.append(o)
            continue
        if o.get("detector_version"):
            versioned += 1
            kept.append(o)
            continue
        r = verdict.get(o.get("observation_id"))
        if r and r["status"] == "mismatch":
            quarantine.append({"quarantined_at": stamp, "reason": r["reason"],
                               "audit_status": "mismatch", "record": copy.deepcopy(o)})
            if scan_status is not None:
                scan_status.pop(o.get("source_observation_id"), None)
            removed += 1
            continue
        if r and r["status"] == "verified":
            o["detector_version"] = LEGACY_VERIFIED_VERSION
            stamped += 1
        else:
            unverifiable += 1
        kept.append(o)
    observations[:] = kept
    return {"removed": removed, "verified_stamped": stamped,
            "kept_unverified": unverifiable, "versioned_untouched": versioned}


def _gh_creds():
    from modules.alpha_validation import _get_gh_creds
    return _get_gh_creds()


def _load_store(path: str, default):
    try:
        from modules.gh_storage import load_json_from_github
        token, repo = _gh_creds()
        if token and repo:
            data, _ = load_json_from_github(token, repo, path)
            if isinstance(data, type(default)):
                return data
    except Exception as e:
        log.warning(f"Could not load {path}: {e}")
    return default


def _save_store(path: str, data, message: str) -> bool:
    try:
        from modules.gh_storage import save_json_to_github
        token, repo = _gh_creds()
        if token and repo:
            return bool(save_json_to_github(token, repo, path, data, message=message))
    except Exception as e:
        log.warning(f"Could not save {path}: {e}")
    return False


def load_scan_status() -> dict:
    """source observation_id -> scan status record. {} if unavailable."""
    return _load_store(SCAN_STATUS_PATH, {})


def save_scan_status(status: dict) -> bool:
    return _save_store(SCAN_STATUS_PATH, status,
                       f"Update invalidation_scan_status.json ({len(status)} sources)")


def load_quarantine() -> list:
    return _load_store(QUARANTINE_PATH, [])


def save_quarantine(quarantine: list) -> bool:
    return _save_store(QUARANTINE_PATH, quarantine,
                       f"Update invalidation_quarantine.json ({len(quarantine)} records)")


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


def _iso(d) -> str:
    return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]


def log_invalidation_events(observations: list, near_miss_pct: float = 5.0,
                             max_tickers: int = 200,
                             scan_status: Optional[dict] = None,
                             fetch: Optional[Callable] = None,
                             today: Optional[date] = None) -> dict:
    """
    Walks real daily price history since discovery for discovery
    observations that have a stop_price and no event yet. Logs at most one
    event per source: a breach if price ever closed below stop_price, else a
    near-miss if it came within near_miss_pct without breaching, else
    nothing (level not tested yet — recorded as a scan status, not an event).

    SCANNED vs DETECTED: `scanned` counts sources examined this run;
    `new_events` counts events detected. The two differ whenever a source
    was examined and had no event (or no usable price data).

    Rotation: eligible sources are those with no event and not already
    scanned today. Never-scanned sources come first, then the least
    recently scanned (discovery order breaks ties), capped at max_tickers —
    so repeated runs sweep the whole population instead of re-fetching the
    same first max_tickers forever.

    `scan_status` (source observation_id -> record) is updated IN PLACE for
    every source examined; the caller persists it. New events are appended
    to `observations` in place; caller persists via save_observations().
    `fetch` / `today` are injectable for tests; defaults are live yfinance
    and the current date.
    """
    if fetch is None and yf is None:
        return {"scanned": 0, "new_events": 0, "breaches": 0, "near_misses": 0,
                "mechanical_breaches": 0, "no_event": 0, "no_data": 0, "skipped": 0,
                "remaining_unscanned": 0, "error": "yfinance not available this session"}
    fetch_fn = fetch or _fetch_price_path
    if scan_status is None:
        scan_status = {}
    today_iso = _iso(today or datetime.now().date())

    discovery_obs = [o for o in observations if o.get("observation_type", "discovery") == "discovery"]
    event_sources = {
        o.get("source_observation_id") for o in observations
        if o.get("observation_type") in _EVENT_TYPES
    }
    pool = []
    for idx, o in enumerate(discovery_obs):
        oid = o.get("observation_id")
        if o.get("stop_price") is None or oid in event_sources:
            continue
        st = scan_status.get(oid)
        if st is not None and st.get("scanned_on", "") >= today_iso:
            continue  # already examined today — daily bars can't have changed
        rank = (0, "", idx) if st is None else (1, st.get("scanned_on", ""), idx)
        pool.append((rank, o))
    pool.sort(key=lambda t: t[0])
    candidates = [o for _, o in pool[:max_tickers]]
    remaining = max(0, len(pool) - len(candidates))

    def _record(oid, ticker, result, bars=0, last_bar=None):
        scan_status[oid] = {"ticker": ticker, "scanned_on": today_iso, "result": result,
                            "bars": bars, "last_bar": last_bar,
                            "detector_version": DETECTOR_VERSION}

    new_events = []
    breaches = near_misses = mechanical = no_event = no_data = skipped = 0

    for obs in candidates:
        ticker, stop_price, ts = obs.get("ticker"), obs.get("stop_price"), obs.get("timestamp")
        oid = obs.get("observation_id")
        if not ticker or not ts or not stop_price:
            _record(oid, ticker, "invalid_input")
            skipped += 1
            continue
        try:
            disc_date = pd.to_datetime(str(ts).split(" ")[0]).date()
            close = fetch_fn(ticker, disc_date)
            if close is None or len(close) < 2:
                _record(oid, ticker, "no_data")
                no_data += 1
                skipped += 1
                continue

            found = _detect_first_event(close, float(stop_price), near_miss_pct)
            if found is None:
                _record(oid, ticker, "no_event", bars=len(close), last_bar=_iso(close.index[-1]))
                no_event += 1
                continue  # hasn't tested the level yet either way — scanned, not detected

            event_type, event_date, event_price = found
            basis = event_basis_for(event_type, obs.get("entry_price"), stop_price, near_miss_pct)
            is_mech = basis == BASIS_ALREADY_BELOW
            new_events.append({
                "observation_id": str(uuid.uuid4()),
                "ticker": ticker,
                "market": obs.get("market", "US"),
                "timestamp": event_date.strftime("%Y-%m-%d") + " 00:00:00",
                "entry_price": round(event_price, 2),
                "observation_type": event_type,
                "source_observation_id": oid,
                "stage": obs.get("stage"),
                "setup_id": obs.get("setup_id"),
                "market_regime": obs.get("market_regime"),
                "market_features": obs.get("market_features"),
                "distance_to_stop_pct": round((event_price / float(stop_price) - 1) * 100, 2),
                "days_since_discovery": (event_date.date() - disc_date).days,
                "detector_version": DETECTOR_VERSION,
                "event_basis": basis,
                "is_mechanical": is_mech,
                "logged_on": today_iso,
            })
            _record(oid, ticker, "breach" if event_type == "invalidation_breach" else "near_miss",
                    bars=len(close), last_bar=_iso(close.index[-1]))
            if event_type == "invalidation_breach":
                breaches += 1
                mechanical += int(is_mech)
            else:
                near_misses += 1

        except Exception as e:
            log.warning(f"Invalidation event scan skipped for {ticker}: {e}")
            _record(oid, ticker, "error")
            skipped += 1

    if new_events:
        observations.extend(new_events)

    return {"scanned": len(candidates), "new_events": len(new_events),
            "breaches": breaches, "near_misses": near_misses,
            "mechanical_breaches": mechanical, "no_event": no_event,
            "no_data": no_data, "skipped": skipped, "remaining_unscanned": remaining}


def compute_scan_coverage(observations: list, scan_status: Optional[dict] = None,
                          near_miss_pct: float = 5.0, later_than: Optional[str] = None) -> dict:
    """
    Coverage, with SCANNED and DETECTED kept separate.
      scanned_with_status — has a status record (examined by the new scan)
      scanned_any         — has a status record OR a logged event (a legacy
                            event proves the source was examined at some point)
      detected            — has a logged event (breach or near-miss)
    `later_than` (YYYY-MM-DD) isolates discoveries logged after a cohort date.
    """
    scan_status = scan_status or {}
    discovery_obs = [o for o in observations if o.get("observation_type", "discovery") == "discovery"]
    by_id = {o.get("observation_id"): o for o in observations if o.get("observation_id")}
    events = [o for o in observations if o.get("observation_type") in _EVENT_TYPES]
    event_by_source = {e.get("source_observation_id"): e for e in events}

    with_stop = [(i, o) for i, o in enumerate(discovery_obs) if o.get("stop_price") is not None]

    def _stats(subset):
        ids = [o.get("observation_id") for _, o in subset]
        return {
            "eligible": len(ids),
            "scanned_with_status": sum(1 for x in ids if x in scan_status),
            "scanned_any": sum(1 for x in ids if x in scan_status or x in event_by_source),
            "detected": sum(1 for x in ids if x in event_by_source),
        }

    def _mech_eligible(o):
        try:
            return float(o["entry_price"]) <= float(o["stop_price"]) * (1 + near_miss_pct / 100)
        except Exception:
            return False

    overall = _stats(with_stop)
    mech = _stats([(i, o) for i, o in with_stop if _mech_eligible(o)])
    later = _stats([(i, o) for i, o in with_stop
                    if later_than and str(o.get("timestamp", ""))[:10] > later_than])

    reached = [i for i, o in with_stop
               if o.get("observation_id") in scan_status or o.get("observation_id") in event_by_source]
    bases = {}
    for e in events:
        if e.get("observation_type") == "invalidation_breach":
            b = derive_event_basis(e, by_id.get(e.get("source_observation_id")), near_miss_pct)
            bases[b] = bases.get(b, 0) + 1
    return {
        "overall": overall,
        "mechanically_eligible": mech,
        "later_discoveries": later,
        "max_source_position_reached": max(reached) if reached else None,
        "discovery_count": len(discovery_obs),
        "events_breach": sum(1 for e in events if e["observation_type"] == "invalidation_breach"),
        "events_near_miss": sum(1 for e in events if e["observation_type"] == "invalidation_near_miss"),
        "breach_by_basis": bases,
        "status_by_result": {r: sum(1 for s in scan_status.values() if s.get("result") == r)
                             for r in sorted({s.get("result") for s in scan_status.values()})},
    }


def compute_invalidation_alpha_overview(observations: list, horizon: str = "20D",
                                        scan_status: Optional[dict] = None) -> dict:
    """
    Pure computation for ui/alpha_lab.py's Invalidation Alpha tab —
    keeps that file's stated design of being pure display logic, same
    as every other Phase 3+ metric it shows. Returns breach/near-miss
    Alpha Metrics at `horizon` (via the existing, unmodified
    compute_alpha_metrics()), plus coverage counts so the UI can show
    how much of the discovery population has a stop_price at all and
    how much of that hasn't been through an event scan yet.

    Existing keys are unchanged. Added: `breach_genuine` (breaches whose
    basis is crossed_after_discovery) and `breach_mechanical_n`, so the
    mechanical day-0 population is visible instead of blended in.
    """
    from modules.alpha_metrics import compute_alpha_metrics

    scan_status = scan_status or {}
    by_id = {o.get("observation_id"): o for o in observations if o.get("observation_id")}
    discovery_obs = [o for o in observations if o.get("observation_type", "discovery") == "discovery"]
    with_stop = [o for o in discovery_obs if o.get("stop_price") is not None]
    breach_obs = [o for o in observations if o.get("observation_type") == "invalidation_breach"]
    near_miss_obs = [o for o in observations if o.get("observation_type") == "invalidation_near_miss"]
    logged_source_ids = {o.get("source_observation_id") for o in breach_obs + near_miss_obs}
    examined_ids = logged_source_ids | set(scan_status)
    not_yet_scanned = [o for o in with_stop if o.get("observation_id") not in examined_ids]

    genuine, mechanical_n = [], 0
    for e in breach_obs:
        basis = derive_event_basis(e, by_id.get(e.get("source_observation_id")))
        if basis == BASIS_ALREADY_BELOW:
            mechanical_n += 1
        else:
            genuine.append(e)

    return {
        "total_discovery_observations": len(discovery_obs),
        "with_stop_price": len(with_stop),
        "not_yet_scanned": len(not_yet_scanned),
        "breach": compute_alpha_metrics(breach_obs, horizon),
        "near_miss": compute_alpha_metrics(near_miss_obs, horizon),
        "breach_genuine": compute_alpha_metrics(genuine, horizon),
        "breach_mechanical_n": mechanical_n,
    }
