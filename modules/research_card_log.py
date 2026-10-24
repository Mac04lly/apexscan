"""
modules/research_card_log.py — Research card snapshot log (Change 3, Stage 2)

Turns a research card into an append-only record so the original thesis can later be compared with
what actually happened. Deliberately separate from data/alpha_observations.json: card records live
in their own file (data/research_cards.json) and are never mixed into the discovery, invalidation or
APEX10 stores or their metrics.

Design rules
  * APPEND-ONLY. `append_card` returns a NEW list; existing records are never altered or removed, and
    a card for the same ticker and observation date is refused, not overwritten.
  * ENGINE-COMPATIBLE. A record carries ticker, entry_price (last complete close) and timestamp
    (that bar's date), so the existing outcome engine resolves it UNCHANGED, freezing each horizon
    once and never rewriting it. The engine's MFE/MAE tell us whether the stop, target or trigger
    level was touched within a horizon (`touch_flags`); the order of events is not known.
  * SAFE STORE. The shared GitHub helper returns (None, None) both for "file missing" and for
    "load failed", which would let one transient read error overwrite the whole history. This module
    reads with explicit `ok` / `missing` / `error` states and REFUSES to write on `error`.
  * HONEST EVALUATION. Two hypotheses are pre-registered below, before any card exists. They are
    judged on excess return versus SPY (NOT distance-matched, which needs a same-date cohort that card
    logs do not have), with the Change 2 harness's rules: no interval below 6 independent windows,
    no verdict below 8 cards. Cards are logged by the user's own choice, so they are NOT a random
    sample of the scanner's universe; the log says so.
"""
from __future__ import annotations

import copy
import json
import logging
import uuid
from datetime import datetime
from typing import Callable, Optional

from modules.eval_harness import (ADEQUATE_CLUSTERS, MIN_CLUSTERS_FOR_CI, MIN_MATCHED_EVENTS,
                                  cluster_robust_mean, horizon_days, independence_summary,
                                  window_clusters)

log = logging.getLogger(__name__)

CARD_LOG_PATH = "data/research_cards.json"
RECORD_TYPE = "research_card"
NOT_LOGGABLE = ("insufficient_data", "unvalidated_market")

# Pre-registered BEFORE any card was logged. Do not edit after data exists; add a new id instead.
HYPOTHESES = (
    {"id": "H1", "label": "trend_pullback", "horizon": "20D", "registered": "2026-10-22",
     "text": "Cards described as a pullback within an uptrend structure have positive average "
             "20-day excess return versus SPY."},
    {"id": "H2", "label": "breakout_candidate", "horizon": "20D", "registered": "2026-10-22",
     "text": "Cards described as a breakout candidate have positive average 20-day excess "
             "return versus SPY."},
)
SELECTION_NOTE = ("Cards are logged by your own choice, so they are not a random sample of the scanner's "
                  "universe. Any result describes the cards you chose to log.")
CONTROL_NOTE = "Control: SPY. Not matched on distance to the 200-day average (card logs have no same-date cohort)."


# ── records ───────────────────────────────────────────────────────────────
def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def compact_card(card: dict) -> dict:
    """The part of a card worth keeping with the record: what it said, not every display field."""
    plan, tr, st = card["plan"], card["trend"], card["structure"]
    inv = plan.get("invalidation") or {}
    tgt = plan.get("target") or {}
    sz = plan.get("sizing") or {}
    return {
        "card_version": card["card_version"], "status": plan["status"],
        "setup": card["setup"]["label"], "setup_why": card["setup"]["why"],
        "observation_date": card["observation_date"],
        "close": card["price"]["close"], "bars": card["data"]["bars"],
        "structure": st["structure"], "break_of_structure": st["break_of_structure"],
        "swings": st["swings"][-4:],
        "trend": {"ema20_dist_pct": tr["ema20"]["distance_pct"], "ema50_dist_pct": tr["ema50"]["distance_pct"],
                  "sma200_dist_pct": tr["sma200"]["distance_pct"], "alignment": tr["alignment"],
                  "weekly_above_40w": tr["weekly"].get("above_sma40w")},
        "rvol_last": card["volume"]["rvol_last"], "rsi14": card["momentum"]["rsi14"],
        "divergence": card["momentum"]["divergence"],
        "plan": {"entry_condition": plan["entry_condition"], "entry_ref": plan.get("entry_ref"),
                 "stop": inv.get("stop"), "swing_date": inv.get("swing_date"), "risk_pct": inv.get("risk_pct"),
                 "target": tgt.get("price"), "target_reason": plan.get("target_reason"),
                 "reward_risk": plan.get("reward_risk"), "shares_illustration": sz.get("shares")},
        "earnings_flag": card["risk_context"]["earnings"]["flag"],
        "evidence_for": card["evidence_for"], "evidence_against": card["evidence_against"],
        "missing_data": card["missing_data"],
    }


def make_record(card: dict, logged_at: Optional[str] = None, new_id: Optional[str] = None) -> Optional[dict]:
    """A loggable record, or None for cards that say nothing (insufficient data / unvalidated market)."""
    if card["plan"]["status"] in NOT_LOGGABLE or not card.get("setup"):
        return None
    close = card["price"]["close"]
    plan = card["plan"]
    inv = plan.get("invalidation") or {}
    tgt = plan.get("target") or {}

    def pct(level):
        return None if level is None or not close else _r((level / close - 1) * 100)
    return {
        "observation_id": new_id or str(uuid.uuid4()),
        "observation_type": RECORD_TYPE,
        "ticker": card["ticker"], "market": "US",
        "timestamp": f"{card['observation_date']} 00:00:00",
        "entry_price": close,
        "setup_id": card["setup"]["label"], "status": plan["status"],
        "card_version": card["card_version"], "logged_at": logged_at,
        "entry_ref": plan.get("entry_ref"), "stop_price": inv.get("stop"), "target_price": tgt.get("price"),
        "risk_pct": inv.get("risk_pct"), "reward_risk": plan.get("reward_risk"),
        "trigger_pct_from_close": pct(plan.get("entry_ref")),
        "stop_pct_from_close": pct(inv.get("stop")),
        "target_pct_from_close": pct(tgt.get("price")),
        "card": compact_card(card),
        "outcomes": {},
    }


def record_key(rec: dict) -> tuple:
    return (rec.get("ticker"), str(rec.get("timestamp", ""))[:10])


def append_card(records: list, record: dict) -> tuple:
    """Returns (new_list, status). Never mutates `records` or any record in it. A card for a ticker and
    observation date already in the log is refused ('duplicate')."""
    if any(record_key(r) == record_key(record) for r in records):
        return list(records), "duplicate"
    return list(records) + [record], "added"


# ── what happened afterwards ──────────────────────────────────────────────
def touch_flags(record: dict, horizon: str) -> dict:
    """Did the intraday extremes within `horizon` reach the stop / target / trigger level? Derived from the
    engine's MAE and MFE (measured from the logged close). Touched is not the same as closed beyond, and
    the order in which levels were touched is not known."""
    out = (record.get("outcomes") or {}).get(horizon)
    res = {"stop_touched": None, "target_touched": None, "trigger_touched": None}
    if not out or out.get("mae_%") is None or out.get("mfe_%") is None:
        return res
    if record.get("stop_pct_from_close") is not None:
        res["stop_touched"] = out["mae_%"] <= record["stop_pct_from_close"]
    if record.get("target_pct_from_close") is not None:
        res["target_touched"] = out["mfe_%"] >= record["target_pct_from_close"]
    t = record.get("trigger_pct_from_close")
    if t is not None:
        res["trigger_touched"] = True if t <= 0 else out["mfe_%"] >= t
    return res


def resolve_card_outcomes(records: list, max_records: int = 50, compute: Optional[Callable] = None) -> int:
    """Resolves pending horizons for up to `max_records` records using the EXISTING outcome engine, unchanged.
    Mutates records in place (only their `outcomes`); returns how many records gained something."""
    if compute is None:
        from modules.outcome_engine import compute_outcomes_for_observation as compute
    cache, n_done, tried = {}, 0, 0
    for rec in records:
        if tried >= max_records:
            break
        tried += 1
        if compute(rec, cache):
            n_done += 1
    return n_done


# ── summaries and the pre-registered hypotheses ───────────────────────────
def evaluate_hypotheses(records: list) -> list:
    out = []
    for h in HYPOTHESES:
        key = h["horizon"]
        vals, dates = [], []
        for r in records:
            if r.get("setup_id") != h["label"]:
                continue
            o = (r.get("outcomes") or {}).get(key)
            if o and o.get("excess_return_%") is not None:
                vals.append(o["excess_return_%"])
                dates.append(str(r["timestamp"])[:10])
        row = {"id": h["id"], "label": h["label"], "horizon": key, "text": h["text"], "n": len(vals),
               "windows": 0, "mean_excess_%": None, "ci": None, "control": "SPY (not distance-matched)"}
        if vals:
            st = cluster_robust_mean(vals, window_clusters(dates, horizon_days(key)))
            row.update({"windows": st["k"], "mean_excess_%": _r(st["mean"], 2),
                        "ci": None if st["ci"] is None else (_r(st["ci"][0], 2), _r(st["ci"][1], 2))})
        if row["n"] < MIN_MATCHED_EVENTS:
            row["verdict"] = f"Too few resolved cards ({row['n']}) for any verdict"
        elif row["windows"] < MIN_CLUSTERS_FOR_CI:
            row["verdict"] = f"Insufficient independent windows ({row['windows']}): descriptive only"
        else:
            tag = "adequate" if row["windows"] >= ADEQUATE_CLUSTERS else "exploratory"
            lo, hi = row["ci"]
            row["verdict"] = (f"Outperformed SPY ({tag}, {row['windows']} windows)" if lo > 0 else
                              f"Underperformed SPY ({tag}, {row['windows']} windows)" if hi < 0 else
                              f"No detectable difference from SPY ({tag}, {row['windows']} windows)")
        out.append(row)
    return out


def summarize_log(records: list) -> dict:
    by_label, by_status = {}, {}
    for r in records:
        by_label[r.get("setup_id")] = by_label.get(r.get("setup_id"), 0) + 1
        by_status[r.get("status")] = by_status.get(r.get("status"), 0) + 1
    resolved = {h: sum(1 for r in records if (r.get("outcomes") or {}).get(h, {}).get("forward_return_%") is not None)
                for h in ("5D", "10D", "20D")}
    return {"n_cards": len(records), "by_setup": by_label, "by_status": by_status, "resolved": resolved,
            "independence_20D": independence_summary(records, 20, "20D"),
            "hypotheses": evaluate_hypotheses(records), "selection_note": SELECTION_NOTE,
            "control_note": CONTROL_NOTE}


# ── store: explicit ok / missing / error ──────────────────────────────────
def _creds() -> tuple:
    try:
        from modules.alpha_validation import _get_gh_creds
        return _get_gh_creds()
    except Exception:
        return "", ""


def _fetch_remote(token: str, repo: str, path: str) -> tuple:
    """('ok', data) | ('missing', None) | ('error', None). Unlike the shared helper, a failed read is NOT
    reported as a missing file."""
    try:
        import requests
        from modules.gh_storage import GITHUB_API, _headers
        url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
        meta = requests.get(url, headers=_headers(token), params={"ref": "main"}, timeout=10)
        if meta.status_code == 404:
            return "missing", None
        meta.raise_for_status()
        raw = requests.get(url, headers=_headers(token, accept="application/vnd.github.raw"),
                           params={"ref": "main"}, timeout=30)
        if raw.status_code == 404:
            return "missing", None
        raw.raise_for_status()
        return "ok", json.loads(raw.text)
    except Exception as e:
        log.warning(f"Card log read failed: {e}")
        return "error", None


def _push(token: str, repo: str, path: str, data, message: str) -> bool:
    from modules.gh_storage import save_json_to_github
    return bool(save_json_to_github(token, repo, path, data, message=message))


def load_log_state() -> tuple:
    """('ok', records) | ('missing', []) | ('error', []) | ('no_credentials', [])."""
    token, repo = _creds()
    if not token or not repo:
        return "no_credentials", []
    status, data = _fetch_remote(token, repo, CARD_LOG_PATH)
    if status == "ok" and not isinstance(data, list):
        return "error", []                       # a malformed file is never silently replaced
    if status == "missing":
        return "missing", []
    return status, (data if status == "ok" else [])


def append_and_save(card: dict, logged_at: Optional[str] = None) -> dict:
    """Load -> append (dedup) -> save, in one call. Writes ONLY on a clean read ('ok' or 'missing') and
    only if the new list is exactly one record longer. Returns {'status', 'n'}; statuses: added, duplicate,
    not_loggable, no_credentials, read_failed, save_failed."""
    rec = make_record(card, logged_at=logged_at)
    if rec is None:
        return {"status": "not_loggable", "n": None}
    token, repo = _creds()
    if not token or not repo:
        return {"status": "no_credentials", "n": None}
    state, current = _fetch_remote(token, repo, CARD_LOG_PATH)
    if state == "ok" and not isinstance(current, list):
        state = "error"
    if state == "error":
        return {"status": "read_failed", "n": None}
    current = current if state == "ok" else []
    new, status = append_card(current, rec)
    if status == "duplicate":
        return {"status": "duplicate", "n": len(current)}
    assert len(new) == len(current) + 1
    ok = _push(token, repo, CARD_LOG_PATH, new, f"Log research card {rec['ticker']} {rec['timestamp'][:10]}")
    return {"status": "added" if ok else "save_failed", "n": len(new) if ok else len(current)}


def resolve_and_save(max_records: int = 50, compute: Optional[Callable] = None) -> dict:
    """Load -> resolve pending outcomes via the engine -> save. Writes only on a clean read, only if something
    changed, and never changes the number of records."""
    token, repo = _creds()
    if not token or not repo:
        return {"status": "no_credentials", "updated": 0}
    state, current = _fetch_remote(token, repo, CARD_LOG_PATH)
    if state == "ok" and not isinstance(current, list):
        state = "error"
    if state == "error":
        return {"status": "read_failed", "updated": 0}
    if state == "missing" or not current:
        return {"status": "empty", "updated": 0}
    work = copy.deepcopy(current)
    n = resolve_card_outcomes(work, max_records=max_records, compute=compute)
    if n == 0:
        return {"status": "nothing_due", "updated": 0}
    assert len(work) == len(current)
    ok = _push(token, repo, CARD_LOG_PATH, work, f"Update research card outcomes ({n} record(s))")
    return {"status": "updated" if ok else "save_failed", "updated": n if ok else 0}
