"""
modules/forward_hypotheses.py — Pre-registered FORWARD hypotheses on discoveries (H3)

Pure functions; reads the existing observation list and writes nothing. It adds no logging of its own: the
sample is simply the discoveries ApexScan's normal scans record from now on, which are the scanner's own
output, not hand-picked cards.

Why pre-register at all. H3 was suggested by an in-sample look at one scan (2026-08-06: names with a Stage 1
base and a score of at least 70 averaged +4.4%, 55% up, n=83, against -3.1% for the whole scan). That is one
market window and one of many slices that could have been cut, so by itself it proves nothing. Fixing the
definition, the horizon, the benchmark and the pass rule BEFORE any new data exists, and excluding everything
logged on or before the registration date, turns the hint into something that can be tested honestly.

Rules (identical in spirit to the Change 2 harness):
  * forward-only: only observations stamped AFTER `registered` count;
  * 20-day excess return versus SPY, taken from the frozen outcomes the existing engine wrote;
  * independence unit = one 20-trading-day window; no interval below 6 windows, no verdict below 8
    observations, "adequate" from 12 windows;
  * a descriptive contrast against all OTHER discoveries in the same period is shown WITHOUT any interval,
    because beating SPY is not the same as beating the scanner's average;
  * three hypotheses (H1, H2 in the card log, and this one) are now being watched, so a lone "pass" among
    them deserves extra scepticism.
Do not edit a registered entry after data exists; add a new id instead.
"""
from __future__ import annotations

from typing import Optional

from modules.eval_harness import (ADEQUATE_CLUSTERS, MIN_CLUSTERS_FOR_CI, MIN_MATCHED_EVENTS,
                                  cluster_robust_mean, horizon_days, window_clusters)

REGISTERED = "2026-10-26"

HYPOTHESES = (
    {"id": "H3", "registered": REGISTERED, "horizon": "20D", "stage_prefix": "1", "min_score": 70.0,
     "text": (f"Discoveries logged after {REGISTERED} with a Stage 1 base and an Apex score of at least 70 have "
              "positive average 20-day excess return versus SPY."),
     "motivation": ("Suggested by an in-sample look at the 2026-08-06 scan (83 names, +4.4% mean, 55% up). That data, "
                    f"and everything logged on or before {REGISTERED}, is excluded from the test.")},
)

NOTE = ("Excess return is versus SPY. The contrast with other discoveries is descriptive only. H1, H2 and H3 are all "
        "being watched, so one apparent pass among them calls for extra scepticism, and no claim is made before 12 "
        "independent windows.")


def _is_discovery(o: dict) -> bool:
    return o.get("observation_type", "discovery") == "discovery" and o.get("model_version") == "APEX-9.0"


def in_scope(o: dict, hyp: dict) -> bool:
    """A discovery stamped strictly after the registration date."""
    return _is_discovery(o) and bool(o.get("timestamp")) and str(o["timestamp"])[:10] > hyp["registered"]


def matches(o: dict, hyp: dict) -> bool:
    score = o.get("apex_score")
    return (str(o.get("stage", "")).startswith(hyp["stage_prefix"]) and score is not None
            and float(score) >= hyp["min_score"])


def _excess(o: dict, key: str) -> Optional[float]:
    out = (o.get("outcomes") or {}).get(key)
    if out and out.get("excess_return_%") is not None:
        return float(out["excess_return_%"])
    return None


def verdict_text(n: int, windows: int, ci: Optional[tuple]) -> str:
    if n < MIN_MATCHED_EVENTS:
        return f"Too few resolved observations ({n}) for any verdict"
    if windows < MIN_CLUSTERS_FOR_CI or ci is None:
        return f"Insufficient independent windows ({windows}): descriptive only"
    tag = "adequate" if windows >= ADEQUATE_CLUSTERS else "exploratory"
    lo, hi = ci
    if lo > 0:
        return f"Outperformed SPY ({tag}, {windows} windows)"
    if hi < 0:
        return f"Underperformed SPY ({tag}, {windows} windows)"
    return f"No detectable difference from SPY ({tag}, {windows} windows)"


def evaluate_forward(observations: list, hyp: dict) -> dict:
    key = hyp["horizon"]
    scoped = [o for o in observations if in_scope(o, hyp)]
    group = [o for o in scoped if matches(o, hyp)]
    others = [o for o in scoped if not matches(o, hyp)]
    g_vals, g_dates = [], []
    for o in group:
        v = _excess(o, key)
        if v is not None:
            g_vals.append(v)
            g_dates.append(str(o["timestamp"])[:10])
    o_vals = [v for v in (_excess(o, key) for o in others) if v is not None]
    res = {"id": hyp["id"], "text": hyp["text"], "motivation": hyp["motivation"], "registered": hyp["registered"],
           "horizon": key, "control": "SPY", "n_in_scope": len(group), "n_resolved": len(g_vals), "windows": 0,
           "mean_excess_%": None, "ci": None, "others_n": len(o_vals), "others_mean_excess_%": None,
           "difference_%": None,
           "first_date": min((str(o["timestamp"])[:10] for o in group), default=None),
           "last_date": max((str(o["timestamp"])[:10] for o in group), default=None)}
    if g_vals:
        st = cluster_robust_mean(g_vals, window_clusters(g_dates, horizon_days(key)))
        res.update({"windows": st["k"], "mean_excess_%": round(st["mean"], 2),
                    "ci": None if st["ci"] is None else (round(st["ci"][0], 2), round(st["ci"][1], 2))})
    if o_vals:
        res["others_mean_excess_%"] = round(sum(o_vals) / len(o_vals), 2)
        if g_vals:
            res["difference_%"] = round(res["mean_excess_%"] - res["others_mean_excess_%"], 2)
    res["verdict"] = verdict_text(res["n_resolved"], res["windows"], res["ci"])
    return res


def evaluate_all(observations: list) -> list:
    return [evaluate_forward(observations, h) for h in HYPOTHESES]
