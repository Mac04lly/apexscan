"""
modules/eval_harness.py — Evaluation harness (Change 2)

Pure functions. No network, no storage, no dependency on scanner.py or
dashboard.py, and it never writes anything. It exists so that claims about
ApexScan's signals are made against the right yardstick:

  1. INDEPENDENCE. 656 observations stamped on a handful of dates, evaluated over
     20-trading-day windows that overlap each other, are not 656 independent
     tests: they share one market episode. `independence_summary` counts
     independent windows instead of rows and labels the evidence honestly.

  2. A MATCHED CONTROL. A raw return of -5% means nothing when the whole
     cohort lost 5%. For each event the control is the leave-one-out average
     return of cohort stocks that were in the SAME distance-to-200MA quintile
     (distance to the stop is the dominant confound found in the audit) over the
     SAME calendar window (same trading-day offset from the cohort's entry).

  3. CLUSTER-ROBUST INFERENCE. Events are grouped into non-overlapping
     horizon-length windows; the standard error is clustered on those windows
     (CR1, t distribution with G-1 degrees of freedom). With too few windows no
     interval is shown at all — a precise-looking interval built on two
     independent windows would be a false claim.

  4. SELF-VALIDATION. `simulate_matched_test` measures this harness's own
     false-positive rate on pure noise and its power on an injected effect, so the
     machinery is shown to work before anything is trusted to it.

Known approximations (documented, not hidden): cohort cumulative returns are
linearly interpolated between the outcome engine's fixed horizons; adjacent
windows overlap at their boundary, so cluster independence is approximate;
holidays use a built-in 2026–2027 NYSE list; returns are gross of costs.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np

# ── constants ─────────────────────────────────────────────────────────────
MIN_CLUSTERS_FOR_CI = 6       # fewer independent windows than this -> no interval is shown
ADEQUATE_CLUSTERS = 12        # the bar set in the evaluation plan for a claim
MIN_MATCHED_EVENTS = 8        # fewer matched events than this -> no verdict
MIN_COHORT = 30               # cohort members (with the feature) needed to build a control
MIN_CONTROL_MEMBERS = 15      # leave-one-out control members needed for one event
DISTANCE_KEY = "vs_200ma_%"   # technical_features key used for matching

# Full-day NYSE closures (2026–2027). Outside this range weekends are the only closures assumed.
NYSE_HOLIDAYS = [
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
]

# Two-sided 95% Student-t critical values, df -> t. Values between table keys use the
# next-lower key (slightly wider = conservative). Avoids a scipy dependency.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
         23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
         30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t_crit_95(df: int) -> float:
    if df < 1:
        return float("inf")
    if df >= 1000:
        return 1.960
    keys = [k for k in _T975 if k <= df]
    return _T975[max(keys)]


# ── dates / windows ───────────────────────────────────────────────────────
def horizon_days(horizon) -> int:
    """'20D' -> 20; 20 -> 20."""
    if isinstance(horizon, (int, np.integer)):
        return int(horizon)
    return int(str(horizon).rstrip("Dd"))


def _day(s) -> np.datetime64:
    return np.datetime64(str(s)[:10], "D")


def trading_days_between(start, end) -> int:
    """Whole trading days from `start` to `end` (negative if end < start)."""
    return int(np.busday_count(_day(start), _day(end), holidays=NYSE_HOLIDAYS))


def first_trading_day(date) -> np.datetime64:
    """Row 0 of the outcome engine: the first trading day on/after `date`."""
    return np.busday_offset(_day(date), 0, roll="forward", holidays=NYSE_HOLIDAYS)


def window_clusters(entry_dates: list, h: int, anchor=None) -> list:
    """Assign each entry date to a block of `h` trading days. Entries in the same block
    share (overlapping) forward windows; blocks are the independence unit. `anchor`
    defaults to the earliest entry date."""
    if not entry_dates:
        return []
    anchor = anchor if anchor is not None else min(str(d)[:10] for d in entry_dates)
    return [max(0, trading_days_between(anchor, d)) // max(1, int(h)) for d in entry_dates]


def independence_label(k: int) -> str:
    if k < MIN_CLUSTERS_FOR_CI:
        return "Insufficient independent windows"
    if k < ADEQUATE_CLUSTERS:
        return "Exploratory"
    return "Adequate"


def independence_summary(observations: list, h: int, horizon_key: Optional[str] = None) -> dict:
    """Counts independent evidence, not rows. If `horizon_key` is given, only observations
    with a frozen outcome at that horizon are counted (matching what the tables use)."""
    rows = []
    for o in observations:
        if horizon_key is not None:
            out = (o.get("outcomes") or {}).get(horizon_key)
            if not out or out.get("forward_return_%") is None:
                continue
        if o.get("timestamp"):
            rows.append(str(o["timestamp"])[:10])
    n = len(rows)
    if n == 0:
        return {"n_rows": 0, "n_dates": 0, "n_windows": 0, "largest_date_share_%": None,
                "label": independence_label(0), "note": "No resolved observations."}
    counts = defaultdict(int)
    for d in rows:
        counts[d] += 1
    k = len(set(window_clusters(rows, h)))
    share = round(100 * max(counts.values()) / n, 1)
    label = independence_label(k)
    if k < MIN_CLUSTERS_FOR_CI:
        note = (f"These rows come from {len(counts)} signal date(s), {share}% from the largest, "
                f"and share overlapping {h}-day windows: one market episode, not {n} independent "
                f"tests. Treat as descriptive.")
    elif k < ADEQUATE_CLUSTERS:
        note = (f"{k} independent {h}-day windows: enough for an exploratory read, not for a claim "
                f"(the plan requires {ADEQUATE_CLUSTERS}).")
    else:
        note = f"{k} independent {h}-day windows."
    return {"n_rows": n, "n_dates": len(counts), "n_windows": k,
            "largest_date_share_%": share, "label": label, "note": note}


# ── inference ─────────────────────────────────────────────────────────────
def cluster_robust_mean(values, clusters) -> dict:
    """Mean with a cluster-robust (CR1) standard error and a 95% t interval on G-1 df.
    The interval is withheld (None) when there are fewer than MIN_CLUSTERS_FOR_CI clusters."""
    x = np.asarray(values, float)
    g = np.asarray(clusters)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    n = len(x)
    if n == 0:
        return {"n": 0, "k": 0, "mean": None, "se": None, "ci": None, "label": independence_label(0)}
    mean = float(x.mean())
    ids = np.unique(g)
    k = len(ids)
    out = {"n": n, "k": k, "mean": mean, "se": None, "ci": None, "label": independence_label(k)}
    if k < 2:
        return out
    resid = x - mean
    s = np.array([resid[g == c].sum() for c in ids])
    var = (k / (k - 1)) * float((s ** 2).sum()) / (n ** 2)
    se = math.sqrt(var)
    out["se"] = se
    if k >= MIN_CLUSTERS_FOR_CI:
        m = t_crit_95(k - 1) * se
        out["ci"] = (mean - m, mean + m)
    return out


def naive_mean_ci(values) -> Optional[tuple]:
    """The row-level normal interval the Alpha Lab tables use, which treats every row as
    independent. Kept only to show how much narrower it is than the clustered one."""
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if len(x) < 2:
        return None
    m = 1.96 * x.std(ddof=1) / math.sqrt(len(x))
    return (float(x.mean() - m), float(x.mean() + m))


# ── matched control ───────────────────────────────────────────────────────
def _horizon_grid(universe: list) -> list:
    hs = set()
    for o in universe:
        for key in (o.get("outcomes") or {}):
            if key.endswith("D") and key[:-1].isdigit():
                hs.add(int(key[:-1]))
    return [0] + sorted(hs)


def _interp_matrix(R: np.ndarray, grid: np.ndarray, t: float) -> np.ndarray:
    """Per-member cumulative return at trading-day offset t (linear between the outcome
    engine's fixed horizons). NaN where either bracketing value is unresolved."""
    if t > grid[-1] or t < 0:
        return np.full(R.shape[0], np.nan)
    i = int(np.searchsorted(grid, t))
    if grid[i] == t:
        return R[:, i]
    lo, hi = i - 1, i
    return R[:, lo] + (R[:, hi] - R[:, lo]) * (t - grid[lo]) / (grid[hi] - grid[lo])


def build_cohort_controls(universe: list, distance_key: str = DISTANCE_KEY,
                          n_buckets: int = 5, min_cohort: int = MIN_COHORT) -> dict:
    """Per entry-date cohort: members' frozen cumulative returns, plus distance-quintile
    membership. `universe` = discovery observations that carry frozen outcomes."""
    disc = [o for o in universe
            if o.get("observation_type", "discovery") == "discovery"
            and o.get("observation_id") and o.get("outcomes") and o.get("timestamp")]
    grid = _horizon_grid(disc)
    by_date = defaultdict(list)
    for o in disc:
        feat = (o.get("technical_features") or {}).get(distance_key)
        if isinstance(feat, (int, float)) and not isinstance(feat, bool) and math.isfinite(feat):
            by_date[str(o["timestamp"])[:10]].append((o, float(feat)))
    cohorts = {}
    for d, members in by_date.items():
        if len(members) < min_cohort:
            continue
        R = np.full((len(members), len(grid)), np.nan)
        R[:, 0] = 0.0
        for i, (o, _) in enumerate(members):
            for key, val in (o.get("outcomes") or {}).items():
                if key.endswith("D") and key[:-1].isdigit() and val and val.get("forward_return_%") is not None:
                    R[i, grid.index(int(key[:-1]))] = val["forward_return_%"]
        x = np.array([f for _, f in members])
        edges = np.quantile(x, np.linspace(0, 1, n_buckets + 1)[1:-1])
        cohorts[d] = {"index": {o["observation_id"]: i for i, (o, _) in enumerate(members)},
                      "R": R, "bucket": np.searchsorted(edges, x, side="right"), "edges": edges}
    return {"grid": np.array(grid, float), "cohorts": cohorts, "distance_key": distance_key}


def control_window_return(controls: dict, cohort_date: str, source_id: str, a: int, h: int,
                          min_members: int = MIN_CONTROL_MEMBERS) -> Optional[tuple]:
    """Mean return (%) of same-quintile cohort members over trading days [a, a+h], EXCLUDING
    the event's own stock. Returns (mean, n_members) or None when no defensible control exists."""
    c = controls["cohorts"].get(cohort_date)
    if not c:
        return None
    i = c["index"].get(source_id)
    if i is None or a < 0:
        return None
    grid = controls["grid"]
    r0 = _interp_matrix(c["R"], grid, a)
    r1 = _interp_matrix(c["R"], grid, a + h)
    mask = c["bucket"] == c["bucket"][i]
    mask[i] = False                                   # leave-one-out
    mask &= np.isfinite(r0) & np.isfinite(r1)
    if int(mask.sum()) < min_members:
        return None
    w = ((1 + r1[mask] / 100.0) / (1 + r0[mask] / 100.0) - 1) * 100.0
    return float(w.mean()), int(mask.sum())


def evaluate_events(groups: dict, controls: dict, source_lookup: dict, h: int) -> dict:
    """For each named group of events: event return minus matched-control return over the
    same window, with a cluster-robust interval (withheld when there are too few independent
    windows). `source_lookup`: observation_id -> the discovery observation an event came from."""
    key = f"{h}D"
    all_dates = [str(e["timestamp"])[:10] for evs in groups.values() for e in evs if e.get("timestamp")]
    anchor = min(all_dates) if all_dates else None
    results = {}
    for label, events in groups.items():
        diffs, ev_r, ct_r, dates = [], [], [], []
        n_with_outcome = 0
        for e in events:
            out = (e.get("outcomes") or {}).get(key)
            if not out or out.get("forward_return_%") is None:
                continue
            n_with_outcome += 1
            src = source_lookup.get(e.get("source_observation_id"))
            if not src or not src.get("timestamp"):
                continue
            cohort_date = str(src["timestamp"])[:10]
            a = trading_days_between(first_trading_day(cohort_date), str(e["timestamp"])[:10])
            ctl = control_window_return(controls, cohort_date, src["observation_id"], a, h)
            if ctl is None:
                continue
            ev_r.append(out["forward_return_%"])
            ct_r.append(ctl[0])
            diffs.append(out["forward_return_%"] - ctl[0])
            dates.append(str(e["timestamp"])[:10])
        n_matched = len(diffs)
        res = {"label": label, "horizon": key, "n_events": len(events),
               "n_with_outcome": n_with_outcome, "n_matched": n_matched,
               "n_unmatched": n_with_outcome - n_matched,
               "mean_event_%": None, "mean_control_%": None, "mean_diff_%": None,
               "se": None, "ci": None, "naive_ci": None, "n_windows": 0,
               "independence": independence_label(0)}
        if n_matched:
            cl = window_clusters(dates, h, anchor=anchor)
            stats = cluster_robust_mean(diffs, cl)
            res.update({"mean_event_%": float(np.mean(ev_r)), "mean_control_%": float(np.mean(ct_r)),
                        "mean_diff_%": stats["mean"], "se": stats["se"], "ci": stats["ci"],
                        "naive_ci": naive_mean_ci(diffs), "n_windows": stats["k"],
                        "independence": stats["label"]})
        res["verdict"] = verdict(res)
        results[label] = res
    return results


def verdict(res: dict) -> str:
    if res["n_matched"] < MIN_MATCHED_EVENTS:
        return f"Too few matched events ({res['n_matched']}) for any verdict"
    k = res["n_windows"]
    if k < MIN_CLUSTERS_FOR_CI:
        return (f"Insufficient independent windows ({k}): descriptive only, no interval shown")
    lo, hi = res["ci"]
    tag = "adequate" if k >= ADEQUATE_CLUSTERS else "exploratory"
    if lo > 0:
        return f"Outperformed matched control ({tag}, {k} windows)"
    if hi < 0:
        return f"Underperformed matched control ({tag}, {k} windows)"
    return f"No detectable difference from matched control ({tag}, {k} windows)"


def evaluate_invalidation_events(observations: list, horizons=(5, 10, 20)) -> list:
    """Evaluates the logged invalidation events, split by basis, against the matched control.
    Returns one row dict per (group, horizon)."""
    from modules.invalidation_alpha import derive_event_basis

    by_id = {o.get("observation_id"): o for o in observations if o.get("observation_id")}
    events = [o for o in observations
              if o.get("observation_type") in ("invalidation_breach", "invalidation_near_miss")]
    names = {"crossed_after_discovery": "Breach — genuine crossing",
             "already_below_at_discovery": "Breach — mechanical (already below stop)",
             "approached_after_discovery": "Near-miss — approached after discovery",
             "within_band_at_discovery": "Near-miss — already in band at discovery"}
    groups = defaultdict(list)
    for e in events:
        basis = derive_event_basis(e, by_id.get(e.get("source_observation_id")))
        groups[names.get(basis, "Unclassified")].append(e)
    universe = [o for o in observations if o.get("observation_type", "discovery") == "discovery"]
    controls = build_cohort_controls(universe)
    rows = []
    for h in horizons:
        res = evaluate_events(dict(groups), controls, by_id, int(h))
        for name in names.values():
            if name in res:
                rows.append(res[name])
    return rows


# ── self-validation by simulation ─────────────────────────────────────────
def simulate_matched_test(n_sims: int = 500, n_days: int = 240, h: int = 20, n_events: int = 300,
                          effect: float = 0.0, sigma_factor: float = 1.0, sigma_idio: float = 12.0,
                          beta_sd: float = 0.3, seed: int = 11) -> dict:
    """Runs the harness's own statistics on synthetic data with a KNOWN truth and returns the
    rate at which each estimator 'finds' an effect (95% interval excludes 0).

    World: a daily market factor (sd `sigma_factor` %/day) drives every stock through its beta
    (beta ~ N(1, beta_sd)), so windows are correlated and adjacent windows overlap. Each event is
    an entry on a random day with idiosyncratic noise (sd `sigma_idio` %) and `effect` added.
    The matched control sees the same factor at beta 1 (imperfect matching leaves a residual
    (beta-1)*factor, as in real data).

      harness  : (event - control), cluster-robust CI on h-day window blocks, needs >= 6 windows
      naive    : raw event return, row-level CI vs 0 (what a table implies)
      matched_row_ci : (event - control) but row-level CI, isolating the effect of clustering
    """
    rng = np.random.default_rng(seed)
    found = {"harness": 0, "naive": 0, "matched_row_ci": 0}
    withheld = 0
    for _ in range(n_sims):
        f = rng.normal(0.0, sigma_factor, n_days + h + 1)
        cum = np.concatenate([[0.0], np.cumsum(f)])
        t = rng.integers(0, n_days, n_events)
        F = cum[t + h] - cum[t]                                    # factor over each event's window
        beta = rng.normal(1.0, beta_sd, n_events)
        ev = beta * F + rng.normal(0.0, sigma_idio, n_events) + effect
        diff = ev - F
        clusters = t // h
        r = cluster_robust_mean(diff, clusters)
        if r["ci"] is None:
            withheld += 1
        elif r["ci"][0] > 0 or r["ci"][1] < 0:
            found["harness"] += 1
        for name, vals in (("naive", ev), ("matched_row_ci", diff)):
            ci = naive_mean_ci(vals)
            if ci and (ci[0] > 0 or ci[1] < 0):
                found[name] += 1
    out = {k: v / n_sims for k, v in found.items()}
    out["harness_withheld"] = withheld / n_sims
    out["n_sims"] = n_sims
    return out


def power_curve(effects=(0.0, 1.0, 2.0, 3.0, 4.0), **kw) -> dict:
    """Detection rate of the harness at each injected effect size (in % points)."""
    return {e: simulate_matched_test(effect=e, **kw)["harness"] for e in effects}
