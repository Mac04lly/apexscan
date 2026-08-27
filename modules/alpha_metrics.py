"""
modules/alpha_metrics.py — Alpha Metrics (V9 Phase 3)

Given a set of observations that already have frozen Phase 2 outcomes,
computes the statistics needed to objectively answer "has this score
bucket / setup / feature historically produced superior outcomes":
expectancy, win rate, median return, profit factor, a 95% confidence
interval, and a plain-language sample-size classification.

This module only ever reads observations — it never writes to
alpha_observations.json, and it has no dependency on scanner.py,
dashboard.py, or any live scan. It's a pure function of whatever
observation list (with outcomes) you give it, which makes it safe to
test in isolation, and safe to reuse later for setup-based (Phase 4) or
feature-based (Phase 5) grouping without any changes to this file.
"""
from __future__ import annotations
import math
from typing import Optional

# Same n=30 threshold already used elsewhere in ApexScan's own honesty
# discipline (Discovery Tracker's Sample Readiness indicator) — kept
# consistent rather than inventing a different bar here.
def classify_sample_size(n: int) -> str:
    if n < 10:
        return "Too Small — not enough data to draw any conclusion"
    if n < 30:
        return "Emerging — directional only, not yet statistically reliable"
    if n < 100:
        return "Meaningful — a real pattern, worth attention"
    return "Robust — a well-supported sample"


def _confidence_interval_95(returns: list) -> Optional[tuple]:
    """Standard normal-approximation 95% CI on the mean. Returns None if
    the sample is too small (<2) for a standard deviation to exist at all."""
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)
    margin = 1.96 * se
    return (round(mean - margin, 3), round(mean + margin, 3))


def compute_alpha_metrics(observations: list, horizon: str = "20D") -> dict:
    """
    Computes Alpha Metrics for one horizon across a list of observations.
    Only observations with a FROZEN outcome for this exact horizon are
    included — an observation still waiting on that horizon contributes
    nothing (not a zero, not an estimate; genuinely excluded).

    Returns a dict with n, expectancy_%, win_rate_%, median_return_%,
    profit_factor, confidence_interval_95, sample_classification, and
    avg_excess_return_% (mean outperformance vs. the S&P 500 over the
    same window, when benchmark data was available for those trades).
    """
    returns = []
    excess_returns = []
    for obs in observations or []:
        outcome = (obs.get("outcomes") or {}).get(horizon)
        if not outcome or outcome.get("forward_return_%") is None:
            continue
        returns.append(float(outcome["forward_return_%"]))
        if outcome.get("excess_return_%") is not None:
            excess_returns.append(float(outcome["excess_return_%"]))

    n = len(returns)
    if n == 0:
        return {
            "horizon": horizon, "n": 0, "expectancy_%": None, "win_rate_%": None,
            "median_return_%": None, "profit_factor": None,
            "confidence_interval_95": None, "avg_excess_return_%": None,
            "sample_classification": classify_sample_size(0),
        }

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    win_rate = round(len(wins) / n * 100, 1)
    expectancy = round(sum(returns) / n, 3)  # mean return per trade, this horizon

    sorted_returns = sorted(returns)
    mid = n // 2
    median = sorted_returns[mid] if n % 2 == 1 else round((sorted_returns[mid - 1] + sorted_returns[mid]) / 2, 3)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_profit > 0 else None)

    return {
        "horizon": horizon,
        "n": n,
        "expectancy_%": expectancy,
        "win_rate_%": win_rate,
        "median_return_%": median,
        "profit_factor": profit_factor,
        "confidence_interval_95": _confidence_interval_95(returns),
        "avg_excess_return_%": round(sum(excess_returns) / len(excess_returns), 3) if excess_returns else None,
        "sample_classification": classify_sample_size(n),
    }


def compute_alpha_metrics_by_score_bucket(observations: list, horizon: str = "20D",
                                          buckets: Optional[list] = None) -> list:
    """
    Same computation as compute_alpha_metrics(), but grouped by
    apex_score_raw bucket — the direct evolution of Discovery Tracker's
    existing 'Does Apex Score Predict Returns?' table, now using fixed-
    horizon outcomes with MFE/MAE/benchmark-excess and real confidence
    intervals instead of a single ad-hoc 'whatever day you happened to
    check' price comparison.
    """
    if buckets is None:
        buckets = [("150+", 150, 10_000), ("125-149", 125, 150), ("100-124", 100, 125),
                   ("80-99", 80, 100), ("65-79", 65, 80), ("50-64", 50, 65), ("<50", 0, 50)]

    results = []
    for label, lo, hi in buckets:
        bucket_obs = [
            o for o in observations
            if o.get("apex_score_raw") is not None and lo <= float(o["apex_score_raw"]) < hi
        ]
        metrics = compute_alpha_metrics(bucket_obs, horizon)
        metrics["score_bucket"] = label
        results.append(metrics)
    return results


# ══════════════════════════════════════════════════════════════════════════
# V9 PHASE 4 — SETUP RESEARCH
#
# "Does ApexScan know which setup types work?" Every AlphaObservation
# already carries a setup_id (assigned once, at discovery, by
# alpha_validation.derive_setup_id — Phase 1). This section just groups
# by that existing field; it introduces no new taxonomy of its own.
#
# Honesty note: derive_setup_id() currently produces a SUBSET of the full
# taxonomy the V9 spec sketches (no S2-PULLBACK-200, S2-GAP, or S4-RETEST
# yet — nothing in the current scanner distinguishes those from their
# closest sibling setup). Whatever setup_id values actually appear in the
# data is exactly what gets reported here — nothing is backfilled or
# assumed for setups the scanner doesn't yet distinguish.
# ══════════════════════════════════════════════════════════════════════════

# Human-readable labels for the setup_id values derive_setup_id() actually
# produces today. A setup_id with no entry here still works fine — it's
# just displayed as-is, so this list never needs to be kept airtight.
SETUP_LABELS = {
    "S1-BASE": "Stage 1 — Base",
    "S1-EARLY-ALPHA": "Stage 1 — Early Alpha (low-ADR base)",
    "S2-PULLBACK-50": "Stage 2 — Pullback to 50MA",
    "S2-MA-RECLAIM": "Stage 2 — Fresh 200MA Reclaim",
    "S2-BREAKOUT-VOL": "Stage 2 — Breakout (volume confirmed)",
    "S2-BREAKOUT-NOVOL": "Stage 2 — Breakout (no volume confirmation)",
    "S2-HIGH-RS": "Stage 2 — High Relative Strength",
    "S2-CONSOLIDATION": "Stage 2 — Consolidation",
    "S3-TOP": "Stage 3 — Top / Distribution",
    "S4-BREAKDOWN": "Stage 4 — Breakdown",
}


def compute_alpha_metrics_by_setup(observations: list, horizon: str = "20D") -> list:
    """
    Groups observations by their (already-assigned) setup_id and computes
    Alpha Metrics per group. This is the direct answer to Phase 4's
    success criterion: "ApexScan knows which setup types work."

    Every setup_id actually present in `observations` gets a row, even
    ones with zero resolved outcomes yet (n=0, sample_classification
    'Too Small') — a setup with no rows here simply hasn't produced a
    discovery yet, which is itself useful to see, not something to hide.
    """
    setup_ids = sorted({o.get("setup_id") for o in observations if o.get("setup_id")})
    results = []
    for setup_id in setup_ids:
        bucket_obs = [o for o in observations if o.get("setup_id") == setup_id]
        metrics = compute_alpha_metrics(bucket_obs, horizon)
        metrics["setup_id"] = setup_id
        metrics["setup_label"] = SETUP_LABELS.get(setup_id, setup_id)
        results.append(metrics)
    # Setups with resolved evidence first (largest sample first); untested
    # setups sink to the bottom rather than cluttering the top of the table.
    results.sort(key=lambda r: r["n"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════
# V9 PHASE 5 — FEATURE RESEARCH
#
# "Which existing features have predictive association?" Every feature
# ApexScan already computes (technical, fundamental, valuation) gets
# bucketed and measured the same way the score itself was in Phase 3.
#
# Per the spec's own explicit instruction (V9 spec §15): results here
# describe HISTORICAL ASSOCIATION, never causation. A feature showing a
# wide spread between its best- and worst-performing bucket has
# historically been associated with better/worse subsequent excess
# returns in ApexScan's own observed universe — it is not a claim that
# the feature CAUSES those returns.
# ══════════════════════════════════════════════════════════════════════════

FEATURE_REGISTRY = {
    "rs_3m":                {"label": "Relative Strength (3M, raw)", "path": "technical_features.rs_3m", "type": "continuous"},
    "rs_percentile":        {"label": "RS Percentile (within scan)", "path": "technical_features.rs_percentile", "type": "continuous"},
    "perf_3m":              {"label": "3M Performance", "path": "technical_features.perf_3m_%", "type": "continuous"},
    "perf_1m":              {"label": "1M Performance", "path": "technical_features.perf_1m_%", "type": "continuous"},
    "vs_200ma":             {"label": "Distance vs 200MA", "path": "technical_features.vs_200ma_%", "type": "continuous"},
    "vs_50ma":              {"label": "Distance vs 50MA", "path": "technical_features.vs_50ma_%", "type": "continuous"},
    "volume_ratio":         {"label": "Volume Expansion (x avg)", "path": "technical_features.volume_ratio", "type": "continuous"},
    "volume_persistence":   {"label": "Order Flow Persistence Score", "path": "technical_features.volume_persistence", "type": "continuous"},
    "adr":                  {"label": "Average Daily Range %", "path": "technical_features.adr_%", "type": "continuous"},
    "stage":                {"label": "Stage", "path": "stage", "type": "categorical"},
    "market_structure":     {"label": "Market Structure", "path": "technical_features.market_structure", "type": "categorical"},
    "vwap_position":        {"label": "VWAP Position", "path": "technical_features.vwap_position", "type": "categorical"},
    "breakout_status":      {"label": "Breaking Out", "path": "technical_features.breakout_status", "type": "categorical"},
    "weekly_confirmation":  {"label": "Weekly Confirmation", "path": "technical_features.weekly_confirmation", "type": "categorical"},
    "hh_hl":                {"label": "Higher-High / Higher-Low Structure", "path": "technical_features.hh_hl", "type": "categorical"},
    "revenue_growth":       {"label": "Revenue Growth", "path": "fundamental_features.revenue_growth", "type": "continuous"},
    "earnings_growth":      {"label": "Earnings Growth", "path": "fundamental_features.earnings_growth", "type": "continuous"},
    "roe":                  {"label": "ROE", "path": "fundamental_features.roe", "type": "continuous"},
    "debt_to_equity":       {"label": "Debt / Equity", "path": "fundamental_features.debt_to_equity", "type": "continuous"},
    "institutional_ownership": {"label": "Institutional Ownership", "path": "fundamental_features.institutional_ownership", "type": "continuous"},
    "beta":                 {"label": "Beta", "path": "fundamental_features.beta", "type": "continuous"},
    "pe_ratio":             {"label": "P/E Ratio", "path": "valuation_features.pe_ratio", "type": "continuous"},
    "peg_ratio":            {"label": "PEG Ratio", "path": "valuation_features.peg_ratio", "type": "continuous"},
}


def get_feature_value(observation: dict, path: str):
    """Reads a dotted path (e.g. 'technical_features.rs_3m') out of an
    observation. Returns None on any missing/malformed segment — a
    feature ApexScan hasn't populated yet (most fundamentals, currently)
    is treated as genuinely absent, never coerced to zero or skipped
    silently in a way that would bias a bucket's average."""
    try:
        node = observation
        for part in path.split("."):
            if not isinstance(node, dict):
                return None
            node = node.get(part)
        if node is None or (isinstance(node, float) and math.isnan(node)):
            return None
        return node
    except Exception:
        return None


def _bucket_label(lo: float, hi: float, is_last: bool) -> str:
    lo_s = f"{lo:.1f}" if abs(lo) < 1000 else f"{lo:,.0f}"
    hi_s = f"{hi:.1f}" if abs(hi) < 1000 else f"{hi:,.0f}"
    return f"{lo_s} – {hi_s}" if not is_last else f"{lo_s} – {hi_s} (top)"


def compute_alpha_metrics_by_feature(observations: list, horizon: str = "20D",
                                     feature_key: str = "rs_3m", n_buckets: int = 5) -> list:
    """
    Buckets observations by one feature and computes Alpha Metrics per
    bucket — the Phase 5 equivalent of compute_alpha_metrics_by_score_bucket
    for any feature in FEATURE_REGISTRY.

    Bucket edges (for continuous features) are computed from exactly the
    observations that have BOTH a value for this feature AND a frozen
    outcome at this horizon — so every bucket reported is immediately
    populated by construction, and the boundaries reflect the population
    actually being analyzed for this horizon, not the full unfiltered
    dataset (which would include still-pending observations the buckets
    can't say anything about yet).
    """
    if feature_key not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature_key: {feature_key!r}")
    spec = FEATURE_REGISTRY[feature_key]
    path, ftype, label = spec["path"], spec["type"], spec["label"]

    usable = []
    for o in observations or []:
        val = get_feature_value(o, path)
        outcome = (o.get("outcomes") or {}).get(horizon)
        if val is None or not outcome or outcome.get("forward_return_%") is None:
            continue
        usable.append((o, val))

    if not usable:
        return []

    if ftype == "categorical":
        cats = sorted({str(v) for _, v in usable})
        results = []
        for cat in cats:
            bucket_obs = [o for o, v in usable if str(v) == cat]
            metrics = compute_alpha_metrics(bucket_obs, horizon)
            metrics.update({"feature_key": feature_key, "feature_label": label, "bucket_label": cat})
            results.append(metrics)
        return results

    # Continuous: rank-based quantile bucketing. Sorts by value and splits
    # into n_buckets contiguous, roughly-equal-SIZE chunks by rank —
    # deliberately not a value-edge method (e.g. pandas qcut edges), which
    # can silently collapse a bucket to zero width (and drop every
    # observation that would have landed in it) whenever a quantile
    # boundary lands on a repeated/clustered value. Rank-based chunking
    # guarantees every observation with a valid value lands in exactly
    # one bucket, and the bucket count sums to the full usable sample —
    # verified by test_feature_bucketing_excludes_observations_without_outcome
    # and test_rank_feature_alpha_computes_spread_when_data_sufficient.
    numeric = []
    for o, v in usable:
        try:
            numeric.append((o, float(v)))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return []

    numeric.sort(key=lambda p: p[1])
    n = len(numeric)
    effective_buckets = max(1, min(n_buckets, n))

    results = []
    for i in range(effective_buckets):
        lo_idx = (i * n) // effective_buckets
        hi_idx = ((i + 1) * n) // effective_buckets
        chunk = numeric[lo_idx:hi_idx]
        if not chunk:
            continue
        chunk_vals = [v for _, v in chunk]
        lo, hi = min(chunk_vals), max(chunk_vals)
        is_last = (i == effective_buckets - 1)
        bucket_obs = [o for o, _ in chunk]
        metrics = compute_alpha_metrics(bucket_obs, horizon)
        metrics.update({
            "feature_key": feature_key, "feature_label": label,
            "bucket_label": _bucket_label(lo, hi, is_last),
            "bucket_lo": lo, "bucket_hi": hi,
        })
        results.append(metrics)
    return results


def rank_feature_alpha(observations: list, horizon: str = "20D",
                       feature_keys: Optional[list] = None, min_n: int = 10) -> list:
    """
    Ranks features by how much their outcome varies across buckets — the
    spread between the best- and worst-performing bucket's expectancy,
    among buckets with at least `min_n` resolved observations.

    This deliberately does NOT assume a direction (e.g. "higher RS is
    better") for any feature — some features (debt/equity, P/E) don't
    have an obvious 'higher is better' reading, and baking that in would
    be an assumption dressed up as a finding. It reports which bucket
    happened to score best, plain and simple; the reader draws the
    interpretation.

    A feature with fewer than 2 buckets meeting `min_n` is reported with
    spread_%=None and is_conclusive=False — explicitly flagged as not
    yet decidable, rather than silently omitted or given a misleadingly
    precise number from a thin sample.
    """
    keys = feature_keys or list(FEATURE_REGISTRY.keys())
    results = []
    for key in keys:
        buckets = compute_alpha_metrics_by_feature(observations, horizon, key)
        qualifying = [b for b in buckets if b["n"] >= min_n and b["expectancy_%"] is not None]
        total_n = sum(b["n"] for b in buckets)
        if len(qualifying) < 2:
            results.append({
                "feature_key": key, "feature_label": FEATURE_REGISTRY[key]["label"],
                "spread_%": None, "best_bucket_label": None, "best_bucket_expectancy_%": None,
                "worst_bucket_label": None, "n_qualifying_buckets": len(qualifying),
                "total_n": total_n, "is_conclusive": False,
            })
            continue
        best = max(qualifying, key=lambda b: b["expectancy_%"])
        worst = min(qualifying, key=lambda b: b["expectancy_%"])
        results.append({
            "feature_key": key, "feature_label": FEATURE_REGISTRY[key]["label"],
            "spread_%": round(best["expectancy_%"] - worst["expectancy_%"], 3),
            "best_bucket_label": best["bucket_label"], "best_bucket_expectancy_%": best["expectancy_%"],
            "worst_bucket_label": worst["bucket_label"], "worst_bucket_expectancy_%": worst["expectancy_%"],
            "n_qualifying_buckets": len(qualifying), "total_n": total_n, "is_conclusive": True,
        })
    results.sort(key=lambda r: (r["is_conclusive"], abs(r["spread_%"]) if r["spread_%"] is not None else -1),
                reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════
# V9 PHASE 6 — CONDITIONAL RESEARCH
#
# "ApexScan knows WHEN its signals work." Cross a feature against regime,
# sector/theme, or setup — e.g. "RS in its top bucket, broken out by
# market regime." Every condition value actually observed gets a row,
# same honesty rule as everywhere else in this file: an unseen
# combination isn't fabricated, it's just absent from the table.
# ══════════════════════════════════════════════════════════════════════════

CONDITION_PATHS = {
    "regime": "market_regime",
    "sector": "market_features.theme",
    "setup":  "setup_id",
}


def compute_conditional_alpha(observations: list, horizon: str = "20D", feature_key: str = "rs_percentile",
                              condition_type: str = "regime", threshold: Optional[float] = None) -> dict:
    """
    Filters to observations in the feature's TOP bucket (either an
    explicit `threshold`, i.e. feature_value >= threshold, or — if no
    threshold is given — the top quintile of the feature's own observed
    distribution among observations resolved at this horizon), then
    breaks that filtered set down by `condition_type` (regime / sector /
    setup). Returns one row per condition value actually present.
    """
    if condition_type not in CONDITION_PATHS:
        raise ValueError(f"Unknown condition_type: {condition_type!r}")
    if feature_key not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature_key: {feature_key!r}")

    fpath = FEATURE_REGISTRY[feature_key]["path"]
    flabel = FEATURE_REGISTRY[feature_key]["label"]
    cpath = CONDITION_PATHS[condition_type]

    resolved = [o for o in observations or []
                if (o.get("outcomes") or {}).get(horizon)
                and (o["outcomes"][horizon].get("forward_return_%") is not None)]

    if threshold is None:
        buckets = compute_alpha_metrics_by_feature(resolved, horizon, feature_key)
        numeric_buckets = [b for b in buckets if "bucket_hi" in b]
        if not numeric_buckets:
            return {"feature_key": feature_key, "feature_label": flabel,
                    "condition_type": condition_type, "filter_description": "n/a — no data yet", "rows": []}
        top = max(numeric_buckets, key=lambda b: b["bucket_hi"])
        threshold = top["bucket_lo"]
        filter_desc = f"{flabel} in {top['bucket_label']} (top quintile of observed distribution)"
    else:
        filter_desc = f"{flabel} ≥ {threshold}"

    qualifying = []
    for o in resolved:
        val = get_feature_value(o, fpath)
        try:
            if val is not None and float(val) >= threshold:
                qualifying.append(o)
        except (TypeError, ValueError):
            continue

    condition_values = sorted({str(get_feature_value(o, cpath)) for o in qualifying
                               if get_feature_value(o, cpath) is not None})
    rows = []
    for cv in condition_values:
        group = [o for o in qualifying if str(get_feature_value(o, cpath)) == cv]
        metrics = compute_alpha_metrics(group, horizon)
        metrics["condition_value"] = cv
        rows.append(metrics)
    rows.sort(key=lambda r: r["n"], reverse=True)

    return {"feature_key": feature_key, "feature_label": flabel, "condition_type": condition_type,
            "filter_description": filter_desc, "rows": rows}


# Predefined combinations only (V9 spec §16 explicitly warns against
# testing many combinations — multiple-testing bias — so this list is
# deliberately short and fixed, not generated by searching for whatever
# looks good). Each condition is (dotted_path, operator, value).
# "rs_percentile" (0–100, this-scan-relative) is used instead of raw
# rs_3m for threshold-style conditions, since rs_3m's own scale varies
# wildly with the benchmark's return and isn't comparable across scans —
# see FIELD_COVERAGE in alpha_validation.py.
PRESET_COMBINATIONS = [
    {"label": "RS + Stage 2", "conditions": [
        ("technical_features.rs_percentile", ">=", 80), ("stage", "startswith", "2")]},
    {"label": "RS + Volume Expansion", "conditions": [
        ("technical_features.rs_percentile", ">=", 80), ("technical_features.volume_ratio", ">=", 1.4)]},
    {"label": "Stage 2 + Volume Expansion", "conditions": [
        ("stage", "startswith", "2"), ("technical_features.volume_ratio", ">=", 1.4)]},
    {"label": "Stage 2 + RS + Volume", "conditions": [
        ("stage", "startswith", "2"), ("technical_features.rs_percentile", ">=", 80),
        ("technical_features.volume_ratio", ">=", 1.4)]},
    {"label": "Fundamentals + Technical (Growth + RS)", "conditions": [
        ("fundamental_features.revenue_growth", ">=", 10), ("technical_features.rs_percentile", ">=", 80)]},
    {"label": "Institutional Ownership + RS", "conditions": [
        ("fundamental_features.institutional_ownership", ">=", 50), ("technical_features.rs_percentile", ">=", 80)]},
]


def _matches_conditions(observation: dict, conditions: list) -> bool:
    for path, op, target in conditions:
        val = get_feature_value(observation, path)
        if val is None:
            return False
        try:
            if op == ">=":
                if not (float(val) >= target):
                    return False
            elif op == "<=":
                if not (float(val) <= target):
                    return False
            elif op == "==":
                if str(val) != str(target):
                    return False
            elif op == "startswith":
                if not str(val).startswith(str(target)):
                    return False
            else:
                return False
        except (TypeError, ValueError):
            return False
    return True


def compute_combination_alpha(observations: list, horizon: str, conditions: list) -> dict:
    filtered = [o for o in observations or [] if _matches_conditions(o, conditions)]
    metrics = compute_alpha_metrics(filtered, horizon)
    metrics["conditions"] = conditions
    return metrics


def get_combination_alpha_table(observations: list, horizon: str = "20D") -> list:
    """Runs every PRESET_COMBINATIONS entry and returns one row each —
    the Phase 6 answer to V9 spec §16's combination-testing section."""
    rows = []
    for combo in PRESET_COMBINATIONS:
        metrics = compute_combination_alpha(observations, horizon, combo["conditions"])
        metrics["label"] = combo["label"]
        rows.append(metrics)
    return rows


# ══════════════════════════════════════════════════════════════════════════
# V9 PHASE 7 SUPPORT — OVERVIEW & RESEARCH FINDINGS
#
# Pure computation feeding ui/alpha_lab.py. Findings are generated live
# from current data on every call — nothing here is persisted as a
# standing "finding #027" registry (V9 spec §39 sketches
# data/alpha_findings.json as a possible future store; deliberately not
# built yet, since a finding is fully re-derivable from the observations
# + outcomes that already ARE persisted, and adding a second store that
# must be kept in sync with the first is exactly the kind of premature
# infrastructure the spec's own §49 warns against). If this becomes a
# real UI need (e.g. a finding a person wants to comment on or pin),
# that's the trigger to add persistence — not before.
# ══════════════════════════════════════════════════════════════════════════

def compute_alpha_lab_overview(observations: list, horizon: str = "20D") -> dict:
    resolved = [o for o in observations or [] if o.get("outcomes")]
    overall = compute_alpha_metrics(observations, horizon)

    setups = [s for s in compute_alpha_metrics_by_setup(observations, horizon) if s["n"] >= 10]
    best_setup = max(setups, key=lambda s: s["expectancy_%"]) if setups else None

    feature_ranks = [f for f in rank_feature_alpha(observations, horizon) if f["is_conclusive"]]
    best_feature = feature_ranks[0] if feature_ranks else None
    worst_feature = feature_ranks[-1] if feature_ranks else None

    return {
        "horizon": horizon,
        "total_observations": len(observations or []),
        "resolved_observations": len(resolved),
        "unresolved_observations": len(observations or []) - len(resolved),
        "overall": overall,
        "best_setup": best_setup,
        "best_feature": best_feature,
        "worst_feature": worst_feature,
    }


def generate_research_findings(observations: list, horizon: str = "20D", min_n: int = 10, top_k: int = 8) -> list:
    """
    Auto-generates plain-language findings from setups, feature buckets,
    and preset combinations that meet `min_n` — per V9 spec §29's
    "Finding #027"-style format, minus the persistent numbering (see
    module docstring above for why). Every finding carries a `source`
    dict pointing back to exactly what produced it, so nothing here is a
    black-box claim — a reader can always trace a finding to the
    underlying setup_id / feature bucket / combination.
    """
    candidates = []

    for s in compute_alpha_metrics_by_setup(observations, horizon):
        if s["n"] < min_n or s["expectancy_%"] is None:
            continue
        candidates.append({
            "statement": (f"{s['setup_label']} setups have historically produced "
                         f"{s['expectancy_%']:+.1f}% average {horizon} return "
                         f"(win rate {s['win_rate_%']:.0f}%, n={s['n']})."),
            "n": s["n"], "expectancy_%": s["expectancy_%"], "win_rate_%": s["win_rate_%"],
            "evidence": s["sample_classification"], "source": {"type": "setup", "key": s["setup_id"]},
        })

    for key in FEATURE_REGISTRY:
        for b in compute_alpha_metrics_by_feature(observations, horizon, key):
            if b["n"] < min_n or b["expectancy_%"] is None:
                continue
            is_range = "bucket_lo" in b
            descriptor = f"in the {b['bucket_label']} range" if is_range else f"of '{b['bucket_label']}'"
            candidates.append({
                "statement": (f"{b['feature_label']} {descriptor} has "
                             f"historically been associated with {b['expectancy_%']:+.1f}% "
                             f"average {horizon} return (win rate {b['win_rate_%']:.0f}%, n={b['n']})."),
                "n": b["n"], "expectancy_%": b["expectancy_%"], "win_rate_%": b["win_rate_%"],
                "evidence": b["sample_classification"],
                "source": {"type": "feature", "key": key, "bucket": b["bucket_label"]},
            })

    for c in get_combination_alpha_table(observations, horizon):
        if c["n"] < min_n or c["expectancy_%"] is None:
            continue
        candidates.append({
            "statement": (f"{c['label']} has historically produced {c['expectancy_%']:+.1f}% "
                         f"average {horizon} return (win rate {c['win_rate_%']:.0f}%, n={c['n']})."),
            "n": c["n"], "expectancy_%": c["expectancy_%"], "win_rate_%": c["win_rate_%"],
            "evidence": c["sample_classification"], "source": {"type": "combination", "key": c["label"]},
        })

    candidates.sort(key=lambda c: (c["n"], abs(c["expectancy_%"])), reverse=True)
    return candidates[:top_k]
