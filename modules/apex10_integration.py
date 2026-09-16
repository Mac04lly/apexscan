"""
modules/apex10_integration.py — Apex the Great X: live scan wiring

This is the piece that makes Stages C/D/E actually produce real radar
entries instead of sitting as untested-in-production backend code.
Called from scanner.run_scan(), AFTER the existing scan has already
fetched history and finalized its results — reuses that data
completely. This module makes ZERO network calls of its own.

════════════════════════════════════════════════════════════════════════
SAFETY CONTRACT — this function is called from inside the live,
already-working scan path. It must be structurally incapable of
breaking that scan.
════════════════════════════════════════════════════════════════════════
- Off by default: requires cfg["apex10"]["enabled"] == True. Absent
  config, or apex10 modules failing to import for any reason, both
  return None immediately — the caller (scanner.run_scan) wraps this in
  its own try/except too, matching the exact defensive pattern already
  used there for the optional AI-enrichment block, so this is
  belt-and-suspenders, not the only safety net.
- Per-ticker computation is individually wrapped — one ticker's feature
  computation raising an exception skips that ticker only, never aborts
  the batch.
- Persistence (GitHub write) happens once for the whole batch via
  modules.apex10_tracker.run_daily_radar_update(), not once per ticker.

════════════════════════════════════════════════════════════════════════
WHY CANDIDATES = `results` (the scan's own filtered output), NOT THE
FULL FETCHED UNIVERSE
════════════════════════════════════════════════════════════════════════
`batch_hist` (passed in from run_scan) actually contains every ticker
in the scanned universe, not just the ones that passed filters — since
computing apex10 features is pure CPU (no network calls), scanning the
full universe would cost nothing in API risk. It was deliberately NOT
done that way for this first wiring: persisting a radar entry for every
ticker in the universe every day (hundreds of mostly-DISTANT-state
entries) would bloat data/alpha_observations.json with noise, and
that's a real cost (storage, GitHub push size, every future Alpha Lab
query getting slower) for close to zero benefit. Starting from `results`
— the scan's own already-filtered, already-useful population — is the
conservative first step. Widening this to the full universe is a valid
FUTURE decision, not something to default into silently.
"""
from __future__ import annotations
import logging
from typing import Optional

log = logging.getLogger("apexscan.apex10_integration")

DEFAULT_MIN_RADAR_SCORE = 50  # only worth tracking if at least "developing"-ish


def process_scan_for_radar(results: list, batch_hist: dict, benchmark_close,
                           cfg: dict, market: str = "US") -> Optional[dict]:
    """
    For every ticker in `results`: computes Stage C features
    (as_of_date=None — "live, as of today") from the SAME hist already
    in `batch_hist`, scores it via Stage D, and creates/updates its
    radar entry via Stage E. One persistence call for the whole batch.

    Returns a summary dict, or None if apex10 is disabled/unavailable —
    the caller can safely ignore a None return.
    """
    apex10_cfg = (cfg or {}).get("apex10", {})
    if not apex10_cfg.get("enabled", False):
        return None

    try:
        from modules.apex10_features import compute_precursor_features
        from modules.apex10_radar import (
            compute_apex10_score, compute_liquidity_gate, compute_breakout_trigger_gates,
        )
        from modules.apex10_tracker import run_daily_radar_update
    except Exception as e:
        log.warning(f"Apex the Great X modules unavailable, skipping radar update: {e}")
        return None

    if benchmark_close is None or len(benchmark_close) == 0:
        log.warning("Apex the Great X: no benchmark data available, skipping radar update.")
        return None

    min_radar_score = apex10_cfg.get("min_radar_score", DEFAULT_MIN_RADAR_SCORE)
    candidates = []
    skipped_no_hist = 0
    skipped_error = 0

    for data in results or []:
        ticker = data.get("ticker")
        if not ticker:
            continue
        hist = batch_hist.get(ticker) if batch_hist else None
        if hist is None or len(hist) < 21:
            skipped_no_hist += 1
            continue

        try:
            features = compute_precursor_features(hist, benchmark_close, as_of_date=None)
            if "relative_strength" not in features:
                continue

            score_result = compute_apex10_score(features)
            score = score_result.get("score")
            if score is None or score < min_radar_score:
                continue

            current_price = float(hist["Close"].iloc[-1])
            liquidity_gate = compute_liquidity_gate(
                features.get("volume", {}).get("avg_volume_20"), current_price, market=market)
            trigger_gates = compute_breakout_trigger_gates(features, liquidity_gate)

            candidates.append({
                "ticker": ticker, "current_price": current_price,
                "features": features, "score_result": score_result,
                "liquidity_gate": liquidity_gate, "trigger_gates": trigger_gates,
            })
        except Exception as e:
            skipped_error += 1
            log.debug(f"Apex the Great X feature computation failed for {ticker}: {e}")
            continue

    summary = {
        "candidates_evaluated": len(results or []), "candidates_qualified": len(candidates),
        "skipped_no_hist": skipped_no_hist, "skipped_error": skipped_error,
        "min_radar_score": min_radar_score,
    }

    if not candidates:
        summary.update({"created": 0, "updated": 0})
        return summary

    try:
        batch_result = run_daily_radar_update(candidates, market=market)
        summary.update(batch_result)
    except Exception as e:
        log.warning(f"Apex the Great X radar persistence failed: {e}")
        summary["persistence_error"] = str(e)

    return summary


def process_short_scan_for_radar(results: list, batch_hist: dict, benchmark_close,
                                 cfg: dict, market: str = "US") -> Optional[dict]:
    """
    Short-side mirror of process_scan_for_radar() — same safety contract,
    same reasoning, same "off by default / per-ticker isolation / one
    persistence call per batch" structure, so not re-derived in full
    here (see this module's docstring and process_scan_for_radar above).

    Gated on the SAME cfg["apex10"]["enabled"] flag as the long side —
    confirmed against ui/alpha_lab.py's own short-radar empty-state text
    ("accumulate automatically once apex10.enabled is true... nothing
    else needed"), which already promised this contract before this
    function existed to fulfill it.

    `results` here is run_short_scan()'s OWN filtered output (scored by
    its independent legacy short_score / analyze_stock_short formula,
    completely unrelated scoring logic). Exactly like the long side,
    this function does NOT use that score for anything — it recomputes
    fresh Stage-C/D short features and an apex10_short_score for each
    candidate ticker and persists THAT, so the radar always reflects
    the apex10_short_radar model's own opinion, not the legacy scanner's.
    """
    apex10_cfg = (cfg or {}).get("apex10", {})
    if not apex10_cfg.get("enabled", False):
        return None

    try:
        from modules.apex10_short_features import compute_short_precursor_features
        from modules.apex10_short_radar import (
            compute_apex10_short_score, compute_breakdown_trigger_gates,
        )
        from modules.apex10_radar import compute_liquidity_gate
        from modules.apex10_short_tracker import run_daily_short_radar_update
    except Exception as e:
        log.warning(f"Apex the Great X (short) modules unavailable, skipping radar update: {e}")
        return None

    if benchmark_close is None or len(benchmark_close) == 0:
        log.warning("Apex the Great X (short): no benchmark data available, skipping radar update.")
        return None

    min_radar_score = apex10_cfg.get("min_radar_score", DEFAULT_MIN_RADAR_SCORE)
    candidates = []
    skipped_no_hist = 0
    skipped_error = 0

    for data in results or []:
        ticker = data.get("ticker")
        if not ticker:
            continue
        hist = batch_hist.get(ticker) if batch_hist else None
        if hist is None or len(hist) < 21:
            skipped_no_hist += 1
            continue

        try:
            features = compute_short_precursor_features(hist, benchmark_close, as_of_date=None)
            if "relative_strength" not in features:
                continue

            score_result = compute_apex10_short_score(features)
            score = score_result.get("score")
            if score is None or score < min_radar_score:
                continue

            current_price = float(hist["Close"].iloc[-1])
            liquidity_gate = compute_liquidity_gate(
                features.get("volume", {}).get("avg_volume_20"), current_price, market=market)
            trigger_gates = compute_breakdown_trigger_gates(features, liquidity_gate)

            candidates.append({
                "ticker": ticker, "current_price": current_price,
                "features": features, "score_result": score_result,
                "liquidity_gate": liquidity_gate, "trigger_gates": trigger_gates,
            })
        except Exception as e:
            skipped_error += 1
            log.debug(f"Apex the Great X (short) feature computation failed for {ticker}: {e}")
            continue

    summary = {
        "candidates_evaluated": len(results or []), "candidates_qualified": len(candidates),
        "skipped_no_hist": skipped_no_hist, "skipped_error": skipped_error,
        "min_radar_score": min_radar_score,
    }

    if not candidates:
        summary.update({"created": 0, "updated": 0})
        return summary

    try:
        batch_result = run_daily_short_radar_update(candidates, market=market)
        summary.update(batch_result)
    except Exception as e:
        log.warning(f"Apex the Great X (short) radar persistence failed: {e}")
        summary["persistence_error"] = str(e)

    return summary
