"""
modules/alpha_validation.py — Alpha Observation System (V9 Phase 1)

Captures an IMMUTABLE snapshot of every feature ApexScan already computes,
at the exact moment a stock qualifies as a discovery — so later research
into "does X actually predict returns" is never contaminated by hindsight.
Golden rule: an observation's fields are frozen at creation. If a stock's
RS/ROE/score changes later, that's a NEW fact belonging to a later check,
never a retroactive edit of this observation.

This is purely additive: a new persistent store (data/alpha_observations.json,
GitHub-backed via modules/gh_storage.py, same pattern as every other store
in this app), populated by a new hook called alongside — not instead of —
the existing Discovery Tracker logging. Nothing here reads from or writes
to discoveries.json, and nothing about the existing scanner/scoring/UI
changes as a result of this file existing.

Honesty note: a handful of the fields the V9 spec calls for aren't
computed anywhere in the current scanner (base_depth, a true market-wide
RS percentile, institutional-ownership CHANGE at the moment of
discovery). Rather than fabricate them, they're captured as None with the
gap documented in FIELD_COVERAGE below, or mapped from the closest real
existing proxy (documented per-field). Nothing here invents a number that
isn't real.
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd

from modules.gh_storage import load_json_from_github, save_json_to_github
from modules.model_registry import get_model_version, get_strategy_version

# Fields the V9 spec asks for that aren't available from the current
# scanner and are NOT approximated — always None, honestly, rather than
# guessed. Documented here so it's a known, visible gap, not a silent one.
FIELD_COVERAGE = {
    "base_depth": "Not computed anywhere in scanner.py yet — always None.",
    "rs_percentile": "Computed at observation time as this stock's percentile "
                      "rank of rs_3m within the SAME scan's results — not a "
                      "market-wide percentile, since ApexScan doesn't score "
                      "every US-listed stock every scan.",
    "institutional_ownership_change": "Pulled from fundamentals_history.json "
                      "if a prior snapshot exists for this ticker; None on a "
                      "stock's first-ever observation.",
    "volume_ratio": "Mapped from vol_surge_x (5-day vs 50-day average volume).",
    "volatility": "Mapped from adr_% (average daily range) as the closest "
                  "existing proxy — not a formal realized-volatility figure.",
}


def _get_gh_creds():
    """Alpha observations reuse the same GitHub credentials already
    configured for every other persistent store — pulled at call time
    from Streamlit secrets, not duplicated/hardcoded here."""
    try:
        import streamlit as st
        return st.secrets.get("github_token", ""), st.secrets.get("github_repo", "")
    except Exception:
        return "", ""


def derive_setup_id(row: dict) -> str:
    """
    Maps the EXISTING scanner signals to a standardized setup taxonomy —
    per the V9 spec's own instruction, derived from what the scanner
    already detects, not invented independently. Priority order mirrors
    generate_entry_plan()'s logic in dashboard.py, since that's already
    the established, tested read of "what kind of setup is this."
    """
    stage       = str(row.get("stage", ""))
    above200    = bool(row.get("above_200ma"))
    ma50gt200   = bool(row.get("ma50_gt_ma200"))
    breaking_out = bool(row.get("breaking_out"))
    vol_surge   = row.get("vol_surge_x")
    fresh200    = bool(row.get("fresh_200ma_cross"))
    fresh50     = bool(row.get("fresh_50ma_cross"))
    pullback50  = bool(row.get("pullback_to_50ma"))
    low_adr_base = bool(row.get("low_adr_base"))
    rs3         = row.get("rs_3m")

    if stage.startswith("3"):
        return "S3-TOP"
    if stage.startswith("4") or not above200:
        return "S4-BREAKDOWN"

    if breaking_out:
        try:
            return "S2-BREAKOUT-VOL" if (vol_surge is not None and float(vol_surge) > 1.4) else "S2-BREAKOUT-NOVOL"
        except (TypeError, ValueError):
            return "S2-BREAKOUT-NOVOL"
    if fresh200:
        return "S2-MA-RECLAIM"
    if pullback50 or fresh50:
        return "S2-PULLBACK-50"
    if rs3 is not None:
        try:
            if float(rs3) > 100 and ma50gt200:
                return "S2-HIGH-RS"
        except (TypeError, ValueError):
            pass
    if low_adr_base and stage.startswith("1"):
        return "S1-EARLY-ALPHA"
    if stage.startswith("1"):
        return "S1-BASE"
    if ma50gt200:
        return "S2-CONSOLIDATION"
    return "S1-BASE"


def _compute_rs_percentile(ticker: str, scan_df: Optional[pd.DataFrame]) -> Optional[float]:
    """This stock's percentile rank of rs_3m within the SAME scan's
    results — see FIELD_COVERAGE for why this isn't a market-wide figure."""
    if scan_df is None or scan_df.empty or "rs_3m" not in scan_df.columns:
        return None
    try:
        rs_vals = pd.to_numeric(scan_df["rs_3m"], errors="coerce").dropna()
        this_row = scan_df[scan_df["ticker"] == ticker]
        if this_row.empty or rs_vals.empty:
            return None
        this_rs = pd.to_numeric(this_row.iloc[0].get("rs_3m"), errors="coerce")
        if pd.isna(this_rs):
            return None
        return round((rs_vals < this_rs).mean() * 100, 1)
    except Exception:
        return None


def build_observation(row: dict, scan_df: Optional[pd.DataFrame] = None,
                      strategy: str = "swing", market_regime: Optional[str] = None,
                      fund_history: Optional[dict] = None) -> dict:
    """
    Builds one immutable AlphaObservation from a scan result row, at the
    exact moment it qualifies as a discovery. Every value here is either
    read directly from what the scanner already computed for this exact
    scan, or explicitly marked None per FIELD_COVERAGE — never guessed.
    """
    def _g(key, default=None):
        v = row.get(key, default)
        try:
            return default if (v is None or (isinstance(v, float) and pd.isna(v))) else v
        except Exception:
            return v if v is not None else default

    ticker = row.get("ticker")

    inst_change = None
    if fund_history and ticker in fund_history:
        prior = fund_history[ticker].get("fields", {})
        prior_inst = prior.get("institutional_ownership")
        cur_inst = _g("institutional_ownership")
        if prior_inst is not None and cur_inst is not None:
            try:
                inst_change = round(float(cur_inst) - float(prior_inst), 4)
            except (TypeError, ValueError):
                pass

    technical_features = {
        "perf_3m_%": _g("perf_3m_%"), "perf_1m_%": _g("perf_1m_%"), "perf_1w_%": _g("perf_1w_%"),
        "rs_3m": _g("rs_3m"), "rs_percentile": _compute_rs_percentile(ticker, scan_df),
        "stage": _g("stage"), "vs_50ma_%": _g("vs_50ma_%"), "vs_200ma_%": _g("vs_200ma_%"),
        "ma50_gt_ma200": _g("ma50_gt_ma200"),
        "volume_ratio": _g("vol_surge_x"), "volume_persistence": _g("of_score"),
        "vwap_position": _g("vwap_position"), "market_structure": _g("ms_structure"),
        "hh_hl": _g("ms_hh_hl"), "breakout_status": _g("breaking_out"),
        "base_depth": None,  # see FIELD_COVERAGE
        "adr_%": _g("adr_%"), "volatility": _g("adr_%"),  # see FIELD_COVERAGE
        "price_action_pattern": _g("pa_patterns"),
        "weekly_confirmation": _g("weekly_confirmed"),
        "early_entry_condition": _g("early_entry_type"),
    }
    fundamental_features = {
        "revenue_growth": _g("revenue_growth"), "earnings_growth": _g("earnings_growth"),
        "eps_growth_%": _g("eps_growth_%"), "roe": _g("roe"),
        "gross_margin": _g("gross_margin"), "operating_margin": _g("operating_margin"),
        "free_cash_flow": _g("free_cash_flow"), "debt_to_equity": _g("debt_to_equity"),
        "current_ratio": _g("current_ratio"), "quick_ratio": _g("quick_ratio"),
        "net_cash_position": _g("net_cash_position"),
        "institutional_ownership": _g("institutional_ownership"),
        "institutional_ownership_change": inst_change,
        "short_pct_float": _g("short_pct_float"), "beta": _g("beta"),
    }
    valuation_features = {
        "pe_ratio": _g("pe_ratio"), "forward_pe": _g("forward_pe"),
        "ev_to_ebitda": _g("ev_to_ebitda"), "peg_ratio": _g("peg_ratio"),
        "valuation_vs_sector_%": _g("valuation_vs_sector_%"),
    }
    risk_features = {
        "risk_score": _g("risk_score"), "risk_label": _g("risk_label"),
        "liquidity_warn": _g("liquidity_warn"), "avg_volume_30d": _g("avg_volume_30d"),
    }
    market_features = {
        "market_regime": market_regime,
        "theme": _g("theme"), "mcap_category": _g("mcap_category"),
        "market": _g("market", "US"),
    }

    return {
        "observation_id": str(uuid.uuid4()),
        "ticker": ticker,
        "market": _g("market", "US"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": get_model_version(),
        "strategy_version": get_strategy_version(strategy),
        "apex_score": _g("apex_score"),
        "apex_score_raw": _g("apex_score_raw"),
        "stage": _g("stage"),
        "setup_id": derive_setup_id(row),
        "strategy": strategy,
        "market_regime": market_regime,
        "technical_features": technical_features,
        "fundamental_features": fundamental_features,
        "valuation_features": valuation_features,
        "risk_features": risk_features,
        "market_features": market_features,
        "entry_price": _g("price"),
        "stop_price": None,      # left for a future phase's caller to populate
        "target_price": None,
        "risk_reward": None,
        "benchmark_price": None,
    }


def load_observations() -> list:
    token, repo = _get_gh_creds()
    if token and repo:
        data, _ = load_json_from_github(token, repo, "data/alpha_observations.json")
        if isinstance(data, list):
            return data
    return []


def save_observations(observations: list):
    token, repo = _get_gh_creds()
    if token and repo:
        save_json_to_github(token, repo, "data/alpha_observations.json", observations,
                             message=f"Update alpha_observations.json ({len(observations)} observations)")


def log_new_observations(scan_df: pd.DataFrame, strategy: str = "swing",
                         market_regime: Optional[str] = None,
                         fund_history: Optional[dict] = None) -> int:
    """
    Records one immutable AlphaObservation per NEW discovery in this scan
    (a ticker already observed is skipped — an observation is captured
    once, at first discovery, never updated). Returns the count of new
    observations added. Never raises — a failure here must never break
    the scan or Discovery Tracker logging it runs alongside.
    """
    if scan_df is None or scan_df.empty:
        return 0
    try:
        existing = load_observations()
        existing_tickers = {o["ticker"] for o in existing}
        new_obs = []
        for _, row in scan_df.iterrows():
            rd = row.to_dict()
            ticker = rd.get("ticker")
            if not ticker or ticker in existing_tickers:
                continue
            new_obs.append(build_observation(rd, scan_df, strategy, market_regime, fund_history))
            existing_tickers.add(ticker)
        if new_obs:
            save_observations(existing + new_obs)
        return len(new_obs)
    except Exception:
        return 0
