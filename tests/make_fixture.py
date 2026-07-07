"""
tests/make_fixture.py

Generates a small, realistic fake scan report and drops it into reports/
so the smoke test (and manual local testing) can exercise every tab's
data-driven code paths WITHOUT needing live API keys or a real scan.

Run this once before running the smoke test:
    python tests/make_fixture.py

It writes reports/scan_99999999_9999.csv (a date far in the future so it's
always picked up as the "latest" report by load_latest_report(), and is
never confused with a real scan). Safe to delete any time — it does not
touch your real scan history.
"""
import sys
import random
from pathlib import Path
import pandas as pd

random.seed(7)

# Every column ApexScan's COLUMN_META references, so every tab that reads
# scan data (Leaderboard, Deep Dive, Interpretation, Pre-Buy Checklist,
# Correction Watchlist, Watchlists, Scan Delta) has real values to work with.
TICKERS = ["NVDA", "AAPL", "HIMS", "RKLB", "SMCI", "PLTR", "MSTR", "DUOL"]

def _row(i, tk):
    is_gem = tk in ("HIMS", "RKLB")
    stage = "Stage 2 ✅ (Uptrend)" if i % 3 != 0 else "Stage 1 ⏳ (Basing)"
    return {
        "rank": i + 1,
        "ticker": tk,
        "market": "US",
        "theme": "ai_semis" if i % 2 == 0 else "biotech_health",
        "price": round(50 + i * 17.3, 2),
        "stage": stage,
        "perf_1m_%": round(random.uniform(-5, 15), 1),
        "perf_3m_%": round(random.uniform(-10, 45), 1),
        "perf_6m_%": round(random.uniform(-15, 80), 1),
        "rs_3m": round(random.uniform(40, 180), 0),
        "rs_6m": round(random.uniform(40, 180), 0),
        "rs_r2500_3m": round(random.uniform(40, 160), 0),
        "rs_r2500_6m": round(random.uniform(40, 160), 0),
        "rs_r3000g_3m": round(random.uniform(40, 160), 0),
        "rs_r3000g_6m": round(random.uniform(40, 160), 0),
        "rs_multi_leader": i == 0,
        "adr_%": round(random.uniform(1.5, 8), 1),
        "vs_50ma_%": round(random.uniform(-5, 20), 1),
        "vs_200ma_%": round(random.uniform(-10, 40), 1),
        "volume": random.randint(500_000, 20_000_000),
        "vol_filter": random.randint(500_000, 20_000_000),
        "vol_surge_x": round(random.uniform(0.8, 3.2), 1),
        "above_50ma": True,
        "above_200ma": True,
        "ma50_gt_ma200": True,
        "near_52wh": i % 2 == 0,
        "pct_off_high_%": round(random.uniform(-30, 0), 1),
        "pattern": random.choice(["Flat Base Breakout", "Tight Base", "Handle Forming", "Pullback to 50MA", "None"]),
        "breaking_out": i == 0,
        "news_count": random.randint(0, 12),
        "sentiment": random.choice(["Positive", "Neutral", "N/A"]),
        "earn_momentum": random.choice(["Strong", "Moderate", "Weak"]),
        "eps_growth_%": round(random.uniform(-10, 60), 1),
        "eps_surprise_%": round(random.uniform(-5, 25), 1),
        "eps_accel": i % 2 == 0,
        "consec_beats": random.randint(0, 5),
        "rev_growth_%": round(random.uniform(0, 40), 1),
        "eps_score": random.randint(0, 15),
        "eps_trend": "1.2 -> 1.4 -> 1.6 -> 1.9",
        "analyst_target": round(50 + i * 17.3 * 1.15, 2),
        "pe_ratio": round(random.uniform(15, 90), 1),
        "peg_ratio": round(random.uniform(0.5, 2.5), 2),
        "eps_details": "Q1 1.2 | Q2 1.4 | Q3 1.6 | Q4 1.9",
        "next_earnings": "2026-08-15",
        "of_bias": random.choice(["Strong Bullish", "Bullish", "Neutral", "Bearish"]),
        "of_up_vol_ratio": round(random.uniform(0.6, 2.2), 2),
        "of_bullish_days": round(random.uniform(30, 80), 0),
        "of_consec_up": random.randint(0, 6),
        "of_score": random.randint(0, 8),
        "vwap": round(48 + i * 17.3, 2),
        "vwap_upper": round(52 + i * 17.3, 2),
        "vwap_lower": round(44 + i * 17.3, 2),
        "vs_vwap_%": round(random.uniform(-5, 8), 1),
        "vwap_position": random.choice(["Above VWAP", "Extended Above VWAP", "Below VWAP"]),
        "vwap_slope": random.choice(["Rising", "Flat", "Falling"]),
        "vwap_score": random.randint(0, 4),
        "ms_structure": random.choice(["Bullish (HH/HL)", "Bearish (LH/LL)", "Transitioning"]),
        "ms_hh_hl": i % 2 == 0,
        "ms_bos": i == 0,
        "ms_swing_high": round(55 + i * 17.3, 2),
        "ms_swing_low": round(45 + i * 17.3, 2),
        "pa_patterns": random.choice(["Bullish SFP (Bear Trap)", "Bullish Engulfing", "None", "PA Confluence"]),
        "pa_engulfing": random.choice(["Bullish", "None"]),
        "pa_sfp": random.choice(["Bullish SFP (Bear Trap)", "None"]),
        "pa_inside_day": i % 3 == 0,
        "pa_context": random.choice(["Bullish", "None"]),
        "pa_score": random.randint(0, 5),
        "weekly_stage": stage,
        "weekly_above_10wma": True,
        "weekly_above_40wma": True,
        "weekly_10gt40": True,
        "weekly_rs": round(random.uniform(40, 180), 0),
        "weekly_base_tight": i % 3 == 0,
        "weekly_base_depth_%": round(random.uniform(5, 30), 0),
        "weekly_hh_hl": i % 2 == 0,
        "weekly_trending_up": True,
        "weekly_consec_up_wks": random.randint(0, 5),
        "weekly_confirmed": i % 3 != 0,
        "weekly_contradicts": i % 5 == 0,
        "weekly_score": random.randint(-15, 10),
        "early_entry": i % 3 == 0,
        "early_entry_type": random.choice(["Fresh 200MA Cross", "Pullback to 50MA", "Low-ADR Base", "–"]),
        "fresh_200ma_cross": i == 1,
        "fresh_50ma_cross": i == 2,
        "pullback_to_50ma": i == 3,
        "low_adr_base": i % 3 == 0,
        "early_entry_score": random.randint(0, 10),
        "days_since_200ma_cross": random.randint(1, 40),
        "apex_score": round(random.uniform(30, 95), 0),
        "scanned_at": "2026-07-05 09:31:00",
        "market_cap": random.randint(200_000_000, 900_000_000_000),
        "market_cap_bn": round(random.uniform(0.2, 900), 1),
        "mcap_category": random.choice(["Micro Cap", "Small Cap", "Mid Cap", "Large Cap", "Mega Cap"]),
        "is_gem": is_gem,
        "liquidity_score": random.randint(0, 3),
        "liquidity_warn": False,
        "avg_volume_30d": random.randint(300_000, 15_000_000),
        "changes": "First scan",
        "is_new": True,
        "delta_score": None,
    }

def main():
    rows = [_row(i, tk) for i, tk in enumerate(TICKERS)]
    fixture_df = pd.DataFrame(rows)

    repo_root = Path(__file__).resolve().parent.parent
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    out_path = reports_dir / "scan_99999999_9999.csv"
    fixture_df.to_csv(out_path, index=False)
    print(f"✅ Wrote fixture scan with {len(fixture_df)} tickers to: {out_path}")
    print("   This file sorts last alphabetically/by-date, so load_latest_report()")
    print("   will pick it up as the most recent scan during smoke testing.")

    # ── Second fixture scan, with a COMPLETELY DIFFERENT ticker set ──────
    # This deliberately recreates the exact scenario that crashed the Scan
    # Delta tab: when comparing two scans where every single ticker is
    # either brand new or dropped out, the "Δ Score" column ends up all
    # None. Without a dtype fix, pandas infers dtype=object for that whole
    # column, and .nlargest()/.nsmallest() raise a TypeError even on an
    # empty column. Keep this second fixture around specifically so that
    # regression can never silently come back.
    OTHER_TICKERS = ["COIN", "SHOP", "ABNB", "UBER"]
    other_rows = [_row(i, tk) for i, tk in enumerate(OTHER_TICKERS)]
    other_df = pd.DataFrame(other_rows)
    out_path_2 = reports_dir / "scan_99999999_9998.csv"
    other_df.to_csv(out_path_2, index=False)
    print(f"✅ Wrote second fixture scan (non-overlapping tickers) to: {out_path_2}")
    print("   Together these two files let the smoke test exercise the Scan Delta")
    print("   tab's 'all tickers are new/dropped' edge case.")
    print()
    print("   Delete both any time with:")
    print("     rm reports/scan_99999999_9999.csv reports/scan_99999999_9998.csv")

if __name__ == "__main__":
    main()
