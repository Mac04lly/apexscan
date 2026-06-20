"""
modules/ai_briefing.py — AI Daily Briefing via Claude API
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

BRIEFINGS_DIR = "data/briefings"
Path(BRIEFINGS_DIR).mkdir(parents=True, exist_ok=True)


def generate_briefing(df: pd.DataFrame, sector_df: pd.DataFrame = None, save: bool = True) -> str:
    """Generate a morning market briefing using Claude API."""

    # Build summary of top setups
    if df.empty:
        return "No scan data available. Run a Live Scan first."

    top = df.head(10)
    setups = []
    for _, row in top.iterrows():
        setups.append(
            f"- {row.get('ticker','?')}: Score={row.get('apex_score','?'):.0f}, "
            f"Stage={row.get('stage','?')}, 3M={row.get('perf_3m_%','?'):.1f}%, "
            f"RS={row.get('rs_3m','?'):.0f}, OF={row.get('of_bias','?')}, "
            f"VWAP={row.get('vwap_position','?')}, Pattern={row.get('pattern','?')}, "
            f"PA={row.get('pa_patterns','None')}"
        )

    breakouts = df[df.get("breaking_out", pd.Series([False]*len(df))) == True]["ticker"].tolist() if "breaking_out" in df.columns else []
    stage2_count = int(df["stage"].str.contains("2 ✅", na=False).sum()) if "stage" in df.columns else 0

    sector_text = ""
    if sector_df is not None and not sector_df.empty:
        try:
            top3 = sector_df.nlargest(3, "1W %")["Sector"].tolist()
            bot3 = sector_df.nsmallest(3, "1W %")["Sector"].tolist()
            sector_text = f"\nSector rotation — flowing INTO: {', '.join(top3)}. Flowing OUT OF: {', '.join(bot3)}."
        except Exception:
            sector_text = ""

    prompt = f"""You are a professional stock market analyst writing a concise morning briefing for a momentum trader.

Today's scan found {len(df)} setups passing filters. {stage2_count} are in Stage 2 uptrends.
{f"Active breakouts: {', '.join(breakouts)}" if breakouts else "No active breakouts today."}
{sector_text}

Top 10 setups by Apex Score:
{chr(10).join(setups)}

Write a professional morning briefing covering:
1. Overall market conditions based on the scan (2-3 sentences)
2. Top 3 highest-priority setups and exactly why they stand out
3. Key themes showing momentum
4. One specific risk reminder relevant to today's data
5. A one-sentence action plan for the day

Keep it concise, actionable, and direct. Use ** for bold. No fluff."""

    try:
        api_key = _get_api_key()
        if not api_key:
            return _fallback_briefing(df, breakouts, stage2_count, sector_text)

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        briefing = data["content"][0]["text"]

    except Exception as e:
        briefing = _fallback_briefing(df, breakouts, stage2_count, sector_text)

    if save and briefing:
        filename = f"{BRIEFINGS_DIR}/briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(briefing)

    return briefing


def _get_api_key() -> str:
    """Get Claude API key from Streamlit secrets or environment."""
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            key = st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("anthropic_api_key", "")
            if key:
                return key
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")


def _fallback_briefing(df, breakouts, stage2_count, sector_text) -> str:
    """Generate a rule-based briefing when Claude API is unavailable."""
    top3 = df.head(3)
    lines = [
        f"**ApexScan Morning Briefing — {datetime.now().strftime('%A, %B %d %Y')}**\n",
        f"**Market Conditions:** Today's scan found {len(df)} setups passing filters, "
        f"with {stage2_count} stocks in confirmed Stage 2 uptrends. "
        f"{'Breakout conditions are active.' if breakouts else 'No active breakouts — patience required.'}"
    ]
    if sector_text:
        lines.append(f"\n**Sector Rotation:**{sector_text}")
    lines.append("\n**Top Setups:**")
    for _, row in top3.iterrows():
        lines.append(
            f"- **{row.get('ticker','?')}** (Score: {row.get('apex_score',0):.0f}) — "
            f"{row.get('stage','?')} | {row.get('of_bias','?')} flow | "
            f"{row.get('vwap_position','?')} | Pattern: {row.get('pattern','?')}"
        )
    if breakouts:
        lines.append(f"\n**🚀 Active Breakouts:** {', '.join(breakouts)} — monitor closely for volume confirmation.")
    lines.append("\n**⚠️ Risk Reminder:** Never risk more than 1-2% of account on any single trade. Set stops immediately on entry.")
    lines.append("\n**Action Plan:** Focus only on Stage 2 stocks with high RS scores and confirmed order flow bias.")
    lines.append("\n\n*Note: Add ANTHROPIC_API_KEY to Streamlit secrets for AI-generated briefings.*")
    return "\n".join(lines)


def load_latest_briefing() -> str:
    """Load the most recent saved briefing."""
    try:
        files = sorted(Path(BRIEFINGS_DIR).glob("briefing_*.txt"), reverse=True)
        if files:
            with open(files[0], encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return ""
