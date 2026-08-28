"""ai/prompts.py — prompt templates. Never includes secrets."""
from __future__ import annotations
from typing import Dict, List


def build_market_brief_prompt(diagnostics: dict, sector_rotation: List[Dict], regime: str) -> str:
    sectors_txt = "\n".join(
        f"- {s['theme']}: avg score {s['avg_score']}/100 across {s['count']} stocks"
        for s in sector_rotation
    ) or "No sector data available."

    return f"""You are a market analyst writing a short, plain-English brief for a stock scanner's users.

Scan breadth: {diagnostics.get('passed', 0)} of {diagnostics.get('scanned', 0)} scanned stocks passed all filters.
Rule-based market regime classification: {regime}

Sector performance this scan (by average Apex Score):
{sectors_txt}

Write a concise (120-180 word) market brief covering:
1. Overall market tone based on the regime and breadth above
2. Which sectors are leading vs lagging
3. One suggested strategy focus for today (swing trading, long-term accumulation, or defensive/wait)

Do not invent data not given above. Do not give specific buy/sell instructions for individual named stocks."""


def build_stock_analysis_prompt(stock: Dict) -> str:
    return f"""You are analysing one stock from a deterministic technical/fundamental scan. All figures below come directly from the scan — do not invent additional data.

Ticker: {stock.get('ticker')}
Theme/Sector: {stock.get('theme')}
Apex Score: {stock.get('apex_score')}/100
Stage: {stock.get('stage')}
3-month return: {stock.get('perf_3m_%')}%
Relative Strength vs S&P 500: {stock.get('rs_3m')}
Order flow bias: {stock.get('of_bias')}
VWAP position: {stock.get('vwap_position')}
Pattern: {stock.get('pattern')}
Breaking out: {stock.get('breaking_out')}
Earnings momentum: {stock.get('earn_momentum')}
EPS growth: {stock.get('eps_growth_%')}

Write a 3-4 sentence AI summary covering: key strength, key weakness/risk, and a recommendation (Strong Buy / Buy / Watch / Avoid) consistent with the Apex Score. Be specific to the numbers given, not generic."""


def build_decision_explanation_prompt(stock: Dict, evidence: Dict) -> str:
    """
    V9 Phase 11 — AI Explanation. Per the spec (§46): the AI layer comes
    AFTER the deterministic decision and validated evidence, and its job
    is to explain them in plain English — never to invent a price,
    fundamental, historical statistic, signal, or confidence level of
    its own. Every number in this prompt is pulled directly from
    scanner.py's output and modules/alpha_metrics.py's already-computed
    Alpha Lab evidence; nothing is estimated here.
    """
    setup_line = "No matching setup evidence yet (too new / not enough resolved observations)."
    if evidence.get("setup"):
        s = evidence["setup"]
        setup_line = (f"Setup type: {s.get('setup_label', s.get('setup_id'))}. Historical record for "
                     f"this setup at this horizon: {s.get('n', 0)} resolved observations, "
                     f"win rate {s.get('win_rate_%')}%, expectancy {s.get('expectancy_%')}%, "
                     f"sample classification: {s.get('sample_classification')}.")

    return f"""You are explaining ONE deterministic stock decision that ApexScan has already made, using ONLY the numbers given below. Do not invent, estimate, or infer any price, fundamental, historical statistic, signal, or confidence level that is not explicitly provided here — if something isn't given, say it isn't available yet rather than guessing.

Ticker: {stock.get('ticker')}
Apex Score: {stock.get('apex_score')}/100 (raw, uncapped: {stock.get('apex_score_raw')})
Stage: {stock.get('stage')}
{setup_line}

Write 3-5 sentences that:
1. State what the score and setup mean in plain English.
2. State how much historical evidence backs this specific setup (the sample size and classification above) — and if the evidence is "Too Small" or there isn't any yet, say so plainly rather than implying confidence that doesn't exist.
3. Do NOT issue a buy/sell recommendation — only explain what is already known and how reliable it is. This is an explanation of evidence, not new advice."""
