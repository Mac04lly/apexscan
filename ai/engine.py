"""
ai/engine.py — ApexScan AI Investment Intelligence Engine (Phase 4, core)

The AI layer NEVER replaces the scanner. It only interprets results the
deterministic scanner already produced. If disabled, unconfigured, or if
any call fails, the scanner continues to work exactly as before — nothing
here is allowed to raise up into scanner.py.

Cost note: this engine makes ZERO API calls, and therefore costs nothing,
unless BOTH of the following are true in config.yaml:
  ai_enabled: true
  openai_api_key: <a real key, not the YOUR_... placeholder>
Out of the box (ai_enabled: false) this entire module is inert.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .providers import OpenAIProvider
from .prompts import build_market_brief_prompt, build_stock_analysis_prompt, build_decision_explanation_prompt
from .cache import AICache

log = logging.getLogger("apexscan.ai")


class InvestmentIntelligenceEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.enabled = self._check_enabled()
        self.cache = AICache()
        self.provider: Optional[OpenAIProvider] = None
        if self.enabled:
            self.provider = OpenAIProvider(
                api_key=self.cfg.get("openai_api_key", ""),
                model=self.cfg.get("ai_model", "gpt-4o-mini"),
                temperature=self.cfg.get("ai_temperature", 0.2),
                max_tokens=self.cfg.get("ai_max_tokens", 600),
            )

    def _check_enabled(self) -> bool:
        if not self.cfg.get("ai_enabled", False):
            return False
        key = self.cfg.get("openai_api_key", "") or ""
        if not key or key.startswith("YOUR_"):
            log.info("AI disabled: ai_enabled=true but no valid openai_api_key configured.")
            return False
        return True

    def market_brief(self, diagnostics_summary: dict, results_df: Optional[pd.DataFrame] = None) -> str:
        if not self.enabled or not self.provider:
            return ""
        try:
            regime = classify_market_regime(results_df)
            sector_rotation = summarise_sector_rotation(results_df)
            cache_key = self.cache.key("market_brief", diagnostics_summary, sector_rotation, regime)
            cached = self.cache.get(cache_key)
            if cached:
                return cached
            prompt = build_market_brief_prompt(diagnostics_summary, sector_rotation, regime)
            text = self.provider.complete(prompt)
            if text:
                self.cache.set(cache_key, text)
            return text or ""
        except Exception as e:
            log.warning(f"Market brief generation failed, continuing without it: {e}")
            return ""

    def analyze_stock(self, stock: Dict[str, Any]) -> str:
        if not self.enabled or not self.provider:
            return ""
        try:
            cache_key = self.cache.key("stock", stock.get("ticker"), stock.get("apex_score"),
                                        stock.get("scanned_at"))
            cached = self.cache.get(cache_key)
            if cached:
                return cached
            prompt = build_stock_analysis_prompt(stock)
            text = self.provider.complete(prompt)
            if text:
                self.cache.set(cache_key, text)
            return text or ""
        except Exception as e:
            log.debug(f"AI stock analysis failed for {stock.get('ticker')}: {e}")
            return ""

    def explain_decision(self, stock: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        """
        V9 Phase 11 — explains an already-made deterministic decision
        using only the real score/setup/evidence numbers passed in
        `evidence` (built by ui/alpha_lab.py from modules/alpha_metrics.py
        output). Architecture per spec §46:

            DATA -> DETERMINISTIC ANALYSIS -> VALIDATED MODEL ->
            APEX DECISION -> AI EXPLANATION

        Never the reverse (DATA -> LLM -> BUY). If disabled/unconfigured,
        returns "" exactly like every other method here — the decision
        and its evidence remain fully visible in the UI without this;
        this only adds a plain-English narration of numbers that already
        exist.
        """
        if not self.enabled or not self.provider:
            return ""
        try:
            cache_key = self.cache.key("explain", stock.get("ticker"), stock.get("apex_score_raw"),
                                        (evidence.get("setup") or {}).get("n"))
            cached = self.cache.get(cache_key)
            if cached:
                return cached
            prompt = build_decision_explanation_prompt(stock, evidence)
            text = self.provider.complete(prompt)
            if text:
                self.cache.set(cache_key, text)
            return text or ""
        except Exception as e:
            log.debug(f"AI decision explanation failed for {stock.get('ticker')}: {e}")
            return ""


def classify_market_regime(results_df: Optional[pd.DataFrame]) -> str:
    """Rule-based regime classification from scan breadth. Free — no API call."""
    if results_df is None or results_df.empty or "stage" not in results_df.columns:
        return "Unknown"
    total = len(results_df)
    stage2 = results_df["stage"].astype(str).str.contains("2 ✅", na=False).sum()
    stage4 = results_df["stage"].astype(str).str.contains("4 🔴", na=False).sum()
    breadth = stage2 / total if total else 0
    if breadth >= 0.6:
        return "Bull"
    if breadth <= 0.15 and stage4 > stage2:
        return "Bear"
    if 0.15 < breadth < 0.4:
        return "Correction"
    return "Sideways"


def summarise_sector_rotation(results_df: Optional[pd.DataFrame], top_n: int = 5) -> List[Dict]:
    """Deterministic sector rotation ranking — no API call."""
    if results_df is None or results_df.empty or "theme" not in results_df.columns:
        return []
    try:
        grp = (results_df.groupby("theme")["apex_score"]
               .agg(["mean", "count"]).sort_values("mean", ascending=False))
        return [
            {"theme": t, "avg_score": round(float(r["mean"]), 1), "count": int(r["count"])}
            for t, r in grp.head(top_n).iterrows()
        ]
    except Exception:
        return []
