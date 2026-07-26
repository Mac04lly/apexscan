"""Strategy variants layered on the stable ApexScan technical analysis output."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Type


class BaseStrategy(ABC):
    name = "base"
    @abstractmethod
    def evaluate(self, stock: Dict[str, Any]) -> Dict[str, Any]: ...

    def _result(self, stock: Dict[str, Any], score: float, recommendation: str) -> Dict[str, Any]:
        stock["strategy"] = self.name
        stock["strategy_score"] = round(max(0, min(100, score)), 1)
        stock["strategy_recommendation"] = recommendation
        return stock

class SwingStrategy(BaseStrategy):
    name = "swing"
    def evaluate(self, stock): return self._result(stock, stock.get("apex_score", 0), "Buy" if stock.get("apex_score", 0) >= 60 else "Watch")

class PositionStrategy(BaseStrategy):
    name = "position"
    def evaluate(self, stock):
        score = 45 + 20 * bool(stock.get("above_200ma")) + 15 * bool(stock.get("ma50_gt_200")) + 20 * bool(stock.get("weekly_trending_up"))
        return self._result(stock, score, "Buy" if score >= 75 else "Watch")

class LongTermStrategy(BaseStrategy):
    name = "long_term"
    def evaluate(self, stock):
        fields = ("eps_growth_%", "rev_growth_%", "pe_ratio", "peg_ratio")
        available = sum(stock.get(k) is not None for k in fields)
        score = min(100, stock.get("apex_score", 0) * .45 + available * 12 + 20 * bool(stock.get("above_200ma")))
        recommendation = "Strong Buy" if score >= 80 else "Buy" if score >= 65 else "Hold" if score >= 45 else "Avoid"
        return self._result(stock, score, recommendation)

class DividendStrategy(BaseStrategy):
    name = "dividend"
    def evaluate(self, stock):
        yield_ = float(stock.get("dividend_yield", 0) or 0)
        score = min(100, 30 + stock.get("apex_score", 0) * .35 + min(yield_ * 8, 35))
        return self._result(stock, score, "Buy" if score >= 65 else "Watch")

class ValueStrategy(BaseStrategy):
    name = "value"
    def evaluate(self, stock):
        pe, peg = stock.get("pe_ratio"), stock.get("peg_ratio")
        value = 25 + stock.get("apex_score", 0) * .35
        if pe is not None and 0 < pe <= 20: value += 20
        if peg is not None and 0 < peg <= 1.5: value += 20
        return self._result(stock, value, "Buy" if value >= 65 else "Watch")

STRATEGIES: Dict[str, Type[BaseStrategy]] = {s.name: s for s in (SwingStrategy, PositionStrategy, LongTermStrategy, DividendStrategy, ValueStrategy)}
def get_strategy(name: str = "swing") -> BaseStrategy:
    return STRATEGIES.get((name or "swing").lower(), SwingStrategy)()
