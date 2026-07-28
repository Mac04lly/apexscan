"""Strategy variants layered on the stable ApexScan technical analysis output.
Fundamentals used here (roe, debt_to_equity, free_cash_flow, dividend_yield,
payout_ratio, price_to_sales, ev_to_ebitda, operating_margin, profit_margin,
revenue_growth, earnings_growth) come from a free yfinance .info call made
only for stocks that already passed the scan (see scanner.py Pass 3) — no
paid API involved. When unavailable (None), each strategy falls back to the
Apex Score alone so scoring never breaks on missing data."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Type


def _safe(v, default=0.0):
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


class BaseStrategy(ABC):
    name = "base"

    @abstractmethod
    def evaluate(self, stock: Dict[str, Any]) -> Dict[str, Any]: ...

    def _result(self, stock, score, recommendation, breakdown=None):
        stock["strategy"] = self.name
        stock["strategy_score"] = round(max(0, min(100, score)), 1)
        stock["strategy_recommendation"] = recommendation
        if breakdown:
            stock["strategy_breakdown"] = breakdown
        return stock


class SwingStrategy(BaseStrategy):
    name = "swing"

    def evaluate(self, stock):
        apex = stock.get("apex_score", 0)
        return self._result(stock, apex, "Buy" if apex >= 60 else "Watch")


class PositionStrategy(BaseStrategy):
    """2-12 months. Stage 2 + weekly trend + RS leadership + order-flow-based
    institutional accumulation proxy + earnings growth."""
    name = "position"

    def evaluate(self, stock):
        stage2  = bool(stock.get("above_200ma")) and bool(stock.get("ma50_gt_ma200"))
        weekly  = bool(stock.get("weekly_trending_up")) or bool(stock.get("weekly_confirmed"))
        rs_lead = _safe(stock.get("rs_3m")) > 100
        accum   = "Bullish" in str(stock.get("of_bias", ""))
        earn_ok = _safe(stock.get("eps_growth_%")) > 0 or "Strong" in str(stock.get("earn_momentum", ""))

        score = 30 * stage2 + 25 * weekly + 20 * rs_lead + 15 * accum + 10 * earn_ok
        recommendation = "Buy" if score >= 75 else "Watch" if score >= 50 else "Avoid"
        breakdown = {
            "stage_2": stage2, "weekly_uptrend": weekly, "rs_leadership": rs_lead,
            "institutional_accumulation": accum, "earnings_growth": earn_ok,
        }
        return self._result(stock, score, recommendation, breakdown)


class LongTermStrategy(BaseStrategy):
    """1-10 years. Weighs real fundamentals (ROE, revenue/earnings growth,
    debt, FCF, PEG) alongside the technical Apex Score."""
    name = "long_term"

    def evaluate(self, stock):
        apex        = _safe(stock.get("apex_score"))
        roe         = stock.get("roe")
        rev_growth  = stock.get("revenue_growth")
        earn_growth = stock.get("earnings_growth")
        debt_to_eq  = stock.get("debt_to_equity")
        fcf         = stock.get("free_cash_flow")
        peg         = stock.get("peg_ratio")

        score = apex * 0.30
        fields_seen = 0

        if roe is not None:
            fields_seen += 1
            if _safe(roe) > 0.20: score += 15
            elif _safe(roe) > 0.10: score += 8

        if rev_growth is not None:
            fields_seen += 1
            if _safe(rev_growth) > 0.15: score += 15
            elif _safe(rev_growth) > 0.05: score += 8

        if earn_growth is not None:
            fields_seen += 1
            if _safe(earn_growth) > 0.15: score += 15
            elif _safe(earn_growth) > 0: score += 6

        if debt_to_eq is not None:
            fields_seen += 1
            if _safe(debt_to_eq) < 100: score += 10
            elif _safe(debt_to_eq) < 200: score += 4

        if fcf is not None:
            fields_seen += 1
            if _safe(fcf) > 0: score += 10

        if peg is not None:
            fields_seen += 1
            if 0 < _safe(peg) <= 1.5: score += 10
            elif 0 < _safe(peg) <= 2.5: score += 4

        if fields_seen == 0:
            score = apex

        recommendation = (
            "Strong Buy" if score >= 80 else
            "Buy"        if score >= 65 else
            "Hold"       if score >= 45 else
            "Avoid"
        )
        breakdown = {
            "roe": roe, "revenue_growth": rev_growth, "earnings_growth": earn_growth,
            "debt_to_equity": debt_to_eq,
            "free_cash_flow_positive": (_safe(fcf) > 0) if fcf is not None else None,
            "peg_ratio": peg, "fundamentals_available": fields_seen,
        }
        return self._result(stock, score, recommendation, breakdown)


class DividendStrategy(BaseStrategy):
    """Dividend yield, payout sustainability, and 5y average yield trend."""
    name = "dividend"

    def evaluate(self, stock):
        apex   = _safe(stock.get("apex_score"))
        yld    = stock.get("dividend_yield")
        payout = stock.get("payout_ratio")
        yld5y  = stock.get("dividend_5y_avg")

        if yld is None:
            return self._result(stock, apex * 0.3, "Not a Dividend Candidate",
                                 {"dividend_yield": None})

        score = 20 + apex * 0.20
        yld_pct = _safe(yld) * 100 if _safe(yld) < 1 else _safe(yld)
        if yld_pct >= 2: score += min(30, yld_pct * 6)

        safe_payout = payout is not None and 0 < _safe(payout) < 0.75
        if safe_payout: score += 20
        elif payout is not None and _safe(payout) >= 0.9: score -= 15

        if yld5y is not None and _safe(yld5y) > 0 and yld_pct >= _safe(yld5y) * 100 * 0.9:
            score += 10

        recommendation = "Buy" if score >= 65 else "Watch" if score >= 45 else "Avoid"
        breakdown = {
            "dividend_yield_%": round(yld_pct, 2), "payout_ratio": payout,
            "payout_sustainable": safe_payout, "vs_5y_avg_yield": yld5y,
        }
        return self._result(stock, score, recommendation, breakdown)


class ValueStrategy(BaseStrategy):
    """PE, PEG, EV/EBITDA, and Price/Sales as a margin-of-safety proxy."""
    name = "value"

    def evaluate(self, stock):
        apex      = _safe(stock.get("apex_score"))
        pe        = stock.get("pe_ratio")
        peg       = stock.get("peg_ratio")
        ev_ebitda = stock.get("ev_to_ebitda")
        ps        = stock.get("price_to_sales")

        score = 20 + apex * 0.25
        cheap_signals = 0
        checked = 0

        if pe is not None:
            checked += 1
            if 0 < _safe(pe) <= 20: score += 15; cheap_signals += 1
            elif 0 < _safe(pe) <= 30: score += 6

        if peg is not None:
            checked += 1
            if 0 < _safe(peg) <= 1.0: score += 15; cheap_signals += 1
            elif 0 < _safe(peg) <= 1.5: score += 8

        if ev_ebitda is not None:
            checked += 1
            if 0 < _safe(ev_ebitda) <= 10: score += 15; cheap_signals += 1
            elif 0 < _safe(ev_ebitda) <= 15: score += 6

        if ps is not None:
            checked += 1
            if 0 < _safe(ps) <= 2: score += 10; cheap_signals += 1

        if checked == 0:
            score = apex

        recommendation = "Buy" if (score >= 65 and cheap_signals >= 2) else "Watch" if score >= 50 else "Avoid"
        breakdown = {
            "pe_ratio": pe, "peg_ratio": peg, "ev_to_ebitda": ev_ebitda,
            "price_to_sales": ps, "cheap_signals": cheap_signals, "fields_checked": checked,
        }
        return self._result(stock, score, recommendation, breakdown)


STRATEGIES: Dict[str, Type[BaseStrategy]] = {
    s.name: s for s in (SwingStrategy, PositionStrategy, LongTermStrategy, DividendStrategy, ValueStrategy)
}

def get_strategy(name: str = "swing") -> BaseStrategy:
    return STRATEGIES.get((name or "swing").lower(), SwingStrategy)()
