"""
modules/risk_calc.py — Position Sizing & Risk Calculator
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class TradeSetup:
    account_size: float
    risk_pct: float
    entry_price: float
    stop_price: float
    target_price: Optional[float] = None
    commission: float = 0.0


@dataclass
class PositionResult:
    shares: int
    position_size: float
    risk_per_trade: float
    risk_pct_actual: float
    stop_distance: float
    stop_pct: float
    reward_risk: Optional[float]
    target_gain: Optional[float]
    target_gain_pct: Optional[float]
    breakeven_price: float
    warning: Optional[str]


def calculate_position(setup: TradeSetup) -> PositionResult:
    if setup.stop_price >= setup.entry_price:
        raise ValueError("Stop loss must be below entry price.")

    risk_dollars   = setup.account_size * (setup.risk_pct / 100)
    stop_distance  = setup.entry_price - setup.stop_price
    stop_pct       = (stop_distance / setup.entry_price) * 100
    shares         = max(1, math.floor(risk_dollars / stop_distance))
    position_size  = shares * setup.entry_price
    actual_risk    = shares * stop_distance + setup.commission
    actual_risk_pct = (actual_risk / setup.account_size) * 100
    breakeven      = setup.entry_price + (setup.commission / shares if shares else 0)

    warning = None
    if position_size > setup.account_size * 0.25:
        warning = f"Position size (${position_size:,.0f}) is over 25% of account — consider reducing."

    reward_risk = target_gain = target_gain_pct = None
    if setup.target_price and setup.target_price > setup.entry_price:
        gain_per_share  = setup.target_price - setup.entry_price
        target_gain     = round(gain_per_share * shares, 2)
        target_gain_pct = round((gain_per_share / setup.entry_price) * 100, 1)
        reward_risk     = round(gain_per_share / stop_distance, 2)

    return PositionResult(
        shares          = shares,
        position_size   = round(position_size, 2),
        risk_per_trade  = round(actual_risk, 2),
        risk_pct_actual = round(actual_risk_pct, 2),
        stop_distance   = round(stop_distance, 2),
        stop_pct        = round(stop_pct, 1),
        reward_risk     = reward_risk,
        target_gain     = target_gain,
        target_gain_pct = target_gain_pct,
        breakeven_price = round(breakeven, 4),
        warning         = warning,
    )


def pyramiding_plan(setup: TradeSetup, add_pct: float = 5.0, n_adds: int = 2) -> list:
    plan = [{"add": "Initial", "price": setup.entry_price,
             "position_$": setup.entry_price * calculate_position(setup).shares,
             "cumulative_$": setup.entry_price * calculate_position(setup).shares}]
    base_result = calculate_position(setup)
    cumulative  = base_result.position_size

    for i in range(1, n_adds + 1):
        add_price   = round(setup.entry_price * (1 + (add_pct / 100) * i), 2)
        add_shares  = max(1, math.floor(base_result.shares / (i + 1)))
        add_cost    = round(add_price * add_shares, 2)
        cumulative += add_cost
        plan.append({
            "add":           f"Add #{i}",
            "price":         add_price,
            "shares":        add_shares,
            "position_$":    add_cost,
            "cumulative_$":  round(cumulative, 2),
        })
    return plan
