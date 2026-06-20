"""
modules/backtester.py — Historical Strategy Backtester
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


def backtest_ticker(
    ticker: str,
    start: str,
    end: str,
    min_score: float = 40,
    require_stage2: bool = True,
    max_hold_days: int = 60,
    req_of: bool = False,
    req_vwap: bool = False,
    req_hh_hl: bool = False,
    req_pa: bool = False,
) -> dict:
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end)
        if len(hist) < 60:
            return {"error": f"Not enough data for {ticker}"}

        close  = hist["Close"]
        ma50   = close.rolling(50).mean()
        ma200  = close.rolling(200).mean()
        volume = hist["Volume"]
        avg_vol = volume.rolling(50).mean()

        trades = []
        in_trade = False
        entry_price = entry_date = None
        hold_days = 0

        for i in range(200, len(hist)):
            price   = close.iloc[i]
            m50     = ma50.iloc[i]
            m200    = ma200.iloc[i]
            date    = hist.index[i]

            # Simple score proxy
            perf_3m = (price / close.iloc[max(0, i-63)] - 1) * 100 if i >= 63 else 0
            stage2  = price > m50 > m200

            score = 0
            if perf_3m > 15:  score += 40
            if stage2:        score += 25
            if price > close.rolling(252).max().iloc[i] * 0.85: score += 10

            entry_signal = (
                score >= min_score and
                (not require_stage2 or stage2)
            )

            if not in_trade and entry_signal:
                in_trade    = True
                entry_price = price
                entry_date  = date
                hold_days   = 0

            elif in_trade:
                hold_days += 1
                exit_signal = price < m50 or hold_days >= max_hold_days
                if exit_signal:
                    ret = round((price / entry_price - 1) * 100, 2)
                    trades.append({
                        "ticker":      ticker,
                        "entry_date":  str(entry_date.date()),
                        "exit_date":   str(date.date()),
                        "entry_price": round(entry_price, 2),
                        "exit_price":  round(price, 2),
                        "hold_days":   hold_days,
                        "return_%":    ret,
                        "exit_reason": "Below 50MA" if price < m50 else "Max hold",
                    })
                    in_trade = False

        if not trades:
            return {"trades": [], "summary": {}}

        df = pd.DataFrame(trades)
        wins = (df["return_%"] > 0).sum()
        summary = {
            "total_trades": len(df),
            "win_rate_%":   round(wins / len(df) * 100, 1),
            "avg_return_%": round(df["return_%"].mean(), 1),
            "best_trade_%": round(df["return_%"].max(), 1),
            "worst_trade_%":round(df["return_%"].min(), 1),
            "total_return_%":round(df["return_%"].sum(), 1),
        }
        return {"trades": trades, "summary": summary}

    except Exception as e:
        return {"error": str(e)}


def backtest_portfolio(
    tickers: list,
    start: str,
    end: str,
    min_score: float = 40,
    max_hold_days: int = 60,
    req_of: bool = False,
    req_vwap: bool = False,
    req_hh_hl: bool = False,
    req_pa: bool = False,
) -> tuple:
    all_trades = []
    for ticker in tickers:
        result = backtest_ticker(ticker, start, end, min_score, True, max_hold_days)
        if "trades" in result and result["trades"]:
            all_trades.extend(result["trades"])

    if not all_trades:
        return pd.DataFrame(), {}

    df = pd.DataFrame(all_trades).sort_values("entry_date")
    wins = (df["return_%"] > 0).sum()
    best_row = df.loc[df["return_%"].idxmax()]
    agg = {
        "total_trades":  len(df),
        "win_rate_%":    round(wins / len(df) * 100, 1),
        "avg_return_%":  round(df["return_%"].mean(), 1),
        "best_trade":    f"{best_row['ticker']} +{best_row['return_%']:.1f}%",
        "total_return_%":round(df["return_%"].sum(), 1),
    }
    return df, agg
