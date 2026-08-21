"""
modules/backtester.py — Historical Strategy Backtester

Two fixes from the original version:
1. req_of/req_vwap/req_hh_hl/req_pa were accepted as parameters but never
   actually used anywhere — the confirmation checkboxes in the UI silently
   did nothing. They now genuinely gate trade entry, reusing the SAME
   functions the live scanner uses (order_flow_persistence, compute_vwap,
   detect_market_structure, detect_price_action_patterns) — not a separate
   re-implementation that could drift from what's actually live.
2. backtest_portfolio() never forwarded these four flags down to
   backtest_ticker() at all, even in the original code — multi-ticker
   backtests had this bug twice over. Fixed.

The entry score also now includes a genuine relative-strength component
(vs. the S&P 500, same weighting as the live Apex Score), not just the
original 3-factor proxy (3m performance / Stage 2 / near 52-week high).
It's still a deliberately simplified stand-in for the full live score —
things like weekly-timeframe confirmation and EPS momentum aren't
practical to backtest day-by-day without re-fetching fundamentals data
for every historical date — but it now shares real logic with the live
scanner for the pieces that matter most, rather than being a from-scratch
guess at what the scanner does.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from scanner import (
        order_flow_persistence, compute_vwap, detect_market_structure,
        detect_price_action_patterns, compute_rs, get_benchmark,
    )
    _HAS_SCANNER_FUNCS = True
except Exception:
    _HAS_SCANNER_FUNCS = False


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
    # Fail loudly, not silently, if a confirmation filter is requested but
    # the shared scanner logic isn't available to actually check it —
    # the whole point of this fix is to stop confirmation checkboxes from
    # quietly doing nothing.
    if (req_of or req_vwap or req_hh_hl or req_pa) and not _HAS_SCANNER_FUNCS:
        return {"error": "Confirmation filters (Order Flow / VWAP / Structure / Price Action) "
                          "require scanner.py's functions, which failed to import here. "
                          "Try again without those filters checked, or check the deployment."}

    try:
        hist = yf.Ticker(ticker).history(start=start, end=end)
        if len(hist) < 260:
            return {"error": f"Not enough data for {ticker} — need at least 260 trading days "
                              f"for a fair 200-day-average-based backtest."}

        close  = hist["Close"]
        ma50   = close.rolling(50).mean()
        ma200  = close.rolling(200).mean()
        high52 = close.rolling(252).max()

        # Benchmark fetched once (not per-day) for the RS component.
        bench_close = None
        if _HAS_SCANNER_FUNCS:
            try:
                bench_close = get_benchmark("^GSPC", period="max")
            except Exception:
                bench_close = None

        trades = []
        in_trade = False
        entry_price = entry_date = None
        hold_days = 0

        for i in range(200, len(hist)):
            price   = close.iloc[i]
            m50     = ma50.iloc[i]
            m200    = ma200.iloc[i]
            date    = hist.index[i]

            perf_3m = (price / close.iloc[max(0, i - 63)] - 1) * 100 if i >= 63 else 0
            stage2  = price > m50 > m200

            score = 0
            if perf_3m > 15:  score += 40
            if stage2:        score += 25
            if price > high52.iloc[i] * 0.85: score += 10

            # ── Relative strength vs. S&P 500 — same weighting as the live
            # Apex Score, using the scanner's own compute_rs(). ──
            if bench_close is not None and i >= 63:
                try:
                    rs = compute_rs(close.iloc[:i + 1], bench_close, 63)
                    if rs > 100:   score += 25
                    elif rs > 50:  score += 12
                except Exception:
                    pass

            # ── Confirmation filters — only computed when actually
            # requested, to keep the backtest fast when they're off. ──
            of_ok = vwap_ok = hh_hl_ok = pa_ok = True
            if req_of or req_vwap or req_hh_hl or req_pa:
                _window = hist.iloc[:i + 1]

                if req_of:
                    try:
                        of_data = order_flow_persistence(_window, 10)
                        of_ok = of_data["of_directional_bias"] in ("Bullish", "Strong Bullish")
                    except Exception:
                        of_ok = False

                if req_vwap:
                    try:
                        vwap_data = compute_vwap(_window, 20)
                        vwap_ok = vwap_data["vwap_position"] == "Above VWAP"
                    except Exception:
                        vwap_ok = False

                if req_hh_hl:
                    try:
                        ms_data = detect_market_structure(_window, 5)
                        hh_hl_ok = bool(ms_data.get("ms_hh_hl"))
                    except Exception:
                        hh_hl_ok = False

                if req_pa:
                    try:
                        pa_data = detect_price_action_patterns(_window)
                        pa_ok = bool(pa_data.get("pa_patterns"))
                    except Exception:
                        pa_ok = False

            entry_signal = (
                score >= min_score and
                (not require_stage2 or stage2) and
                of_ok and vwap_ok and hh_hl_ok and pa_ok
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
            "total_trades":   len(df),
            "win_rate_%":     round(wins / len(df) * 100, 1),
            "avg_return_%":   round(df["return_%"].mean(), 1),
            "best_trade_%":   round(df["return_%"].max(), 1),
            "worst_trade_%":  round(df["return_%"].min(), 1),
            "total_return_%": round(df["return_%"].sum(), 1),
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
        # Previously did NOT forward req_of/req_vwap/req_hh_hl/req_pa at
        # all — multi-ticker backtests ignored these filters entirely,
        # on top of backtest_ticker() itself ignoring them. Fixed here.
        result = backtest_ticker(ticker, start, end, min_score, True, max_hold_days,
                                  req_of, req_vwap, req_hh_hl, req_pa)
        if "trades" in result and result["trades"]:
            all_trades.extend(result["trades"])

    if not all_trades:
        return pd.DataFrame(), {}

    df = pd.DataFrame(all_trades).sort_values("entry_date")
    wins = (df["return_%"] > 0).sum()
    best_row = df.loc[df["return_%"].idxmax()]
    agg = {
        "total_trades":   len(df),
        "win_rate_%":     round(wins / len(df) * 100, 1),
        "avg_return_%":   round(df["return_%"].mean(), 1),
        "best_trade":     f"{best_row['ticker']} +{best_row['return_%']:.1f}%",
        "total_return_%": round(df["return_%"].sum(), 1),
    }
    return df, agg
