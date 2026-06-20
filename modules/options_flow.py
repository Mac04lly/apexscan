"""
modules/options_flow.py — Options Chain Scanner
"""

import yfinance as yf
import pandas as pd
from datetime import datetime


def scan_options_flow(ticker: str, min_volume: int = 200) -> pd.DataFrame:
    try:
        tk   = yf.Ticker(ticker)
        spot = tk.history(period="1d")["Close"].iloc[-1]
        exps = tk.options
        if not exps:
            return pd.DataFrame()

        rows = []
        for exp in exps[:3]:
            try:
                chain = tk.option_chain(exp)
                for opt_type, df in [("CALL", chain.calls), ("PUT", chain.puts)]:
                    for _, row in df.iterrows():
                        vol = int(row.get("volume", 0) or 0)
                        oi  = int(row.get("openInterest", 0) or 0)
                        if vol < min_volume:
                            continue
                        iv      = round(float(row.get("impliedVolatility", 0) or 0) * 100, 1)
                        strike  = float(row.get("strike", 0))
                        last    = float(row.get("lastPrice", 0) or 0)
                        notional= vol * last * 100
                        vol_oi  = round(vol / oi, 2) if oi > 0 else 0
                        unusual = vol_oi > 3 or notional > 500000
                        moneyness = round((spot / strike - 1) * 100, 1) if strike else 0

                        if opt_type == "CALL":
                            sentiment = "Bullish" if unusual else "Neutral Calls"
                        else:
                            sentiment = "Bearish" if unusual else "Neutral Puts"

                        rows.append({
                            "ticker":        ticker,
                            "type":          opt_type,
                            "expiry":        exp,
                            "strike":        strike,
                            "spot":          round(spot, 2),
                            "moneyness_%":   moneyness,
                            "volume":        vol,
                            "open_interest": oi,
                            "vol/OI":        vol_oi,
                            "IV_%":          iv,
                            "last_price":    last,
                            "notional_$":    round(notional, 0),
                            "unusual":       unusual,
                            "sentiment":     sentiment,
                        })
            except Exception:
                continue

        return pd.DataFrame(rows).sort_values("notional_$", ascending=False) if rows else pd.DataFrame()

    except Exception:
        return pd.DataFrame()


def scan_multiple(tickers: list, min_volume: int = 200) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        df = scan_options_flow(ticker, min_volume)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
