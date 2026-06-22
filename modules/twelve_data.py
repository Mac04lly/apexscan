"""
modules/twelve_data.py — Twelve Data API Integration
Provides: real-time quotes, RSI, MACD, Bollinger Bands, earnings calendar
Free tier: 800 requests/day, 8 requests/minute
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/twelve_data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Rate limiting ─────────────────────────────────────────────────────────────
_last_call_time = 0.0
_calls_this_minute = 0
_minute_start = 0.0
MIN_PAUSE = 7.5  # seconds between calls (8/min free tier = 7.5s safe)


def _rate_limit():
    global _last_call_time, _calls_this_minute, _minute_start
    now = time.time()
    if now - _minute_start >= 60:
        _calls_this_minute = 0
        _minute_start = now
    if _calls_this_minute >= 7:
        sleep_time = 60 - (now - _minute_start) + 1
        if sleep_time > 0:
            log.info(f"Twelve Data rate limit — sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        _calls_this_minute = 0
        _minute_start = time.time()
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_PAUSE:
        time.sleep(MIN_PAUSE - elapsed)
    _last_call_time = time.time()
    _calls_this_minute += 1


def _cache_path(ticker: str, endpoint: str) -> Path:
    return CACHE_DIR / f"{ticker}_{endpoint}.json"


def _cache_valid(path: Path, hours: int = 4) -> bool:
    if not path.exists():
        return False
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds()
    return age < hours * 3600


def _load_cache(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(path: Path, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# REAL-TIME QUOTE
# ══════════════════════════════════════════════════════════════════════════════

def get_realtime_quote(ticker: str, api_key: str) -> Dict:
    """
    Get real-time price quote from Twelve Data.
    Returns: price, open, high, low, close, volume, change, change_pct, timestamp
    Cache: 5 minutes (quotes need to be fresh)
    """
    if not api_key or api_key.startswith("YOUR_"):
        return {}

    cache = _cache_path(ticker, "quote")
    if _cache_valid(cache, hours=0.083):  # 5 min cache
        data = _load_cache(cache)
        if data:
            return data

    try:
        _rate_limit()
        url = "https://api.twelvedata.com/quote"
        resp = requests.get(url, params={
            "symbol":   ticker,
            "apikey":   api_key,
            "dp":       2,
        }, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        if raw.get("status") == "error" or "code" in raw:
            log.debug(f"Twelve Data quote error {ticker}: {raw.get('message','')}")
            return {}

        result = {
            "price":       float(raw.get("close", 0)),
            "open":        float(raw.get("open", 0)),
            "high":        float(raw.get("high", 0)),
            "low":         float(raw.get("low", 0)),
            "prev_close":  float(raw.get("previous_close", 0)),
            "volume":      int(float(raw.get("volume", 0))),
            "change":      float(raw.get("change", 0)),
            "change_pct":  float(raw.get("percent_change", 0)),
            "52w_high":    float(raw.get("fifty_two_week", {}).get("high", 0)),
            "52w_low":     float(raw.get("fifty_two_week", {}).get("low", 0)),
            "market_cap":  raw.get("market_cap"),
            "timestamp":   raw.get("datetime", ""),
            "is_market_open": raw.get("is_market_open", False),
            "exchange":    raw.get("exchange", ""),
            "source":      "twelve_data",
        }
        _save_cache(cache, result)
        return result

    except Exception as e:
        log.debug(f"Twelve Data quote failed {ticker}: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def get_rsi(ticker: str, api_key: str, period: int = 14, cache_hours: int = 4) -> Dict:
    """
    RSI from Twelve Data.
    Returns: rsi_value, signal (Overbought/Oversold/Neutral), series (list for chart)
    """
    if not api_key or api_key.startswith("YOUR_"):
        return {}

    cache = _cache_path(ticker, f"rsi_{period}")
    if _cache_valid(cache, cache_hours):
        data = _load_cache(cache)
        if data:
            return data

    try:
        _rate_limit()
        resp = requests.get("https://api.twelvedata.com/rsi", params={
            "symbol":      ticker,
            "interval":    "1day",
            "time_period": period,
            "outputsize":  30,
            "apikey":      api_key,
        }, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        if raw.get("status") == "error":
            return {}

        values = raw.get("values", [])
        if not values:
            return {}

        latest_rsi = float(values[0]["rsi"])
        series     = [float(v["rsi"]) for v in reversed(values)]

        signal = "Neutral"
        if latest_rsi >= 70:   signal = "Overbought ⚠️"
        elif latest_rsi <= 30: signal = "Oversold 💡"
        elif latest_rsi >= 60: signal = "Bullish Momentum"
        elif latest_rsi <= 40: signal = "Bearish Momentum"

        result = {
            "rsi":    round(latest_rsi, 1),
            "signal": signal,
            "series": series,
            "dates":  [v["datetime"] for v in reversed(values)],
        }
        _save_cache(cache, result)
        return result

    except Exception as e:
        log.debug(f"Twelve Data RSI failed {ticker}: {e}")
        return {}


def get_macd(ticker: str, api_key: str, cache_hours: int = 4) -> Dict:
    """
    MACD from Twelve Data (12, 26, 9).
    Returns: macd, signal, histogram, crossover signal
    """
    if not api_key or api_key.startswith("YOUR_"):
        return {}

    cache = _cache_path(ticker, "macd")
    if _cache_valid(cache, cache_hours):
        data = _load_cache(cache)
        if data:
            return data

    try:
        _rate_limit()
        resp = requests.get("https://api.twelvedata.com/macd", params={
            "symbol":        ticker,
            "interval":      "1day",
            "fast_period":   12,
            "slow_period":   26,
            "signal_period": 9,
            "outputsize":    30,
            "apikey":        api_key,
        }, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        if raw.get("status") == "error":
            return {}

        values = raw.get("values", [])
        if len(values) < 2:
            return {}

        latest  = values[0]
        prev    = values[1]
        macd_v  = float(latest["macd"])
        sig_v   = float(latest["macd_signal"])
        hist_v  = float(latest["macd_hist"])
        prev_h  = float(prev["macd_hist"])

        # Crossover detection
        crossover = "None"
        if hist_v > 0 and prev_h <= 0:  crossover = "Bullish Crossover 🟢"
        elif hist_v < 0 and prev_h >= 0: crossover = "Bearish Crossover 🔴"
        elif hist_v > 0:                 crossover = "Bullish Zone"
        else:                            crossover = "Bearish Zone"

        result = {
            "macd":      round(macd_v, 4),
            "signal":    round(sig_v, 4),
            "histogram": round(hist_v, 4),
            "crossover": crossover,
            "macd_series": [float(v["macd"])        for v in reversed(values)],
            "sig_series":  [float(v["macd_signal"]) for v in reversed(values)],
            "hist_series": [float(v["macd_hist"])   for v in reversed(values)],
            "dates":       [v["datetime"]            for v in reversed(values)],
        }
        _save_cache(cache, result)
        return result

    except Exception as e:
        log.debug(f"Twelve Data MACD failed {ticker}: {e}")
        return {}


def get_bollinger_bands(ticker: str, api_key: str, period: int = 20,
                        std_dev: float = 2.0, cache_hours: int = 4) -> Dict:
    """
    Bollinger Bands from Twelve Data.
    Returns: upper, middle, lower, bandwidth, squeeze flag, %B position
    """
    if not api_key or api_key.startswith("YOUR_"):
        return {}

    cache = _cache_path(ticker, f"bbands_{period}")
    if _cache_valid(cache, cache_hours):
        data = _load_cache(cache)
        if data:
            return data

    try:
        _rate_limit()
        resp = requests.get("https://api.twelvedata.com/bbands", params={
            "symbol":     ticker,
            "interval":   "1day",
            "time_period": period,
            "sd":          std_dev,
            "outputsize":  30,
            "apikey":      api_key,
        }, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        if raw.get("status") == "error":
            return {}

        values = raw.get("values", [])
        if not values:
            return {}

        latest = values[0]
        upper  = float(latest["upper_band"])
        middle = float(latest["middle_band"])
        lower  = float(latest["lower_band"])
        bw     = round((upper - lower) / middle * 100, 2)

        # Bandwidth vs 20-period average for squeeze detection
        bw_series = [(float(v["upper_band"]) - float(v["lower_band"])) /
                     float(v["middle_band"]) * 100 for v in values]
        avg_bw    = sum(bw_series) / len(bw_series)
        squeeze   = bw < avg_bw * 0.7  # bandwidth 30% below average = squeeze

        result = {
            "upper":     round(upper, 2),
            "middle":    round(middle, 2),
            "lower":     round(lower, 2),
            "bandwidth": bw,
            "squeeze":   squeeze,
            "avg_bw":    round(avg_bw, 2),
            "upper_series":  [float(v["upper_band"])  for v in reversed(values)],
            "middle_series": [float(v["middle_band"]) for v in reversed(values)],
            "lower_series":  [float(v["lower_band"])  for v in reversed(values)],
            "dates":         [v["datetime"]            for v in reversed(values)],
        }
        _save_cache(cache, result)
        return result

    except Exception as e:
        log.debug(f"Twelve Data BBands failed {ticker}: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# BATCH INDICATORS (single call for all 3 — saves API quota)
# ══════════════════════════════════════════════════════════════════════════════

def get_all_indicators(ticker: str, api_key: str, cache_hours: int = 4) -> Dict:
    """
    Fetch RSI + MACD + BBands for a ticker.
    Returns combined dict. Uses cache aggressively to protect free tier quota.
    """
    cache = _cache_path(ticker, "all_indicators")
    if _cache_valid(cache, cache_hours):
        data = _load_cache(cache)
        if data:
            return data

    result = {
        "rsi":    get_rsi(ticker, api_key, cache_hours=cache_hours),
        "macd":   get_macd(ticker, api_key, cache_hours=cache_hours),
        "bbands": get_bollinger_bands(ticker, api_key, cache_hours=cache_hours),
    }

    # Composite technical signal
    signals = []
    rsi_v   = result["rsi"].get("rsi")
    macd_x  = result["macd"].get("crossover", "")
    squeeze = result["bbands"].get("squeeze", False)

    if rsi_v:
        if rsi_v > 60 and "Bullish" in macd_x:
            signals.append("🟢 Strong momentum — RSI bullish + MACD aligned")
        elif rsi_v > 70:
            signals.append("⚠️ Overbought — consider waiting for pullback")
        elif rsi_v < 30:
            signals.append("💡 Oversold — potential reversal zone")
        elif rsi_v < 40 and "Bearish" in macd_x:
            signals.append("🔴 Weak — RSI bearish + MACD negative")

    if squeeze:
        signals.append("🔄 Bollinger Squeeze — volatility contraction, breakout incoming")

    if "Bullish Crossover" in macd_x:
        signals.append("🟢 MACD Bullish Crossover — momentum turning up")
    elif "Bearish Crossover" in macd_x:
        signals.append("🔴 MACD Bearish Crossover — momentum turning down")

    result["composite_signals"] = signals
    _save_cache(cache, result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# EARNINGS CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

def get_earnings_calendar(tickers: List[str], api_key: str) -> Dict[str, str]:
    """
    Get upcoming earnings dates for a list of tickers.
    Returns dict: {ticker: date_string}
    Cache: 24 hours
    """
    if not api_key or api_key.startswith("YOUR_"):
        return {}

    cache = _cache_path("batch", "earnings_calendar")
    if _cache_valid(cache, hours=24):
        data = _load_cache(cache)
        if data:
            return data

    result = {}
    try:
        _rate_limit()
        resp = requests.get("https://api.twelvedata.com/earnings", params={
            "apikey":     api_key,
            "outputsize": 50,
        }, timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        earnings = raw.get("earnings", [])
        for item in earnings:
            sym  = item.get("symbol", "")
            date = item.get("date", "")
            if sym in tickers and date:
                result[sym] = date

        _save_cache(cache, result)

    except Exception as e:
        log.debug(f"Twelve Data earnings calendar failed: {e}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# QUOTA TRACKER
# ══════════════════════════════════════════════════════════════════════════════

_quota_file = Path("data/twelve_data_quota.json")

def get_quota_status() -> Dict:
    """Track daily API usage to avoid hitting free tier limit."""
    today = datetime.now().strftime("%Y-%m-%d")
    if _quota_file.exists():
        try:
            with open(_quota_file) as f:
                data = json.load(f)
            if data.get("date") == today:
                used      = data.get("calls_used", 0)
                remaining = max(0, 800 - used)
                return {
                    "date":      today,
                    "calls_used": used,
                    "remaining": remaining,
                    "pct_used":  round(used / 800 * 100, 1),
                    "status":    "🟢 OK" if remaining > 200 else ("🟡 Low" if remaining > 50 else "🔴 Near Limit"),
                }
        except Exception:
            pass
    return {"date": today, "calls_used": 0, "remaining": 800, "pct_used": 0, "status": "🟢 OK"}


def increment_quota():
    today = datetime.now().strftime("%Y-%m-%d")
    status = get_quota_status()
    new_count = status["calls_used"] + 1
    try:
        with open(_quota_file, "w") as f:
            json.dump({"date": today, "calls_used": new_count}, f)
    except Exception:
        pass

