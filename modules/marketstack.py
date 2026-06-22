"""
modules/marketstack.py — Marketstack API Integration
Provides: EOD historical data, intraday quotes (backup to yfinance)
Free tier: 100 requests/month — use ONLY as fallback/backup
Strategy: cache aggressively (7 days), only call when yfinance fails
"""

import requests
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/marketstack_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://api.marketstack.com/v1"


def _cache_path(ticker: str, endpoint: str) -> Path:
    return CACHE_DIR / f"{ticker}_{endpoint}.json"


def _cache_valid(path: Path, hours: int = 168) -> bool:
    """Default 7-day cache to protect the 100 req/month quota."""
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


# ── Quota tracker ─────────────────────────────────────────────────────────────
_quota_file = Path("data/marketstack_quota.json")

def get_quota_status() -> Dict:
    """Track monthly API usage — free tier is only 100/month."""
    month = datetime.now().strftime("%Y-%m")
    if _quota_file.exists():
        try:
            with open(_quota_file) as f:
                data = json.load(f)
            if data.get("month") == month:
                used      = data.get("calls_used", 0)
                remaining = max(0, 100 - used)
                return {
                    "month":      month,
                    "calls_used": used,
                    "remaining":  remaining,
                    "pct_used":   round(used / 100 * 100, 1),
                    "status":     "🟢 OK" if remaining > 20 else ("🟡 Low" if remaining > 5 else "🔴 Near Limit"),
                    "warning":    remaining <= 10,
                }
        except Exception:
            pass
    return {"month": month, "calls_used": 0, "remaining": 100,
            "pct_used": 0, "status": "🟢 OK", "warning": False}


def _increment_quota():
    month = datetime.now().strftime("%Y-%m")
    status = get_quota_status()
    try:
        with open(_quota_file, "w") as f:
            json.dump({"month": month, "calls_used": status["calls_used"] + 1}, f)
    except Exception:
        pass


def _can_make_call(api_key: str) -> bool:
    """Only make a call if we have quota remaining and a valid key."""
    if not api_key or api_key.startswith("YOUR_"):
        return False
    status = get_quota_status()
    if status["remaining"] <= 0:
        log.warning("Marketstack monthly quota exhausted — skipping API call, using cache only")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# EOD HISTORICAL DATA (main use case — yfinance backup)
# ══════════════════════════════════════════════════════════════════════════════

def get_eod_data(ticker: str, api_key: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Get end-of-day OHLCV data from Marketstack.
    Used as FALLBACK when yfinance fails for a ticker.
    Cache: 7 days (protects monthly quota aggressively).
    Returns: DataFrame with columns [Open, High, Low, Close, Volume] or None
    """
    cache = _cache_path(ticker, f"eod_{days}d")

    # Always check cache first — protect the quota
    if _cache_valid(cache, hours=168):
        data = _load_cache(cache)
        if data:
            try:
                df = pd.DataFrame(data)
                df.index = pd.to_datetime(df["date"])
                df = df.rename(columns={
                    "open": "Open", "high": "High",
                    "low": "Low", "close": "Close", "volume": "Volume"
                })
                return df[["Open","High","Low","Close","Volume"]].sort_index()
            except Exception:
                pass

    if not _can_make_call(api_key):
        return None

    try:
        date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        date_to   = datetime.now().strftime("%Y-%m-%d")

        resp = requests.get(f"{BASE_URL}/eod", params={
            "access_key": api_key,
            "symbols":    ticker,
            "date_from":  date_from,
            "date_to":    date_to,
            "limit":      1000,
        }, timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        if "error" in raw:
            log.debug(f"Marketstack EOD error {ticker}: {raw['error'].get('message','')}")
            return None

        data_list = raw.get("data", [])
        if not data_list:
            return None

        _save_cache(cache, data_list)
        _increment_quota()
        log.info(f"Marketstack EOD {ticker}: {len(data_list)} bars fetched")

        df = pd.DataFrame(data_list)
        df.index = pd.to_datetime(df["date"])
        df = df.rename(columns={
            "open": "Open", "high": "High",
            "low":  "Low", "close": "Close", "volume": "Volume"
        })
        return df[["Open","High","Low","Close","Volume"]].sort_index()

    except Exception as e:
        log.debug(f"Marketstack EOD failed {ticker}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LATEST QUOTE (backup to Twelve Data for real-time price)
# ══════════════════════════════════════════════════════════════════════════════

def get_latest_quote(ticker: str, api_key: str) -> Dict:
    """
    Get latest EOD quote from Marketstack.
    Cache: 1 hour (quotes don't need to be real-time from this source).
    Use Twelve Data for real-time — this is the secondary backup.
    """
    cache = _cache_path(ticker, "latest_quote")
    if _cache_valid(cache, hours=1):
        data = _load_cache(cache)
        if data:
            return data

    if not _can_make_call(api_key):
        return {}

    try:
        resp = requests.get(f"{BASE_URL}/eod/latest", params={
            "access_key": api_key,
            "symbols":    ticker,
        }, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        if "error" in raw:
            return {}

        data_list = raw.get("data", [])
        if not data_list:
            return {}

        item   = data_list[0]
        result = {
            "price":   float(item.get("close", 0)),
            "open":    float(item.get("open", 0)),
            "high":    float(item.get("high", 0)),
            "low":     float(item.get("low", 0)),
            "volume":  int(item.get("volume", 0)),
            "date":    item.get("date", ""),
            "source":  "marketstack",
        }
        _save_cache(cache, result)
        _increment_quota()
        return result

    except Exception as e:
        log.debug(f"Marketstack quote failed {ticker}: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SMART FALLBACK — try yfinance first, marketstack if it fails
# ══════════════════════════════════════════════════════════════════════════════

def get_history_with_fallback(ticker: str, ms_api_key: str,
                               period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Try yfinance first. If it returns empty/error, fall back to Marketstack.
    This is the function scanner.py should call instead of raw yf.Ticker().history()
    when a ticker keeps failing.
    """
    import yfinance as yf
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if not hist.empty and len(hist) >= 50:
            return hist
    except Exception:
        pass

    log.info(f"yfinance failed for {ticker} — trying Marketstack fallback")
    days_map = {"3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
    days     = days_map.get(period, 365)
    return get_eod_data(ticker, ms_api_key, days=days)
