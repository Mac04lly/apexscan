"""
modules/ngx_data.py — ApexScan Nigerian Exchange (NGX) Data Layer
Uses NGN Market API for real-time NGX stock data.

NGN Market API: https://ngnmarket.com/developer
Free tier: 3,000 requests/month
Endpoints used: /quotes, /historical, /fundamentals

Falls back to yfinance (.LA suffix) when API unavailable.
"""

import requests
import json
import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/ngx_cache")
NGN_BASE  = "https://api.ngnmarket.com/v1"

# NGX-specific constants
NGX_SUFFIX_YF  = ".LA"          # yfinance Lagos suffix
NGX_CURRENCY   = "NGN"          # Nigerian Naira
NGX_BENCHMARK  = "^NGSE"        # NGX All-Share Index on yfinance


def _ensure_dir():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _cache_path(symbol: str, endpoint: str) -> Path:
    _ensure_dir()
    clean = symbol.replace(".","_").replace("/","_")
    return CACHE_DIR / f"{clean}_{endpoint}.json"


def _cache_valid(path: Path, hours: float = 4) -> bool:
    try:
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < timedelta(hours=hours)
    except Exception:
        return False


def _read_cache(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_cache(path: Path, data):
    try:
        path.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _resolve_key(api_key: str = "") -> str:
    """Get NGN Market API key from param, Streamlit secrets, or config."""
    if api_key and not api_key.startswith("YOUR_"):
        return api_key
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "ngn_market_key" in st.secrets:
            return st.secrets["ngn_market_key"]
    except Exception:
        pass
    return api_key or ""


def _ngn_get(endpoint: str, params: dict, api_key: str,
              timeout: int = 10) -> Optional[dict]:
    """Single NGN Market API call."""
    key = _resolve_key(api_key)
    if not key:
        return None
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type": "application/json"}
    try:
        resp = requests.get(f"{NGN_BASE}/{endpoint}",
                           params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.debug(f"NGN Market API error ({endpoint}): {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# QUOTE & HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def get_ngx_quote(symbol: str, api_key: str = "",
                   cache_minutes: float = 15) -> Optional[Dict]:
    """
    Get real-time NGX quote from NGN Market API.
    Falls back to yfinance if API unavailable.

    symbol: bare NGX ticker e.g. "GTCO" (no .LA suffix needed)
    Returns: price (NGN), change_pct, volume, market_cap etc.
    """
    # Strip .LA suffix if present
    bare = symbol.upper().replace(".LA", "").replace(".LG", "")

    cache = _cache_path(bare, "quote")
    if _cache_valid(cache, cache_minutes / 60):
        d = _read_cache(cache)
        if d:
            return d

    # Try NGN Market API first
    key = _resolve_key(api_key)
    if key:
        raw = _ngn_get("quotes", {"symbol": bare}, api_key)
        if raw:
            try:
                # NGN Market API response format
                data = raw.get("data", raw)
                if isinstance(data, list) and data:
                    data = data[0]

                result = {
                    "ticker":      bare,
                    "ticker_yf":   bare + NGX_SUFFIX_YF,
                    "price_ngn":   float(data.get("close", data.get("price", 0)) or 0),
                    "open_ngn":    float(data.get("open", 0) or 0),
                    "high_ngn":    float(data.get("high", 0) or 0),
                    "low_ngn":     float(data.get("low",  0) or 0),
                    "volume":      int(data.get("volume", 0) or 0),
                    "change_pct":  float(data.get("change_pct",
                                         data.get("percent_change", 0)) or 0),
                    "market_cap":  float(data.get("market_cap", 0) or 0),
                    "pe_ratio":    float(data.get("pe_ratio", 0) or 0),
                    "eps":         float(data.get("eps", 0) or 0),
                    "week_52_high":float(data.get("week_52_high", 0) or 0),
                    "week_52_low": float(data.get("week_52_low",  0) or 0),
                    "source":      "ngn_market_api",
                    "currency":    "NGN",
                    "timestamp":   datetime.now().isoformat(),
                }
                _write_cache(cache, result)
                return result
            except Exception as e:
                log.debug(f"NGN Market quote parse error {bare}: {e}")

    # Fallback: yfinance with .LA suffix
    return get_ngx_quote_yfinance(bare + NGX_SUFFIX_YF)


def get_ngx_quote_yfinance(ticker_la: str) -> Optional[Dict]:
    """Fallback: get NGX quote via yfinance .LA ticker."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker_la).fast_info
        bare = ticker_la.replace(NGX_SUFFIX_YF, "")
        return {
            "ticker":      bare,
            "ticker_yf":   ticker_la,
            "price_ngn":   float(getattr(info, "last_price", 0) or 0),
            "volume":      int(getattr(info,   "three_month_average_volume", 0) or 0),
            "week_52_high":float(getattr(info, "year_high", 0) or 0),
            "week_52_low": float(getattr(info, "year_low",  0) or 0),
            "source":      "yfinance",
            "currency":    "NGN",
            "timestamp":   datetime.now().isoformat(),
        }
    except Exception as e:
        log.debug(f"yfinance NGX fallback error {ticker_la}: {e}")
        return None


def get_ngx_history(symbol: str, api_key: str = "",
                     period_days: int = 365,
                     cache_hours: float = 24) -> Optional[pd.DataFrame]:
    """
    Fetch NGX historical OHLCV data.
    Returns DataFrame with Date index and OHLCV columns (prices in NGN).
    """
    bare = symbol.upper().replace(".LA", "").replace(".LG", "")
    cache = _cache_path(bare, f"hist_{period_days}")

    if _cache_valid(cache, cache_hours):
        d = _read_cache(cache)
        if d:
            try:
                df = pd.DataFrame(d)
                df["Date"] = pd.to_datetime(df["Date"])
                return df.set_index("Date").sort_index()
            except Exception:
                pass

    # Try NGN Market API
    key = _resolve_key(api_key)
    if key:
        date_from = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")
        raw = _ngn_get("historical", {
            "symbol":    bare,
            "from":      date_from,
            "to":        datetime.now().strftime("%Y-%m-%d"),
            "interval":  "1d",
        }, api_key, timeout=15)

        if raw:
            try:
                records = raw.get("data", raw)
                if isinstance(records, list) and records:
                    rows = []
                    for r in records:
                        rows.append({
                            "Date":   r.get("date", r.get("timestamp","")),
                            "Open":   float(r.get("open",  0) or 0),
                            "High":   float(r.get("high",  0) or 0),
                            "Low":    float(r.get("low",   0) or 0),
                            "Close":  float(r.get("close", 0) or 0),
                            "Volume": int(r.get("volume",  0) or 0),
                        })
                    df = pd.DataFrame(rows)
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date").sort_index()
                    _write_cache(cache, df.reset_index().to_dict("records"))
                    log.info(f"NGX history {bare}: {len(df)} bars from API")
                    return df
            except Exception as e:
                log.debug(f"NGN Market history parse {bare}: {e}")

    # Fallback: yfinance
    try:
        import yfinance as yf
        yf_ticker = bare + NGX_SUFFIX_YF
        hist = yf.Ticker(yf_ticker).history(period="1y")
        if not hist.empty:
            hist.index = hist.index.tz_localize(None)
            log.info(f"NGX history {bare}: {len(hist)} bars from yfinance")
            return hist[["Open","High","Low","Close","Volume"]]
    except Exception as e:
        log.debug(f"yfinance NGX history {bare}: {e}")

    return None


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════

def get_ngx_fundamentals(symbol: str, api_key: str = "",
                          cache_hours: float = 168) -> Optional[Dict]:
    """
    Fetch NGX fundamental data from NGN Market API.
    Returns: EPS, PE, dividend yield, market cap, sector.
    Cache: 7 days.
    """
    bare = symbol.upper().replace(".LA","").replace(".LG","")
    cache = _cache_path(bare, "fundamentals")

    if _cache_valid(cache, cache_hours):
        d = _read_cache(cache)
        if d:
            return d

    key = _resolve_key(api_key)
    if not key:
        return None

    raw = _ngn_get("fundamentals", {"symbol": bare}, api_key)
    if not raw:
        return None

    try:
        data = raw.get("data", raw)
        result = {
            "ticker":       bare,
            "eps":          float(data.get("eps",           0) or 0),
            "pe_ratio":     float(data.get("pe_ratio",      0) or 0),
            "pb_ratio":     float(data.get("pb_ratio",      0) or 0),
            "div_yield_%":  float(data.get("dividend_yield",0) or 0),
            "market_cap":   float(data.get("market_cap",    0) or 0),
            "revenue":      float(data.get("revenue",       0) or 0),
            "net_income":   float(data.get("net_income",    0) or 0),
            "sector":       str(data.get("sector",  "")),
            "industry":     str(data.get("industry","")),
            "listed_date":  str(data.get("listed_date", "")),
        }
        _write_cache(cache, result)
        return result
    except Exception as e:
        log.debug(f"NGX fundamentals parse {bare}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def get_ngx_benchmark(api_key: str = "",
                       period_days: int = 365,
                       cache_hours: float = 24) -> pd.Series:
    """
    Get NGX All-Share Index historical closes for RS calculation.
    Falls back to yfinance ^NGSE.
    """
    cache = _cache_path("NGXASI", f"benchmark_{period_days}")
    if _cache_valid(cache, cache_hours):
        d = _read_cache(cache)
        if d:
            try:
                s = pd.Series(d["values"], index=pd.to_datetime(d["dates"]))
                return s.sort_index()
            except Exception:
                pass

    # Try NGN Market API for ASI
    key = _resolve_key(api_key)
    if key:
        date_from = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")
        raw = _ngn_get("historical", {
            "symbol":   "ASI",
            "from":     date_from,
            "to":       datetime.now().strftime("%Y-%m-%d"),
            "interval": "1d",
        }, api_key, timeout=15)
        if raw:
            try:
                records = raw.get("data", raw)
                if records:
                    dates  = [r.get("date","") for r in records]
                    values = [float(r.get("close",0) or 0) for r in records]
                    s = pd.Series(values, index=pd.to_datetime(dates)).sort_index()
                    _write_cache(cache, {"dates": [str(d) for d in s.index], "values": list(s.values)})
                    return s
            except Exception:
                pass

    # Fallback: yfinance
    try:
        import yfinance as yf
        hist = yf.Ticker("^NGSE").history(period="1y")["Close"]
        if not hist.empty:
            hist.index = hist.index.tz_localize(None)
            return hist.sort_index()
    except Exception:
        pass

    return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# ALL TICKERS LIST
# ══════════════════════════════════════════════════════════════════════════════

def get_all_ngx_tickers(api_key: str = "",
                         cache_hours: float = 168) -> List[Dict]:
    """
    Fetch all NGX listed companies from NGN Market API.
    Returns list of {ticker, name, sector, market_cap}.
    Cache: 7 days.
    """
    cache = _cache_path("__all_tickers__", "list")
    if _cache_valid(cache, cache_hours):
        d = _read_cache(cache)
        if d:
            return d

    key = _resolve_key(api_key)
    if not key:
        return []

    raw = _ngn_get("stocks", {}, api_key, timeout=20)
    if not raw:
        return []

    try:
        records = raw.get("data", raw)
        result  = []
        for r in records:
            sym = str(r.get("symbol", r.get("ticker",""))).upper()
            if sym:
                result.append({
                    "ticker":   sym,
                    "name":     str(r.get("name", sym)),
                    "sector":   str(r.get("sector",   "")),
                    "industry": str(r.get("industry", "")),
                    "market_cap": float(r.get("market_cap", 0) or 0),
                })
        if result:
            _write_cache(cache, result)
            log.info(f"NGX all tickers: {len(result)} stocks from API")
        return result
    except Exception as e:
        log.debug(f"NGX all tickers error: {e}")
        return []
