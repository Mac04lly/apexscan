"""
scanner.py — ApexScan Core Engine v10 (US Market Only)
Momentum / Stage Analysis / Theme Rotation /
Order Flow Persistence / Auction Market Theory /
Market Structure / Price Action Patterns
"""

import yfinance as yf
import requests
import pandas as pd
import numpy as np
import time
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from modules.diagnostics import ScanDiagnostics
from strategies import get_strategy

# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/scanner.log"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            for key in ["alpha_vantage_key", "finnhub_key", "twelve_data_key", "marketstack_key", "anthropic_api_key", "ngx_pulse_key", "ngn_market_key"]:
                if key in st.secrets:
                    cfg[key] = st.secrets[key]
    except Exception:
        pass

    return cfg


def build_watchlist(cfg: dict, market: str = "us") -> List[str]:
    theme_key = "ng_themes" if market == "ng" else "us_themes"
    themes    = cfg.get(theme_key, {})
    return list(set(t for theme in themes.values() for t in theme))


def resolve_market(ticker: str, requested_market: str = "auto") -> str:
    requested = (requested_market or "auto").lower()
    if requested in {"ng", "us"}:
        return requested
    return "ng" if ticker.upper().endswith((".LG", ".NG")) else "us"


# ══════════════════════════════════════════════════════════════════════════════
# FINNHUB
# ══════════════════════════════════════════════════════════════════════════════

def get_finnhub_news(ticker: str, api_key: str) -> Dict:
    if not api_key or api_key.startswith("YOUR_"):
        return {"news_count": 0, "sentiment": "N/A"}
    try:
        end   = datetime.now()
        start = end - timedelta(days=7)
        url   = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={start.strftime('%Y-%m-%d')}"
            f"&to={end.strftime('%Y-%m-%d')}&token={api_key}"
        )
        resp     = requests.get(url, timeout=5)
        resp.raise_for_status()
        articles = resp.json()[:5]
        return {"news_count": len(articles),
                "sentiment":  "Positive" if articles else "Neutral"}
    except Exception as e:
        log.debug(f"Finnhub error {ticker}: {e}")
        return {"news_count": 0, "sentiment": "N/A"}


def get_recent_headlines(ticker: str, api_key: str, n: int = 2, days: int = 14) -> List[str]:
    """
    Fetches actual recent headline text (not just a count) — used to give
    qualitative 'why is this stock moving' context for short candidates,
    since a score alone doesn't explain a decline. Reuses the same
    Finnhub endpoint as get_finnhub_news; a separate, additive function
    so the existing count-only one is never touched. Returns an empty
    list on any failure — never raises, never blocks a scan.
    """
    if not api_key or api_key.startswith("YOUR_"):
        return []
    try:
        end   = datetime.now()
        start = end - timedelta(days=days)
        url   = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={start.strftime('%Y-%m-%d')}"
            f"&to={end.strftime('%Y-%m-%d')}&token={api_key}"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        articles = resp.json()[:n]
        return [a.get("headline", "").strip() for a in articles if a.get("headline")]
    except Exception as e:
        log.debug(f"Finnhub headlines error {ticker}: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def performance_pct(series: pd.Series, lookback: int) -> float:
    if len(series) < lookback + 1:
        lookback = len(series) - 1
    if lookback <= 0:
        return 0.0
    return round((series.iloc[-1] / series.iloc[-lookback] - 1) * 100, 1)


def compute_rs(stock_close: pd.Series, bench_close: pd.Series, lookback: int = 63) -> float:
    try:
        s = stock_close.copy()
        b = bench_close.copy()
        if hasattr(s.index, 'tz') and s.index.tz is not None:
            s.index = s.index.tz_convert(None)
        if hasattr(b.index, 'tz') and b.index.tz is not None:
            b.index = b.index.tz_convert(None)
        s.index = pd.to_datetime(s.index).normalize()
        b.index = pd.to_datetime(b.index).normalize()
        common  = s.index.intersection(b.index)
        if len(common) >= lookback:
            sa = s.reindex(common).dropna()
            ba = b.reindex(common).dropna()
            if len(sa) >= lookback and len(ba) >= lookback:
                sr = sa.iloc[-1] / sa.iloc[-lookback] - 1
                br = ba.iloc[-1] / ba.iloc[-lookback] - 1
                if br != 0:
                    return round((sr / abs(br)) * 100, 1)
        if len(s) >= lookback and len(b) >= lookback:
            sr = s.iloc[-1] / s.iloc[-lookback] - 1
            br = b.iloc[-1] / b.iloc[-lookback] - 1
            if br != 0:
                return round((sr / abs(br)) * 100, 1)
        return 0.0
    except Exception as ex:
        log.debug(f"RS error: {ex}")
        return 0.0


def detect_stage(price: float, ma50: float, ma200: float) -> str:
    if price > ma50 > ma200:   return "2 ✅ Uptrend"
    if price > ma200 > ma50:   return "1 ⏳ Base"
    if price < ma50 < ma200:   return "4 🔴 Downtrend"
    if price < ma200:          return "3 ⚠️ Top/Decline"
    if price > ma50 and price > ma200:
        # Price has gapped above BOTH averages before the 50-day has had
        # time to cross above the 200-day — the signature of a sudden,
        # sharp catalyst move (earnings surprise, drug trial results,
        # etc.), not a gradual trend. Genuinely bullish right now, but
        # the moving-average structure itself hasn't confirmed it yet.
        return "2 ⚡ Uptrend (gap)"
    return "? Unknown"


def detect_early_entry(close: pd.Series, ma50: float, ma200: float,
                       hist: pd.DataFrame) -> dict:
    result = {
        "early_entry":          False,
        "early_entry_type":     "",
        "fresh_200ma_cross":    False,
        "fresh_50ma_cross":     False,
        "pullback_to_50ma":     False,
        "low_adr_base":         False,
        "early_entry_score":    0,
        "days_since_200ma_cross": None,
    }

    if len(close) < 15:
        return result

    current = close.iloc[-1]
    signals = []
    score   = 0

    close_arr  = close.values
    ma200_arr  = close.rolling(200).mean().values
    cross_day  = None
    for i in range(max(1, len(close_arr)-15), len(close_arr)):
        if (close_arr[i] > ma200_arr[i] and close_arr[i-1] <= ma200_arr[i-1]
                and not pd.isna(ma200_arr[i]) and not pd.isna(ma200_arr[i-1])):
            cross_day = len(close_arr) - i
            break

    if cross_day is not None and cross_day <= 10:
        result["fresh_200ma_cross"]           = True
        result["days_since_200ma_cross"]      = cross_day
        signals.append(f"Fresh 200MA Cross ({cross_day}d ago)")
        score += 8

    ma50_arr = close.rolling(50).mean().values
    for i in range(max(1, len(close_arr)-5), len(close_arr)):
        if (close_arr[i] > ma50_arr[i] and close_arr[i-1] <= ma50_arr[i-1]
                and not pd.isna(ma50_arr[i]) and not pd.isna(ma50_arr[i-1])):
            result["fresh_50ma_cross"] = True
            signals.append("Fresh 50MA Cross")
            score += 4
            break

    if ma50 > 0 and current > ma200:
        dist_50 = abs(current / ma50 - 1) * 100
        if dist_50 <= 3.0:
            result["pullback_to_50ma"] = True
            signals.append(f"Pullback to 50MA ({dist_50:.1f}% away)")
            score += 5

    if len(hist) >= 20:
        _adr = ((hist["High"] - hist["Low"]) / hist["Close"] * 100).iloc[-20:].mean()
        if _adr < 3.0:
            result["low_adr_base"] = True
            signals.append(f"Low-ADR Base ({_adr:.1f}%)")
            score += 3

    if len(hist) >= 4:
        consec_inside = 0
        for k in range(-1, -4, -1):
            try:
                if (hist["High"].iloc[k] <= hist["High"].iloc[k-1] and
                        hist["Low"].iloc[k] >= hist["Low"].iloc[k-1]):
                    consec_inside += 1
                else:
                    break
            except IndexError:
                break
        if consec_inside >= 2:
            signals.append(f"{consec_inside}x Inside Day Compression")
            score += 2

    if score > 0:
        result["early_entry"]       = True
        result["early_entry_type"]  = " | ".join(signals) if signals else ""
        result["early_entry_score"] = min(10, score)

    return result


def adr_pct(hist: pd.DataFrame, lookback: int = 20) -> float:
    if len(hist) < lookback:
        return 0.0
    w = hist.iloc[-lookback:]
    return round(((w["High"] - w["Low"]) / w["Close"] * 100).mean(), 2)


def price_vs_ma(price: float, ma: float) -> float:
    if not ma or ma == 0:
        return 0.0
    return round((price / ma - 1) * 100, 1)


def detect_base_breakout(hist: pd.DataFrame, lookback_weeks: int = 8) -> Tuple[bool, str]:
    bars = min(lookback_weeks * 5, len(hist) - 1)
    if bars < 15:
        return False, "Insufficient data"
    window    = hist.iloc[-bars:]
    current   = hist["Close"].iloc[-1]
    high      = window["High"].max()
    low       = window["Low"].min()
    avg_vol   = window["Volume"].mean()
    today_vol = hist["Volume"].iloc[-1]
    depth     = (high - low) / high * 100
    vol_surge = today_vol > avg_vol * 1.4
    recent_range = (hist["High"].iloc[-15:].max() - hist["Low"].iloc[-15:].min()) / hist["Close"].iloc[-15:].mean() * 100
    prior_range  = (hist["High"].iloc[-bars:-15].max() - hist["Low"].iloc[-bars:-15].min()) / hist["Close"].iloc[-bars:-15].mean() * 100 if bars > 15 else recent_range
    contracting  = recent_range < prior_range * 0.75
    pivot        = current >= high * 0.995
    if depth < 12 and pivot and vol_surge:          return True,  "Flat Base Breakout"
    if depth < 12 and pivot:                        return False, "Tight — Watch Vol"
    if 12 <= depth <= 30 and pivot and vol_surge:   return True,  "Cup Breakout"
    if 12 <= depth <= 30 and contracting and current >= high * 0.90:
        return False, f"Handle Forming ({depth:.0f}% base)"
    if depth < 20 and contracting:                  return False, f"Tight Base ({depth:.0f}%)"
    if current >= high * 0.95:                      return False, "Near High (No Vol)"
    if depth > 40:                                  return False, f"Deep Correction ({depth:.0f}%)"
    return False, f"Basing ({depth:.0f}%)"


def volume_surge_ratio(hist: pd.DataFrame, short: int = 5, long: int = 50) -> float:
    if len(hist) < long:
        return 1.0
    return round(hist["Volume"].iloc[-short:].mean() / hist["Volume"].iloc[-long:].mean(), 2)


def earnings_momentum_proxy(news_count: int, perf_3m: float) -> str:
    if news_count >= 3 and perf_3m > 20: return "Strong"
    if news_count >= 1 or perf_3m > 10:  return "Moderate"
    return "Weak"


# ── Alpha Vantage — top-level import with fallback ────────────────────────────
try:
    from modules.alpha_vantage import analyse_eps, _cache_path, _cache_valid
    _AV_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _AV_AVAILABLE = False
    def analyse_eps(*a, **k): return None
    def _cache_path(*a, **k): return type("P", (), {"exists": lambda s: False, "stat": lambda s: type("S", (), {"st_mtime": 0})()})()
    def _cache_valid(*a, **k): return False

# ── Benchmark Cache ────────────────────────────────────────────────────────────
_bench_cache: Dict[str, pd.Series] = {}

# ── NGX Pulse module ────────────────────────────────────────────────────────
try:
    from modules.ngx_pulse import (
        get_ngx_history_for_scan as ngxp_get_history,
        get_ngx_index as ngxp_get_index,
        get_all_stocks_lookup as ngxp_get_lookup,
        validate_api_key as ngxp_validate,
    )
    _HAS_NGX_PULSE = True
except ImportError:
    _HAS_NGX_PULSE = False
    def ngxp_get_history(ticker, api_key): return None
    def ngxp_get_index(api_key): return None
    def ngxp_get_lookup(api_key): return {}
    def ngxp_validate(api_key): return {"ok": False, "message": "NGX Pulse module not installed"}

# ── NGN Market module ───────────────────────────────────────────────────────
try:
    from modules.ngn_market import (
        get_ngn_equity_history as ngnm_get_history,
        get_ngn_asi_index as ngnm_get_index,
        validate_api_key as ngnm_validate,
    )
    _HAS_NGN_MARKET = True
except ImportError:
    _HAS_NGN_MARKET = False
    def ngnm_get_history(ticker, api_key): return None
    def ngnm_get_index(api_key): return None
    def ngnm_validate(api_key): return {"ok": False, "message": "NGN Market module not installed"}

# ── Market Cap Cache (Emerging Gems) ──────────────────────────────────────────
_mcap_cache: Dict[str, dict] = {}

def get_market_cap_data(ticker: str) -> Dict:
    if ticker in _mcap_cache:
        return _mcap_cache[ticker]
    result = {
        "market_cap": None, "market_cap_bn": None,
        "avg_volume_30d": None, "is_gem": False,
        "liquidity_warn": False, "mcap_category": "Unknown",
    }
    try:
        info    = yf.Ticker(ticker).fast_info
        mcap    = getattr(info, "market_cap", None)
        avg_vol = getattr(info, "three_month_average_volume", None)
        if mcap:
            result["market_cap"]    = mcap
            result["market_cap_bn"] = round(mcap / 1e9, 2)
            if mcap < 300_000_000:       result["mcap_category"] = "Micro Cap"
            elif mcap < 2_000_000_000:   result["mcap_category"] = "Small Cap"
            elif mcap < 10_000_000_000:  result["mcap_category"] = "Mid Cap"
            elif mcap < 200_000_000_000: result["mcap_category"] = "Large Cap"
            else:                        result["mcap_category"] = "Mega Cap"
            result["is_gem"] = 100_000_000 <= mcap <= 5_000_000_000
        if avg_vol:
            result["avg_volume_30d"] = int(avg_vol)
            result["liquidity_warn"] = avg_vol < 300_000
    except Exception:
        pass
    _mcap_cache[ticker] = result
    return result
_fundamentals_cache: Dict[str, dict] = {}

def get_fundamentals_data(ticker: str) -> Dict:
    """
    Extended fundamentals for strategy scoring (Long-Term/Dividend/Value).
    Uses yf.Ticker(ticker).info — a free HTTP call, no paid API involved.
    Only called for stocks that already passed the scan gates (Pass 3),
    not the full universe, to keep request volume reasonable. Cached per session.

    Note: Asset Turnover and Inventory Turnover are NOT included — they
    aren't in yfinance's free .info endpoint and would need separate
    balance-sheet/income-statement calls per stock, adding real latency
    to every scan. Left out deliberately rather than added unreliably.
    """
    if ticker in _fundamentals_cache:
        return _fundamentals_cache[ticker]
    result = {
        "roe": None, "roa": None, "debt_to_equity": None,
        "free_cash_flow": None, "dividend_yield": None, "payout_ratio": None,
        "dividend_5y_avg": None, "price_to_sales": None, "ev_to_ebitda": None,
        "operating_margin": None, "profit_margin": None,
        "revenue_growth": None, "earnings_growth": None,
        # ── new additions ──
        "gross_margin": None, "current_ratio": None, "quick_ratio": None,
        "net_cash_position": None, "operating_cashflow": None,
        "institutional_ownership": None, "short_pct_float": None,
        "beta": None, "forward_pe": None,
    }
    try:
        info = yf.Ticker(ticker).info or {}
        result["roe"]              = info.get("returnOnEquity")
        result["roa"]              = info.get("returnOnAssets")
        result["debt_to_equity"]   = info.get("debtToEquity")
        result["free_cash_flow"]   = info.get("freeCashflow")
        result["dividend_yield"]   = info.get("dividendYield")
        result["payout_ratio"]     = info.get("payoutRatio")
        result["dividend_5y_avg"]  = info.get("fiveYearAvgDividendYield")
        result["price_to_sales"]   = info.get("priceToSalesTrailing12Months")
        result["ev_to_ebitda"]     = info.get("enterpriseToEbitda")
        result["operating_margin"] = info.get("operatingMargins")
        result["profit_margin"]    = info.get("profitMargins")
        result["revenue_growth"]   = info.get("revenueGrowth")
        result["earnings_growth"]  = info.get("earningsGrowth")

        # ── new additions ──
        result["gross_margin"]     = info.get("grossMargins")
        result["current_ratio"]    = info.get("currentRatio")
        result["quick_ratio"]      = info.get("quickRatio")
        result["operating_cashflow"] = info.get("operatingCashflow")
        result["institutional_ownership"] = info.get("heldPercentInstitutions")
        result["short_pct_float"]  = info.get("shortPercentOfFloat")
        result["beta"]             = info.get("beta")
        result["forward_pe"]       = info.get("forwardPE")

        total_cash = info.get("totalCash")
        total_debt = info.get("totalDebt")
        if total_cash is not None and total_debt is not None:
            result["net_cash_position"] = total_cash - total_debt

    except Exception as e:
        log.debug(f"Fundamentals fetch failed {ticker}: {e}")
    _fundamentals_cache[ticker] = result
    return result
def gem_score_boost(score: float, rs3: float, breaking_out: bool,
                    of_score: int, pa_score: int, gem_cfg: dict,
                    ee_score: int = 0, mcap: float = None) -> float:
    boosts = gem_cfg.get("score_boosts", {})
    bonus  = 0

    if rs3 >= boosts.get("rs_bonus_threshold", 150):
        bonus += boosts.get("rs_bonus_points", 5)
    elif rs3 >= 100:
        bonus += 3

    if breaking_out:
        bonus += boosts.get("breakout_bonus", 5) + 3

    bonus += of_score * (boosts.get("of_persistence_multiplier", 1.5) - 1)
    bonus += pa_score * (boosts.get("pa_patterns_multiplier", 1.4) - 1)
    bonus += ee_score * 1.5

    if mcap is not None:
        if mcap < 300_000_000:       bonus += 6
        elif mcap < 1_000_000_000:   bonus += 4
        elif mcap < 2_000_000_000:   bonus += 2

    return round(min(100, score + bonus), 1)

_BENCH_FALLBACKS = {
    "^R25I": "SMMD",
    "^RAG":  "IWZ",
    "^RUA":  "IWV",
    "^RUT":  "IWM",
    "^RLG":  "IWF",
}

def get_benchmark(symbol: str = "^GSPC", period: str = "1y") -> pd.Series:
    if symbol not in _bench_cache:
        _symbols_to_try = [symbol]
        if symbol in _BENCH_FALLBACKS:
            _symbols_to_try.append(_BENCH_FALLBACKS[symbol])

        _loaded = False
        for _sym in _symbols_to_try:
            try:
                data = yf.Ticker(_sym).history(period=period)["Close"]
                if hasattr(data.index, 'tz') and data.index.tz is not None:
                    data.index = data.index.tz_convert(None)
                data.index = pd.to_datetime(data.index).normalize()
                if len(data) > 50:
                    _bench_cache[symbol] = data
                    _src = f"{_sym}" if _sym == symbol else f"{_sym} (fallback for {symbol})"
                    log.info(f"Benchmark loaded: {_src} — {len(data)} bars")
                    _loaded = True
                    break
            except Exception as e:
                log.warning(f"Benchmark {_sym} failed: {e} — trying fallback…")

        if not _loaded:
            log.warning(f"All benchmark attempts failed for {symbol} — RS vs this benchmark will be None")
            _bench_cache[symbol] = pd.Series(dtype=float)

    return _bench_cache[symbol]


def batch_fetch_history(tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV history for many tickers in a handful of batched requests
    instead of one HTTP call per ticker. Yahoo blocks/rate-limits individual
    per-ticker requests far more aggressively on cloud-hosted IPs (Streamlit
    Cloud, etc.) than it does bulk download calls. Any ticker missing or too
    short from the batch result silently falls through to analyze_stock()'s
    existing per-ticker fetch as a fallback — this never removes a data path,
    only adds a faster one in front of it.
    """
    result: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return result

    chunk_size = 100
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    for chunk in chunks:
        try:
            raw = yf.download(
                chunk, period=period, group_by="ticker",
                threads=True, progress=False, auto_adjust=True,
            )
        except Exception as e:
            log.warning(f"Batch download failed for a chunk of {len(chunk)}: {e}")
            continue

        if raw is None or raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            for tk in chunk:
                try:
                    sub = raw[tk].dropna(how="all")
                    if len(sub) > 0:
                        result[tk] = sub
                except Exception:
                    continue
        elif len(chunk) == 1:
            sub = raw.dropna(how="all")
            if len(sub) > 0:
                result[chunk[0]] = sub

    log.info(f"Batch fetch: {len(result)}/{len(tickers)} tickers returned data")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ORDER FLOW PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def order_flow_persistence(hist: pd.DataFrame, lookback: int = 10) -> Dict:
    if len(hist) < lookback + 1:
        return {
            "of_persistence_score": 0,
            "of_directional_bias":  "Neutral",
            "of_up_vol_ratio":      1.0,
            "of_bullish_days_%":    50.0,
            "of_consecutive_up":    0,
        }

    window   = hist.iloc[-lookback:].copy()
    prev_cls = hist["Close"].iloc[-lookback-1:-1].values
    window["up"] = window["Close"].values > prev_cls

    up_days     = int(window["up"].sum())
    bullish_pct = round(up_days / lookback * 100, 1)
    vol_up      = window.loc[window["up"],  "Volume"].sum()
    vol_down    = window.loc[~window["up"], "Volume"].sum()
    up_vol_ratio = round(vol_up / vol_down, 2) if vol_down > 0 else 5.0

    max_run = cur_run = 0
    for u in window["up"]:
        cur_run = cur_run + 1 if u else 0
        max_run = max(max_run, cur_run)

    if bullish_pct >= 70 and up_vol_ratio >= 1.5:   bias = "Strong Bullish"
    elif bullish_pct >= 60:                          bias = "Bullish"
    elif bullish_pct <= 30 and up_vol_ratio <= 0.7: bias = "Strong Bearish"
    elif bullish_pct <= 40:                          bias = "Bearish"
    else:                                            bias = "Neutral"

    score = 0
    if bullish_pct >= 70:     score += 4
    elif bullish_pct >= 60:   score += 2
    if up_vol_ratio >= 2.0:   score += 3
    elif up_vol_ratio >= 1.5: score += 2
    elif up_vol_ratio >= 1.2: score += 1
    if max_run >= 4:          score += 1

    return {
        "of_persistence_score": min(8, score),
        "of_directional_bias":  bias,
        "of_up_vol_ratio":      up_vol_ratio,
        "of_bullish_days_%":    bullish_pct,
        "of_consecutive_up":    max_run,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VWAP / AUCTION MARKET THEORY
# ══════════════════════════════════════════════════════════════════════════════

def compute_vwap(hist: pd.DataFrame, lookback: int = 20) -> Dict:
    if len(hist) < lookback:
        return {
            "vwap": None, "vwap_upper": None, "vwap_lower": None,
            "vs_vwap_%": 0.0, "vwap_position": "Unknown",
            "vwap_slope": "Flat", "vwap_score": 0,
        }

    w = hist.iloc[-lookback:].copy()
    w["typical"] = (w["High"] + w["Low"] + w["Close"]) / 3
    w["tp_vol"]  = w["typical"] * w["Volume"]
    vwap         = w["tp_vol"].sum() / w["Volume"].sum()

    w["dev_sq"]  = ((w["typical"] - vwap) ** 2) * w["Volume"]
    vwap_std     = np.sqrt(w["dev_sq"].sum() / w["Volume"].sum())
    upper_band   = vwap + vwap_std
    lower_band   = vwap - vwap_std
    current      = hist["Close"].iloc[-1]
    vs_vwap      = round((current / vwap - 1) * 100, 2)

    if current > upper_band:    position = "Extended Above VWAP"
    elif current > vwap:        position = "Above VWAP"
    elif current > lower_band:  position = "Below VWAP"
    else:                       position = "Extended Below VWAP"

    if len(hist) >= lookback + 5:
        w_prev = hist.iloc[-lookback-5:-5].copy()
        w_prev["typical"] = (w_prev["High"] + w_prev["Low"] + w_prev["Close"]) / 3
        w_prev["tp_vol"]  = w_prev["typical"] * w_prev["Volume"]
        vwap_prev  = w_prev["tp_vol"].sum() / w_prev["Volume"].sum()
        slope_pct  = (vwap - vwap_prev) / vwap_prev * 100
        slope = "Rising" if slope_pct > 0.5 else ("Falling" if slope_pct < -0.5 else "Flat")
    else:
        slope = "Flat"

    score = 0
    if current > vwap and slope == "Rising": score = 4
    elif current > vwap:                     score = 2
    elif current > lower_band:               score = 1
    if len(hist) >= 2 and hist["Close"].iloc[-2] < vwap <= current:
        score = min(4, score + 1)

    return {
        "vwap":          round(vwap, 2),
        "vwap_upper":    round(upper_band, 2),
        "vwap_lower":    round(lower_band, 2),
        "vs_vwap_%":     vs_vwap,
        "vwap_position": position,
        "vwap_slope":    slope,
        "vwap_score":    score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MARKET STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def detect_market_structure(hist: pd.DataFrame, swing_lookback: int = 5) -> Dict:
    if len(hist) < swing_lookback * 4 + 10:
        return {
            "ms_structure": "Insufficient Data", "ms_hh_hl": False,
            "ms_lh_ll": False, "ms_break_of_struct": False,
            "ms_last_swing_high": None, "ms_last_swing_low": None,
        }

    n = swing_lookback
    highs = hist["High"].values
    lows  = hist["Low"].values
    swing_highs, swing_lows = [], []

    for i in range(n, len(hist) - n):
        if all(highs[i] >= highs[i-j] for j in range(1, n+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, n+1)):
            swing_highs.append((i, highs[i]))
        if all(lows[i] <= lows[i-j] for j in range(1, n+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, n+1)):
            swing_lows.append((i, lows[i]))

    last_sh = swing_highs[-1][1] if swing_highs else None
    last_sl = swing_lows[-1][1]  if swing_lows  else None
    hh_hl = lh_ll = False

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        sh_vals = [s[1] for s in swing_highs[-2:]]
        sl_vals = [s[1] for s in swing_lows[-2:]]
        hh_hl   = sh_vals[-1] > sh_vals[-2] and sl_vals[-1] > sl_vals[-2]
        lh_ll   = sh_vals[-1] < sh_vals[-2] and sl_vals[-1] < sl_vals[-2]

    current = hist["Close"].iloc[-1]
    bos     = bool((last_sh and current > last_sh) or (last_sl and current < last_sl))
    structure = "Bullish (HH/HL)" if hh_hl else ("Bearish (LH/LL)" if lh_ll else "Transitioning")

    return {
        "ms_structure":       structure,
        "ms_hh_hl":           hh_hl,
        "ms_lh_ll":           lh_ll,
        "ms_break_of_struct": bos,
        "ms_last_swing_high": round(last_sh, 2) if last_sh else None,
        "ms_last_swing_low":  round(last_sl, 2) if last_sl else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PRICE ACTION PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

def detect_price_action_patterns(hist: pd.DataFrame) -> Dict:
    if len(hist) < 10:
        return {
            "pa_patterns": [], "pa_engulfing": None, "pa_sfp": None,
            "pa_inside_day": False, "pa_context_candle": None, "pa_score": 0,
        }

    c0 = hist.iloc[-1]
    c1 = hist.iloc[-2]
    patterns, engulfing, sfp, inside_day, context_c, score = [], None, None, False, None, 0

    o0, h0, l0, close0, v0 = c0["Open"], c0["High"], c0["Low"], c0["Close"], c0["Volume"]
    o1, h1, l1, close1     = c1["Open"], c1["High"], c1["Low"], c1["Close"]
    avg_vol = hist["Volume"].iloc[-20:].mean()

    c1_bearish = close1 < o1
    c1_bullish = close1 > o1
    c0_bullish = close0 > o0
    c0_bearish = close0 < o0

    if c1_bearish and c0_bullish and close0 > o1 and o0 < close1:
        engulfing = "Bullish"; patterns.append("Bullish Engulfing"); score += 2
    if c1_bullish and c0_bearish and o0 > close1 and close0 < o1:
        engulfing = "Bearish"; patterns.append("Bearish Engulfing")

    lookback_sfp = hist.iloc[-20:-1]
    recent_high  = lookback_sfp["High"].max()
    recent_low   = lookback_sfp["Low"].min()

    if l0 < recent_low and close0 > recent_low:
        sfp = "Bullish SFP"; patterns.append("Bullish SFP (Bear Trap)"); score += 3
    if h0 > recent_high and close0 < recent_high:
        sfp = "Bearish SFP"; patterns.append("Bearish SFP (Bull Trap)")

    if h0 <= h1 and l0 >= l1:
        inside_day = True; patterns.append("Inside Day (Compression)"); score += 1

    candle_range = h0 - l0
    if candle_range > 0 and v0 > avg_vol * 1.2:
        close_pct = (close0 - l0) / candle_range
        if close_pct >= 0.75:
            context_c = "Bullish"; patterns.append("Bullish Context Candle"); score += 2
        elif close_pct <= 0.25:
            context_c = "Bearish"; patterns.append("Bearish Context Candle")

    bull_signals = sum([engulfing=="Bullish", sfp=="Bullish SFP",
                        inside_day and c0_bullish, context_c=="Bullish"])
    if bull_signals >= 2:
        patterns.append("PA Confluence"); score = min(5, score + 1)

    return {
        "pa_patterns":       patterns,
        "pa_engulfing":      engulfing,
        "pa_sfp":            sfp,
        "pa_inside_day":     inside_day,
        "pa_context_candle": context_c,
        "pa_score":          min(5, score),
    }


# ══════════════════════════════════════════════════════════════════════════════
# WEEKLY TIMEFRAME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

_weekly_cache: Dict[str, dict] = {}

def analyse_weekly(ticker: str, bench_close: pd.Series) -> dict:
    if ticker in _weekly_cache:
        return _weekly_cache[ticker]

    result = {
        "weekly_stage":          "Unknown",
        "weekly_above_10wma":    False,
        "weekly_above_40wma":    False,
        "weekly_10gt40":         False,
        "weekly_rs":             None,
        "weekly_base_tight":     False,
        "weekly_trending_up":    False,
        "weekly_hh_hl":          False,
        "weekly_confirmed":      False,
        "weekly_contradicts":    False,
        "weekly_score":          0,
        "weekly_base_depth_%":   None,
        "weekly_consec_up_wks":  0,
    }

    try:
        hist_w = yf.Ticker(ticker).history(period="2y", interval="1wk")
        if len(hist_w) < 40:
            _weekly_cache[ticker] = result
            return result

        if hasattr(hist_w.index, "tz") and hist_w.index.tz is not None:
            hist_w.index = hist_w.index.tz_convert(None)
        hist_w.index = pd.to_datetime(hist_w.index).normalize()

        close_w = hist_w["Close"]
        cur_w   = float(close_w.iloc[-1])

        ma10w = float(close_w.rolling(10).mean().iloc[-1])
        ma40w = float(close_w.rolling(40).mean().iloc[-1])

        above_10w = cur_w > ma10w
        above_40w = cur_w > ma40w
        ma10_gt40 = ma10w > ma40w

        result["weekly_above_10wma"] = above_10w
        result["weekly_above_40wma"] = above_40w
        result["weekly_10gt40"]      = ma10_gt40

        if cur_w > ma10w > ma40w:
            result["weekly_stage"] = "2 ✅ Weekly Uptrend"
        elif cur_w > ma40w:
            result["weekly_stage"] = "1 ⏳ Weekly Base"
        elif cur_w < ma10w < ma40w:
            result["weekly_stage"] = "4 🔴 Weekly Downtrend"
        else:
            result["weekly_stage"] = "3 ⚠️ Weekly Topping"

        if bench_close is not None and len(bench_close) > 0:
            try:
                bench_w = bench_close.resample("W").last().dropna()
                common  = close_w.index.intersection(bench_w.index)
                if len(common) >= 13:
                    cw_a = close_w.reindex(common).dropna()
                    bw_a = bench_w.reindex(common).dropna()
                    if len(cw_a) >= 13:
                        sr = float(cw_a.iloc[-1] / cw_a.iloc[-13] - 1)
                        br = float(bw_a.iloc[-1] / bw_a.iloc[-13] - 1)
                        result["weekly_rs"] = round((sr / abs(br)) * 100, 1) if br != 0 else None
            except Exception:
                pass

        last8w = hist_w.iloc[-8:]
        if len(last8w) >= 5:
            wk_high = float(last8w["High"].max())
            wk_low  = float(last8w["Low"].min())
            wk_depth = (wk_high - wk_low) / wk_high * 100 if wk_high > 0 else 100
            result["weekly_base_depth_%"] = round(wk_depth, 1)
            result["weekly_base_tight"]   = wk_depth < 15

        consec_up = 0
        for i in range(-1, -min(6, len(hist_w)+1), -1):
            try:
                if float(hist_w["Close"].iloc[i]) > float(hist_w["Close"].iloc[i-1]):
                    consec_up += 1
                else:
                    break
            except IndexError:
                break
        result["weekly_consec_up_wks"] = consec_up
        result["weekly_trending_up"]   = consec_up >= 2

        if len(hist_w) >= 6:
            highs6 = hist_w["High"].iloc[-6:].values
            lows6  = hist_w["Low"].iloc[-6:].values
            wk_hh  = highs6[-1] > highs6[-3] > highs6[-5]
            wk_hl  = lows6[-1]  > lows6[-3]  > lows6[-5]
            result["weekly_hh_hl"] = bool(wk_hh and wk_hl)

        confirmed = (
            above_40w and
            ma10_gt40 and
            (result["weekly_rs"] is None or result["weekly_rs"] > 0)
        )
        contradicts = (
            not above_40w and
            not above_10w and
            ma10w < ma40w
        )

        result["weekly_confirmed"]   = confirmed
        result["weekly_contradicts"] = contradicts

        wscore = 0
        if above_40w and ma10_gt40:          wscore += 4
        elif above_40w:                       wscore += 2
        if result["weekly_hh_hl"]:           wscore += 2
        if result["weekly_base_tight"]:      wscore += 2
        if result["weekly_trending_up"]:     wscore += 1
        if result.get("weekly_rs") and result["weekly_rs"] > 100: wscore += 1

        result["weekly_score"] = min(10, wscore)

    except Exception as e:
        log.debug(f"Weekly analysis failed {ticker}: {e}")

    _weekly_cache[ticker] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE STOCK
# ══════════════════════════════════════════════════════════════════════════════

def analyze_stock(ticker: str, cfg: dict,
                  market: str = "auto",
                  hist_override: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """
    Full ApexScan analysis for one ticker.
    market="auto" detects from ticker suffix (.LG = NGX, else US).
    hist_override — pre-fetched OHLCV DataFrame from a batch download, used
    instead of an individual yf.Ticker(...).history() call when available.
    """
    market = resolve_market(ticker, market)
    _mkt_key     = "ng" if market == "ng" else "us"
    _is_ngx      = (market == "ng")
    _mkt_label   = "NGX" if _is_ngx else "US"
    thresholds   = cfg["thresholds"].get(_mkt_key, cfg["thresholds"]["us"])
    bench_symbol = cfg["benchmarks"].get(_mkt_key, cfg["benchmarks"]["us"])

    _ngx_bench   = None

    if not _is_ngx:
        bench_r2500  = cfg.get("benchmarks", {}).get("russell_2500", "^R25I")
        bench_r3000g = cfg.get("benchmarks", {}).get("russell_3000_growth", "^RAG")
    else:
        bench_r2500 = bench_r3000g = None
    bench_r2500   = cfg.get("benchmarks", {}).get("russell_2500",       "^R25I")
    bench_r3000g  = cfg.get("benchmarks", {}).get("russell_3000_growth", "^RAG")
    finnhub_key  = cfg.get("finnhub_key", "")
    av_key       = cfg.get("alpha_vantage_key", "")
    av_cfg       = cfg.get("alpha_vantage", {})
    ngxp_key     = cfg.get("ngx_pulse_key", "")
    ngnm_key     = cfg.get("ngn_market_key", "")
    use_av       = bool(av_key and not av_key.startswith("YOUR_")) and not _is_ngx

    try:
        if _is_ngx:
            hist = None
            if _HAS_NGX_PULSE and ngxp_key:
                hist = ngxp_get_history(ticker, ngxp_key)
            if (hist is None or len(hist) == 0) and _HAS_NGN_MARKET and ngnm_key:
                hist = ngnm_get_history(ticker, ngnm_key)
            if hist is None or len(hist) == 0:
                hist = yf.Ticker(ticker).history(period=cfg["scan"]["history_period"])
            if _ngx_bench is None:
                _ngx_idx = ngxp_get_index(ngxp_key) if _HAS_NGX_PULSE else None
                if _ngx_idx is None and _HAS_NGN_MARKET and ngnm_key:
                    _ngx_idx = ngnm_get_index(ngnm_key)
                _ngx_bench = _ngx_idx["df"]["Close"] if _ngx_idx else None
        else:
            _min_bars_check = cfg["scan"]["min_history_bars"]
            if hist_override is not None and len(hist_override) >= _min_bars_check:
                hist = hist_override
            else:
                hist = yf.Ticker(ticker).history(period=cfg["scan"]["history_period"])

        _min_bars = 30 if _is_ngx else cfg["scan"]["min_history_bars"]
        if len(hist) < _min_bars:
            log.debug(f"{ticker}: only {len(hist)} bars (min={_min_bars}), skipping")
            return None

        close         = hist["Close"]
        current_price = close.iloc[-1]

        perf_1m = performance_pct(close, 21)
        perf_3m = performance_pct(close, 63)
        perf_6m = performance_pct(close, 126)

        high_52w     = close.rolling(252).max().iloc[-1]
        if pd.isna(high_52w) or high_52w == 0:
            high_52w = float(close.max())
        near_52wh    = current_price >= high_52w * thresholds["near_52w_high"]
        pct_off_high = round((current_price / high_52w - 1) * 100, 1) if high_52w else 0.0

        ma50         = close.rolling(50).mean().iloc[-1]
        ma200        = close.rolling(200).mean().iloc[-1]
        if pd.isna(ma200):
            log.debug(f"{ticker}: only {len(close)} bars — 200MA not computable, skipping")
            return None
        above_50ma   = bool(current_price > ma50)
        above_200ma  = bool(current_price > ma200)
        ma50_gt_200  = bool(ma50 > ma200)
        stage        = detect_stage(current_price, ma50, ma200)
        vs_50ma_pct  = price_vs_ma(current_price, ma50)
        vs_200ma_pct = price_vs_ma(current_price, ma200)

        # ── Moving-average slope — is the trend turning, not just where
        # price sits relative to it. Computed live from history already
        # fetched, no cross-session tracking needed. A "flattening or
        # turning higher" 50/200-day average, while still building a base,
        # is a real pre-breakout tell that a static above/below check misses.
        ma50_series  = close.rolling(50).mean()
        ma200_series = close.rolling(200).mean()
        ma50_slope_pct, ma200_slope_pct = None, None
        ma50_turning_up, ma200_turning_up = False, False
        if len(ma50_series.dropna()) > 10:
            _ma50_then = ma50_series.iloc[-11]
            if pd.notna(_ma50_then) and _ma50_then != 0:
                ma50_slope_pct = round((ma50 / _ma50_then - 1) * 100, 2)
                ma50_turning_up = ma50_slope_pct > -0.5   # flat or rising, not still declining
        if len(ma200_series.dropna()) > 20:
            _ma200_then = ma200_series.iloc[-21]
            if pd.notna(_ma200_then) and _ma200_then != 0:
                ma200_slope_pct = round((ma200 / _ma200_then - 1) * 100, 2)
                ma200_turning_up = ma200_slope_pct > -0.3

        if _is_ngx and _ngx_bench is not None and len(_ngx_bench) > 63:
            bench = _ngx_bench
            log.info(f"{ticker}: using NGX All-Share benchmark for RS")
        else:
            bench = get_benchmark(bench_symbol)
        rs_3m      = compute_rs(close, bench, 63)
        rs_6m      = compute_rs(close, bench, 126)

        if not _is_ngx and bench_r2500 and bench_r3000g:
            _bench_r2500  = get_benchmark(bench_r2500,  cfg["scan"]["history_period"])
            _bench_r3000g = get_benchmark(bench_r3000g, cfg["scan"]["history_period"])
            rs_r2500      = compute_rs(close, _bench_r2500,  63)  if len(_bench_r2500)  > 63  else None
            rs_r2500_6m   = compute_rs(close, _bench_r2500, 126)  if len(_bench_r2500)  > 126 else None
            rs_r3000g     = compute_rs(close, _bench_r3000g, 63)  if len(_bench_r3000g) > 63  else None
            rs_r3000g_6m  = compute_rs(close, _bench_r3000g, 126) if len(_bench_r3000g) > 126 else None
        else:
            rs_r2500 = rs_r2500_6m = rs_r3000g = rs_r3000g_6m = None

        rs_multi_leader = (
            not _is_ngx and
            rs_3m is not None and rs_3m > 100 and
            (rs_r2500  is None or rs_r2500  > 100) and
            (rs_r3000g is None or rs_r3000g > 100)
        )

        adr          = adr_pct(hist, 20)
        vol_today    = int(hist["Volume"].iloc[-1])
        vol_avg_20   = int(hist["Volume"].rolling(20).mean().iloc[-1]) if len(hist) >= 20 else vol_today
        vol_filter   = max(vol_today, vol_avg_20)
        vol_surge    = volume_surge_ratio(hist)
        breaking_out, pattern = detect_base_breakout(hist)

        fh       = get_finnhub_news(ticker, finnhub_key)
        av_data  = None

        if use_av:
            try:
                av_data = analyse_eps(
                    ticker, av_key,
                    lookback_q   = av_cfg.get("eps_lookback_quarters", 4),
                    min_growth   = av_cfg.get("min_eps_growth", 15),
                    min_surprise = av_cfg.get("min_surprise_pct", 5),
                    cache_hours  = av_cfg.get("cache_hours", 168),
                )
            except Exception as av_err:
                log.debug(f"AV error {ticker}: {av_err}")

        earn_mom = (av_data["eps_momentum"]
                    if av_data and av_data.get("eps_momentum") not in (None, "Unknown")
                    else earnings_momentum_proxy(fh["news_count"], perf_3m))

        _theme_src = cfg["ng_themes"] if _is_ngx else cfg["us_themes"]
        _cfg_theme = next((k for k, v in _theme_src.items() if ticker in v), None)

        _GICS_MAP = {
            "Energy": [
                "XOM","CVX","COP","SLB","BKR","HAL","PSX","VLO","MPC","EOG",
                "PXD","DVN","OXY","FANG","HES","APA","NOV","WHD","TRGP","KMI",
                "WMB","OKE","EPD","ET","PAA","MMP","LNG","AR","EQT","RRC",
                "CRC","SM","CIVI","MGY","ESTE","REX","FLNG","GMLP","SLNG",
                "BP","SHEL","TTE","ENB","TRP","SU","CVE","IMO","CNQ","MEG",
            ],
            "Materials": [
                "LIN","APD","SHW","ECL","IFF","PPG","RPM","FMC","CF","MOS","NTR",
                "NUE","STLD","CMC","RS","ATI","FCX","SCCO","AA","CLF","MP","ALB",
                "LAC","LTHM","SQM","VALE","RIO","BHP","GOLD","NEM","AEM","PAAS",
                "DOW","DD","LYB","HUN","CE","EMN","OLN","ASH","TROX","IOSP",
                "AG","EXK","SILV","CDE","HL","GPL","MUX","AUY","KGC","GATO",
                "MAG","SVM","FSM","ERO","ATX","VZLA","SAND","WPM","OR","RGLD",
            ],
            "Industrials": [
                "BA","RTX","LMT","NOC","GD","HII","TDG","HWM","GE","HON","MMM",
                "CAT","DE","EMR","ETN","PH","ITW","ROK","AME","ROP","CPRT","EXPD",
                "UPS","FDX","GXO","XPO","CHRW","JBHT","SAIA","TFII","ZTO",
                "DAL","UAL","AAL","LUV","ALK","SAVE","H","MAR","HLT","WH","CHH",
                "NCLH","RCL","CCL","EXPE","BKNG","ABNB","TRIP","MTN","VAIL",
                "WM","RSG","CTAS","VRSK","LDOS","SAIC","BAH","CACI","ACN",
                "AXON","TDY","HXL","KTOS","DRS","RKLB","ACHR","JOBY","LUNR",
            ],
            "Utilities": [
                "NEE","D","SO","DUK","AEP","SRE","PCG","XEL","AWK","ES","EXC",
                "ED","PPL","ETR","FE","AEE","CMS","DTE","LNT","PNW","WEC","NI",
                "BEP","BEPC","AES","NRG","CEG","VST","PEG","CNP","EVRG","AVA",
                "IDACORP","OGE","SPWR","NOVA","RUN","ENPH","FSLR","PLUG",
            ],
            "Healthcare": [
                "UNH","CI","CVS","HCA","MCK","CAH","DHR","TMO","ABT","MDT","SYK",
                "BSX","EW","ZBH","BDX","BAX","STE","HOLX","IQV","CRL","MTD","WAT",
                "LH","DGX","CTLT","VTRS","RPRX","JAZZ","ALKS","ITCI","ACAD",
                "LLY","ABBV","BMY","PFE","JNJ","MRK","AZN","NVO","GSK","SNY",
                "MRNA","BNTX","REGN","BIIB","GILD","IDXX","DXCM","ISRG","ILMN","VRTX",
                "ALNY","SGEN","BMRN","INCY","EXAS","RARE","NTLA","BEAM","CRSP","EDIT",
                "HIMS","TMDX","RXRX","SAGE","AUPH","AVXL","SNDX","PRAX","IMVT",
                "DNLI","KRTX","VRNA","AKRO","TARS","NKTR","ACAD","ARQT","GOSS",
            ],
            "Financials": [
                "JPM","GS","MS","BAC","WFC","C","AXP","BLK","SCHW","ICE","CME",
                "SPGI","MCO","AMP","PGR","MET","TRV","AFL","ALL","CB","HIG","L",
                "BX","KKR","APO","CG","ARES","TPG","BN","BAM","TROW","IVZ","BEN",
                "WTW","AON","MMC","USB","PNC","TFC","FITB","KEY","CFG","RF","HBAN",
                "COIN","HOOD","SOFI","AFRM","UPST","DAVE","OPEN","UWMC","MSTR",
                "V","MA","PYPL","SQ","BILL","SMAR","INTL","IBKR","LPLA","RJF",
            ],
            "Consumer Discretionary": [
                "AMZN","TSLA","HD","TGT","LOW","MCD","SBUX","YUM","CMG","DPZ",
                "QSR","EAT","DRI","TXRH","BLMN","BJRI","CAKE","SHAK","WING","PLNT",
                "BJ","FIVE","OLLI","F","GM","STLA","HOG","RACE","TM","HMC",
                "MGA","LEA","BWA","ALV","NKE","DECK","SKX","CROX","PVH","RL",
                "TPR","TIF","VFC","HBI","UA","LULU","ONON","CELH","RIVN",
                "LYFT","UBER","DASH","DKNG","RBLX","MTCH","ABNB","BKNG","EXPE",
                "ROST","DLTR","DG","BURL","TJX","COST","WMT","DUOL","CAVA",
            ],
            "Consumer Staples": [
                "PG","KO","PEP","PM","MO","CL","KMB","CHD","CLX","HRL",
                "SJM","CAG","CPB","GIS","K","MKC","HSY","TR","MDLZ","KHC",
                "STZ","BF-B","TAP","SAM","BUD","DEO","BTI","MNST","CELH",
                "WMT","COST","TGT","KR","SFM","GO","CASY","ATD",
            ],
            "Information Technology": [
                "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","QCOM","TXN","INTU",
                "AMD","ARM","AMAT","LRCX","KLAC","MU","MRVL","SMCI","CDNS","SNPS",
                "PANW","CRWD","FTNT","ZS","NET","DDOG","SNOW","PLTR","NOW","WDAY",
                "TEAM","HUBS","MDB","GTLB","PATH","AI","APPN","VEEV","BILL","TTD",
                "IBM","HPQ","HPE","DELL","NCR","CDW","WIT","INFY","CTSH","EPAM",
                "GLOB","DXC","CACI","LDOS","SAIC","BAH","ACN","IONQ","SOUN","BTDR",
                "TSM","ASML","ON","MPWR","ADI","MCHP","SWKS","QRVO","WOLF","ONTO",
            ],
            "Communication Services": [
                "META","GOOGL","GOOG","NFLX","SPOT","ROKU","TTD","SNAP","PINS","TWTR",
                "RDDT","MTCH","IAC","ZG","DASH","LYFT","UBER","ABNB","BKNG","EXPE",
                "DIS","PARA","WBD","FOXA","FOX","NWSA","NWS","NYT","SIRI","LSXMA",
                "T","VZ","TMUS","LUMN","FYBR","ATUS","CABO","CHTR","CMCSA",
                "EA","TTWO","ATVI","RBLX","U","DKNG","HOOD",
            ],
            "Real Estate": [
                "PLD","AMT","CCI","SBAC","EQIX","DLR","O","SPG","PSA","EXR",
                "AVB","EQR","UDR","ESS","MAA","CPT","NNN","VICI","MGM","WYNN","LVS",
                "HST","RHP","PK","SHO","PLYA","APLE","CLDT","CPLG","RLJ","XHR",
                "ARE","BXP","SLG","VNO","KIM","REG","FRT","WRE","AIV","NHI",
                "WELL","VTR","PEAK","HR","DOC","SBRA","LTC","NHC","CTRE","GMRE",
            ],
        }

        _TICKER_TO_GICS = {}
        for _sector, _tickers in _GICS_MAP.items():
            for _t in _tickers:
                _TICKER_TO_GICS[_t] = _sector

        if _cfg_theme:
            theme = _cfg_theme
        elif ticker in _TICKER_TO_GICS:
            theme = _TICKER_TO_GICS[ticker]
        else:
            try:
                _yf_info = yf.Ticker(ticker).fast_info
                _yf_sector = getattr(_yf_info, "sector", None)
                if not _yf_sector:
                    _yf_info2 = yf.Ticker(ticker).info
                    _yf_sector = _yf_info2.get("sector", None)
                _SECTOR_ALIASES = {
                    "Technology":               "Information Technology",
                    "Financial Services":       "Financials",
                    "Consumer Cyclical":        "Consumer Discretionary",
                    "Consumer Defensive":       "Consumer Staples",
                    "Basic Materials":          "Materials",
                    "Communication Services":   "Communication Services",
                    "Healthcare":               "Healthcare",
                    "Industrials":              "Industrials",
                    "Energy":                   "Energy",
                    "Utilities":                "Utilities",
                    "Real Estate":              "Real Estate",
                }
                theme = _SECTOR_ALIASES.get(_yf_sector, _yf_sector or "Information Technology")
            except Exception:
                theme = "Information Technology"

        gem_cfg   = cfg.get("emerging_gems", {})
        mcap_data = get_market_cap_data(ticker)
        is_gem    = mcap_data["is_gem"] or theme == "emerging_gems"
        avg_vol_30d = mcap_data.get("avg_volume_30d")
        liq_score = 3
        if avg_vol_30d:
            if avg_vol_30d >= 1_000_000:  liq_score = 3
            elif avg_vol_30d >= 300_000:  liq_score = 2
            elif avg_vol_30d >= 100_000:  liq_score = 1
            else:                         liq_score = 0

        of_data  = order_flow_persistence(hist, cfg.get("advanced", {}).get("of_lookback", 10))
        vwap_data= compute_vwap(hist, cfg.get("advanced", {}).get("vwap_lookback", 20))
        ms_data  = detect_market_structure(hist, cfg.get("advanced", {}).get("swing_lookback", 5))
        pa_data  = detect_price_action_patterns(hist)
        ee_data  = detect_early_entry(close, ma50, ma200, hist)
        wk_data  = analyse_weekly(ticker, bench)

        score = 0

        if perf_3m > thresholds["min_3m_perf"]:    score += min(40, perf_3m)

        if rs_3m > thresholds["rs_rating_min"]:     score += 25
        elif rs_3m > 50:                            score += 12

        if rs_r2500 is not None and rs_r2500 > 100: score += 3
        if rs_r3000g is not None and rs_r3000g > 100: score += 3
        if rs_multi_leader:                         score += 4

        if above_200ma and ma50_gt_200:             score += 15
        elif above_200ma:                           score += 7

        if near_52wh:                               score += 10
        if breaking_out:                            score += 10

        score += of_data["of_persistence_score"]
        score += pa_data["pa_score"]
        score += vwap_data["vwap_score"]
        if ms_data["ms_hh_hl"]:                    score += 2
        if ms_data["ms_break_of_struct"] and ms_data["ms_hh_hl"]: score += 1

        score += ee_data["early_entry_score"]

        if wk_data["weekly_confirmed"]:
            score += wk_data["weekly_score"]

        if av_data:                                 score += av_data.get("eps_score", 0)

        if not above_200ma and not ma50_gt_200:     score -= 20
        elif not above_200ma:                       score -= 10

        if perf_3m < 0:                             score -= 10
        elif perf_3m < thresholds["min_3m_perf"]:  score -= 5

        if rs_3m < 0:                               score -= 10
        elif rs_3m < 50:                            score -= 5

        if of_data["of_directional_bias"] == "Strong Bearish": score -= 5
        elif of_data["of_directional_bias"] == "Bearish":      score -= 2

        if pct_off_high < -40:                      score -= 10
        elif pct_off_high < -25:                    score -= 5

        if wk_data["weekly_contradicts"]:
            score -= 15
        elif not wk_data["weekly_confirmed"] and not wk_data["weekly_contradicts"]:
            score -= 3

        # Preserve the full-resolution score BEFORE the 0-100 display clamp.
        # Many genuinely different-quality stocks can tie at the visible
        # 100 ceiling (the raw total routinely exceeds 150+); score_raw
        # keeps them properly differentiated for ranking/analysis without
        # changing the displayed apex_score or any existing scan threshold.
        score_raw = score
        score = max(0, min(100, round(score, 1)))

        if is_gem:
            score = gem_score_boost(
                score, rs_3m, breaking_out,
                of_data["of_persistence_score"],
                pa_data["pa_score"],
                gem_cfg,
                ee_score = ee_data["early_entry_score"],
                mcap     = mcap_data.get("market_cap"),
            )
            # Mirror gem_score_boost's bonus formula for the raw score too —
            # duplicated intentionally rather than refactoring the existing,
            # already-tested function, so its behavior is never at risk.
            _boosts = gem_cfg.get("score_boosts", {})
            _raw_bonus = 0
            if rs_3m >= _boosts.get("rs_bonus_threshold", 150):
                _raw_bonus += _boosts.get("rs_bonus_points", 5)
            elif rs_3m >= 100:
                _raw_bonus += 3
            if breaking_out:
                _raw_bonus += _boosts.get("breakout_bonus", 5) + 3
            _raw_bonus += of_data["of_persistence_score"] * (_boosts.get("of_persistence_multiplier", 1.5) - 1)
            _raw_bonus += pa_data["pa_score"] * (_boosts.get("pa_patterns_multiplier", 1.4) - 1)
            _raw_bonus += ee_data["early_entry_score"] * 1.5
            _mcap_val = mcap_data.get("market_cap")
            if _mcap_val is not None:
                if _mcap_val < 300_000_000:       _raw_bonus += 6
                elif _mcap_val < 1_000_000_000:   _raw_bonus += 4
                elif _mcap_val < 2_000_000_000:   _raw_bonus += 2
            score_raw += _raw_bonus

        score_raw = round(score_raw, 1)

        pa_summary = " | ".join(pa_data["pa_patterns"]) if pa_data["pa_patterns"] else "None"

        return {
            "ticker":          ticker,
            "market":          _mkt_label,
            "theme":           theme,
            "price":           round(current_price, 2),
            "stage":           stage,
            "perf_1m_%":       perf_1m,
            "perf_3m_%":       perf_3m,
            "perf_6m_%":       perf_6m,
            "rs_3m":           rs_3m,
            "rs_6m":           rs_6m,
            "rs_r2500_3m":     rs_r2500,
            "rs_r2500_6m":     rs_r2500_6m,
            "rs_r3000g_3m":    rs_r3000g,
            "rs_r3000g_6m":    rs_r3000g_6m,
            "rs_multi_leader": rs_multi_leader,
            "adr_%":           adr,
            "vs_50ma_%":       vs_50ma_pct,
            "vs_200ma_%":      vs_200ma_pct,
            "ma50_slope_%":    ma50_slope_pct,
            "ma200_slope_%":   ma200_slope_pct,
            "ma50_turning_up": ma50_turning_up,
            "ma200_turning_up": ma200_turning_up,
            "volume":          vol_today,
            "vol_filter":      vol_filter,
            "vol_surge_x":     vol_surge,
            "above_50ma":      above_50ma,
            "above_200ma":     above_200ma,
            "ma50_gt_ma200":   ma50_gt_200,
            "near_52wh":       near_52wh,
            "pct_off_high_%":  float(pct_off_high) if pct_off_high is not None else 0.0,
            "pattern":         pattern,
            "breaking_out":    breaking_out,
            "news_count":      fh["news_count"],
            "sentiment":       fh["sentiment"],
            "earn_momentum":   earn_mom,
            "eps_growth_%":    av_data.get("eps_growth_pct")     if av_data else None,
            "eps_surprise_%":  av_data.get("eps_surprise_pct")   if av_data else None,
            "eps_accel":       av_data.get("eps_acceleration")   if av_data else None,
            "consec_beats":    av_data.get("consecutive_beats")  if av_data else None,
            "rev_growth_%":    av_data.get("revenue_growth_pct") if av_data else None,
            "eps_score":       av_data.get("eps_score", 0)       if av_data else 0,
            "eps_trend":       av_data.get("eps_trend", [])      if av_data else [],
            "analyst_target":  av_data.get("analyst_target")     if av_data else None,
            "pe_ratio":        av_data.get("pe_ratio")           if av_data else None,
            "peg_ratio":       av_data.get("peg_ratio")          if av_data else None,
            "eps_details":     av_data.get("details", "–")       if av_data else "AV key not set",
            "next_earnings":   av_data.get("next_earnings_date") if av_data else None,
            "of_bias":         of_data["of_directional_bias"],
            "of_up_vol_ratio": of_data["of_up_vol_ratio"],
            "of_bullish_days": of_data["of_bullish_days_%"],
            "of_consec_up":    of_data["of_consecutive_up"],
            "of_score":        of_data["of_persistence_score"],
            "vwap":            vwap_data["vwap"],
            "vwap_upper":      vwap_data["vwap_upper"],
            "vwap_lower":      vwap_data["vwap_lower"],
            "vs_vwap_%":       vwap_data["vs_vwap_%"],
            "vwap_position":   vwap_data["vwap_position"],
            "vwap_slope":      vwap_data["vwap_slope"],
            "vwap_score":      vwap_data["vwap_score"],
            "ms_structure":    ms_data["ms_structure"],
            "ms_hh_hl":        ms_data["ms_hh_hl"],
            "ms_bos":          ms_data["ms_break_of_struct"],
            "ms_swing_high":   ms_data["ms_last_swing_high"],
            "ms_swing_low":    ms_data["ms_last_swing_low"],
            "pa_patterns":     pa_summary,
            "pa_engulfing":    pa_data["pa_engulfing"],
            "pa_sfp":          pa_data["pa_sfp"],
            "pa_inside_day":   pa_data["pa_inside_day"],
            "pa_context":      pa_data["pa_context_candle"],
            "pa_score":        pa_data["pa_score"],
            "early_entry":             ee_data["early_entry"],
            "early_entry_type":        ee_data["early_entry_type"],
            "fresh_200ma_cross":       ee_data["fresh_200ma_cross"],
            "fresh_50ma_cross":        ee_data["fresh_50ma_cross"],
            "pullback_to_50ma":        ee_data["pullback_to_50ma"],
            "low_adr_base":            ee_data["low_adr_base"],
            "early_entry_score":       ee_data["early_entry_score"],
            "days_since_200ma_cross":  ee_data["days_since_200ma_cross"],
            "weekly_stage":            wk_data["weekly_stage"],
            "weekly_above_10wma":      wk_data["weekly_above_10wma"],
            "weekly_above_40wma":      wk_data["weekly_above_40wma"],
            "weekly_10gt40":           wk_data["weekly_10gt40"],
            "weekly_rs":               wk_data["weekly_rs"],
            "weekly_base_tight":       wk_data["weekly_base_tight"],
            "weekly_base_depth_%":     wk_data["weekly_base_depth_%"],
            "weekly_hh_hl":            wk_data["weekly_hh_hl"],
            "weekly_trending_up":      wk_data["weekly_trending_up"],
            "weekly_consec_up_wks":    wk_data["weekly_consec_up_wks"],
            "weekly_confirmed":        wk_data["weekly_confirmed"],
            "weekly_contradicts":      wk_data["weekly_contradicts"],
            "weekly_score":            wk_data["weekly_score"],
            "apex_score":      score,
            "apex_score_raw":  score_raw,
            "scanned_at":      datetime.now().strftime("%Y-%m-%d %H:%M"),
            "market_cap":      mcap_data["market_cap"],
            "market_cap_bn":   mcap_data["market_cap_bn"],
            "mcap_category":   mcap_data["mcap_category"],
            "is_gem":          is_gem,
            "liquidity_score": liq_score,
            "liquidity_warn":  mcap_data["liquidity_warn"],
            "avg_volume_30d":  avg_vol_30d,
        }

    except Exception as e:
        log.warning(f"Error on {ticker}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FULL SCAN (two-pass: fast price scan then targeted AV enrichment)
# ══════════════════════════════════════════════════════════════════════════════

def run_scan(cfg: dict, markets: List[str] = None,
             universe_override: list = None,
             market: str = "us", strategy: str = "swing") -> pd.DataFrame:
    if universe_override is not None:
        if universe_override and isinstance(universe_override[0], dict):
            tickers = [t["ticker"] for t in universe_override if "ticker" in t]
        else:
            tickers = list(universe_override)
    elif market == "all":
        _us_ticks = build_watchlist(cfg, "us")
        _ng_ticks = build_watchlist(cfg, "ng")
        tickers   = _us_ticks + _ng_ticks
    elif market == "ng":
        tickers = build_watchlist(cfg, "ng")
    else:
        tickers = build_watchlist(cfg, "us")
    requested_market = market.lower()
    diagnostics = ScanDiagnostics()
    selected_strategy = get_strategy(strategy)
    log.info(f"Scanning {len(tickers)} US tickers…")

    _bench_cache.clear()
    _mcap_cache.clear()
    _weekly_cache.clear()
    _fundamentals_cache.clear()
    log.info("Session caches cleared.")

    if requested_market in {"us", "all"}:
        _primary_bench = cfg["benchmarks"]["us"]
        _r2500_sym = cfg.get("benchmarks", {}).get("russell_2500", "^R25I")
        _r3000g_sym = cfg.get("benchmarks", {}).get("russell_3000_growth", "^RAG")
        _period = cfg["scan"]["history_period"]
        get_benchmark(_primary_bench, _period)
        get_benchmark(_r2500_sym, _period)
        get_benchmark(_r3000g_sym, _period)
        log.info("US benchmarks loaded: %s | %s | %s", _primary_bench, _r2500_sym, _r3000g_sym)

    if requested_market == "ng":
        _bench_cache.clear()
        log.info("NGX scan: removed US-only benchmark preload; NGX All-Share is used.")

    _batch_tickers = [t for t in tickers if not t.upper().endswith((".LG", ".NG"))]
    _batch_hist = batch_fetch_history(_batch_tickers, cfg["scan"]["history_period"])

    av_key   = cfg.get("alpha_vantage_key", "")
    av_cfg   = cfg.get("alpha_vantage", {})
    use_av   = bool(av_key and not av_key.startswith("YOUR_"))
    av_pause = av_cfg.get("rate_limit_pause", 13)
    av_max   = av_cfg.get("max_av_calls_per_scan", 8)
    cache_h  = av_cfg.get("cache_hours", 168)

    if use_av:
        log.info(f"Alpha Vantage active — top {av_max} tickers, cache={cache_h}h")
    else:
        log.info("Alpha Vantage not configured — using earnings proxy")

    results = []
    pause   = cfg["scan"]["rate_limit_pause"]

    log.info("Pass 1: technical scan…")
    for i, ticker in enumerate(tickers):
        diagnostics.scanned += 1
        time.sleep(0.35)
        if i > 0 and i % 8 == 0:
            time.sleep(pause)

        cfg_no_av = {**cfg, "alpha_vantage_key": ""}
        _ticker_market = resolve_market(ticker, requested_market)
        data = analyze_stock(ticker, cfg_no_av, market=_ticker_market,
                              hist_override=_batch_hist.get(ticker))
        if data is None:
            diagnostics.fail("history")
            continue

        _fmkt     = "ng" if data.get("market") == "NGX" else "us"
        min_score = cfg["thresholds"].get(_fmkt, cfg["thresholds"]["us"])["score_filter"]
        min_vol   = cfg["thresholds"].get(_fmkt, cfg["thresholds"]["us"])["min_volume"]

        _is_gem_stock = data.get("is_gem", False)

        if _ticker_market == "ng":
            _stage_ok = data["above_50ma"]
        else:
            _stage_ok = data["above_200ma"]
        _gem_stage_ok = _is_gem_stock and data["above_50ma"]
        if not _stage_ok and not _gem_stage_ok:
            diagnostics.reject("stage")
            continue

        if _ticker_market == "ng":
            _min_perf = cfg["thresholds"].get("ng", {}).get("min_3m_perf", 0)
        else:
            _min_perf = 0.0 if _is_gem_stock else cfg["thresholds"].get(_fmkt, cfg["thresholds"]["us"]).get("min_3m_perf", 5)
        if data["perf_3m_%"] < _min_perf:
            diagnostics.reject("performance")
            continue

        _min_rs = -50 if _is_gem_stock else -30
        if data["rs_3m"] < _min_rs:
            diagnostics.reject("relative_strength")
            continue

        _min_vol_eff = 100_000 if _is_gem_stock else min_vol
        _min_score_eff = max(20, min_score - 15) if _is_gem_stock else min_score

        if data.get("vol_filter", data["volume"]) < _min_vol_eff:
            diagnostics.reject("volume")
            continue
        if data["apex_score"] < _min_score_eff:
            diagnostics.reject("score")
            continue

        results.append(data)
        diagnostics.passed += 1

    if not results:
        log.warning("No results passed filters.")
        diagnostics.log_summary(log)
        return pd.DataFrame()

    # Sort by displayed score first (keeps existing threshold/UI behavior
    # identical), then by raw score as a tiebreaker — many stocks tie at
    # the visible 100 ceiling, and without this, that tie is broken by
    # arbitrary list order rather than true underlying quality.
    results.sort(key=lambda x: (x["apex_score"], x.get("apex_score_raw", x["apex_score"])), reverse=True)

    if use_av and av_max > 0:
        api_calls_made = 0
        log.info(f"Pass 2: AV enrichment for top {av_max} tickers…")

        for data in results:
            if api_calls_made >= av_max:
                break

            ticker    = data["ticker"]
            has_cache = (
                _cache_valid(_cache_path(ticker, "earnings"), cache_h) and
                _cache_valid(_cache_path(ticker, "income"),   cache_h) and
                _cache_valid(_cache_path(ticker, "overview"), cache_h)
            )

            if not has_cache and api_calls_made > 0:
                time.sleep(av_pause)

            try:
                av_data = analyse_eps(
                    ticker, av_key,
                    lookback_q   = av_cfg.get("eps_lookback_quarters", 4),
                    min_growth   = av_cfg.get("min_eps_growth", 15),
                    min_surprise = av_cfg.get("min_surprise_pct", 5),
                    cache_hours  = cache_h,
                )

                if av_data and av_data.get("eps_momentum") not in (None, "Unknown"):
                    data["earn_momentum"]  = av_data["eps_momentum"]
                    data["eps_growth_%"]   = av_data.get("eps_growth_pct")
                    data["eps_surprise_%"] = av_data.get("eps_surprise_pct")
                    data["eps_accel"]      = av_data.get("eps_acceleration")
                    data["consec_beats"]   = av_data.get("consecutive_beats")
                    data["rev_growth_%"]   = av_data.get("revenue_growth_pct")
                    data["eps_score"]      = av_data.get("eps_score", 0)
                    data["eps_trend"]      = av_data.get("eps_trend", [])
                    data["analyst_target"] = av_data.get("analyst_target")
                    data["pe_ratio"]       = av_data.get("pe_ratio")
                    data["peg_ratio"]      = av_data.get("peg_ratio")
                    data["eps_details"]    = av_data.get("details", "–")
                    data["apex_score"]     = min(100, round(
                        data["apex_score"] + av_data.get("eps_score", 0), 1))

                    src = "cache" if has_cache else "API"
                    log.info(f"  ✓ {ticker:<16} EPS={av_data.get('eps_momentum','?')} "
                             f"growth={av_data.get('eps_growth_pct','?')}% [{src}]")

                if not has_cache:
                    api_calls_made += 1

            except Exception as av_err:
                log.warning(f"  AV error {ticker}: {av_err}")

    # ── Pass 3: extended fundamentals enrichment (free yfinance .info call,
    #    only for stocks that already passed — not the full universe) ──────
    fund_cfg = cfg.get("fundamentals", {})
    max_fund_calls = fund_cfg.get("max_calls_per_scan", 30)
    log.info(f"Pass 3: fundamentals enrichment for top {max_fund_calls} tickers…")
    for data in results[:max_fund_calls]:
        try:
            fdata = get_fundamentals_data(data["ticker"])
            data.update(fdata)
        except Exception as fe:
            log.debug(f"Fundamentals enrichment error {data.get('ticker')}: {fe}")

    # ── Apply the selected strategy AFTER all enrichment so it sees the
    #    fullest data available (AV EPS + fundamentals), not just Pass 1 data.
    results = [selected_strategy.evaluate(r) for r in results]

    df = pd.DataFrame(results)
    if "apex_score_raw" in df.columns:
        df = df.sort_values(["apex_score", "apex_score_raw"], ascending=[False, False]).reset_index(drop=True)
    else:
        df = df.sort_values("apex_score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    diagnostics.log_summary(log)
    df.attrs["scan_diagnostics"] = diagnostics.summary()
    try:
        from ai.engine import InvestmentIntelligenceEngine
        engine = InvestmentIntelligenceEngine(cfg)
        if engine.enabled:
            df.attrs["ai_market_brief"] = engine.market_brief(diagnostics.summary(), df)
            df["ai_summary"] = [engine.analyze_stock(row) for row in df.to_dict("records")]
    except Exception as ai_error:
        log.warning("AI enrichment skipped; scanner results are preserved: %s", ai_error)

    # ── Apex the Great X — live radar update (connective wiring) ───────────
    # Reuses THIS scan's already-fetched history (_batch_hist) and the
    # already-loaded US benchmark — makes ZERO additional network calls.
    # Off by default (cfg["apex10"]["enabled"]); US-only for now, matching
    # _batch_hist itself (NGX tickers were never in it — see _batch_tickers
    # above). Wrapped identically to the AI-enrichment block above: any
    # failure here is logged and swallowed, never allowed to affect the
    # scan results this function returns.
    if requested_market in {"us", "all"}:
        try:
            from modules.apex10_integration import process_scan_for_radar
            _bench_close = get_benchmark(_primary_bench, _period)
            apex10_summary = process_scan_for_radar(results, _batch_hist, _bench_close, cfg, market="US")
            if apex10_summary is not None:
                df.attrs["apex10_radar_summary"] = apex10_summary
                log.info(f"Apex the Great X radar update: {apex10_summary}")
        except Exception as apex10_error:
            log.warning("Apex the Great X radar update skipped; scanner results are preserved: %s", apex10_error)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SHORT-SIDE SCANNER (Phase: Short Candidates)
# Entirely new, self-contained pipeline — does NOT call, modify, or share
# gating logic with analyze_stock()/run_scan(). The long-only scan above is
# completely untouched by anything in this section. Reuses the same proven
# lower-level building blocks (get_benchmark, compute_rs, order flow, VWAP,
# market structure) computed the identical way, scored in the opposite
# direction. US equities only.
# ══════════════════════════════════════════════════════════════════════════════

def detect_breakdown(hist: pd.DataFrame, lookback_weeks: int = 8) -> Tuple[bool, str]:
    """Bearish mirror of detect_base_breakout() — a support level breaking
    DOWN on above-average volume, instead of a base breaking up."""
    bars = min(lookback_weeks * 5, len(hist) - 1)
    if bars < 15:
        return False, "Insufficient data"
    window    = hist.iloc[-bars:]
    current   = hist["Close"].iloc[-1]
    high      = window["High"].max()
    low       = window["Low"].min()
    avg_vol   = window["Volume"].mean()
    today_vol = hist["Volume"].iloc[-1]
    depth     = (high - low) / high * 100 if high else 0
    vol_surge = today_vol > avg_vol * 1.4
    pivot     = current <= low * 1.005
    if depth < 12 and pivot and vol_surge:  return True,  "Support Breakdown"
    if depth < 12 and pivot:                return False, "Tight — Watch Vol"
    if current <= low * 1.05:               return False, "Near Support (No Vol)"
    return False, f"Distributing ({depth:.0f}%)"


def analyze_stock_short(ticker: str, cfg: dict,
                        hist_override: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """
    Short-candidate analysis — the bearish mirror of analyze_stock(). A
    completely separate function so the long-only pipeline is never at
    risk of being altered. US equities only (no NGX branch).
    """
    thresholds   = cfg["thresholds"]["us"]
    bench_symbol = cfg["benchmarks"]["us"]

    try:
        if hist_override is not None and len(hist_override) > 0:
            hist = hist_override
        else:
            hist = yf.Ticker(ticker).history(period=cfg["scan"]["history_period"])

        if len(hist) < cfg["scan"]["min_history_bars"]:
            return None

        close         = hist["Close"]
        current_price = close.iloc[-1]

        perf_1w = performance_pct(close, 5)
        perf_1m = performance_pct(close, 21)
        perf_3m = performance_pct(close, 63)
        perf_6m = performance_pct(close, 126)

        low_52w = close.rolling(min(252, len(close))).min().iloc[-1]
        if pd.isna(low_52w) or low_52w == 0:
            low_52w = float(close.min())
        near_52wl   = bool(current_price <= low_52w * 1.05)
        pct_off_low = round((current_price / low_52w - 1) * 100, 1) if low_52w else 0.0

        ma50  = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        if pd.isna(ma200):
            log.debug(f"{ticker}: 200MA not computable, skipping (short scan)")
            return None

        below_50ma  = bool(current_price < ma50)
        below_200ma = bool(current_price < ma200)
        ma50_lt_200 = bool(ma50 < ma200)
        stage       = detect_stage(current_price, ma50, ma200)
        vs_200ma_pct = round((current_price / ma200 - 1) * 100, 2) if ma200 else None

        # ── Hard gate: only a confirmed downtrend qualifies as a short
        # candidate — mirrors the long side's Stage 2 requirement exactly,
        # just inverted. This is the ONLY reason a stock below its 200MA
        # now appears anywhere in ApexScan's output.
        if not (below_200ma and ma50_lt_200):
            return None

        bench = get_benchmark(bench_symbol)
        rs_3m = compute_rs(close, bench, 63)
        rs_6m = compute_rs(close, bench, 126)

        adr        = adr_pct(hist, 20)
        vol_today  = int(hist["Volume"].iloc[-1])
        vol_avg_20 = int(hist["Volume"].rolling(20).mean().iloc[-1]) if len(hist) >= 20 else vol_today
        breaking_down, pattern = detect_breakdown(hist)

        of_data   = order_flow_persistence(hist, cfg.get("advanced", {}).get("of_lookback", 10))
        vwap_data = compute_vwap(hist, cfg.get("advanced", {}).get("vwap_lookback", 20))
        ms_data   = detect_market_structure(hist, cfg.get("advanced", {}).get("swing_lookback", 5))

        mcap_data = get_market_cap_data(ticker)
        # Shorting illiquid names is materially more dangerous than going
        # long them (hard to borrow, hard to cover cleanly) — hard exclude
        # rather than just a warning flag.
        if mcap_data.get("liquidity_warn"):
            return None

        theme = "Unknown"
        try:
            _yf_info   = yf.Ticker(ticker).fast_info
            _yf_sector = getattr(_yf_info, "sector", None)
            theme = _yf_sector or "Unknown"
        except Exception:
            pass

        # ── Short Score — bearish mirror of the Apex Score's weighting ──
        score = 0
        if perf_3m < -thresholds.get("min_3m_perf", 5): score += min(40, abs(perf_3m))
        if rs_3m < -thresholds.get("rs_rating_min", 20): score += 25
        elif rs_3m < -10:                                 score += 12
        if below_200ma and ma50_lt_200:                   score += 15
        if near_52wl:                                      score += 10
        if breaking_down:                                  score += 10
        if of_data["of_directional_bias"] in ("Bearish", "Strong Bearish"):
            score += of_data["of_persistence_score"]
        if ms_data.get("ms_lh_ll"):                        score += 3

        # Penalties — signs the downtrend may be exhausting/reversing
        if perf_3m > 0:  score -= 15
        if rs_3m > 0:    score -= 10
        if of_data["of_directional_bias"] in ("Bullish", "Strong Bullish"): score -= 8
        if pct_off_low > 30: score -= 5  # already bounced meaningfully off the low

        score = max(0, min(100, round(score, 1)))

        # Recent headlines — qualitative "why" context for a decline, not
        # just the numeric score. Fetched here (not earlier) since it's
        # only worth the API call for a candidate that already passed
        # every other gate — keeps the scan fast for the full universe.
        _finnhub_key = cfg.get("finnhub_key", "")
        _headlines = get_recent_headlines(ticker, _finnhub_key, n=2, days=14) if _finnhub_key else []

        return {
            "ticker": ticker, "price": round(current_price, 2), "stage": stage,
            "perf_1w_%": perf_1w, "perf_1m_%": perf_1m, "perf_3m_%": perf_3m, "perf_6m_%": perf_6m,
            "rs_3m": rs_3m, "rs_6m": rs_6m,
            "below_50ma": below_50ma, "below_200ma": below_200ma, "ma50_lt_ma200": ma50_lt_200,
            "vs_200ma_%": vs_200ma_pct,
            "near_52wl": near_52wl, "pct_off_low_%": pct_off_low,
            "adr_%": adr, "volume": vol_today, "vol_avg_20": vol_avg_20,
            "breaking_down": breaking_down, "pattern": pattern,
            "of_bias": of_data["of_directional_bias"], "of_score": of_data["of_persistence_score"],
            "vwap_position": vwap_data["vwap_position"], "vs_vwap_%": vwap_data["vs_vwap_%"],
            "ms_structure": ms_data["ms_structure"], "ms_lh_ll": ms_data.get("ms_lh_ll", False),
            "theme": theme, "market_cap": mcap_data.get("market_cap"),
            "mcap_category": mcap_data.get("mcap_category"),
            "avg_volume_30d": mcap_data.get("avg_volume_30d"),
            "short_pct_float": None,  # filled in by dashboard.py's fundamentals enrichment, same as the long side
            "recent_headlines": _headlines,
            "short_score": score,
            "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    except Exception as e:
        log.warning(f"Short analysis error on {ticker}: {e}")
        return None


def run_short_scan(cfg: dict, universe_override: list = None) -> pd.DataFrame:
    """
    Standalone short-candidate scan. Does NOT call run_scan() and shares
    no gating/scoring logic with it — completely independent pipeline,
    US equities only, so the long-only scan is never at risk of being
    altered by anything here.
    """
    if universe_override is not None:
        if universe_override and isinstance(universe_override[0], dict):
            tickers = [t["ticker"] for t in universe_override if "ticker" in t]
        else:
            tickers = list(universe_override)
    else:
        tickers = build_watchlist(cfg, "us")

    tickers = [t for t in tickers if not t.upper().endswith((".LG", ".NG"))]
    log.info(f"Short scan: {len(tickers)} US tickers…")

    _bench_cache.clear()
    _mcap_cache.clear()
    get_benchmark(cfg["benchmarks"]["us"], cfg["scan"]["history_period"])

    _batch_hist = batch_fetch_history(tickers, cfg["scan"]["history_period"])

    results = []
    pause = cfg["scan"]["rate_limit_pause"]
    min_score = cfg.get("short_scan", {}).get("min_score", 40)

    for i, ticker in enumerate(tickers):
        time.sleep(0.35)
        if i > 0 and i % 8 == 0:
            time.sleep(pause)
        data = analyze_stock_short(ticker, cfg, hist_override=_batch_hist.get(ticker))
        if data is None:
            continue
        if data["short_score"] < min_score:
            continue
        results.append(data)

    if not results:
        log.warning("Short scan: no candidates passed filters.")
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("short_score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    log.info(f"Short scan complete: {len(df)} candidates.")
    return df


def save_report(df: pd.DataFrame, report_dir: str = "reports") -> str:
    Path(report_dir).mkdir(exist_ok=True)
    filename = f"{report_dir}/scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(filename, encoding="utf-8")
    log.info(f"Saved → {filename}")
    return filename


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ApexScan — US Stock Scanner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--top",    type=int, default=20)
    parser.add_argument("--market", choices=["us", "ng", "all"], default="us")
    parser.add_argument("--strategy", choices=["swing", "position", "long_term", "dividend", "value"], default="swing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df  = run_scan(cfg, market=args.market, strategy=args.strategy)

    if not df.empty:
        cols = ["ticker","theme","price","stage","perf_3m_%",
                "rs_3m","of_bias","vwap_position","pa_patterns","apex_score"]
        print(f"\n{'='*80}")
        print(f"  TOP {args.top} SETUPS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*80}")
        print(df[cols].head(args.top).to_string())
        save_report(df)
    else:
        print("No setups matched current filters.")
