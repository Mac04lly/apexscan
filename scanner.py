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

    # Override with Streamlit secrets when running on Streamlit Cloud
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            if "alpha_vantage_key" in st.secrets:
                cfg["alpha_vantage_key"] = st.secrets["alpha_vantage_key"]
            if "finnhub_key" in st.secrets:
                cfg["finnhub_key"] = st.secrets["finnhub_key"]
    except Exception:
        pass  # Not running in Streamlit context (e.g. CLI), use config.yaml only

    return cfg


def build_watchlist(cfg: dict) -> List[str]:
    return list(set(t for theme in cfg["us_themes"].values() for t in theme))


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
    return "? Unknown"


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

# ── Market Cap Cache (Emerging Gems) ──────────────────────────────────────────
_mcap_cache: Dict[str, dict] = {}

def get_market_cap_data(ticker: str) -> Dict:
    """Fetch market cap and liquidity data. Cached per session."""
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


def gem_score_boost(score: float, rs3: float, breaking_out: bool,
                    of_score: int, pa_score: int, gem_cfg: dict) -> float:
    """Apply score boosts for emerging gems with strong signals."""
    boosts = gem_cfg.get("score_boosts", {})
    bonus  = 0
    if rs3 >= boosts.get("rs_bonus_threshold", 150):
        bonus += boosts.get("rs_bonus_points", 5)
    if breaking_out:
        bonus += boosts.get("breakout_bonus", 5)
    bonus += of_score * (boosts.get("of_persistence_multiplier", 1.25) - 1)
    bonus += pa_score * (boosts.get("pa_patterns_multiplier", 1.20) - 1)
    return round(min(100, score + bonus), 1)

def get_benchmark(symbol: str = "^GSPC", period: str = "1y") -> pd.Series:
    if symbol not in _bench_cache:
        try:
            data = yf.Ticker(symbol).history(period=period)["Close"]
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_convert(None)
            data.index = pd.to_datetime(data.index).normalize()
            _bench_cache[symbol] = data
            log.info(f"Benchmark {symbol}: {len(data)} bars")
        except Exception as e:
            log.warning(f"Benchmark {symbol} failed: {e}")
            _bench_cache[symbol] = pd.Series(dtype=float)
    return _bench_cache[symbol]


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
# ANALYZE STOCK
# ══════════════════════════════════════════════════════════════════════════════

def analyze_stock(ticker: str, cfg: dict) -> Optional[Dict]:
    """
    Full ApexScan analysis for one ticker.
    Includes: momentum, stage, RS, order flow, VWAP, market structure,
    price action patterns, and Alpha Vantage EPS fundamentals.
    """
    thresholds   = cfg["thresholds"]["us"]
    bench_symbol = cfg["benchmarks"]["us"]
    finnhub_key  = cfg.get("finnhub_key", "")
    av_key       = cfg.get("alpha_vantage_key", "")
    av_cfg       = cfg.get("alpha_vantage", {})
    use_av       = bool(av_key and not av_key.startswith("YOUR_"))

    try:
        hist = yf.Ticker(ticker).history(period=cfg["scan"]["history_period"])
        if len(hist) < cfg["scan"]["min_history_bars"]:
            log.debug(f"{ticker}: only {len(hist)} bars, skipping")
            return None

        close         = hist["Close"]
        current_price = close.iloc[-1]

        perf_1m = performance_pct(close, 21)
        perf_3m = performance_pct(close, 63)
        perf_6m = performance_pct(close, 126)

        high_52w     = close.rolling(252).max().iloc[-1]
        near_52wh    = current_price >= high_52w * thresholds["near_52w_high"]
        pct_off_high = round((current_price / high_52w - 1) * 100, 1) if high_52w else 0

        ma50         = close.rolling(50).mean().iloc[-1]
        ma200        = close.rolling(200).mean().iloc[-1]
        above_50ma   = bool(current_price > ma50)
        above_200ma  = bool(current_price > ma200)
        ma50_gt_200  = bool(ma50 > ma200)
        stage        = detect_stage(current_price, ma50, ma200)
        vs_50ma_pct  = price_vs_ma(current_price, ma50)
        vs_200ma_pct = price_vs_ma(current_price, ma200)

        bench = get_benchmark(bench_symbol)
        rs_3m = compute_rs(close, bench, 63)
        rs_6m = compute_rs(close, bench, 126)

        adr          = adr_pct(hist, 20)
        vol_today    = int(hist["Volume"].iloc[-1])
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

        theme    = next((k for k, v in cfg["us_themes"].items() if ticker in v), "other")

        # ── Market Cap & Liquidity (Emerging Gems) ────────────────────────
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

        # ── Apex Score ────────────────────────────────────────────────────
        score = 0
        if perf_3m > thresholds["min_3m_perf"]:    score += min(40, perf_3m)
        if rs_3m > thresholds["rs_rating_min"]:     score += 25
        elif rs_3m > 50:                            score += 12
        if above_200ma and ma50_gt_200:             score += 15
        elif above_200ma:                           score += 7
        if near_52wh:                               score += 10
        if breaking_out:                            score += 10
        score += of_data["of_persistence_score"]    # 0-8
        score += pa_data["pa_score"]                # 0-5
        score += vwap_data["vwap_score"]            # 0-4
        if ms_data["ms_hh_hl"]:                    score += 2
        if ms_data["ms_break_of_struct"] and ms_data["ms_hh_hl"]: score += 1
        if av_data:                                 score += av_data.get("eps_score", 0)  # 0-15
        score = min(100, round(score, 1))

        # Apply gem score boosts for emerging gems with strong signals
        if is_gem:
            score = gem_score_boost(
                score, rs_3m, breaking_out,
                of_data["of_persistence_score"],
                pa_data["pa_score"],
                gem_cfg,
            )

        pa_summary = " | ".join(pa_data["pa_patterns"]) if pa_data["pa_patterns"] else "None"

        return {
            "ticker":          ticker,
            "market":          "US",
            "theme":           theme,
            "price":           round(current_price, 2),
            "stage":           stage,
            "perf_1m_%":       perf_1m,
            "perf_3m_%":       perf_3m,
            "perf_6m_%":       perf_6m,
            "rs_3m":           rs_3m,
            "rs_6m":           rs_6m,
            "adr_%":           adr,
            "vs_50ma_%":       vs_50ma_pct,
            "vs_200ma_%":      vs_200ma_pct,
            "volume":          vol_today,
            "vol_surge_x":     vol_surge,
            "above_50ma":      above_50ma,
            "above_200ma":     above_200ma,
            "ma50_gt_ma200":   ma50_gt_200,
            "near_52wh":       near_52wh,
            "pct_off_high_%":  pct_off_high,
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
            "apex_score":      score,
            "scanned_at":      datetime.now().strftime("%Y-%m-%d %H:%M"),
            # ── Emerging Gems ─────────────────────────────────────────────
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
             ticker_list: List[str] = None,
             max_tickers: int = None,
             progress_callback=None) -> pd.DataFrame:
    """
    Full scan engine.
    ticker_list: explicit list of tickers to scan (from universe module)
    max_tickers: optional cap on number of tickers
    progress_callback: fn(current, total, passing) for live progress updates
    """
    # Use provided ticker_list, or build from config themes as fallback
    if ticker_list:
        tickers = ticker_list
    else:
        tickers = build_watchlist(cfg)

    # Optional cap
    if max_tickers and len(tickers) > max_tickers:
        tickers = tickers[:max_tickers]

    log.info(f"Scanning {len(tickers)} tickers…")

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

    results   = []
    pause     = cfg["scan"]["rate_limit_pause"]
    total     = len(tickers)
    errors    = 0

    # ── Pass 1: price/technical scan, no AV ──────────────────────────────
    log.info(f"Pass 1: technical scan of {total} tickers…")
    for i, ticker in enumerate(tickers):
        # Rate limiting
        if i > 0 and i % 10 == 0:
            time.sleep(pause)

        # Progress update every 25 tickers
        if progress_callback and i % 25 == 0:
            try:
                progress_callback(i, total, len(results))
            except Exception:
                pass

        # Log every 100
        if i > 0 and i % 100 == 0:
            log.info(f"  {i}/{total} scanned — {len(results)} passing filters")

        cfg_no_av = {**cfg, "alpha_vantage_key": ""}
        try:
            data = analyze_stock(ticker, cfg_no_av)
        except Exception as e:
            errors += 1
            log.debug(f"Error {ticker}: {e}")
            continue

        if data is None:
            continue

        min_score = cfg["thresholds"]["us"]["score_filter"]
        min_vol   = cfg["thresholds"]["us"]["min_volume"]
        if data["apex_score"] >= min_score and data["volume"] >= min_vol:
            results.append(data)

    # Final progress update
    if progress_callback:
        try:
            progress_callback(total, total, len(results))
        except Exception:
            pass

    log.info(f"Pass 1 complete: {len(results)} passing from {total} tickers ({errors} errors)")

    if not results:
        log.warning("No results passed filters.")
        return pd.DataFrame()

    results.sort(key=lambda x: x["apex_score"], reverse=True)

    # ── Pass 2: AV enrichment for top N tickers ───────────────────────────
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

    df = pd.DataFrame(results).sort_values("apex_score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    df  = run_scan(cfg)

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
