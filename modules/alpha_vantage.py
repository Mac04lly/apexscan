"""
modules/alpha_vantage.py — ApexScan Fundamental Data Layer
Real EPS growth, earnings surprises, revenue momentum, and analyst estimates
via Alpha Vantage API.

Free tier: 25 requests/day, 5/minute.
Premium: higher limits — adjust rate_limit_pause in config.yaml accordingly.

Data cached locally in data/av_cache/ for 24h to preserve API quota.
"""

import requests
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/av_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AV_BASE = "https://www.alphavantage.co/query"


# ══════════════════════════════════════════════════════════════════════════════
# CACHE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(ticker: str, endpoint: str) -> Path:
    return CACHE_DIR / f"{ticker}_{endpoint}.json"


def _cache_valid(path: Path, hours: int = 24) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=hours)


def _read_cache(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except:
        return None


def _write_cache(path: Path, data: dict):
    try:
        path.write_text(json.dumps(data, indent=2))
    except:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# RAW API FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

def _av_get(params: dict, api_key: str) -> Optional[dict]:
    """Make one Alpha Vantage API call with error handling."""
    params["apikey"] = api_key
    try:
        resp = requests.get(AV_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # AV returns error messages inside the JSON
        if "Error Message" in data:
            log.warning(f"AV error: {data['Error Message']}")
            return None
        if "Note" in data:
            log.warning(f"AV rate limit hit: {data['Note']}")
            return None
        if "Information" in data:
            log.warning(f"AV limit: {data['Information']}")
            return None
        return data

    except Exception as e:
        log.warning(f"AV request failed: {e}")
        return None


def fetch_earnings(ticker: str, api_key: str, cache_hours: int = 24) -> Optional[dict]:
    """
    Fetch quarterly EPS actuals vs estimates (EARNINGS endpoint).
    Returns the raw AV response dict or None.
    """
    cache = _cache_path(ticker, "earnings")
    if _cache_valid(cache, cache_hours):
        return _read_cache(cache)

    data = _av_get({"function": "EARNINGS", "symbol": ticker}, api_key)
    if data:
        _write_cache(cache, data)
    return data


def fetch_income_statement(ticker: str, api_key: str, cache_hours: int = 24) -> Optional[dict]:
    """Fetch quarterly income statements for revenue momentum."""
    cache = _cache_path(ticker, "income")
    if _cache_valid(cache, cache_hours):
        return _read_cache(cache)

    data = _av_get({"function": "INCOME_STATEMENT", "symbol": ticker}, api_key)
    if data:
        _write_cache(cache, data)
    return data


def fetch_overview(ticker: str, api_key: str, cache_hours: int = 24) -> Optional[dict]:
    """Fetch company overview — PE, PEG, margins, analyst target."""
    cache = _cache_path(ticker, "overview")
    if _cache_valid(cache, cache_hours):
        return _read_cache(cache)

    data = _av_get({"function": "OVERVIEW", "symbol": ticker}, api_key)
    if data:
        _write_cache(cache, data)
    return data


def fetch_earnings_calendar(api_key: str, horizon: str = "3month") -> Optional[list]:
    """
    Fetch upcoming earnings calendar for all tickers.
    horizon: '3month' | '6month' | '12month'
    Returns list of dicts with symbol, reportDate, estimate, currency.
    """
    cache = _cache_path("__calendar__", f"upcoming_{horizon}")
    if _cache_valid(cache, 12):  # refresh twice daily
        return _read_cache(cache)

    try:
        params = {
            "function": "EARNINGS_CALENDAR",
            "horizon":  horizon,
            "apikey":   api_key,
        }
        resp = requests.get(AV_BASE, params=params, timeout=20)
        resp.raise_for_status()

        # AV returns CSV for this endpoint
        lines   = resp.text.strip().split("\n")
        headers = lines[0].split(",")
        records = []
        for line in lines[1:]:
            vals = line.split(",")
            if len(vals) == len(headers):
                records.append(dict(zip(headers, vals)))

        _write_cache(cache, records)
        return records

    except Exception as e:
        log.warning(f"Earnings calendar fetch failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_eps(ticker: str, api_key: str,
                lookback_q: int = 4,
                min_growth: float = 15.0,
                min_surprise: float = 5.0,
                cache_hours: int = 24) -> Dict:
    """
    Compute real EPS momentum from Alpha Vantage EARNINGS data.

    Returns:
        eps_growth_pct:     YoY EPS growth latest quarter (%)
        eps_acceleration:   Is growth accelerating vs prior quarter? (bool)
        eps_surprise_pct:   Last quarter's actual vs estimate beat/miss (%)
        consecutive_beats:  How many consecutive quarters beat estimates
        revenue_growth_pct: YoY revenue growth (from income statement)
        eps_momentum:       'Strong' / 'Moderate' / 'Weak' / 'Declining'
        eps_score:          0–15 pts contribution to Apex Score
        eps_trend:          list of last 4 quarters EPS for sparkline
        next_earnings_date: next scheduled report date (str or None)
        analyst_target:     analyst price target from overview
        pe_ratio:           trailing PE
        peg_ratio:          PEG ratio
        details:            human-readable summary string
    """
    result = {
        "eps_growth_pct":     None,
        "eps_acceleration":   False,
        "eps_surprise_pct":   None,
        "consecutive_beats":  0,
        "revenue_growth_pct": None,
        "eps_momentum":       "Unknown",
        "eps_score":          0,
        "eps_trend":          [],
        "next_earnings_date": None,
        "analyst_target":     None,
        "pe_ratio":           None,
        "peg_ratio":          None,
        "details":            "No data",
    }

    # ── Earnings data ─────────────────────────────────────────────────────
    earn_data = fetch_earnings(ticker, api_key, cache_hours)
    if not earn_data:
        result["details"] = "Alpha Vantage quota reached or ticker not found"
        return result

    quarterly = earn_data.get("quarterlyEarnings", [])
    if not quarterly:
        result["details"] = "No quarterly earnings data available"
        return result

    # Parse most recent N quarters
    parsed = []
    for q in quarterly[:lookback_q + 4]:
        try:
            actual   = float(q.get("reportedEPS",   "0") or 0)
            estimate = float(q.get("estimatedEPS",  "0") or 0)
            surprise = float(q.get("surprisePercentage", "0") or 0)
            date_str = q.get("fiscalDateEnding", "")
            parsed.append({
                "date":     date_str,
                "actual":   actual,
                "estimate": estimate,
                "surprise": surprise,
            })
        except:
            continue

    if len(parsed) < 2:
        result["details"] = "Insufficient earnings history"
        return result

    # EPS trend (last 4 quarters for sparkline, most recent first)
    result["eps_trend"] = [round(p["actual"], 2) for p in parsed[:4]]

    # YoY EPS growth (Q1 this year vs Q1 last year = index 0 vs index 4)
    if len(parsed) >= 5:
        latest    = parsed[0]["actual"]
        year_ago  = parsed[4]["actual"]
        if year_ago and year_ago != 0:
            growth = round((latest / abs(year_ago) - 1) * 100, 1)
            result["eps_growth_pct"] = growth

    # QoQ acceleration (is growth rate higher than prior quarter's growth rate?)
    if len(parsed) >= 6:
        try:
            prev_growth = (parsed[1]["actual"] / abs(parsed[5]["actual"]) - 1) * 100 if parsed[5]["actual"] else 0
            curr_growth = result["eps_growth_pct"] or 0
            result["eps_acceleration"] = curr_growth > prev_growth
        except:
            pass

    # Latest surprise
    result["eps_surprise_pct"] = round(parsed[0]["surprise"], 1) if parsed else None

    # Consecutive beats
    beats = 0
    for p in parsed:
        if p["surprise"] > 0:
            beats += 1
        else:
            break
    result["consecutive_beats"] = beats

    # ── Revenue growth (income statement) ─────────────────────────────────
    income = fetch_income_statement(ticker, api_key, cache_hours)
    if income:
        qtrs = income.get("quarterlyReports", [])
        if len(qtrs) >= 5:
            try:
                rev_now  = float(qtrs[0].get("totalRevenue", 0) or 0)
                rev_yago = float(qtrs[4].get("totalRevenue", 0) or 0)
                if rev_yago > 0:
                    result["revenue_growth_pct"] = round((rev_now / rev_yago - 1) * 100, 1)
            except:
                pass

    # ── Overview (PE, PEG, target) ────────────────────────────────────────
    overview = fetch_overview(ticker, api_key, cache_hours)
    if overview:
        try:
            result["analyst_target"] = float(overview.get("AnalystTargetPrice", 0) or 0) or None
            result["pe_ratio"]       = float(overview.get("TrailingPE", 0) or 0) or None
            result["peg_ratio"]      = float(overview.get("PEGRatio", 0) or 0) or None
        except:
            pass

    # ── EPS Momentum classification ────────────────────────────────────────
    growth   = result["eps_growth_pct"]   or 0
    surprise = result["eps_surprise_pct"] or 0
    accel    = result["eps_acceleration"]
    rev_g    = result["revenue_growth_pct"] or 0

    if growth >= min_growth and accel and surprise >= min_surprise:
        result["eps_momentum"] = "Strong"
    elif growth >= min_growth and surprise >= 0:
        result["eps_momentum"] = "Moderate"
    elif growth >= 0:
        result["eps_momentum"] = "Weak"
    else:
        result["eps_momentum"] = "Declining"

    # ── EPS Score (0–15 pts) ───────────────────────────────────────────────
    score = 0
    if growth >= 50:                    score += 6
    elif growth >= 25:                  score += 4
    elif growth >= min_growth:          score += 2
    if accel:                           score += 3
    if beats >= 4:                      score += 3
    elif beats >= 2:                    score += 2
    elif beats >= 1:                    score += 1
    if surprise >= 20:                  score += 2
    elif surprise >= min_surprise:      score += 1
    if rev_g >= 20:                     score += 1
    result["eps_score"] = min(15, score)

    # ── Details string ────────────────────────────────────────────────────
    parts = []
    if growth is not None:
        parts.append(f"EPS growth YoY: {growth:+.1f}%")
    if surprise is not None:
        parts.append(f"Last surprise: {surprise:+.1f}%")
    if beats:
        parts.append(f"{beats} consecutive beats")
    if accel:
        parts.append("Accelerating ✅")
    if rev_g:
        parts.append(f"Revenue growth: {rev_g:+.1f}%")
    result["details"] = " | ".join(parts) if parts else "Data parsed"

    return result


def get_upcoming_earnings_for_watchlist(
    tickers: List[str],
    api_key: str,
) -> Dict[str, str]:
    """
    Get next earnings date for each ticker in the watchlist
    from the AV earnings calendar endpoint (1 API call for all tickers).
    Returns {ticker: date_string} dict.
    """
    calendar = fetch_earnings_calendar(api_key, horizon="3month")
    if not calendar:
        return {}

    ticker_set = set(t.upper() for t in tickers)
    result = {}
    for entry in calendar:
        sym  = entry.get("symbol","").upper()
        date = entry.get("reportDate","")
        if sym in ticker_set and date:
            result[sym] = date

    return result


def clear_cache(ticker: str = None):
    """Clear cached AV data. Pass ticker to clear one, or None to clear all."""
    if ticker:
        for f in CACHE_DIR.glob(f"{ticker}_*.json"):
            f.unlink()
        log.info(f"AV cache cleared for {ticker}")
    else:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
        log.info("AV cache fully cleared")

