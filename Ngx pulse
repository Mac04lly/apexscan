"""
modules/ngx_pulse.py — NGX Pulse API Data Provider
=====================================================
NOTE: This is NGX Pulse (ngxpulse.ng) — a DIFFERENT service from NGN Market
(api.ngnmarket.com, see modules/ngn_market.py). Do not confuse the two:
different base URL, different auth scheme, different key, different plans.

Base URL:  https://www.ngxpulse.ng
Auth:      header  X-API-Key: <your_key>   (NOT a Bearer token)
Docs:      https://ngxpulse.ng/api

Coverage: 150+ NGX-listed equities, full NGX index universe (incl. ASI back
to 1996), NGX ETFs, bonds, disclosures, dividends, NASD OTC, forex, news.

Personal (free) tier limits: 10 requests/min, 100 requests/day.
Equities/market endpoints refresh during market hours; ETF/index snapshots
are DB-backed and refresh every 30 minutes.

Rate budget strategy for the free Personal tier (100/day):
  - The bulk "all stocks" endpoint returns ALL 150+ equities' current price,
    volume, market cap, sector and P/E in ONE call — this is used as the
    primary source for current snapshot / sector / market-cap data instead
    of hitting a per-symbol endpoint 150 times.
  - Per-symbol historical price calls are cached 23h and used sparingly —
    full OHLCV bar history for the technical scan still leans on the free
    yfinance .LG fallback so the 100/day budget isn't blown on one scan.
  - ASI / index history is cached 23h (it's historical, doesn't change).
"""

import os
import json
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List

log = logging.getLogger("apexscan.ngxpulse")

# ── Constants ──────────────────────────────────────────────────────────────────
NGXP_BASE_URL   = "https://www.ngxpulse.ng"
NGXP_CACHE_DIR  = Path(__file__).resolve().parent.parent / "data" / "ngxpulse_cache"
NGXP_CACHE_TTL  = 23 * 3600     # 23h for historical/semi-static data
NGXP_SNAP_TTL   = 20 * 60       # 20min for live snapshot data (matches API refresh cadence)
NGXP_RATE_PAUSE = 1.0           # generous pacing under the 10/min Personal limit
NGXP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_mem: Dict[str, dict] = {}


# ══════════════════════════════════════════════════════════════════════════════
# CACHE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cpath(key: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in key.upper())
    return NGXP_CACHE_DIR / f"{safe}.json"

def _cread(key: str, ttl: int = None) -> Optional[dict]:
    ttl = ttl or NGXP_CACHE_TTL
    if key in _mem:
        if time.time() - _mem[key].get("_ts", 0) < ttl:
            return _mem[key].get("data")
    path = _cpath(key)
    try:
        if path.exists():
            with open(path) as f:
                stored = json.load(f)
            if time.time() - stored.get("_ts", 0) < ttl:
                _mem[key] = stored
                return stored.get("data")
    except Exception:
        pass
    return None

def _cwrite(key: str, data) -> None:
    payload = {"_ts": time.time(), "data": data}
    _mem[key] = payload
    try:
        path = _cpath(key)
        tmp  = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, default=str)
        import shutil
        shutil.move(str(tmp), str(path))
    except Exception as e:
        log.debug(f"Cache write error {key}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# HTTP HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _get(endpoint: str, api_key: str, params: dict = None,
         ttl: int = None, require_key: bool = True) -> Optional[dict]:
    """
    GET request to the NGX Pulse API with caching and rate limiting.
    Auth: X-API-Key header (per the official docs — NOT Bearer).
    """
    if require_key and (not api_key or api_key.startswith("YOUR_")):
        return None

    cache_key = endpoint + str(sorted((params or {}).items()))
    cached    = _cread(cache_key, ttl)
    if cached is not None:
        log.debug(f"Cache hit: {endpoint}")
        return cached

    url     = f"{NGXP_BASE_URL}{endpoint}"
    headers = {
        "X-API-Key":     api_key or "",
        "Content-Type":  "application/json",
    }

    try:
        time.sleep(NGXP_RATE_PAUSE)
        resp = requests.get(url, headers=headers, params=params or {}, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            _cwrite(cache_key, data)
            log.info(f"NGX Pulse API: {endpoint} → {resp.status_code}")
            return data
        elif resp.status_code == 429:
            log.warning("NGX Pulse rate limit hit (10/min or 100/day on Personal) — backing off")
            time.sleep(30)
            return None
        elif resp.status_code == 401:
            log.error("NGX Pulse auth error 401: check X-API-Key header / key value")
            return None
        elif resp.status_code == 404:
            log.debug(f"NGX Pulse {endpoint}: 404 not found")
            return None
        else:
            log.warning(f"NGX Pulse {endpoint}: HTTP {resp.status_code} — {resp.text[:200]}")
            return None
    except requests.exceptions.ConnectionError:
        log.warning("NGX Pulse: connection error — offline or API down")
        return None
    except Exception as e:
        log.debug(f"NGX Pulse request error {endpoint}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def validate_api_key(api_key: str) -> Dict:
    """Validate the API key by hitting the market-status endpoint (cheap, always available)."""
    if not api_key or api_key.startswith("YOUR_"):
        return {"ok": False, "message": "API key not configured"}

    raw = _get("/api/ngxdata/market-status", api_key, ttl=60)
    if raw is not None:
        return {
            "ok":      True,
            "message": "✅ NGX Pulse API connected (Personal: 100 req/day, 10 req/min)",
            "plan":    "Personal",
        }
    return {
        "ok":      False,
        "message": "❌ NGX Pulse: could not validate key — check X-API-Key header value",
    }


def get_all_stocks(api_key: str) -> Optional[List[Dict]]:
    """
    Fetch current price/volume/market-cap/sector/P-E for ALL 150+ NGX equities
    in a single call. This is the primary, quota-cheap way to get a live
    snapshot + sector classification for the whole scan universe.
    Cached ~20 minutes (matches the API's own refresh cadence).
    """
    raw = _get("/api/ngxdata/stocks", api_key, ttl=NGXP_SNAP_TTL)
    if raw is None:
        return None
    data = raw.get("data") if isinstance(raw, dict) else raw
    return data if isinstance(data, list) else None


def get_all_stocks_lookup(api_key: str) -> Dict[str, Dict]:
    """Convenience wrapper: symbol → snapshot dict, for O(1) lookups during a scan."""
    rows = get_all_stocks(api_key) or []
    out = {}
    for row in rows:
        sym = str(row.get("symbol", "")).upper().replace(".LG", "")
        if sym:
            out[sym] = row
    return out


def get_stock_price(symbol: str, api_key: str) -> Optional[Dict]:
    """Latest snapshot for a single ticker (price, change%, volume, sector, P/E)."""
    clean = symbol.upper().replace(".LG", "").strip()
    raw = _get(f"/api/ngxdata/prices/{clean}", api_key, ttl=NGXP_SNAP_TTL)
    return raw if isinstance(raw, dict) else None


def get_equity_history(symbol: str, api_key: str, days: int = 400) -> Optional[pd.DataFrame]:
    """
    Attempt to fetch daily OHLCV history for a ticker via the prices endpoint
    with a date range. The documented response shows a single latest-snapshot
    object; some plans/tickers may return an array when from/to is supplied.
    Parsed defensively — returns None (caller should fall back to yfinance)
    if the response isn't a usable time series.
    """
    clean     = symbol.upper().replace(".LG", "").strip()
    cache_key = f"hist_{clean}_{days}"
    cached = _cread(cache_key, NGXP_CACHE_TTL)
    if cached:
        try:
            df = pd.DataFrame(cached)
            df.index = pd.to_datetime(df.index)
            if len(df) > 20:
                return df.sort_index()
        except Exception:
            pass

    end   = datetime.now()
    start = end - timedelta(days=days)
    raw = _get(f"/api/ngxdata/prices/{clean}", api_key,
               {"from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d")},
               ttl=NGXP_CACHE_TTL)
    if raw is None:
        return None

    records = None
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        records = raw.get("history") or raw.get("data") or raw.get("prices")

    if not records or not isinstance(records, list) or len(records) < 20:
        return None  # snapshot-only response — caller falls back to yfinance

    try:
        df = pd.DataFrame(records)
        date_col  = _find_col(df, ["date", "trade_date", "datetime"])
        open_col  = _find_col(df, ["open"])
        high_col  = _find_col(df, ["high"])
        low_col   = _find_col(df, ["low"])
        close_col = _find_col(df, ["close", "current_price", "price"])
        vol_col   = _find_col(df, ["volume"])
        if not date_col or not close_col:
            return None
        result = pd.DataFrame({
            "Open":   pd.to_numeric(df.get(open_col,  df[close_col]), errors="coerce"),
            "High":   pd.to_numeric(df.get(high_col,  df[close_col]), errors="coerce"),
            "Low":    pd.to_numeric(df.get(low_col,   df[close_col]), errors="coerce"),
            "Close":  pd.to_numeric(df[close_col],                     errors="coerce"),
            "Volume": pd.to_numeric(df.get(vol_col,   0),              errors="coerce"),
        }, index=pd.to_datetime(df[date_col]))
        result = result.dropna(subset=["Close"]).sort_index()
        if len(result) > 20:
            _cwrite(cache_key, result.to_dict())
            return result
    except Exception as e:
        log.debug(f"NGX Pulse history parse error {clean}: {e}")
    return None


def _num(row: dict, keys: List[str], default=None):
    """Best-effort numeric extraction trying several possible field-name variants,
    since the free-tier response schema for /api/ngxdata/stocks isn't fully
    documented and field names may differ slightly from what we assume."""
    for k in keys:
        for rk in row.keys():
            if rk.lower().replace(" ", "_") == k:
                try:
                    v = row[rk]
                    if v is None or v == "":
                        continue
                    return float(str(v).replace(",", "").replace("%", ""))
                except (ValueError, TypeError):
                    continue
    return default


def _txt(row: dict, keys: List[str], default=""):
    for k in keys:
        for rk in row.keys():
            if rk.lower().replace(" ", "_") == k:
                v = row[rk]
                if v not in (None, ""):
                    return str(v)
    return default


NGXP_ACCUM_DIR = Path(__file__).resolve().parent.parent / "data" / "ngxpulse_accum"
NGXP_ACCUM_DIR.mkdir(parents=True, exist_ok=True)


def accumulate_daily_snapshots(api_key: str) -> int:
    """
    Append today's snapshot (close/volume) for every NGX ticker to a small local
    CSV, one row per ticker per day (idempotent — re-running same day is a no-op).

    Why this exists: the free/Personal NGX Pulse tier only reliably returns a
    single latest-snapshot object per symbol, not a real OHLCV history array —
    so `get_equity_history` above returns None for virtually every ticker no
    matter how the key is configured. There is currently no free source of
    real NGX daily bar history. This function builds one ourselves: every time
    a scan runs, today's snapshot gets appended per ticker, and after ~50-200
    calendar days of the app being used regularly, real MA/RS/52w-high
    technical analysis becomes possible from this accumulated data — the
    same way any charting service has to bootstrap history for a new source.
    Call this once per scan (not once per ticker) to stay within the bulk
    endpoint's cheap single-call budget.
    """
    rows = get_all_stocks(api_key)
    if not rows:
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    written = 0
    for row in rows:
        sym = str(row.get("symbol", "")).upper().replace(".LG", "").strip()
        if not sym:
            continue
        close = _num(row, ["price", "current_price", "last_price", "close", "closing_price"])
        vol   = _num(row, ["volume", "today_volume", "trade_volume", "deal_volume"], default=0)
        if close is None:
            continue
        path = NGXP_ACCUM_DIR / f"{sym}.csv"
        try:
            existing_dates = set()
            if path.exists():
                with open(path) as f:
                    existing_dates = {line.split(",")[0] for line in f.readlines()[1:]}
            if today in existing_dates:
                continue
            is_new = not path.exists()
            with open(path, "a") as f:
                if is_new:
                    f.write("date,close,volume\n")
                f.write(f"{today},{close},{vol}\n")
            written += 1
        except Exception as e:
            log.debug(f"accum write error {sym}: {e}")
    log.info(f"NGX Pulse: accumulated today's snapshot for {written} tickers")
    return written


def get_accumulated_history(symbol: str) -> Optional[pd.DataFrame]:
    """Read back our self-built daily history CSV for a ticker, if enough rows exist."""
    clean = symbol.upper().replace(".LG", "").strip()
    path = NGXP_ACCUM_DIR / f"{clean}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        df = df.rename(columns={"close": "Close", "volume": "Volume"})
        df["Open"] = df["High"] = df["Low"] = df["Close"]
        return df if len(df) >= 5 else None
    except Exception as e:
        log.debug(f"accum read error {clean}: {e}")
        return None


def get_snapshot_scan_data(symbol: str, api_key: str,
                            lookup: Dict[str, Dict] = None) -> Optional[Dict]:
    """
    Normalized single-day snapshot for a ticker — used as a last-resort scan
    input when no real price history is available from any source (NGX Pulse
    history, NGN Market, yfinance, or accumulated local history). Lets the
    scanner surface today's movers (price change %, volume, sector) instead
    of showing zero results while historical coverage builds up.
    """
    clean = symbol.upper().replace(".LG", "").strip()
    row = None
    if lookup is not None:
        row = lookup.get(clean)
    if row is None:
        row = get_stock_price(clean, api_key)
    if not row:
        return None
    price      = _num(row, ["price", "current_price", "last_price", "close", "closing_price"])
    prev_close = _num(row, ["previous_close", "prev_close", "previous_price"])
    change_pct = _num(row, ["change_percent", "percent_change", "day_change_pct", "change_pct", "pct_change"])
    if change_pct is None and price is not None and prev_close:
        change_pct = round((price / prev_close - 1) * 100, 2)
    volume = _num(row, ["volume", "today_volume", "trade_volume", "deal_volume"], default=0)
    sector = _txt(row, ["sector", "industry", "sub_sector"])
    pe     = _num(row, ["pe", "pe_ratio", "p_e", "price_earnings"])
    name   = _txt(row, ["name", "company_name", "company"], default=clean)
    if price is None:
        return None
    return {
        "price": price, "change_pct": change_pct or 0.0, "volume": volume or 0,
        "sector": sector, "pe": pe, "name": name,
    }


def get_history_with_fallback(symbol: str, api_key: str) -> Optional[pd.DataFrame]:
    """Try the real history endpoint first, then our own accumulated daily log."""
    hist = get_equity_history(symbol, api_key, days=400)
    if hist is not None and len(hist) >= 5:
        return hist
    return get_accumulated_history(symbol)


def get_ngx_history_for_scan(ticker: str, api_key: str) -> Optional[pd.DataFrame]:
    """Main entry point called by scanner.py for NGX stocks (mirrors ngn_market's naming).
    Falls back to our self-accumulated daily snapshot log when the API's
    history endpoint doesn't return a real time series (see get_history_with_fallback)."""
    return get_history_with_fallback(ticker, api_key)





def get_market_overview(api_key: str) -> Optional[Dict]:
    """ASI level, market cap, volume, breadth, gainers/losers. Cached ~20 min."""
    raw = _get("/api/ngxdata/market", api_key, ttl=NGXP_SNAP_TTL)
    return raw if isinstance(raw, dict) else None


def get_market_status(api_key: str) -> Optional[Dict]:
    """{'status': 'open'|'closed', 'message': ..., 'timestamp': ...}"""
    raw = _get("/api/ngxdata/market-status", api_key, ttl=300)
    return raw if isinstance(raw, dict) else None


def get_asi_history(api_key: str, days: int = 380) -> Optional[pd.Series]:
    """
    NGX All-Share Index daily history — public/DB-backed endpoint, covers
    1996 to present. This is the best available ASI benchmark source
    (better coverage than NGN Market's ASI endpoint). Cached 23h.
    """
    cache_key = f"asi_history_{days}"
    cached = _cread(cache_key, NGXP_CACHE_TTL)
    if cached:
        try:
            s = pd.Series(cached)
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception:
            pass

    end   = datetime.now()
    start = end - timedelta(days=days)
    raw = _get("/api/ngxdata/indices/asi/history", api_key,
               {"from": start.strftime("%Y-%m-%d")}, ttl=NGXP_CACHE_TTL,
               require_key=False)  # index endpoints are documented as public
    if raw and isinstance(raw, dict):
        hist = raw.get("history")
        if hist and isinstance(hist, list):
            try:
                df = pd.DataFrame(hist)
                if "date" in df.columns and "value" in df.columns:
                    s = pd.Series(
                        pd.to_numeric(df["value"], errors="coerce").values,
                        index=pd.to_datetime(df["date"])
                    ).dropna().sort_index()
                    if len(s) > 50:
                        _cwrite(cache_key, s.to_dict())
                        log.info(f"NGX Pulse ASI: {len(s)} bars fetched")
                        return s
            except Exception as e:
                log.debug(f"ASI parse error: {e}")

    return _yf_asi_fallback(cache_key)


def _yf_asi_fallback(cache_key: str) -> Optional[pd.Series]:
    try:
        import yfinance as yf
        raw = yf.Ticker("^NGXASI").history(period="2y")["Close"]
        if len(raw) > 50:
            if hasattr(raw.index, "tz") and raw.index.tz:
                raw.index = raw.index.tz_convert(None)
            raw.index = pd.to_datetime(raw.index).normalize()
            _cwrite(cache_key, raw.to_dict())
            log.info(f"ASI fallback yfinance: {len(raw)} bars")
            return raw
    except Exception as e:
        log.debug(f"ASI yfinance fallback failed: {e}")
    return None


def get_ngx_index(api_key: str, days: int = 380) -> Optional[Dict]:
    """
    Compatibility wrapper matching the {"df": DataFrame-with-Close} shape
    scanner.py expects for the benchmark series.
    """
    s = get_asi_history(api_key, days=days)
    if s is not None and len(s) > 0:
        return {"df": pd.DataFrame({"Close": s})}
    return None


def get_index(code: str, api_key: str) -> Optional[Dict]:
    """Single NGX index/sector-index snapshot by code or slug (e.g. 'ASI', 'ngx-bnk')."""
    raw = _get(f"/api/ngxdata/indices/{code.lower()}", api_key, ttl=NGXP_SNAP_TTL,
               require_key=False)
    return raw.get("data") if isinstance(raw, dict) else None


def get_dividend_history(symbol: str, api_key: str) -> Optional[List[Dict]]:
    """Full dividend history for a ticker (declaration/ex-div/pay dates, amount, type)."""
    clean = symbol.upper().replace(".LG", "").strip()
    raw = _get(f"/api/ngxdata/dividends/{clean}", api_key, ttl=NGXP_CACHE_TTL)
    if raw is None:
        return None
    data = raw.get("data") if isinstance(raw, dict) else raw
    return data if isinstance(data, list) else None


def get_disclosures(api_key: str) -> Optional[List[Dict]]:
    """Recent corporate disclosures — earnings, dividends, rights issues, board actions."""
    raw = _get("/api/ngxdata/disclosures", api_key, ttl=1800)
    if raw is None:
        return None
    data = raw.get("data") if isinstance(raw, dict) else raw
    return data if isinstance(data, list) else None


def get_market_news(api_key: str) -> Optional[List[Dict]]:
    """Latest Nigerian capital-market news (Nairametrics, BusinessDay, etc.)."""
    raw = _get("/api/news", api_key, ttl=900)
    if raw is None:
        return None
    data = raw.get("data") if isinstance(raw, dict) else raw
    return data if isinstance(data, list) else None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        for col in df.columns:
            if col.lower().strip() == c:
                return col
    return None
