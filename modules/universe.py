"""
modules/universe.py — ApexScan Dynamic Universe Builder
Fetches S&P 500, NASDAQ 100, Russell 2000, S&P 400 components automatically.
Multiple source fallbacks ensure reliability.

Source priority per index:
  S&P 500   : GitHub datasets CSV → Wikipedia → Hardcoded emergency list
  NASDAQ 100: Wikipedia (table 4) → GitHub NASDAQ list
  Russell2000: iShares IWM CSV → Wikipedia
  S&P 400   : iShares IJH CSV

Total universe: 500–2500 tickers depending on preset chosen.
"""

import requests
import pandas as pd
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from io import StringIO

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/universe_cache")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml,text/csv;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Emergency fallback — core tickers always available ────────────────────────
# These 100 tickers are hardcoded so the scanner never returns empty
# even if all network sources fail
EMERGENCY_SP500 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","AVGO","JPM",
    "LLY","V","UNH","XOM","COST","MA","PG","JNJ","HD","ABBV",
    "MRK","NFLX","CRM","BAC","CVX","KO","PEP","WMT","TMO","MCD",
    "CSCO","ORCL","ACN","ABT","ADBE","TXN","LIN","DHR","PM","NKE",
    "NEE","IBM","INTC","RTX","HON","CAT","SPGI","UPS","BLK","AMGN",
    "INTU","LOW","GS","MS","ISRG","SBUX","SYK","MDLZ","TJX","GILD",
    "VRTX","ADP","REGN","MMM","ELV","PLD","AMT","PANW","ANET","MU",
    "ADI","LRCX","KLAC","SNPS","CDNS","MRVL","ARM","CRWD","DDOG","SNOW",
    "NOW","TEAM","WDAY","ZS","NET","PLTR","RKLB","IONQ","AFRM","SOFI",
    "HOOD","HIMS","LLY","NVO","ISRG","SQ","COIN","MSTR","APP","CELH",
]

EMERGENCY_NASDAQ_EXTRA = [
    "ASML","TSM","BKNG","ABNB","PYPL","MELI","MDB","DXCM","PCAR","FAST",
    "CTAS","ODFL","BIIB","ILMN","VRSK","CPRT","IDXX","ROST","DLTR","WBA",
]


def _ensure_dir():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _cache_path(name: str) -> Path:
    _ensure_dir()
    return CACHE_DIR / f"{name}.json"


def _cache_valid(path: Path, hours: float = 24) -> bool:
    try:
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < timedelta(hours=hours)
    except Exception:
        return False


def _read_cache(path: Path) -> Optional[list]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_cache(path: Path, data: list):
    try:
        _ensure_dir()
        path.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _get(url: str, timeout: int = 15) -> Optional[requests.Response]:
    """HTTP GET with standard headers and error handling."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        log.debug(f"GET failed {url}: {e}")
        return None


def _clean_ticker(tk: str) -> str:
    """Normalise ticker — replace dots with dashes (BRK.B → BRK-B)."""
    return str(tk).strip().replace(".", "-").upper()


# ══════════════════════════════════════════════════════════════════════════════
# S&P 500
# ══════════════════════════════════════════════════════════════════════════════

def get_sp500(cache_hours: float = 24) -> List[Dict]:
    """
    Fetch S&P 500 components. ~503 large-cap US tickers.
    Sources tried in order:
      1. GitHub datasets/s-and-p-500-companies (CSV, very reliable)
      2. Wikipedia HTML table
      3. Emergency hardcoded list (100 core tickers)
    """
    cache = _cache_path("sp500")
    if _cache_valid(cache, cache_hours):
        data = _read_cache(cache)
        if data:
            log.info(f"SP500 loaded from cache: {len(data)} tickers")
            return data

    result = []

    # Source 1: GitHub datasets (CSV)
    urls_csv = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/sp500/sp500_symbols.txt",
    ]
    for url in urls_csv:
        resp = _get(url)
        if resp:
            try:
                if url.endswith(".csv"):
                    df = pd.read_csv(StringIO(resp.text))
                    # Column names vary — find ticker column
                    for col in df.columns:
                        if col.lower() in ("symbol","ticker"):
                            for _, row in df.iterrows():
                                tk = _clean_ticker(row[col])
                                if tk and len(tk) <= 6:
                                    result.append({
                                        "ticker":          tk,
                                        "name":            str(row.get("Name", row.get("name", tk))).strip(),
                                        "sector":          str(row.get("Sector", row.get("sector", ""))).strip(),
                                        "sub_industry":    str(row.get("Sub-Industry","")).strip(),
                                        "index":           "SP500",
                                        "market_cap_tier": "large",
                                    })
                            break
                else:
                    # Plain text list
                    for line in resp.text.splitlines():
                        tk = _clean_ticker(line)
                        if tk and len(tk) <= 6 and tk.isalpha():
                            result.append({"ticker": tk, "name": tk, "sector": "",
                                           "sub_industry": "", "index": "SP500",
                                           "market_cap_tier": "large"})

                if result:
                    log.info(f"SP500 from GitHub: {len(result)} tickers")
                    break
            except Exception as e:
                log.debug(f"SP500 CSV parse error: {e}")
                result = []

    # Source 2: Wikipedia HTML
    if not result:
        resp = _get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if resp:
            try:
                tables = pd.read_html(StringIO(resp.text), header=0)
                df = tables[0]
                for col in df.columns:
                    if col.lower() in ("symbol","ticker"):
                        for _, row in df.iterrows():
                            tk = _clean_ticker(row[col])
                            if tk and len(tk) <= 6:
                                result.append({
                                    "ticker":          tk,
                                    "name":            str(row.get("Security", tk)).strip(),
                                    "sector":          str(row.get("GICS Sector","")).strip(),
                                    "sub_industry":    str(row.get("GICS Sub-Industry","")).strip(),
                                    "index":           "SP500",
                                    "market_cap_tier": "large",
                                })
                        break
                if result:
                    log.info(f"SP500 from Wikipedia: {len(result)} tickers")
            except Exception as e:
                log.debug(f"SP500 Wikipedia parse error: {e}")

    # Source 3: Emergency fallback
    if not result:
        log.warning("SP500 all sources failed — using emergency list (100 core tickers)")
        result = [{"ticker": tk, "name": tk, "sector": "", "sub_industry": "",
                   "index": "SP500", "market_cap_tier": "large"}
                  for tk in EMERGENCY_SP500]

    _write_cache(cache, result)
    log.info(f"SP500 final: {len(result)} tickers")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# NASDAQ 100
# ══════════════════════════════════════════════════════════════════════════════

def get_nasdaq100(cache_hours: float = 24) -> List[Dict]:
    """
    Fetch NASDAQ 100 components. ~101 large-cap tech-heavy tickers.
    Sources: Wikipedia → GitHub list → Emergency subset
    """
    cache = _cache_path("nasdaq100")
    if _cache_valid(cache, cache_hours):
        data = _read_cache(cache)
        if data:
            log.info(f"NASDAQ100 loaded from cache: {len(data)} tickers")
            return data

    result = []

    # Source 1: Wikipedia
    resp = _get("https://en.wikipedia.org/wiki/Nasdaq-100")
    if resp:
        try:
            tables = pd.read_html(StringIO(resp.text), header=0)
            # Try each table looking for one with ticker symbols
            for tbl in tables:
                cols_lower = [c.lower() for c in tbl.columns]
                if any(k in " ".join(cols_lower) for k in ["ticker","symbol"]):
                    for col in tbl.columns:
                        if col.lower() in ("ticker","symbol"):
                            for _, row in tbl.iterrows():
                                tk = _clean_ticker(row[col])
                                if tk and 1 < len(tk) <= 6:
                                    result.append({
                                        "ticker":          tk,
                                        "name":            str(row.get("Company", row.get("company", tk))).strip(),
                                        "sector":          str(row.get("Sector","")).strip(),
                                        "sub_industry":    "",
                                        "index":           "NASDAQ100",
                                        "market_cap_tier": "large",
                                    })
                            break
                    if result:
                        break
            if result:
                log.info(f"NASDAQ100 from Wikipedia: {len(result)} tickers")
        except Exception as e:
            log.debug(f"NASDAQ100 Wikipedia error: {e}")

    # Source 2: GitHub
    if not result:
        resp = _get("https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_symbols.txt")
        if resp:
            try:
                lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
                result = [{"ticker": _clean_ticker(tk), "name": tk, "sector": "",
                           "sub_industry": "", "index": "NASDAQ100",
                           "market_cap_tier": "large"}
                          for tk in lines[:110] if len(tk) <= 6]
                if result:
                    log.info(f"NASDAQ100 from GitHub: {len(result)} tickers")
            except Exception:
                pass

    # Emergency fallback
    if not result:
        log.warning("NASDAQ100 all sources failed — using emergency subset")
        result = [{"ticker": tk, "name": tk, "sector": "", "sub_industry": "",
                   "index": "NASDAQ100", "market_cap_tier": "large"}
                  for tk in EMERGENCY_SP500[:50] + EMERGENCY_NASDAQ_EXTRA]

    _write_cache(cache, result)
    log.info(f"NASDAQ100 final: {len(result)} tickers")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# S&P 400 MID-CAP
# ══════════════════════════════════════════════════════════════════════════════

def get_sp400_midcap(cache_hours: float = 48) -> List[Dict]:
    """Fetch S&P 400 Mid-Cap from iShares IJH ETF holdings."""
    cache = _cache_path("sp400")
    if _cache_valid(cache, cache_hours):
        data = _read_cache(cache)
        if data:
            return data

    result = []
    urls = [
        "https://www.ishares.com/us/products/239763/ishares-sp-mid-cap-etf/1467271812596.ajax?fileType=csv&fileName=IJH_holdings&dataType=fund",
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/sp400/sp400_symbols.txt",
    ]

    for url in urls:
        resp = _get(url)
        if not resp:
            continue
        try:
            if "ishares" in url:
                lines = resp.text.split("\n")
                start = next((i for i, l in enumerate(lines)
                              if "Ticker" in l or "Symbol" in l), 0)
                df = pd.read_csv(StringIO("\n".join(lines[start:])), on_bad_lines="skip")
                for col in df.columns:
                    if col.strip().lower() in ("ticker","symbol"):
                        ac_col = next((c for c in df.columns if "asset" in c.lower()), None)
                        for _, row in df.iterrows():
                            tk = _clean_ticker(row[col])
                            ac = str(row.get(ac_col,"Equity")) if ac_col else "Equity"
                            if tk and len(tk) <= 6 and "Equity" in ac and tk.replace("-","").isalpha():
                                result.append({
                                    "ticker": tk, "name": str(row.get("Name",tk)).strip(),
                                    "sector": str(row.get("Sector","")).strip(),
                                    "sub_industry": "", "index": "SP400",
                                    "market_cap_tier": "mid",
                                })
                        break
            else:
                for line in resp.text.splitlines():
                    tk = _clean_ticker(line)
                    if tk and len(tk) <= 6:
                        result.append({"ticker": tk, "name": tk, "sector": "",
                                       "sub_industry": "", "index": "SP400",
                                       "market_cap_tier": "mid"})

            if result:
                log.info(f"SP400 loaded: {len(result)} tickers")
                break
        except Exception as e:
            log.debug(f"SP400 source error: {e}")
            result = []

    if result:
        _write_cache(cache, result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# RUSSELL 2000
# ══════════════════════════════════════════════════════════════════════════════

def get_russell2000(cache_hours: float = 48) -> List[Dict]:
    """Fetch Russell 2000 small-cap components from iShares IWM."""
    cache = _cache_path("russell2000")
    if _cache_valid(cache, cache_hours):
        data = _read_cache(cache)
        if data:
            return data

    result = []
    url = ("https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
           "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund")

    resp = _get(url, timeout=20)
    if resp:
        try:
            lines = resp.text.split("\n")
            start = next((i for i, l in enumerate(lines)
                          if "Ticker" in l or "Symbol" in l), 0)
            df = pd.read_csv(StringIO("\n".join(lines[start:])), on_bad_lines="skip")
            for col in df.columns:
                if col.strip().lower() in ("ticker","symbol"):
                    ac_col = next((c for c in df.columns if "asset" in c.lower()), None)
                    for _, row in df.iterrows():
                        tk = _clean_ticker(row[col])
                        ac = str(row.get(ac_col,"Equity")) if ac_col else "Equity"
                        if tk and len(tk) <= 6 and "Equity" in ac and tk.replace("-","").isalpha():
                            result.append({
                                "ticker": tk, "name": str(row.get("Name",tk)).strip(),
                                "sector": str(row.get("Sector","")).strip(),
                                "sub_industry": "", "index": "RUSSELL2000",
                                "market_cap_tier": "small",
                            })
                    break
            log.info(f"Russell2000 loaded: {len(result)} tickers")
        except Exception as e:
            log.debug(f"Russell2000 parse error: {e}")

    if result:
        _write_cache(cache, result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# DOW 30
# ══════════════════════════════════════════════════════════════════════════════

def get_dow30(cache_hours: float = 168) -> List[Dict]:
    """Fetch Dow Jones 30 from Wikipedia."""
    cache = _cache_path("dow30")
    if _cache_valid(cache, cache_hours):
        data = _read_cache(cache)
        if data:
            return data

    # Hardcode DOW 30 — changes very rarely, not worth a fragile scrape
    dow_tickers = [
        "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
        "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
        "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
    ]
    result = [{"ticker": tk, "name": tk, "sector": "", "sub_industry": "",
               "index": "DOW30", "market_cap_tier": "large"}
              for tk in dow_tickers]
    _write_cache(cache, result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# GICS SECTOR MAPPING
# ══════════════════════════════════════════════════════════════════════════════

GICS_TO_THEME = {
    "Information Technology":  "tech",
    "Communication Services":  "communication",
    "Health Care":             "healthcare",
    "Financials":              "financials",
    "Consumer Discretionary":  "consumer_discretionary",
    "Consumer Staples":        "consumer_staples",
    "Industrials":             "industrials",
    "Energy":                  "energy",
    "Materials":               "materials",
    "Real Estate":             "real_estate",
    "Utilities":               "utilities",
}

UNIVERSE_PRESETS = {
    "custom":     {"label": "Custom (config.yaml themes)",           "fetchers": []},
    "nasdaq100":  {"label": "NASDAQ 100 (~101 tickers)",             "fetchers": ["nasdaq100"]},
    "sp500":      {"label": "S&P 500 (~503 tickers)",                "fetchers": ["sp500"]},
    "sp500+ndx":  {"label": "S&P 500 + NASDAQ 100 (~550)",           "fetchers": ["sp500","nasdaq100"]},
    "large_cap":  {"label": "Large Cap (SP500+NASDAQ+DOW)",          "fetchers": ["sp500","nasdaq100","dow30"]},
    "mid_cap":    {"label": "Mid Cap (S&P 400, ~400)",               "fetchers": ["sp400"]},
    "broad":      {"label": "Broad (SP500+NASDAQ+SP400, ~900)",      "fetchers": ["sp500","nasdaq100","sp400"]},
    "full":       {"label": "Full Universe (SP500+NDX+SP400+R2000)", "fetchers": ["sp500","nasdaq100","sp400","russell2000"]},
}


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_universe(
    preset:           str = "sp500",
    extra_tickers:    List[str] = None,
    exclude_tickers:  List[str] = None,
    min_price:        float = 5.0,
    max_price:        float = 99999.0,
    include_sectors:  List[str] = None,
    exclude_sectors:  List[str] = None,
    market_cap_tiers: List[str] = None,
    cache_hours:      float = 24,
) -> List[Dict]:
    """
    Build a deduplicated ticker universe from one or more index sources.
    Returns list of {ticker, name, sector, index, market_cap_tier, theme}.
    """
    fetcher_map = {
        "sp500":       get_sp500,
        "nasdaq100":   get_nasdaq100,
        "sp400":       get_sp400_midcap,
        "russell2000": get_russell2000,
        "dow30":       get_dow30,
    }

    fetchers   = UNIVERSE_PRESETS.get(preset, UNIVERSE_PRESETS["sp500"])["fetchers"]
    all_items  = []
    seen       = set()

    for fname in fetchers:
        fn   = fetcher_map.get(fname)
        if not fn:
            continue
        try:
            data = fn(cache_hours=cache_hours)
            for item in data:
                tk = item.get("ticker","").strip()
                if tk and tk not in seen:
                    seen.add(tk)
                    item["theme"] = GICS_TO_THEME.get(item.get("sector",""), "other")
                    all_items.append(item)
        except Exception as e:
            log.warning(f"Fetcher {fname} failed: {e}")

    # Extra tickers
    if extra_tickers:
        for tk in extra_tickers:
            tk = tk.strip().upper().replace(".","-")
            if tk and tk not in seen:
                seen.add(tk)
                all_items.append({
                    "ticker": tk, "name": tk, "sector": "",
                    "sub_industry": "", "index": "CUSTOM",
                    "market_cap_tier": "unknown", "theme": "custom",
                })

    # Exclude tickers
    if exclude_tickers:
        excl = {t.upper().replace(".","-") for t in exclude_tickers}
        all_items = [t for t in all_items if t["ticker"] not in excl]

    # Sector filters
    if include_sectors:
        inc = {s.lower() for s in include_sectors}
        all_items = [t for t in all_items
                     if t.get("sector","").lower() in inc
                     or t["index"] == "CUSTOM"]

    if exclude_sectors:
        exc = {s.lower() for s in exclude_sectors}
        all_items = [t for t in all_items
                     if t.get("sector","").lower() not in exc]

    # Market cap tier filter
    if market_cap_tiers:
        tiers = set(market_cap_tiers)
        all_items = [t for t in all_items
                     if t.get("market_cap_tier","unknown") in tiers
                     or t["index"] == "CUSTOM"]

    log.info(f"Universe built: {len(all_items)} tickers (preset={preset})")
    return all_items


def get_universe_stats(universe: List[Dict]) -> Dict:
    """Summary statistics for a built universe."""
    if not universe:
        return {"total": 0, "by_index": {}, "by_sector": {}, "by_tier": {}}

    by_index  = {}
    by_sector = {}
    by_tier   = {}

    for item in universe:
        idx  = item.get("index", "?")
        sec  = item.get("sector", "Unknown") or "Unknown"
        tier = item.get("market_cap_tier", "?")
        by_index[idx]  = by_index.get(idx, 0) + 1
        by_sector[sec] = by_sector.get(sec, 0) + 1
        by_tier[tier]  = by_tier.get(tier, 0) + 1

    return {
        "total":     len(universe),
        "by_index":  dict(sorted(by_index.items(),  key=lambda x: -x[1])),
        "by_sector": dict(sorted(by_sector.items(), key=lambda x: -x[1])),
        "by_tier":   dict(sorted(by_tier.items(),   key=lambda x: -x[1])),
    }


def refresh_universe_cache():
    """Force-clear all cached universe data."""
    _ensure_dir()
    for name in ["sp500","nasdaq100","sp400","russell2000","dow30"]:
        p = _cache_path(name)
        if p.exists():
            p.unlink()
    log.info("Universe cache cleared")
