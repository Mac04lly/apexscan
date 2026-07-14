"""
modules/universe.py — ApexScan Dynamic Universe Builder
"""
import requests, json, logging, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from io import StringIO

log = logging.getLogger(__name__)
CACHE_DIR = Path("data/universe_cache")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

EMERGENCY_SP500 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","AVGO","JPM",
    "LLY","V","UNH","XOM","COST","MA","PG","JNJ","HD","ABBV","MRK","NFLX",
    "CRM","BAC","CVX","KO","PEP","WMT","TMO","MCD","CSCO","ORCL","ACN","ABT",
    "ADBE","TXN","LIN","DHR","PM","NKE","NEE","IBM","INTC","RTX","HON","CAT",
    "SPGI","UPS","BLK","AMGN","INTU","LOW","GS","MS","ISRG","SBUX","SYK",
    "MDLZ","TJX","GILD","VRTX","ADP","REGN","ELV","PLD","AMT","PANW","ANET",
    "MU","ADI","LRCX","KLAC","SNPS","CDNS","MRVL","ARM","CRWD","DDOG","SNOW",
    "NOW","TEAM","WDAY","ZS","NET","PLTR","HIMS","RDDT","CAVA","SOUN","APP",
    "CELH","COIN","MSTR","IONQ","RKLB","AFRM","SOFI","HOOD","ASTS",
]

def _ensure_dir():
    try: CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except: pass

def _cache_path(name):
    _ensure_dir()
    return CACHE_DIR / f"{name}.json"

def _cache_valid(path, hours=24):
    try:
        if not path.exists(): return False
        return (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)) < timedelta(hours=hours)
    except: return False

def _read_cache(path):
    try: return json.loads(path.read_text())
    except: return None

def _write_cache(path, data):
    try: _ensure_dir(); path.write_text(json.dumps(data, indent=2))
    except: pass

def _get(url, timeout=15):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status(); return r
    except Exception as e: log.debug(f"GET {url}: {e}"); return None

def _clean(tk): return str(tk).strip().replace(".","-").upper()

def get_sp500(cache_hours=24):
    cache = _cache_path("sp500")
    if _cache_valid(cache, cache_hours):
        d = _read_cache(cache)
        if d: return d
    result = []
    r = _get("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
    if r:
        try:
            df = pd.read_csv(StringIO(r.text))
            for _, row in df.iterrows():
                tk = _clean(row.get("Symbol", row.get("symbol","")))
                if tk and len(tk) <= 6:
                    result.append({"ticker":tk,"name":str(row.get("Name",tk)).strip(),
                                   "sector":str(row.get("Sector","")).strip(),
                                   "sub_industry":"","index":"SP500","market_cap_tier":"large"})
        except: result = []
    if not result:
        r = _get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if r:
            try:
                tables = pd.read_html(StringIO(r.text), header=0)
                df = tables[0]
                for col in df.columns:
                    if col.lower() in ("symbol","ticker"):
                        for _, row in df.iterrows():
                            tk = _clean(row[col])
                            if tk and len(tk) <= 6:
                                result.append({"ticker":tk,"name":str(row.get("Security",tk)).strip(),
                                               "sector":str(row.get("GICS Sector","")).strip(),
                                               "sub_industry":"","index":"SP500","market_cap_tier":"large"})
                        break
            except: pass
    if not result:
        log.warning("SP500 all sources failed — using emergency list")
        result = [{"ticker":tk,"name":tk,"sector":"","sub_industry":"","index":"SP500","market_cap_tier":"large"} for tk in EMERGENCY_SP500]
    _write_cache(cache, result)
    log.info(f"SP500: {len(result)} tickers"); return result

def get_nasdaq100(cache_hours=24):
    cache = _cache_path("nasdaq100")
    if _cache_valid(cache, cache_hours):
        d = _read_cache(cache)
        if d: return d
    result = []
    r = _get("https://en.wikipedia.org/wiki/Nasdaq-100")
    if r:
        try:
            for tbl in pd.read_html(StringIO(r.text), header=0):
                for col in tbl.columns:
                    if col.lower() in ("ticker","symbol"):
                        for _, row in tbl.iterrows():
                            tk = _clean(row[col])
                            if tk and 1 < len(tk) <= 6:
                                result.append({"ticker":tk,"name":str(row.get("Company",tk)).strip(),
                                               "sector":str(row.get("Sector","")).strip(),
                                               "sub_industry":"","index":"NASDAQ100","market_cap_tier":"large"})
                        if result: break
                if result: break
        except: pass
    if not result:
        result = [{"ticker":tk,"name":tk,"sector":"","sub_industry":"","index":"NASDAQ100","market_cap_tier":"large"} for tk in EMERGENCY_SP500[:101]]
    _write_cache(cache, result)
    log.info(f"NASDAQ100: {len(result)} tickers"); return result

def get_sp400_midcap(cache_hours=48):
    cache = _cache_path("sp400")
    if _cache_valid(cache, cache_hours):
        d = _read_cache(cache)
        if d: return d
    result = []
    r = _get("https://www.ishares.com/us/products/239763/ishares-sp-mid-cap-etf/1467271812596.ajax?fileType=csv&fileName=IJH_holdings&dataType=fund", timeout=20)
    if r:
        try:
            lines = r.text.split("\n")
            start = next((i for i,l in enumerate(lines) if "Ticker" in l or "Symbol" in l), 0)
            df = pd.read_csv(StringIO("\n".join(lines[start:])), on_bad_lines="skip")
            for col in df.columns:
                if col.strip().lower() in ("ticker","symbol"):
                    ac_col = next((c for c in df.columns if "asset" in c.lower()), None)
                    for _, row in df.iterrows():
                        tk = _clean(row[col])
                        ac = str(row.get(ac_col,"Equity")) if ac_col else "Equity"
                        if tk and len(tk)<=6 and "Equity" in ac and tk.replace("-","").isalpha():
                            result.append({"ticker":tk,"name":str(row.get("Name",tk)).strip(),
                                           "sector":str(row.get("Sector","")).strip(),
                                           "sub_industry":"","index":"SP400","market_cap_tier":"mid"})
                    break
        except: pass
    if result: _write_cache(cache, result)
    return result

def get_russell2000(cache_hours=48):
    cache = _cache_path("russell2000")
    if _cache_valid(cache, cache_hours):
        d = _read_cache(cache)
        if d: return d
    result = []
    r = _get("https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund", timeout=20)
    if r:
        try:
            lines = r.text.split("\n")
            start = next((i for i,l in enumerate(lines) if "Ticker" in l or "Symbol" in l), 0)
            df = pd.read_csv(StringIO("\n".join(lines[start:])), on_bad_lines="skip")
            for col in df.columns:
                if col.strip().lower() in ("ticker","symbol"):
                    ac_col = next((c for c in df.columns if "asset" in c.lower()), None)
                    for _, row in df.iterrows():
                        tk = _clean(row[col])
                        ac = str(row.get(ac_col,"Equity")) if ac_col else "Equity"
                        if tk and len(tk)<=6 and "Equity" in ac and tk.replace("-","").isalpha():
                            result.append({"ticker":tk,"name":str(row.get("Name",tk)).strip(),
                                           "sector":str(row.get("Sector","")).strip(),
                                           "sub_industry":"","index":"RUSSELL2000","market_cap_tier":"small"})
                    break
        except: pass
    if result: _write_cache(cache, result)
    return result

def get_dow30(cache_hours=168):
    cache = _cache_path("dow30")
    if _cache_valid(cache, cache_hours):
        d = _read_cache(cache)
        if d: return d
    tickers = ["AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
               "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
               "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT"]
    result = [{"ticker":tk,"name":tk,"sector":"","sub_industry":"","index":"DOW30","market_cap_tier":"large"} for tk in tickers]
    _write_cache(cache, result); return result

GICS_TO_THEME = {
    "Information Technology":"tech","Communication Services":"communication",
    "Health Care":"healthcare","Financials":"financials",
    "Consumer Discretionary":"consumer_discretionary","Consumer Staples":"consumer_staples",
    "Industrials":"industrials","Energy":"energy","Materials":"materials",
    "Real Estate":"real_estate","Utilities":"utilities",
}

UNIVERSE_PRESETS = {
    "custom":    {"label":"Custom (config.yaml)","fetchers":[]},
    "nasdaq100": {"label":"NASDAQ 100","fetchers":["nasdaq100"]},
    "sp500":     {"label":"S&P 500","fetchers":["sp500"]},
    "sp500+ndx": {"label":"S&P 500 + NASDAQ","fetchers":["sp500","nasdaq100"]},
    "large_cap": {"label":"Large Cap (SP500+NDX+DOW)","fetchers":["sp500","nasdaq100","dow30"]},
    "mid_cap":   {"label":"Mid Cap S&P 400","fetchers":["sp400"]},
    "broad":     {"label":"Broad (SP500+NDX+SP400)","fetchers":["sp500","nasdaq100","sp400"]},
    "full":      {"label":"Full Universe","fetchers":["sp500","nasdaq100","sp400","russell2000"]},
}

def build_universe(preset="sp500", extra_tickers=None, exclude_tickers=None,
                   min_price=5.0, max_price=99999.0, include_sectors=None,
                   exclude_sectors=None, market_cap_tiers=None, cache_hours=24):
    fetcher_map = {"sp500":get_sp500,"nasdaq100":get_nasdaq100,"sp400":get_sp400_midcap,
                   "russell2000":get_russell2000,"dow30":get_dow30}
    fetchers  = UNIVERSE_PRESETS.get(preset, UNIVERSE_PRESETS["sp500"])["fetchers"]
    all_items, seen = [], set()
    for fname in fetchers:
        fn = fetcher_map.get(fname)
        if not fn: continue
        try:
            for item in fn(cache_hours=cache_hours):
                tk = item.get("ticker","").strip()
                if tk and tk not in seen:
                    seen.add(tk)
                    item["theme"] = GICS_TO_THEME.get(item.get("sector",""), "other")
                    all_items.append(item)
        except Exception as e:
            log.warning(f"Fetcher {fname} failed: {e}")
    if extra_tickers:
        for tk in extra_tickers:
            tk = tk.strip().upper().replace(".","-")
            if tk and tk not in seen:
                seen.add(tk)
                all_items.append({"ticker":tk,"name":tk,"sector":"","sub_industry":"",
                                   "index":"CUSTOM","market_cap_tier":"unknown","theme":"custom"})
    if exclude_tickers:
        excl = {t.upper().replace(".","-") for t in exclude_tickers}
        all_items = [t for t in all_items if t["ticker"] not in excl]
    if include_sectors:
        inc = {s.lower() for s in include_sectors}
        all_items = [t for t in all_items if t.get("sector","").lower() in inc or t["index"]=="CUSTOM"]
    if exclude_sectors:
        exc = {s.lower() for s in exclude_sectors}
        all_items = [t for t in all_items if t.get("sector","").lower() not in exc]
    if market_cap_tiers:
        tiers = set(market_cap_tiers)
        all_items = [t for t in all_items if t.get("market_cap_tier","unknown") in tiers or t["index"]=="CUSTOM"]
    log.info(f"Universe: {len(all_items)} tickers (preset={preset})")
    return all_items

def get_universe_stats(universe):
    if not universe: return {"total":0,"by_index":{},"by_sector":{},"by_tier":{}}
    by_index, by_sector, by_tier = {}, {}, {}
    for item in universe:
        idx=item.get("index","?"); sec=item.get("sector","Unknown") or "Unknown"; tier=item.get("market_cap_tier","?")
        by_index[idx]=by_index.get(idx,0)+1; by_sector[sec]=by_sector.get(sec,0)+1; by_tier[tier]=by_tier.get(tier,0)+1
    return {"total":len(universe),
            "by_index": dict(sorted(by_index.items(),  key=lambda x:-x[1])),
            "by_sector":dict(sorted(by_sector.items(), key=lambda x:-x[1])),
            "by_tier":  dict(sorted(by_tier.items(),   key=lambda x:-x[1]))}

def refresh_universe_cache():
    _ensure_dir()
    for name in ["sp500","nasdaq100","sp400","russell2000","dow30"]:
        p = _cache_path(name)
        if p.exists(): p.unlink()
    log.info("Universe cache cleared")
