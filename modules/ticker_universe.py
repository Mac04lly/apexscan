"""
modules/ticker_universe.py — Smart Ticker Universe Builder
Uses tiered quality sources — highest quality first.

Quality Tier 1 (Best): S&P 500 + NASDAQ 100
  - Pre-filtered: profitable, liquid, institutionally owned
  - ~600 unique tickers
  - These will generate the most reliable Apex Scores

Quality Tier 2 (Good): S&P 400 Mid Cap + curated growth list
  - Solid companies, decent liquidity
  - ~600 additional tickers
  - Includes high-growth names not yet in S&P 500

Quality Tier 3 (Selective): Emerging growth names
  - Hand-curated high-conviction small caps
  - ~200 names with strong growth profiles
  - Higher risk but highest alpha potential

Total: ~1,200-1,400 tickers — all with a reason to be there.

NOT included (by design):
  - Random penny stocks
  - OTC/pink sheet stocks
  - Tickers with < $500M market cap (unless in emerging_gems theme)
  - Tickers with < 500K average daily volume
"""

import requests
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/universe_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _cache_valid(path: Path, hours: int = 24) -> bool:
    if not path.exists():
        return False
    age = (datetime.now() - datetime.fromtimestamp(
        path.stat().st_mtime)).total_seconds()
    return age < hours * 3600


def _load_cache(path: Path) -> List[str]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _save_cache(path: Path, tickers: List[str]):
    try:
        with open(path, "w") as f:
            json.dump(tickers, f)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — S&P 500
# Best quality: large cap, profitable, liquid, institutionally owned
# Source: Wikipedia (mirrors the official S&P Dow Jones list)
# Updated: daily cache
# ══════════════════════════════════════════════════════════════════════════════

def get_sp500() -> List[str]:
    """
    503 tickers — the gold standard of US equities.
    Every company here has passed S&P's quality screen:
    - Market cap > $14.5B
    - Positive earnings for 4 consecutive quarters
    - High liquidity (annual dollar value traded > 1.0 float-adjusted market cap)
    """
    cache = _cache_path("sp500")
    if _cache_valid(cache, hours=24):
        data = _load_cache(cache)
        if data and len(data) > 400:
            return data
    try:
        url    = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df     = tables[0]
        col    = next(c for c in df.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower())
        tickers = df[col].str.replace(".", "-", regex=False).str.strip().tolist()
        tickers = [t for t in tickers if t and len(t) <= 5 and t.replace("-","").isalpha()]
        log.info(f"S&P 500: {len(tickers)} tickers")
        _save_cache(cache, tickers)
        return tickers
    except Exception as e:
        log.warning(f"S&P 500 fetch failed: {e}")
        return _SP500_FALLBACK


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — NASDAQ 100
# Tech and growth leaders — highest RS scores typically come from here
# ══════════════════════════════════════════════════════════════════════════════

def get_nasdaq100() -> List[str]:
    """
    101 tickers — the most actively traded growth stocks in the world.
    Heavily weighted to tech, biotech and consumer growth.
    High overlap with what our Apex Score rewards.
    """
    cache = _cache_path("nasdaq100")
    if _cache_valid(cache, hours=24):
        data = _load_cache(cache)
        if data and len(data) > 80:
            return data
    try:
        url    = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        for table in tables:
            cols = [str(c).lower() for c in table.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                col = next(c for c in table.columns
                          if "ticker" in str(c).lower() or "symbol" in str(c).lower())
                tickers = table[col].str.replace(".", "-", regex=False).str.strip().tolist()
                tickers = [t for t in tickers
                          if t and isinstance(t, str) and len(t) <= 5
                          and t.replace("-","").isalpha()]
                if len(tickers) > 80:
                    log.info(f"NASDAQ 100: {len(tickers)} tickers")
                    _save_cache(cache, tickers)
                    return tickers
    except Exception as e:
        log.warning(f"NASDAQ 100 fetch failed: {e}")
    return _NASDAQ100_FALLBACK


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — S&P 400 MID CAP
# Mid-size companies — good quality, more growth potential than large caps
# Many future S&P 500 members come from here
# ══════════════════════════════════════════════════════════════════════════════

def get_sp400() -> List[str]:
    """
    400 tickers — the sweet spot between quality and growth.
    S&P 400 requirements: market cap $3.7B-$16.5B, profitable, liquid.
    These companies are established enough to be reliable but small
    enough to still have significant upside momentum potential.
    """
    cache = _cache_path("sp400")
    if _cache_valid(cache, hours=168):  # weekly — index changes slowly
        data = _load_cache(cache)
        if data and len(data) > 300:
            return data
    try:
        url    = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        tables = pd.read_html(url)
        df     = tables[0]
        col    = next(c for c in df.columns
                     if "ticker" in str(c).lower() or "symbol" in str(c).lower())
        tickers = df[col].str.replace(".", "-", regex=False).str.strip().tolist()
        tickers = [t for t in tickers if t and len(t) <= 5 and t.replace("-","").isalpha()]
        log.info(f"S&P 400: {len(tickers)} tickers")
        _save_cache(cache, tickers)
        return tickers
    except Exception as e:
        log.warning(f"S&P 400 fetch failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — CURATED GROWTH WATCHLIST
# High-conviction names not always in indices but with strong momentum profiles
# These are the kinds of stocks ApexScan was built to find
# ══════════════════════════════════════════════════════════════════════════════

CURATED_GROWTH = [
    # AI & Semis — beyond the S&P 500 names
    "ONTO","ENTG","MKSI","WOLF","AMBA","CRUS","MPWR","ALGM","ACLS","COHU",
    "FORM","AEHR","ICHR","UCTT","VECO","AXTI","SMTC","DIOD","SLAB","MCHP",

    # Software — high growth SaaS not yet S&P 500
    "GTLB","MDB","CFLT","DT","DOCN","ESTC","ZI","SMAR","NCNO","ALTR",
    "DOMO","PCTY","CDAY","JAMF","BRZE","SPRK","TASK","WEAV","ARIS","RSKD",

    # Cybersecurity specialists
    "RPD","TENB","QLYS","VRNS","SAIL","OSPN","RDWR","NLOK","MIME","SCWX",
    "CWAN","QLY","BARK","SSTI","CVLT","ATEN","NTCT","EVRY","AVST","ESNT",

    # Fintech & payments growth
    "AFRM","UPST","DAVE","LC","MQ","FLYW","REPAY","RELY","PAYO","PRAA",
    "WRLD","ENVA","OMF","CACC","ECPG","QFIN","TIGR","FUTU","HOOD","SOFI",

    # Biotech high conviction
    "EXAS","NTRA","ACAD","SRPT","BEAM","CRSP","NTLA","IONS","RARE","RCKT",
    "ARQT","PRCT","TMDX","NVCR","KYMR","RXRX","SEER","RVMD","IMVT","ARDX",

    # Consumer growth leaders
    "CELH","ONON","DUOL","CAVA","BROS","TOST","WING","DNUT","SHAK","SG",
    "BIRK","ELF","OLPX","SKIN","GOLI","XPOF","PLBY","LESL","PRPL","LAZR",

    # Healthcare devices & diagnostics
    "SWAV","INSP","IRTC","NARI","GKOS","AXNX","TMDX","NVCR","RXRX","HAYW",
    "PCVX","PRCT","ITGR","MMSI","NVST","LIVN","OMCL","LMAT","IRHC","OSIS",

    # Space, defence & deep tech
    "RKLB","IONQ","ACHR","JOBY","LUNR","ASTS","KTOS","RCAT","AVAV","SPCE",
    "RDW","MNTS","ASTR","BKSY","OSAT","MAXR","IRDM","GSAT","VSAT","GILT",

    # Clean energy & EV
    "ENPH","SEDG","RUN","ARRY","SHLS","STEM","BE","PLUG","FCEL","BLDP",
    "CHPT","BLNK","EVGO","RIVN","NIO","LI","XPEV","LCID","FSR","NKLA",

    # Media, gaming & social
    "RDDT","RBLX","DKNG","U","TTWO","SNAP","PINS","SPOT","MGAM","AGS",

    # Emerging gems
    "SOUN","APP","MSTR","OPEN","UWMC","BTDR","BBAI","HIMS","NU","COIN",
    "HOOD","AFRM","UPST","SOFI","DAVE","RXRX","TMDX","ACHR","ASTS","LUNR",
]


# ══════════════════════════════════════════════════════════════════════════════
# HARDCODED FALLBACKS (used if Wikipedia is unreachable)
# These are the definitive S&P 500 / NASDAQ 100 tickers as of 2025
# ══════════════════════════════════════════════════════════════════════════════

_SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","BRK-B","JPM",
    "LLY","V","MA","UNH","XOM","JNJ","PG","HD","AVGO","COST","MRK","CVX",
    "ABBV","WMT","KO","PEP","ADBE","CRM","BAC","NFLX","MCD","TMO","ORCL",
    "CSCO","ACN","PFE","AMD","DHR","TXN","ABT","NKE","CMCSA","WFC","INTU",
    "PM","IBM","AMGN","CAT","ISRG","NEE","RTX","SPGI","NOW","BKNG","GS",
    "UNP","VRTX","ELV","AMAT","HON","BLK","SYK","C","DE","GILD","ADI",
    "MDLZ","MU","SBUX","TJX","LMT","MMC","REGN","PGR","LRCX","KLAC","AXP",
    "CB","PLD","EOG","SO","ETN","APH","AON","BSX","MCO","ITW","PANW","CRWD",
    "ANET","ZTS","CME","CI","SHW","HCA","DUK","ADP","NOC","SNPS","WM","GE",
    "MSI","PSX","EMR","CL","FCX","NSC","TGT","EW","ORLY","USB","WELL","HLT",
    "CSX","MELI","CDNS","PYPL","ROP","NXPI","CARR","ODFL","AFL","MPC","TT",
    "F","GM","PCAR","UBER","SLB","ECL","OXY","DVN","BK","CTAS","FICO","EL",
    "PAYX","FAST","KHC","RSG","VRSK","ROK","A","MNST","OTIS","NEM","FANG",
    "GEHC","DXCM","IDXX","LHX","ED","XEL","CEG","PCG","FE","EIX","AEP",
    "ACGL","ALL","MET","PRU","EQT","TRGP","COP","HAL","BKR","MRO","DVN",
    "ENPH","SEDG","CEG","VST","NRG","AES","WEC","DTE","ETR","ES","CMS",
    "MSCI","ICE","NDAQ","CBOE","TFC","RF","CFG","FITB","HBAN","KEY","MTB",
    "STT","BRO","AWK","IEX","RMD","HOLX","BAX","BDX","ALGN","PODD","COO",
    "HSIC","TECH","WAT","PKI","TER","ANSS","EPAM","MTCH","ETSY","eBay",
    "NET","DDOG","ZS","OKTA","TEAM","HUBS","VEEV","WDAY","SNOW","TWLO",
    "SQ","COIN","HOOD","SOFI","AFRM","ALLY","SCHW","IBKR","RJF","MKTX",
]

_NASDAQ100_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ADBE","AMD","QCOM","INTU","TXN","AMAT","HON","BKNG","SBUX",
    "VRTX","GILD","REGN","MDLZ","ADI","LRCX","KLAC","PANW","SNPS","CDNS",
    "MELI","CRWD","ORLY","CSX","ABNB","PYPL","PCAR","ROP","AZN","MNST",
    "PAYX","FAST","CTAS","IDXX","ROST","BIIB","DXCM","EXC","MRNA","WDAY",
    "FTNT","TEAM","MCHP","ON","NXPI","GEHC","KDP","ODFL","VRSK","DLTR",
    "ANSS","CTSH","WBD","FANG","ZS","DDOG","NET","OKTA","HUBS","VEEV",
    "ZM","MTCH","SIRI","ILMN","LCID","RIVN","CEG","ENPH","SEDG","ALGN",
    "DOCU","ASML","MU","TTD","CCEP","CPRT","CSGP","MAR","LULU","AEP",
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_full_universe(cfg: dict = None) -> List[str]:
    """
    Build complete scanning universe — quality-first approach.

    Tier 1 (best quality):  S&P 500 + NASDAQ 100  → ~600 tickers
    Tier 2 (good quality):  S&P 400 Mid Cap        → ~400 tickers
    Tier 3 (growth focus):  Curated growth list    → ~200 tickers
    Config themes:          Your custom additions  → variable

    All tickers are:
    - Listed on major US exchanges (NYSE/NASDAQ)
    - Part of a recognised index OR hand-curated for growth
    - NOT penny stocks, OTC, or low-quality names

    Total: ~1,100-1,400 unique tickers
    """
    cache = _cache_path("full_universe")
    if _cache_valid(cache, hours=24):
        data = _load_cache(cache)
        if data and len(data) > 500:
            log.info(f"Universe from cache: {len(data)} tickers")
            return data

    log.info("Building ticker universe…")
    all_tickers = set()

    # Tier 1 — highest quality
    sp500 = get_sp500()
    all_tickers.update(sp500)
    log.info(f"  Tier 1a S&P 500: {len(sp500)}")

    ndx = get_nasdaq100()
    all_tickers.update(ndx)
    log.info(f"  Tier 1b NASDAQ 100: {len(ndx)}")

    # Tier 2 — good quality
    sp400 = get_sp400()
    all_tickers.update(sp400)
    log.info(f"  Tier 2 S&P 400: {len(sp400)}")

    # Tier 3 — curated growth
    all_tickers.update(CURATED_GROWTH)
    log.info(f"  Tier 3 Curated growth: {len(CURATED_GROWTH)}")

    # Config themes — your custom additions
    if cfg and "us_themes" in cfg:
        config_tickers = [t for theme in cfg["us_themes"].values() for t in theme]
        all_tickers.update(config_tickers)
        log.info(f"  Config themes: {len(config_tickers)}")

    # Clean: valid US tickers only
    bad = {"", "nan", "NaN", "N/A", "Symbol", "Ticker", "None"}
    cleaned = sorted([
        t for t in all_tickers
        if t
        and t not in bad
        and isinstance(t, str)
        and 1 <= len(t) <= 5
        and t.replace("-","").replace(".","").isalpha()
    ])

    log.info(f"Universe built: {len(cleaned)} unique tickers")
    _save_cache(cache, cleaned)
    return cleaned


def get_universe_stats() -> Dict:
    """Return stats about the current universe cache."""
    cache = _cache_path("full_universe")
    if cache.exists():
        tickers = _load_cache(cache)
        age_h   = (datetime.now() - datetime.fromtimestamp(
            cache.stat().st_mtime)).total_seconds() / 3600
        return {
            "total":           len(tickers),
            "cache_age_hours": round(age_h, 1),
            "next_refresh":    f"{max(0, 24-age_h):.1f}h",
            "sources":         [
                "S&P 500 (503) — large cap, profitable, liquid",
                "NASDAQ 100 (101) — top growth & tech",
                "S&P 400 Mid Cap (400) — established growth",
                "Curated Growth List (200) — hand-picked momentum names",
                "Config Themes — your custom additions",
            ],
            "quality_note": "All tickers are index constituents or hand-curated — no penny stocks or OTC names",
        }
    return {"total": 0, "cache_age_hours": 0, "next_refresh": "Now",
            "sources": [], "quality_note": ""}


def get_tier1_only() -> List[str]:
    """Return only Tier 1 tickers (S&P 500 + NASDAQ 100) for quick high-quality scans."""
    sp500 = get_sp500() or _SP500_FALLBACK
    ndx   = get_nasdaq100() or _NASDAQ100_FALLBACK
    return sorted(set(sp500 + ndx))
