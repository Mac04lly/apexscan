"""
dashboard.py — ApexScan Streamlit Dashboard v13
"""

import sys
import os

# Ensure project root is on path (required for Streamlit Cloud)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import json

from scanner import load_config, run_scan, save_report

# Import each module independently so one failure doesn't break everything
def _safe_import(module_path, names):
    """Import names from module, returning None for each if import fails."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return {n: getattr(mod, n, None) for n in names}
    except Exception as _e:
        return {n: None for n in names}

_of   = _safe_import("modules.options_flow",      ["scan_options_flow", "scan_multiple"])
_it   = _safe_import("modules.insider_tracker",   ["fetch_insider_trades", "get_insider_summary"])
_bt   = _safe_import("modules.backtester",        ["backtest_ticker", "backtest_portfolio"])
_rc   = _safe_import("modules.risk_calc",         ["TradeSetup", "calculate_position", "pyramiding_plan"])
_ab   = _safe_import("modules.ai_briefing",       ["generate_briefing", "load_latest_briefing"])
_wm   = _safe_import("modules.watchlist_manager", ["load_watchlists", "save_watchlists", "add_ticker",
                                                    "remove_ticker", "create_list", "delete_list",
                                                    "scan_watchlist", "import_tickers", "export_watchlist"])
_al   = _safe_import("modules.alerts",            ["load_alert_settings", "save_alert_settings",
                                                    "test_telegram", "send_email", "dispatch_alert",
                                                    "check_and_fire_alerts", "build_daily_briefing_alert"])
_av   = _safe_import("modules.alpha_vantage",     ["analyse_eps", "get_upcoming_earnings_for_watchlist",
                                                    "_cache_path", "_cache_valid"])

# Expose as module-level names (None if module missing)
scan_options_flow      = _of["scan_options_flow"]
scan_multiple          = _of["scan_multiple"]
fetch_insider_trades   = _it["fetch_insider_trades"]
get_insider_summary    = _it["get_insider_summary"]
backtest_ticker        = _bt["backtest_ticker"]
backtest_portfolio     = _bt["backtest_portfolio"]
TradeSetup             = _rc["TradeSetup"]
calculate_position     = _rc["calculate_position"]
pyramiding_plan        = _rc["pyramiding_plan"]
generate_briefing      = _ab["generate_briefing"]
load_latest_briefing   = _ab["load_latest_briefing"]
load_watchlists        = _wm["load_watchlists"]
save_watchlists        = _wm["save_watchlists"]
add_ticker             = _wm["add_ticker"]
remove_ticker          = _wm["remove_ticker"]
create_list            = _wm["create_list"]
delete_list            = _wm["delete_list"]
scan_watchlist         = _wm["scan_watchlist"]
import_tickers         = _wm["import_tickers"]
export_watchlist       = _wm["export_watchlist"]
load_alert_settings    = _al["load_alert_settings"]
save_alert_settings    = _al["save_alert_settings"]
test_telegram          = _al["test_telegram"]
send_email             = _al["send_email"]
dispatch_alert         = _al["dispatch_alert"]
check_and_fire_alerts  = _al["check_and_fire_alerts"]
build_daily_briefing_alert = _al["build_daily_briefing_alert"]
get_upcoming_earnings_for_watchlist = _av["get_upcoming_earnings_for_watchlist"]

# Fallback: if alert module missing, define minimal stubs so app doesn't crash
if load_alert_settings is None:
    def load_alert_settings(): return {"alerts_enabled": False, "telegram_token": "", "telegram_chat_id": "", "email_from": "", "email_password": "", "email_to": "", "alert_breakouts": True, "alert_stop_breach": True, "alert_earnings": True, "alert_sfp_setup": True, "alert_persistent_flow": True, "alert_vwap_imbalance": True, "min_score_alert": 60}
if save_alert_settings is None:
    def save_alert_settings(s): pass
if load_watchlists is None:
    def load_watchlists(): return {"High Conviction": [], "Monitoring": [], "Earnings Soon": [], "Swing Trades": []}
if save_watchlists is None:
    def save_watchlists(w): pass

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ApexScan — US Stock Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0d1117; color:#e6edf3; }
[data-testid="stSidebar"] { background:#161b22; border-right:1px solid #30363d; }
.metric-card { background:#161b22; border:1px solid #30363d; border-radius:8px;
               padding:16px 20px; margin-bottom:8px; }
.metric-card h3 { margin:0; font-size:0.75rem; color:#8b949e;
                  text-transform:uppercase; letter-spacing:.08em; }
.metric-card .value { font-size:1.7rem; font-weight:700; font-family:'SF Mono',monospace; }
.green{color:#3fb950;} .red{color:#f85149;} .amber{color:#d29922;}
.blue{color:#388bfd;}  .white{color:#e6edf3;}
.alert-box { background:#1a2a1a; border:1px solid #3fb950; border-radius:8px;
             padding:12px 16px; margin:6px 0; }
.warn-box  { background:#2a2200; border:1px solid #d29922; border-radius:8px;
             padding:12px 16px; margin:6px 0; }
.danger-box{ background:#2a1010; border:1px solid #f85149; border-radius:8px;
             padding:12px 16px; margin:6px 0; }
div.stButton > button { background:#21262d; color:#e6edf3;
                        border:1px solid #30363d; border-radius:6px; }
div.stButton > button:hover { border-color:#388bfd; color:#79c0ff; }
</style>
""", unsafe_allow_html=True)

PORTFOLIO_FILE = "data/portfolio.json"
Path("data").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def pct_fmt(v):
    try:
        sign = "+" if float(v) > 0 else ""
        return f"{sign}{float(v):.1f}%"
    except:
        return str(v)

def color_score(v):
    try:
        v = float(v)
        if v >= 70: return "color:#3fb950;font-weight:700"
        if v >= 40: return "color:#d29922;font-weight:600"
        return "color:#8b949e"
    except: return ""

def color_perf(v):
    try:
        v = float(v)
        return "color:#3fb950" if v > 0 else ("color:#f85149" if v < 0 else "")
    except: return ""

def color_rs(v):
    try:
        v = float(v)
        if v >= 100: return "color:#3fb950;font-weight:700"
        if v >= 70:  return "color:#d29922"
        if v > 0:    return "color:#8b949e"
        return "color:#f85149"
    except: return ""

def format_mcap(v):
    try:
        v = float(v)
        if v >= 1e9: return f"${v/1e9:.1f}B"
        if v >= 1e6: return f"${v/1e6:.0f}M"
        return f"${v:,.0f}"
    except: return "–"

def color_mcap_cat(v):
    v = str(v)
    if "Micro" in v: return "color:#f85149;font-size:0.8rem"
    if "Small" in v: return "color:#d29922;font-size:0.8rem"
    if "Mid"   in v: return "color:#388bfd;font-size:0.8rem"
    return "color:#8b949e;font-size:0.8rem"

GEM_DISCLAIMER = """<div style="background:#1a1500;border:1px solid #d29922;border-radius:8px;
padding:10px 16px;margin:6px 0;font-size:0.83rem;">
⚠️ <b style="color:#d29922;">Emerging Gems — Higher Risk / Higher Volatility</b><br>
<span style="color:#c9d1d9;">Small/micro-cap stocks can drop 30–50% on a single bad quarter.
Use <b>0.5–1% max risk per trade</b> (half your normal size).
These are long-term conviction plays — only hold what you can stomach losing short-term.</span>
</div>"""

def load_latest_report() -> pd.DataFrame:
    reports = sorted(Path("reports").glob("scan_*.csv"), reverse=True)
    if reports:
        return pd.read_csv(reports[0], index_col="rank")
    return pd.DataFrame()


def load_previous_report() -> pd.DataFrame:
    """Load the second-most-recent scan report for delta comparison."""
    reports = sorted(Path("reports").glob("scan_*.csv"), reverse=True)
    if len(reports) >= 2:
        return pd.read_csv(reports[1], index_col="rank")
    return pd.DataFrame()


def compute_deltas(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """
    Compare current scan to previous scan and produce a changes summary.
    Returns current df with added delta columns:
      - delta_score:    Apex Score change
      - delta_rs:       RS 3m change
      - delta_3m:       3m performance change
      - delta_vwap:     VWAP position change (text)
      - delta_of:       Order Flow bias change (text)
      - delta_stage:    Stage change (text)
      - delta_pattern:  Pattern change (text)
      - is_new:         True if ticker wasn't in last scan
      - is_gone:        detected via absence (handled separately)
      - change_summary: human-readable summary of what changed
    """
    if previous.empty or current.empty:
        current["delta_score"]   = None
        current["delta_rs"]      = None
        current["delta_3m"]      = None
        current["changes"]       = "No prior scan"
        current["is_new"]        = False
        return current

    prev_idx = previous.set_index("ticker") if "ticker" in previous.columns else previous
    curr     = current.copy()

    delta_scores   = []
    delta_rs_vals  = []
    delta_3m_vals  = []
    change_summaries = []
    is_new_flags   = []

    for _, row in curr.iterrows():
        tk = row.get("ticker","")

        if tk not in prev_idx.index:
            delta_scores.append(None)
            delta_rs_vals.append(None)
            delta_3m_vals.append(None)
            change_summaries.append("🆕 New entry")
            is_new_flags.append(True)
            continue

        prev = prev_idx.loc[tk]
        changes = []
        is_new_flags.append(False)

        # Score delta
        try:
            ds = round(float(row.get("apex_score",0)) - float(prev.get("apex_score",0)), 1)
            delta_scores.append(ds)
            if ds >= 5:   changes.append(f"Score ▲{ds:+.0f}")
            elif ds <= -5: changes.append(f"Score ▼{ds:+.0f}")
        except:
            delta_scores.append(None)

        # RS delta
        try:
            dr = round(float(row.get("rs_3m",0)) - float(prev.get("rs_3m",0)), 0)
            delta_rs_vals.append(dr)
            if dr >= 20:   changes.append(f"RS ▲{dr:+.0f}")
            elif dr <= -20: changes.append(f"RS ▼{dr:+.0f}")
        except:
            delta_rs_vals.append(None)

        # 3m perf delta
        try:
            dp = round(float(row.get("perf_3m_%",0)) - float(prev.get("perf_3m_%",0)), 1)
            delta_3m_vals.append(dp)
            if abs(dp) >= 2: changes.append(f"3m {dp:+.1f}%")
        except:
            delta_3m_vals.append(None)

        # Stage change
        curr_stage = str(row.get("stage",""))
        prev_stage = str(prev.get("stage",""))
        if curr_stage != prev_stage:
            if "2 ✅" in curr_stage:
                changes.append("⬆️ Entered Stage 2")
            elif "4 🔴" in curr_stage:
                changes.append("⬇️ Dropped to Stage 4")
            elif "1 ⏳" in curr_stage and "2 ✅" in prev_stage:
                changes.append("⚠️ Lost Stage 2")
            else:
                changes.append(f"Stage: {prev_stage[:6]}→{curr_stage[:6]}")

        # Order Flow change
        curr_of = str(row.get("of_bias",""))
        prev_of = str(prev.get("of_bias",""))
        if curr_of != prev_of:
            if "Strong Bullish" in curr_of:
                changes.append("📈 Flow→Strong Bull")
            elif "Bullish" in curr_of and "Bearish" in prev_of:
                changes.append("📈 Flow flipped Bull")
            elif "Bearish" in curr_of and "Bullish" in prev_of:
                changes.append("📉 Flow flipped Bear")

        # VWAP position change
        curr_vwap = str(row.get("vwap_position",""))
        prev_vwap = str(prev.get("vwap_position",""))
        if curr_vwap != prev_vwap:
            if "Above" in curr_vwap and "Below" in prev_vwap:
                changes.append("💧 Reclaimed VWAP")
            elif "Below" in curr_vwap and "Above" in prev_vwap:
                changes.append("💧 Lost VWAP")
            elif "Extended Above" in curr_vwap and "Extended Above" not in prev_vwap:
                changes.append("⚡ Extended above VWAP")

        # Pattern change
        curr_pat = str(row.get("pattern",""))
        prev_pat = str(prev.get("pattern",""))
        if curr_pat != prev_pat:
            if "Breakout" in curr_pat:
                changes.append("🚀 Breakout!")
            elif "Handle" in curr_pat and "Handle" not in prev_pat:
                changes.append("📐 Handle forming")
            elif "Tight" in curr_pat and "Tight" not in prev_pat:
                changes.append("🔄 Tightening")

        # Breakout newly fired
        curr_brk = bool(row.get("breaking_out", False))
        prev_brk = bool(prev.get("breaking_out", False))
        if curr_brk and not prev_brk:
            changes.append("🚀 NEW Breakout!")

        # SFP appeared
        curr_sfp = str(row.get("pa_sfp",""))
        prev_sfp = str(prev.get("pa_sfp",""))
        if curr_sfp and curr_sfp != prev_sfp and curr_sfp not in ["","nan","None"]:
            changes.append(f"🎯 {curr_sfp}")

        summary = " | ".join(changes) if changes else "↔ No change"
        change_summaries.append(summary)

    curr["delta_score"]  = delta_scores
    curr["delta_rs"]     = delta_rs_vals
    curr["delta_3m"]     = delta_3m_vals
    curr["changes"]      = change_summaries
    curr["is_new"]       = is_new_flags

    # Tickers that disappeared since last scan
    if "ticker" in previous.columns:
        prev_tickers = set(previous["ticker"].tolist())
        curr_tickers = set(curr["ticker"].tolist())
        gone = prev_tickers - curr_tickers
    else:
        gone = set()

    return curr, gone

@st.cache_data(ttl=300)
def fetch_hist(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        h = yf.Ticker(ticker).history(period=period)
        h["MA50"]  = h["Close"].rolling(50).mean()
        h["MA200"] = h["Close"].rolling(200).mean()
        return h
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_price(ticker: str) -> dict:
    """Return current price, 52w high, MA50, MA200 for a ticker."""
    try:
        h = yf.Ticker(ticker).history(period="1y")
        if h.empty:
            return {}
        close   = h["Close"]
        price   = close.iloc[-1]
        ma50    = close.rolling(50).mean().iloc[-1]
        ma200   = close.rolling(200).mean().iloc[-1]
        high52  = close.rolling(252).max().iloc[-1]
        prev    = close.iloc[-2] if len(close) > 1 else price
        return {
            "price":  round(price, 2),
            "prev":   round(prev, 2),
            "ma50":   round(ma50, 2),
            "ma200":  round(ma200, 2),
            "high52": round(high52, 2),
            "chg_pct": round((price / prev - 1) * 100, 2),
        }
    except:
        return {}

@st.cache_data(ttl=3600)
def fetch_earnings(ticker: str) -> dict:
    """Get next earnings date from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        cal  = yf.Ticker(ticker).calendar
        date = None
        if cal is not None and not cal.empty:
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
                date = pd.to_datetime(val).strftime("%Y-%m-%d") if pd.notna(val) else None
        return {
            "next_earnings": date,
            "eps_est":  info.get("forwardEps"),
            "rev_est":  info.get("revenueEstimate"),
        }
    except:
        return {"next_earnings": None}

@st.cache_data(ttl=600)
@st.cache_data(ttl=600)
def sector_performance() -> pd.DataFrame:
    """Fetch weekly/monthly performance for major US sector ETFs."""
    etfs = {
        "Technology":    "XLK",
        "Financials":    "XLF",
        "Healthcare":    "XLV",
        "Energy":        "XLE",
        "Consumer Disc": "XLY",
        "Industrials":   "XLI",
        "Materials":     "XLB",
        "Real Estate":   "XLRE",
        "Utilities":     "XLU",
        "Comm Services": "XLC",
        "Consumer Stap": "XLP",
    }
    rows = []
    for sector, sym in etfs.items():
        try:
            h = yf.Ticker(sym).history(period="3mo")["Close"].dropna()
            if len(h) < 5:
                continue
            w1 = round((h.iloc[-1] / h.iloc[-5]  - 1) * 100, 2) if len(h) >= 5  else None
            m1 = round((h.iloc[-1] / h.iloc[-21] - 1) * 100, 2) if len(h) >= 21 else None
            m3 = round((h.iloc[-1] / h.iloc[0]   - 1) * 100, 2)
            rows.append({
                "Sector": sector, "ETF": sym,
                "1W %":  w1, "1M %": m1, "3M %": m3,
                "Price": round(h.iloc[-1], 2)
            })
        except Exception as _e:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Ensure numeric columns are float, not object
    for col in ["1W %", "1M %", "3M %"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_portfolio() -> list:
    if Path(PORTFOLIO_FILE).exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return []

def save_portfolio(holdings: list):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(holdings, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📡 ApexScan")
    st.markdown("*momentum · Stage · Theme Rotation*")
    st.divider()
    min_score = st.slider("Min Apex Score", 0, 100, 30, 5)
    min_3m    = st.slider("Min 3m Return %", -50, 100, 0, 5)
    st.divider()

    # ── Scan Mode ─────────────────────────────────────────────────────────
    st.markdown("**Scan Universe**")
    scan_mode = st.radio(
        "Select universe",
        [
            "⚡ Tier 1 — Best Quality (~600)",
            "🔍 Tier 1+2 — Broad (~1,000)",
            "🌐 Full Universe (~1,400)",
        ],
        index=1,
        help=(
            "Tier 1: S&P 500 + NASDAQ 100 — most reliable signals, ~20 min\n"
            "Tier 1+2: + S&P 400 Mid Cap — more growth names, ~35 min\n"
            "Full: + Curated growth list — maximum coverage, ~50 min"
        )
    )

    run_btn  = st.button("🚀 Run Live Scan", use_container_width=True, type="primary")
    load_btn = st.button("📂 Load Last Report", use_container_width=True)
    st.divider()

    # ── Universe stats ─────────────────────────────────────────────────────
    try:
        from modules.ticker_universe import get_universe_stats
        _uni = get_universe_stats()
        if _uni.get("total", 0) > 0:
            st.markdown("**Universe Cache**")
            st.caption(f"📊 {_uni['total']:,} tickers loaded")
            st.caption(f"🔄 Next refresh in {_uni['next_refresh']}")
            st.caption(_uni.get("quality_note",""))
    except Exception:
        st.caption("📊 Universe: S&P 500 + NASDAQ 100 + S&P 400 + Growth")

    st.divider()

    # ── API Key Status ────────────────────────────────────────────────────
    _cfg_check = load_config("config.yaml")
    _av_key    = _cfg_check.get("alpha_vantage_key","")
    _fh_key    = _cfg_check.get("finnhub_key","")
    _td_key    = _cfg_check.get("twelve_data_key","")
    _ms_key    = _cfg_check.get("marketstack_key","")
    _av_ok     = bool(_av_key and not _av_key.startswith("YOUR_"))
    _fh_ok     = bool(_fh_key and not _fh_key.startswith("YOUR_"))
    _td_ok     = bool(_td_key and not _td_key.startswith("YOUR_"))
    _ms_ok     = bool(_ms_key and not _ms_key.startswith("YOUR_"))

    st.markdown("**API Status**")
    st.markdown(f"{'🟢' if _av_ok else '🔴'} Alpha Vantage {'✓ Active' if _av_ok else '✗ Not set'}")
    st.markdown(f"{'🟢' if _fh_ok else '🟡'} Finnhub {'✓ Active' if _fh_ok else 'Optional'}")
    st.markdown(f"{'🟢' if _td_ok else '🟡'} Twelve Data {'✓ Active' if _td_ok else '✗ Not set'}")
    st.markdown(f"{'🟢' if _ms_ok else '🟡'} Marketstack {'✓ Active' if _ms_ok else '✗ Not set'}")

    st.divider()
    st.caption("Data: yfinance · Finnhub · AV · Twelve Data · Marketstack")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

# ── Map scan mode to parameters ────────────────────────────────────────────
_scan_mode_map = {
    "⚡ Tier 1 — Best Quality (~600)":  ("tier1",  600,  "~20 min"),
    "🔍 Tier 1+2 — Broad (~1,000)":    ("tier12", 1000, "~35 min"),
    "🌐 Full Universe (~1,400)":        ("full",   None, "~50 min"),
}
_scan_tier, _max_tickers, _scan_eta = _scan_mode_map[scan_mode]

st.markdown("""
<h1 style="margin:0 0 16px 0;font-size:1.5rem;">
  📡 ApexScan
  <span style="font-size:1rem;color:#8b949e;font-weight:400">
    — US Market Intelligence
  </span>
</h1>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🏆 Leaderboard",
    "📈 Chart Viewer",
    "🌍 Theme Heatmap",
    "💼 Portfolio Tracker",
    "📅 Earnings Calendar",
    "🔄 Sector Rotation",
    "🔍 Stock Deep Dive",
    "🎯 Options Flow",
    "🕵️ Insider Tracker",
    "📊 Dividend Calculator",
    "⏱ Backtester",
    "⚖️ Risk Calculator",
    "🤖 AI Briefing",
    "📋 Watchlists",
    "🔔 Alert Settings",
    "🧠 Interpretation",
    "📖 Guide",
])


# ══════════════════════════════════════════════════════════════════════════════
# LOAD / SCAN DATA
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# LOAD / SCAN DATA
# ══════════════════════════════════════════════════════════════════════════════

df_raw       = pd.DataFrame()
prev_df      = pd.DataFrame()
gone_tickers = set()

if run_btn:
    cfg     = load_config("config.yaml")
    prev_df = load_latest_report()

    # ── Build universe based on selected tier ─────────────────────────────
    try:
        from modules.ticker_universe import (
            build_full_universe, get_tier1_only, get_sp500,
            get_nasdaq100, get_sp400, CURATED_GROWTH
        )
        with st.spinner("Loading ticker universe…"):
            if _scan_tier == "tier1":
                universe = get_tier1_only()
                tier_label = "Tier 1 (S&P 500 + NASDAQ 100)"
            elif _scan_tier == "tier12":
                sp500  = get_sp500()
                ndx    = get_nasdaq100()
                sp400  = get_sp400()
                universe = sorted(set(sp500 + ndx + sp400))
                tier_label = "Tier 1+2 (S&P 500 + NASDAQ 100 + S&P 400)"
            else:
                universe = build_full_universe(cfg)
                tier_label = "Full Universe"
    except Exception as _ue:
        # Fallback to config themes
        universe = list(set(
            t for theme in cfg["us_themes"].values() for t in theme
        ))
        tier_label = "Config Themes (fallback)"

    total_universe = len(universe)
    cap = _max_tickers
    if cap and len(universe) > cap:
        universe = universe[:cap]

    st.info(
        f"**{tier_label}** — scanning **{len(universe):,}** tickers "
        f"(from {total_universe:,} in universe) · Est. time: {_scan_eta}"
    )

    # ── Progress bar ──────────────────────────────────────────────────────
    prog_bar   = st.progress(0)
    prog_text  = st.empty()
    res_badge  = st.empty()

    def _progress_cb(current, total, passing):
        pct = min(int(current / max(total, 1) * 100), 99)
        prog_bar.progress(pct)
        prog_text.caption(f"Scanning {current:,} / {total:,} tickers ({pct}%)")
        res_badge.markdown(f"✅ **{passing}** setups passing filters so far…")

    # ── Run scan ──────────────────────────────────────────────────────────
    with st.spinner(f"Scanning {len(universe):,} tickers…"):
        df_raw = run_scan(
            cfg,
            ticker_list=universe,
            progress_callback=_progress_cb,
        )

    prog_bar.progress(100)
    prog_text.empty()
    res_badge.empty()

    if not df_raw.empty:
        save_report(df_raw)
        st.success(
            f"✅ Scan complete — **{len(df_raw)} setups** found "
            f"from {len(universe):,} tickers scanned!"
        )
        try:
            alert_settings = load_alert_settings()
            if alert_settings.get("alerts_enabled"):
                portfolio_data = load_portfolio()
                fired = check_and_fire_alerts(
                    df_raw, portfolio_data, alert_settings, fetch_price)
                if fired:
                    st.info(f"🔔 {len(fired)} alert(s) sent.")
        except Exception:
            pass
    else:
        st.warning("No setups found. Try lowering the Score Threshold or switching to Full Universe.")
else:
    prev_df = load_previous_report()
    df_raw  = load_latest_report()

# Compute deltas (changes vs previous scan)
if not df_raw.empty and not prev_df.empty:
    try:
        df_raw, gone_tickers = compute_deltas(df_raw, prev_df)
    except Exception as _de:
        df_raw["changes"]     = "–"
        df_raw["is_new"]      = False
        df_raw["delta_score"] = None
        gone_tickers          = set()
elif not df_raw.empty:
    df_raw["changes"]     = "First scan"
    df_raw["is_new"]      = False
    df_raw["delta_score"] = None

df = df_raw.copy()
if not df.empty:
    if "apex_score" in df.columns:
        df = df[pd.to_numeric(df["apex_score"], errors="coerce") >= min_score]
    if "perf_3m_%" in df.columns:
        df = df[pd.to_numeric(df["perf_3m_%"], errors="coerce") >= min_3m]



# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    if df.empty:
        st.info("Click **🚀 Run Live Scan** or **📂 Load Last Report** in the sidebar.")
    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="metric-card"><h3>Tickers Passing</h3><div class="value white">{len(df)}</div></div>', unsafe_allow_html=True)
        with c2:
            top = pd.to_numeric(df["apex_score"], errors="coerce").max()
            st.markdown(f'<div class="metric-card"><h3>Top Score</h3><div class="value green">{top:.0f}</div></div>', unsafe_allow_html=True)
        with c3:
            bo = int(df.get("breaking_out", pd.Series([False]*len(df))).sum()) if "breaking_out" in df.columns else 0
            st.markdown(f'<div class="metric-card"><h3>Breakouts</h3><div class="value amber">{bo}</div></div>', unsafe_allow_html=True)
        with c4:
            stage2 = int((df.get("stage", pd.Series([""] * len(df))).str.contains("2 ✅", na=False)).sum()) if "stage" in df.columns else 0
            st.markdown(f'<div class="metric-card"><h3>Stage 2 Stocks</h3><div class="value blue">{stage2}</div></div>', unsafe_allow_html=True)
        with c5:
            themes_n = df["theme"].nunique() if "theme" in df.columns else 0
            st.markdown(f'<div class="metric-card"><h3>Themes Active</h3><div class="value green">{themes_n}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── New signal filters ─────────────────────────────────────────────
        with st.expander("🔬 Advanced Signal Filters", expanded=False):
            af1, af2, af3, af4 = st.columns(4)
            with af1:
                filter_of = st.selectbox("Order Flow Bias",
                    ["All", "Strong Bullish", "Bullish", "Neutral", "Bearish"], index=0)
            with af2:
                filter_vwap = st.selectbox("VWAP Position",
                    ["All", "Above VWAP", "Extended Above VWAP",
                     "Below VWAP", "Extended Below VWAP"], index=0)
            with af3:
                filter_ms = st.selectbox("Market Structure",
                    ["All", "Bullish (HH/HL)", "Bearish (LH/LL)", "Transitioning"], index=0)
            with af4:
                filter_pa = st.selectbox("PA Pattern",
                    ["All", "Bullish SFP", "Bullish Engulfing",
                     "Inside Day", "Bullish Context Candle", "PA Confluence"], index=0)

        # Apply advanced filters
        df_filtered = df.copy()
        if filter_of != "All" and "of_bias" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["of_bias"] == filter_of]
        if filter_vwap != "All" and "vwap_position" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["vwap_position"] == filter_vwap]
        if filter_ms != "All" and "ms_structure" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["ms_structure"] == filter_ms]
        if filter_pa != "All" and "pa_patterns" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["pa_patterns"].str.contains(filter_pa, na=False)]

        # ── Theme Category Filter ─────────────────────────────────────────
        theme_filter = st.radio(
            "📊 Theme Filter",
            ["🌐 All Themes", "🚀 Growth Leaders", "💎 Emerging Gems"],
            horizontal=True, key="lb_theme_filter"
        )
        if theme_filter == "💎 Emerging Gems":
            st.markdown(GEM_DISCLAIMER, unsafe_allow_html=True)
            if "is_gem" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["is_gem"] == True]
            elif "theme" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["theme"] == "emerging_gems"]
        elif theme_filter == "🚀 Growth Leaders":
            if "market_cap" in df_filtered.columns:
                mcap_num = pd.to_numeric(df_filtered["market_cap"], errors="coerce")
                df_filtered = df_filtered[mcap_num >= 10_000_000_000]

        # Column view toggle
        col_view = st.radio("Column View", ["Standard", "Order Flow", "VWAP & Structure", "Price Action", "Fundamentals"], horizontal=True)

        if col_view == "Standard":
            want = ["ticker","theme","price","mcap_category","stage",
                    "perf_1m_%","perf_3m_%","perf_6m_%",
                    "rs_3m","vol_surge_x","near_52wh","pattern",
                    "earn_momentum","eps_growth_%","eps_surprise_%","consec_beats","apex_score"]
        elif col_view == "Order Flow":
            want = ["ticker","price","of_bias","of_up_vol_ratio",
                    "of_bullish_days","of_consec_up","of_score",
                    "vol_surge_x","ms_structure","apex_score"]
        elif col_view == "VWAP & Structure":
            want = ["ticker","price","vwap","vs_vwap_%","vwap_position",
                    "vwap_slope","vwap_score","ms_structure","ms_hh_hl","ms_bos",
                    "ms_swing_high","ms_swing_low","apex_score"]
        elif col_view == "Price Action":
            want = ["ticker","price","pa_patterns","pa_engulfing","pa_sfp",
                    "pa_inside_day","pa_context","pa_score",
                    "of_bias","vwap_position","apex_score"]
        else:  # Fundamentals
            want = ["ticker","price","earn_momentum",
                    "eps_growth_%","eps_surprise_%","eps_accel",
                    "consec_beats","rev_growth_%","eps_score",
                    "analyst_target","pe_ratio","peg_ratio","apex_score"]

        show_cols = [c for c in want if c in df_filtered.columns]
        disp = df_filtered[show_cols].head(30).copy()
        for col in ["apex_score","perf_1m_%","perf_3m_%","perf_6m_%","rs_3m","rs_6m"]:
            if col in disp.columns:
                disp[col] = pd.to_numeric(disp[col], errors="coerce")

        # Color helpers for new columns
        def color_of_bias(v):
            if "Strong Bullish" in str(v): return "color:#3fb950;font-weight:700"
            if "Bullish" in str(v):        return "color:#3fb950"
            if "Strong Bearish" in str(v): return "color:#f85149;font-weight:700"
            if "Bearish" in str(v):        return "color:#f85149"
            return "color:#8b949e"

        def color_vwap_pos(v):
            if "Extended Above" in str(v): return "color:#d29922"
            if "Above" in str(v):          return "color:#3fb950"
            if "Extended Below" in str(v): return "color:#f85149;font-weight:700"
            if "Below" in str(v):          return "color:#f85149"
            return ""

        def color_ms(v):
            if "Bullish" in str(v): return "color:#3fb950"
            if "Bearish" in str(v): return "color:#f85149"
            return "color:#d29922"

        def color_pa(v):
            if "SFP" in str(v) or "Confluence" in str(v): return "color:#d29922;font-weight:700"
            if "Bullish" in str(v): return "color:#3fb950"
            if "Bearish" in str(v): return "color:#f85149"
            return ""

        fmt_dict = {
            "price": "{:.2f}", "apex_score": "{:.0f}",
            "perf_1m_%": pct_fmt, "perf_3m_%": pct_fmt, "perf_6m_%": pct_fmt,
            "rs_3m": lambda v: f"{v:.0f}" if pd.notna(v) and v != 0 else "–",
            "adr_%": lambda v: f"{v:.1f}%" if pd.notna(v) else "–",
            "vs_50ma_%": pct_fmt, "vs_200ma_%": pct_fmt,
            "vol_surge_x": "{:.1f}x",
            "of_up_vol_ratio": "{:.2f}x",
            "of_bullish_days": "{:.0f}%",
            "vs_vwap_%": pct_fmt,
            "vwap": "${:.2f}",
            "ms_swing_high": lambda v: f"${v:.2f}" if pd.notna(v) else "–",
            "ms_swing_low":  lambda v: f"${v:.2f}" if pd.notna(v) else "–",
            # AV fundamentals
            "eps_growth_%":   lambda v: f"{v:+.1f}%" if pd.notna(v) else "–",
            "eps_surprise_%": lambda v: f"{v:+.1f}%" if pd.notna(v) else "–",
            "rev_growth_%":   lambda v: f"{v:+.1f}%" if pd.notna(v) else "–",
            "eps_score":      lambda v: f"{v}/15"    if pd.notna(v) else "–",
            "analyst_target": lambda v: f"${v:.2f}"  if pd.notna(v) and v else "–",
            "pe_ratio":       lambda v: f"{v:.1f}x"  if pd.notna(v) and v else "–",
            "peg_ratio":      lambda v: f"{v:.2f}"   if pd.notna(v) and v else "–",
            "consec_beats":   lambda v: f"{int(v)}Q" if pd.notna(v) else "–",
        }
        active_fmt = {k: v for k, v in fmt_dict.items() if k in disp.columns}

        map_cols_of   = [c for c in ["of_bias"] if c in disp.columns]
        map_cols_vwap = [c for c in ["vwap_position"] if c in disp.columns]
        map_cols_ms   = [c for c in ["ms_structure"] if c in disp.columns]
        map_cols_pa   = [c for c in ["pa_patterns","pa_sfp","pa_engulfing","pa_context"] if c in disp.columns]

        styled = disp.style.map(color_score, subset=["apex_score"])
        if [c for c in ["perf_1m_%","perf_3m_%","perf_6m_%","vs_50ma_%","vs_200ma_%","vs_vwap_%"] if c in disp.columns]:
            styled = styled.map(color_perf, subset=[c for c in ["perf_1m_%","perf_3m_%","perf_6m_%","vs_50ma_%","vs_200ma_%","vs_vwap_%"] if c in disp.columns])
        if "rs_3m" in disp.columns:
            styled = styled.map(color_rs, subset=["rs_3m"])
        if map_cols_of:   styled = styled.map(color_of_bias,  subset=map_cols_of)
        if map_cols_vwap: styled = styled.map(color_vwap_pos, subset=map_cols_vwap)
        if map_cols_ms:   styled = styled.map(color_ms,       subset=map_cols_ms)
        if map_cols_pa:   styled = styled.map(color_pa,       subset=map_cols_pa)
        styled = styled.format(active_fmt, na_rep="–")

        st.dataframe(styled, use_container_width=True, height=520)

        # New signal summary badges
        if not df_filtered.empty:
            sfp_count  = df_filtered["pa_sfp"].notna().sum() if "pa_sfp" in df_filtered.columns else 0
            of_bull    = (df_filtered.get("of_bias","").str.contains("Bullish", na=False)).sum() if "of_bias" in df_filtered.columns else 0
            vwap_above = (df_filtered.get("vwap_position","").str.contains("Above", na=False)).sum() if "vwap_position" in df_filtered.columns else 0
            hh_hl      = df_filtered["ms_hh_hl"].sum() if "ms_hh_hl" in df_filtered.columns else 0
            st.markdown(
                f"**Signal Summary:** "
                f"🎯 {sfp_count} SFP setups &nbsp;|&nbsp; "
                f"📈 {of_bull} persistent bull flow &nbsp;|&nbsp; "
                f"💧 {vwap_above} above VWAP &nbsp;|&nbsp; "
                f"🏗 {hh_hl} HH/HL structure",
                unsafe_allow_html=True
            )

        top15 = df_filtered.head(15).copy()
        top15["apex_score"] = pd.to_numeric(top15["apex_score"], errors="coerce")
        colors = ["#3fb950" for _ in top15["ticker"]]
        fig = go.Figure(go.Bar(
            x=top15["apex_score"], y=top15["ticker"], orientation="h",
            marker_color=colors, text=top15["apex_score"].round(0).astype("Int64"),
            textposition="outside",
        ))
        fig.update_layout(
            title="Top 15 — Apex Score",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3",
            yaxis=dict(autorange="reversed", gridcolor="#21262d"),
            xaxis=dict(range=[0,115], gridcolor="#21262d"),
            height=400, margin=dict(l=10,r=60,t=40,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Changes Since Last Scan ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔄 Changes Since Last Scan")

        if "changes" not in df_filtered.columns or df_filtered["changes"].eq("No prior scan").all():
            st.info("Run a second scan to start seeing changes between sessions.")
        else:
            # Newly appeared tickers
            new_entries = df_filtered[df_filtered.get("is_new", pd.Series([False]*len(df_filtered))) == True] if "is_new" in df_filtered.columns else pd.DataFrame()
            if not new_entries.empty:
                st.markdown(f"**🆕 New entries this scan:** " +
                    " &nbsp; ".join([f'<span style="background:#1a3a2a;color:#3fb950;padding:2px 8px;border-radius:4px;font-weight:700;">{t}</span>'
                    for t in new_entries["ticker"].tolist()]),
                    unsafe_allow_html=True)

            # Tickers that dropped out
            if gone_tickers:
                st.markdown(f"**❌ Dropped from scan:** " +
                    " &nbsp; ".join([f'<span style="background:#2a1010;color:#f85149;padding:2px 8px;border-radius:4px;">{t}</span>'
                    for t in sorted(gone_tickers)]),
                    unsafe_allow_html=True)

            # Changes table
            change_cols = ["ticker", "price", "apex_score", "delta_score",
                           "stage", "of_bias", "vwap_position", "pa_patterns", "changes"]
            chg_show = [c for c in change_cols if c in df_filtered.columns]
            chg_df   = df_filtered[chg_show].copy()

            # Only show rows that actually changed or are new
            has_change = chg_df["changes"].apply(
                lambda x: x not in ["↔ No change", "No prior scan", "First scan", "–", None, ""]
            ) if "changes" in chg_df.columns else pd.Series([True]*len(chg_df))

            changed_df  = chg_df[has_change]
            unchanged_df = chg_df[~has_change]

            def color_delta(v):
                try:
                    v = float(v)
                    return "color:#3fb950;font-weight:700" if v > 0 else ("color:#f85149;font-weight:700" if v < 0 else "")
                except: return ""

            def color_changes(v):
                v = str(v)
                if "🆕" in v or "▲" in v or "Stage 2" in v or "Reclaimed" in v or "Breakout" in v:
                    return "color:#3fb950"
                if "▼" in v or "Lost" in v or "Stage 4" in v or "Bear" in v:
                    return "color:#f85149"
                if "SFP" in v or "Tighten" in v or "Handle" in v:
                    return "color:#d29922"
                return "color:#8b949e"

            if not changed_df.empty:
                st.markdown(f"**{len(changed_df)} tickers with notable changes:**")

                # Display as styled cards for easier reading
                for _, row in changed_df.head(20).iterrows():
                    tk       = row.get("ticker","")
                    chg_txt  = str(row.get("changes","–"))
                    ds       = row.get("delta_score")
                    score    = row.get("apex_score","–")
                    stage    = str(row.get("stage","–"))
                    of_bias  = str(row.get("of_bias","–"))
                    vwap_p   = str(row.get("vwap_position","–"))

                    # Card border colour based on change direction
                    if any(x in chg_txt for x in ["▲","🆕","Stage 2","Reclaimed","Breakout","🚀","📈"]):
                        border = "#3fb950"
                    elif any(x in chg_txt for x in ["▼","Lost","Stage 4","📉"]):
                        border = "#f85149"
                    elif any(x in chg_txt for x in ["🎯","📐","🔄"]):
                        border = "#d29922"
                    else:
                        border = "#30363d"

                    ds_str = f"({'+' if (ds or 0)>0 else ''}{ds:.0f} pts)" if pd.notna(ds) else ""
                    ds_color = "#3fb950" if (ds or 0) > 0 else ("#f85149" if (ds or 0) < 0 else "#8b949e")

                    st.markdown(
                        f'<div style="background:#0d1117;border-left:3px solid {border};'
                        f'border-radius:0 8px 8px 0;padding:12px 16px;margin:5px 0;'
                        f'display:flex;align-items:center;gap:16px;">'
                        f'<div style="min-width:60px;">'
                        f'<span style="font-weight:800;font-size:1rem;color:#e6edf3;">{tk}</span></div>'
                        f'<div style="min-width:80px;">'
                        f'<span style="color:#8b949e;font-size:0.75rem;">SCORE</span><br>'
                        f'<span style="font-weight:700;">{score}</span> '
                        f'<span style="font-size:0.8rem;color:{ds_color};">{ds_str}</span></div>'
                        f'<div style="min-width:130px;">'
                        f'<span style="color:#8b949e;font-size:0.75rem;">STAGE / OF / VWAP</span><br>'
                        f'<span style="font-size:0.8rem;color:#c9d1d9;">{stage[:10]} · {of_bias[:8]} · {vwap_p[:12]}</span></div>'
                        f'<div style="flex:1;">'
                        f'<span style="color:#8b949e;font-size:0.75rem;">WHAT CHANGED</span><br>'
                        f'<span style="color:{border};font-size:0.88rem;font-weight:600;">{chg_txt}</span></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown('<div class="metric-card" style="text-align:center;color:#8b949e;">↔ No significant changes since last scan — market conditions are stable.</div>', unsafe_allow_html=True)

            if not unchanged_df.empty:
                with st.expander(f"↔ {len(unchanged_df)} tickers with no significant change"):
                    st.dataframe(
                        unchanged_df[["ticker","apex_score","stage","of_bias","vwap_position"]].style
                        .format({"apex_score": "{:.0f}"}, na_rep="–"),
                        use_container_width=True, hide_index=True
                    )

        st.download_button("⬇ Download CSV", df_filtered.to_csv().encode("utf-8"),
            file_name=f"apexscan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHART VIEWER
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("### 📈 Price Chart + Moving Averages + VWAP")
    ticker_opts = df["ticker"].tolist() if not df.empty else ["NVDA","AAPL","TSM","ASML"]
    ca, cb, cc = st.columns([2, 1, 1])
    with ca: sel = st.selectbox("Ticker", ticker_opts)
    with cb: period = st.selectbox("Period", ["3mo","6mo","1y","2y"], index=1)
    with cc:
        show_vwap   = st.checkbox("VWAP", value=True)
        show_swings = st.checkbox("Swing Levels", value=True)

    if sel:
        hist = fetch_hist(sel, period)
        if not hist.empty:
            # Compute VWAP for chart overlay
            hist_vwap = hist.copy()
            hist_vwap["typical"] = (hist_vwap["High"] + hist_vwap["Low"] + hist_vwap["Close"]) / 3
            hist_vwap["tp_vol"]  = hist_vwap["typical"] * hist_vwap["Volume"]
            # Rolling 20-day VWAP
            hist_vwap["VWAP"]   = (hist_vwap["tp_vol"].rolling(20).sum() /
                                    hist_vwap["Volume"].rolling(20).sum())
            hist_vwap["VWAP_U"] = hist_vwap["VWAP"] + hist_vwap["typical"].rolling(20).std()
            hist_vwap["VWAP_L"] = hist_vwap["VWAP"] - hist_vwap["typical"].rolling(20).std()

            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.72,0.28], vertical_spacing=0.04)

            # Candlesticks
            fig2.add_trace(go.Candlestick(
                x=hist_vwap.index, open=hist_vwap["Open"], high=hist_vwap["High"],
                low=hist_vwap["Low"], close=hist_vwap["Close"], name="Price",
                increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            ), row=1, col=1)

            # MAs
            fig2.add_trace(go.Scatter(x=hist_vwap.index, y=hist_vwap["MA50"],
                line=dict(color="#d29922", width=1.5), name="50 MA"), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist_vwap.index, y=hist_vwap["MA200"],
                line=dict(color="#388bfd", width=1.5, dash="dot"), name="200 MA"), row=1, col=1)

            # VWAP overlay
            if show_vwap:
                fig2.add_trace(go.Scatter(
                    x=hist_vwap.index, y=hist_vwap["VWAP"],
                    line=dict(color="#c084fc", width=1.8),
                    name="VWAP (20d)", opacity=0.9,
                ), row=1, col=1)
                fig2.add_trace(go.Scatter(
                    x=hist_vwap.index, y=hist_vwap["VWAP_U"],
                    line=dict(color="#c084fc", width=0.8, dash="dot"),
                    name="VWAP +1σ", opacity=0.5,
                ), row=1, col=1)
                fig2.add_trace(go.Scatter(
                    x=hist_vwap.index, y=hist_vwap["VWAP_L"],
                    line=dict(color="#c084fc", width=0.8, dash="dot"),
                    fill="tonexty", fillcolor="rgba(192,132,252,0.05)",
                    name="VWAP -1σ", opacity=0.5,
                ), row=1, col=1)

            # Swing high/low levels from scan data
            if show_swings and not df.empty and sel in df["ticker"].values:
                row_data = df[df["ticker"] == sel].iloc[0]
                sh = row_data.get("ms_swing_high")
                sl = row_data.get("ms_swing_low")
                if sh and pd.notna(sh):
                    fig2.add_hline(y=sh, line_color="#f85149", line_dash="dash",
                                   line_width=1, opacity=0.7,
                                   annotation_text=f"Swing High ${sh:.2f}",
                                   annotation_position="right", row=1, col=1)
                if sl and pd.notna(sl):
                    fig2.add_hline(y=sl, line_color="#3fb950", line_dash="dash",
                                   line_width=1, opacity=0.7,
                                   annotation_text=f"Swing Low ${sl:.2f}",
                                   annotation_position="right", row=1, col=1)

            # PA pattern annotations on last candle
            if not df.empty and sel in df["ticker"].values:
                row_data = df[df["ticker"] == sel].iloc[0]
                pa = row_data.get("pa_patterns", "")
                if pa and pa != "None":
                    last_date  = hist_vwap.index[-1]
                    last_price = hist_vwap["High"].iloc[-1]
                    fig2.add_annotation(
                        x=last_date, y=last_price * 1.02,
                        text=f"📍 {pa[:40]}",
                        showarrow=True, arrowhead=2,
                        font=dict(color="#d29922", size=10),
                        bgcolor="#1a1a2e", bordercolor="#d29922",
                        row=1, col=1
                    )

            # Volume bars
            vol_colors = ["#3fb950" if hist_vwap["Close"].iloc[i] >= hist_vwap["Open"].iloc[i]
                          else "#f85149" for i in range(len(hist_vwap))]
            fig2.add_trace(go.Bar(x=hist_vwap.index, y=hist_vwap["Volume"],
                marker_color=vol_colors, name="Volume", opacity=0.7), row=2, col=1)

            fig2.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3",
                xaxis_rangeslider_visible=False, height=600,
                margin=dict(l=10,r=10,t=30,b=20), legend=dict(orientation="h",y=1.02),
            )
            fig2.update_yaxes(gridcolor="#21262d")
            fig2.update_xaxes(gridcolor="#21262d")
            st.plotly_chart(fig2, use_container_width=True)

            if not df.empty and sel in df["ticker"].values:
                row_data = df[df["ticker"]==sel].iloc[0]
                # Row 1: existing metrics
                s1,s2,s3,s4,s5 = st.columns(5)
                s1.metric("3m Return",  pct_fmt(row_data.get("perf_3m_%",0)))
                s2.metric("Apex Score", f"{float(row_data.get('apex_score',0)):.0f}")
                s3.metric("RS (3m)",    f"{float(row_data.get('rs_3m',0)):.0f}" if row_data.get("rs_3m",0) else "–")
                s4.metric("Stage",      str(row_data.get("stage","–")))
                s5.metric("ADR %",      f"{float(row_data.get('adr_%',0)):.1f}%")
                # Row 2: new metrics
                n1,n2,n3,n4,n5 = st.columns(5)
                n1.metric("Order Flow",   str(row_data.get("of_bias","–")))
                n2.metric("VWAP Position",str(row_data.get("vwap_position","–")))
                n3.metric("vs VWAP",      pct_fmt(row_data.get("vs_vwap_%",0)))
                n4.metric("Structure",    str(row_data.get("ms_structure","–")))
                n5.metric("PA Patterns",  str(row_data.get("pa_patterns","None"))[:30])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — THEME HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("### 🌍 Theme Rotation Heatmap")
    if df.empty:
        st.info("Run a scan first.")
    else:
        agg = df.groupby(["theme","market"]).agg(
            avg_score=("apex_score","mean"),
            avg_3m=("perf_3m_%","mean"),
            count=("ticker","count"),
            breakouts=("breaking_out","sum"),
        ).reset_index()

        pivot = agg.pivot_table(index="theme", columns="market",
                                values="avg_score", fill_value=0)
        fig3 = px.imshow(pivot, text_auto=".0f", aspect="auto",
            color_continuous_scale=[[0,"#0d1117"],[0.4,"#1f4e79"],
                                    [0.7,"#d29922"],[1,"#3fb950"]],
            title="Avg Apex Score by Theme & Market")
        fig3.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                           font_color="#e6edf3", height=360)
        st.plotly_chart(fig3, use_container_width=True)

        h1,h2 = st.columns(2)
        with h1:
            fig4 = px.scatter(agg, x="avg_3m", y="avg_score", size="count",
                color="market", text="theme", title="Performance vs Score",
                color_discrete_map={"US":"#388bfd"})
            fig4.update_traces(textposition="top center")
            fig4.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_color="#e6edf3", height=380,
                xaxis=dict(title="Avg 3m %",gridcolor="#21262d"),
                yaxis=dict(title="Avg Score",gridcolor="#21262d"))
            st.plotly_chart(fig4, use_container_width=True)
        with h2:
            fig5 = px.bar(agg.sort_values("breakouts",ascending=False),
                x="theme", y="breakouts", color="market", barmode="group",
                title="Active Breakouts by Theme",
                color_discrete_map={"US":"#388bfd"})
            fig5.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_color="#e6edf3", height=380,
                xaxis=dict(tickangle=-30,gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"))
            st.plotly_chart(fig5, use_container_width=True)

        # ── Emerging Gems Spotlight ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💎 Emerging Gems Spotlight")
        st.markdown(GEM_DISCLAIMER, unsafe_allow_html=True)

        gems_df = pd.DataFrame()
        if "theme" in df.columns:
            gems_df = df[df["theme"] == "emerging_gems"].copy()
        if "is_gem" in df.columns:
            extra = df[df["is_gem"] == True].copy()
            gems_df = pd.concat([gems_df, extra]).drop_duplicates(subset=["ticker"]) if not gems_df.empty else extra

        if gems_df.empty:
            st.info("No emerging gems in current scan results. Run a Live Scan — gems are automatically detected from your config.")
        else:
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.markdown(f'<div class="metric-card"><h3>💎 Gems Found</h3><div class="value amber">{len(gems_df)}</div></div>', unsafe_allow_html=True)
            gem_s2 = int(gems_df["stage"].str.contains("2 ✅", na=False).sum()) if "stage" in gems_df.columns else 0
            with g2:
                st.markdown(f'<div class="metric-card"><h3>Stage 2 Gems</h3><div class="value green">{gem_s2}</div></div>', unsafe_allow_html=True)
            gem_brk = int((gems_df.get("breaking_out", pd.Series([False]*len(gems_df))) == True).sum()) if "breaking_out" in gems_df.columns else 0
            with g3:
                st.markdown(f'<div class="metric-card"><h3>Gem Breakouts 🚀</h3><div class="value amber">{gem_brk}</div></div>', unsafe_allow_html=True)
            top_gem_score = pd.to_numeric(gems_df["apex_score"], errors="coerce").max() if not gems_df.empty else 0
            with g4:
                st.markdown(f'<div class="metric-card"><h3>Top Gem Score</h3><div class="value green">{top_gem_score:.0f}</div></div>', unsafe_allow_html=True)

            gem_show = [c for c in ["ticker","mcap_category","market_cap_bn","price","stage",
                                    "perf_3m_%","rs_3m","of_bias","vwap_position",
                                    "pa_patterns","apex_score"] if c in gems_df.columns]
            gem_disp = gems_df[gem_show].copy()
            gem_fmt  = {
                "price":         "${:.2f}",
                "market_cap_bn": lambda v: f"${v:.2f}B" if pd.notna(v) and v else "–",
                "apex_score":    "{:.0f}",
                "perf_3m_%":     pct_fmt,
                "rs_3m":         lambda v: f"{v:.0f}" if pd.notna(v) and v else "–",
            }
            active_gem_fmt = {k: v for k, v in gem_fmt.items() if k in gem_disp.columns}
            st.dataframe(
                gem_disp.style.map(color_score, subset=["apex_score"]).format(active_gem_fmt, na_rep="–"),
                use_container_width=True, hide_index=True
            )

            if len(gems_df) >= 2:
                gems_df["apex_score"] = pd.to_numeric(gems_df["apex_score"], errors="coerce")
                fig_gems = go.Figure(go.Bar(
                    x=gems_df["apex_score"].head(10),
                    y=gems_df["ticker"].head(10),
                    orientation="h", marker_color="#d29922",
                    text=gems_df["apex_score"].head(10).round(0).astype("Int64"),
                    textposition="outside",
                ))
                fig_gems.update_layout(
                    title="💎 Emerging Gems — Apex Score Ranking",
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3",
                    yaxis=dict(autorange="reversed", gridcolor="#21262d"),
                    xaxis=dict(range=[0, 115], gridcolor="#21262d"),
                    height=320, margin=dict(l=10, r=60, t=40, b=20),
                )
                st.plotly_chart(fig_gems, use_container_width=True)
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("### 💼 Portfolio Tracker")
    st.caption("Your holdings are saved locally in data/portfolio.json")

    holdings = load_portfolio()

    # ── Add new holding ───────────────────────────────────────────────────────
    with st.expander("➕ Add a Holding", expanded=len(holdings)==0):
        a1,a2,a3,a4 = st.columns(4)
        with a1: new_ticker = st.text_input("Ticker", placeholder="e.g. NVDA").upper().strip()
        with a2: new_qty    = st.number_input("Shares", min_value=0.01, value=1.0, step=1.0)
        with a3: new_price  = st.number_input("Buy Price ($)", min_value=0.01, value=100.0, step=0.01)
        with a4: new_date   = st.date_input("Buy Date", value=datetime.today())

        if st.button("Add to Portfolio") and new_ticker:
            holdings.append({
                "ticker":    new_ticker,
                "qty":       new_qty,
                "buy_price": new_price,
                "buy_date":  str(new_date),
            })
            save_portfolio(holdings)
            st.success(f"Added {new_ticker}")
            st.rerun()

    if not holdings:
        st.info("No holdings yet. Add your first position above.")
    else:
        # ── Fetch live prices & compute P&L ──────────────────────────────────
        rows = []
        alerts = []

        for h in holdings:
            tk   = h["ticker"]
            qty  = h["qty"]
            cost = h["buy_price"]
            live = fetch_price(tk)

            if not live:
                rows.append({"Ticker": tk, "Qty": qty, "Buy $": cost,
                             "Current $": "–", "P&L $": "–", "P&L %": "–",
                             "Value $": "–", "vs 50MA": "–", "Signal": "No data"})
                continue

            price    = live["price"]
            ma50     = live["ma50"]
            ma200    = live["ma200"]
            chg      = live["chg_pct"]
            pnl_pct  = round((price / cost - 1) * 100, 2)
            pnl_dol  = round((price - cost) * qty, 2)
            value    = round(price * qty, 2)
            vs50     = round((price / ma50 - 1) * 100, 1) if ma50 else 0

            # ── Signal logic ──────────────────────────────────────────────
            if price < ma50 and price < ma200:
                signal = "🔴 SELL — Below Both MAs"
                alerts.append(("danger", tk, f"${price} is below both 50MA (${ma50}) and 200MA (${ma200})"))
            elif price < ma50:
                signal = "⚠️ WATCH — Below 50MA"
                alerts.append(("warn", tk, f"${price} dropped below 50MA (${ma50})"))
            elif vs50 > 20:
                signal = "⚡ Extended — Consider Trimming"
            else:
                signal = "✅ Hold"

            rows.append({
                "Ticker":     tk,
                "Qty":        qty,
                "Buy $":      cost,
                "Current $":  price,
                "Day %":      chg,
                "P&L $":      pnl_dol,
                "P&L %":      pnl_pct,
                "Value $":    value,
                "vs 50MA %":  vs50,
                "Signal":     signal,
            })

        port_df = pd.DataFrame(rows)

        # ── Alerts ────────────────────────────────────────────────────────
        if alerts:
            st.markdown("#### 🚨 Position Alerts")
            for level, tk, msg in alerts:
                box = "danger-box" if level=="danger" else "warn-box"
                icon = "🔴" if level=="danger" else "⚠️"
                st.markdown(f'<div class="{box}">{icon} <b>{tk}</b> — {msg}</div>',
                            unsafe_allow_html=True)
            st.markdown("---")

        # ── Summary KPIs ──────────────────────────────────────────────────
        numeric_rows = port_df[pd.to_numeric(port_df["P&L $"], errors="coerce").notna()]
        total_value  = pd.to_numeric(numeric_rows["Value $"], errors="coerce").sum()
        total_pnl    = pd.to_numeric(numeric_rows["P&L $"],   errors="coerce").sum()
        total_cost   = sum(h["qty"]*h["buy_price"] for h in holdings)
        total_pnl_pct = round((total_pnl / total_cost * 100), 2) if total_cost else 0
        winners = int((pd.to_numeric(numeric_rows["P&L %"], errors="coerce") > 0).sum())

        k1,k2,k3,k4 = st.columns(4)
        pnl_color = "green" if total_pnl >= 0 else "red"
        with k1: st.markdown(f'<div class="metric-card"><h3>Portfolio Value</h3><div class="value white">${total_value:,.0f}</div></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="metric-card"><h3>Total P&L</h3><div class="value {pnl_color}">${total_pnl:+,.0f}</div></div>', unsafe_allow_html=True)
        with k3: st.markdown(f'<div class="metric-card"><h3>Return</h3><div class="value {pnl_color}">{total_pnl_pct:+.1f}%</div></div>', unsafe_allow_html=True)
        with k4: st.markdown(f'<div class="metric-card"><h3>Winners</h3><div class="value amber">{winners}/{len(numeric_rows)}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Holdings table ────────────────────────────────────────────────
        def _color_pnl(v):
            try: return "color:#3fb950" if float(v)>0 else ("color:#f85149" if float(v)<0 else "")
            except: return ""

        styled_p = port_df.style \
            .map(_color_pnl, subset=["P&L $","P&L %","Day %","vs 50MA %"]) \
            .format({
                "Buy $":      "${:.2f}",
                "Current $":  lambda v: f"${v:.2f}" if isinstance(v,(int,float)) else v,
                "Day %":      lambda v: f"{v:+.2f}%" if isinstance(v,(int,float)) else v,
                "P&L $":      lambda v: f"${v:+,.2f}" if isinstance(v,(int,float)) else v,
                "P&L %":      lambda v: f"{v:+.1f}%" if isinstance(v,(int,float)) else v,
                "Value $":    lambda v: f"${v:,.2f}" if isinstance(v,(int,float)) else v,
                "vs 50MA %":  lambda v: f"{v:+.1f}%" if isinstance(v,(int,float)) else v,
            }, na_rep="–")
        st.dataframe(styled_p, use_container_width=True, height=400)

        # ── Pie chart ─────────────────────────────────────────────────────
        pie_data = port_df[pd.to_numeric(port_df["Value $"], errors="coerce").notna()].copy()
        pie_data["Value $"] = pd.to_numeric(pie_data["Value $"])
        if not pie_data.empty:
            fig_pie = px.pie(pie_data, names="Ticker", values="Value $",
                title="Portfolio Allocation",
                color_discrete_sequence=px.colors.sequential.Viridis)
            fig_pie.update_layout(paper_bgcolor="#0d1117", font_color="#e6edf3", height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Remove holding ────────────────────────────────────────────────
        st.markdown("---")
        tickers_in = [h["ticker"] for h in holdings]
        remove_tk = st.selectbox("Remove a position", ["–"] + tickers_in)
        if st.button("Remove") and remove_tk != "–":
            holdings = [h for h in holdings if h["ticker"] != remove_tk]
            save_portfolio(holdings)
            st.success(f"Removed {remove_tk}")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EARNINGS CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("### 📅 Earnings Calendar")
    st.caption("Tracks upcoming earnings for your watchlist. Data from yfinance — may not cover all tickers.")

    cfg_e = load_config("config.yaml")
    all_us = list(set(t for theme in cfg_e["us_themes"].values() for t in theme))

    # Let user pick which tickers to track
    default_watch = df["ticker"].tolist()[:10] if not df.empty else all_us[:10]
    watch_sel = st.multiselect(
        "Select tickers to track",
        options=all_us,
        default=[t for t in default_watch if t in all_us][:8],
    )

    if st.button("🔄 Fetch Earnings Dates") and watch_sel:
        cal_rows = []
        prog = st.progress(0)

        # Try Alpha Vantage earnings calendar first (1 call for all tickers)
        av_key_cal = load_config("config.yaml").get("alpha_vantage_key","")
        av_dates   = {}
        av_eps_map = {}
        if av_key_cal and not av_key_cal.startswith("YOUR_"):
            try:
                # alpha_vantage already imported at top level
                av_dates = get_upcoming_earnings_for_watchlist(watch_sel, av_key_cal)
                st.caption(f"📊 Alpha Vantage: found upcoming dates for {len(av_dates)} tickers")
            except Exception as av_e:
                st.caption(f"AV calendar unavailable: {av_e}")

        for i, tk in enumerate(watch_sel):
            prog.progress((i+1)/len(watch_sel))

            # Get next earnings date — AV first, yfinance fallback
            next_e = av_dates.get(tk)
            eps_est = "–"
            eps_growth_str = "–"

            if not next_e:
                yf_data = fetch_earnings(tk)
                next_e  = yf_data.get("next_earnings")
                eps_est = yf_data.get("eps_est") or "–"

            # Get real EPS data if AV available
            if av_key_cal and not av_key_cal.startswith("YOUR_") and tk not in av_eps_map:
                try:
                    av_fund = analyse_eps(tk, av_key_cal, cache_hours=24)
                    av_eps_map[tk] = av_fund
                    eg = av_fund.get("eps_growth_pct")
                    es = av_fund.get("eps_surprise_pct")
                    cb = av_fund.get("consecutive_beats",0)
                    if eg is not None:
                        eps_growth_str = f"{eg:+.1f}%"
                    if es is not None:
                        eps_est = f"Last surprise: {es:+.1f}% | {cb}Q beats"
                except:
                    pass

            if next_e:
                try:
                    dt = datetime.strptime(next_e, "%Y-%m-%d")
                    days_away = (dt - datetime.now()).days
                except:
                    dt = None
                    days_away = None
            else:
                dt = None
                days_away = None

            urgency = "–"
            if days_away is not None:
                if days_away <= 2:   urgency = "🔴 THIS WEEK"
                elif days_away <= 7: urgency = "🟡 NEXT WEEK"
                elif days_away <= 30:urgency = "🟢 THIS MONTH"
                else:                urgency = f"📅 {days_away}d away"

            cal_rows.append({
                "Ticker":        tk,
                "Next Earnings": next_e or "Unknown",
                "Days Away":     days_away if days_away is not None else "–",
                "Urgency":       urgency,
                "EPS YoY Growth":eps_growth_str,
                "EPS Est/Surprise": eps_est,
            })
        prog.empty()

        cal_df = pd.DataFrame(cal_rows)
        # Sort: known dates first, soonest first
        cal_df["_sort"] = pd.to_numeric(cal_df["Days Away"], errors="coerce")
        cal_df = cal_df.sort_values("_sort").drop(columns=["_sort"])

        # Highlight upcoming
        st.markdown("---")
        urgent = cal_df[cal_df["Urgency"].str.contains("THIS WEEK|NEXT WEEK", na=False)]
        if not urgent.empty:
            st.markdown("#### 🚨 Earnings This Week / Next Week")
            for _, row in urgent.iterrows():
                st.markdown(
                    f'<div class="warn-box">⚠️ <b>{row["Ticker"]}</b> reports on '
                    f'<b>{row["Next Earnings"]}</b> — {row["Days Away"]} days away</div>',
                    unsafe_allow_html=True
                )
            st.markdown("---")

        st.markdown("#### Full Calendar")
        st.dataframe(cal_df, use_container_width=True, hide_index=True)
    else:
        st.info("Select tickers above and click **Fetch Earnings Dates**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SECTOR ROTATION
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("### 🔄 Sector Rotation Dashboard")
    st.caption("Tracks US sector ETF performance to show where institutional money is flowing.")

    if st.button("🔄 Refresh Sector Data"):
        fetch_sector = sector_performance.__wrapped__ if hasattr(sector_performance,"__wrapped__") else sector_performance
        sector_df = sector_performance()
    else:
        sector_df = sector_performance()

    if sector_df.empty:
        st.warning("Could not load sector data — yfinance may be temporarily unavailable. Try the Refresh button in 60 seconds.")
    else:
        # Safe idxmax — drop NAs first to avoid "all NA values" crash
        w_valid  = sector_df["1W %"].dropna()
        m_valid  = sector_df["1M %"].dropna()
        m3_valid = sector_df["3M %"].dropna()

        if w_valid.empty or m3_valid.empty:
            st.warning("Sector ETF data returned empty — yfinance may be rate-limiting. Wait 60 seconds and click Refresh.")
        else:
            best_1w  = sector_df.loc[w_valid.idxmax()]
            worst_1w = sector_df.loc[w_valid.idxmin()]
            best_1m  = sector_df.loc[m_valid.idxmax()] if not m_valid.empty else best_1w
            best_3m  = sector_df.loc[m3_valid.idxmax()]

            r1,r2,r3,r4 = st.columns(4)
            with r1: st.markdown(f'<div class="metric-card"><h3>🏆 Best This Week</h3><div class="value green">{best_1w["Sector"]}</div><div style="color:#8b949e;font-size:.85rem">{best_1w["1W %"]:+.1f}%</div></div>', unsafe_allow_html=True)
            with r2: st.markdown(f'<div class="metric-card"><h3>📉 Worst This Week</h3><div class="value red">{worst_1w["Sector"]}</div><div style="color:#8b949e;font-size:.85rem">{worst_1w["1W %"]:+.1f}%</div></div>', unsafe_allow_html=True)
            with r3: st.markdown(f'<div class="metric-card"><h3>🥇 Best This Month</h3><div class="value amber">{best_1m["Sector"]}</div><div style="color:#8b949e;font-size:.85rem">{best_1m["1M %"]:+.1f}%</div></div>', unsafe_allow_html=True)
            with r4: st.markdown(f'<div class="metric-card"><h3>🚀 Best 3 Months</h3><div class="value blue">{best_3m["Sector"]}</div><div style="color:#8b949e;font-size:.85rem">{best_3m["3M %"]:+.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Bar chart: 1W, 1M, 3M performance ────────────────────────────
        sector_melt = sector_df.melt(
            id_vars="Sector", value_vars=["1W %","1M %","3M %"],
            var_name="Period", value_name="Return %"
        )
        fig_sec = px.bar(
            sector_melt, x="Sector", y="Return %", color="Period",
            barmode="group", title="Sector Performance — 1W / 1M / 3M",
            color_discrete_map={"1W %":"#388bfd","1M %":"#d29922","3M %":"#3fb950"},
        )
        fig_sec.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3",
            xaxis=dict(tickangle=-30, gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", ticksuffix="%"),
            height=420, margin=dict(l=10,r=10,t=40,b=80),
            legend=dict(orientation="h", y=1.05),
        )
        fig_sec.add_hline(y=0, line_color="#30363d", line_width=1)
        st.plotly_chart(fig_sec, use_container_width=True)

        # ── Heatmap: sectors by timeframe ─────────────────────────────────
        heat_data = sector_df.set_index("Sector")[["1W %","1M %","3M %"]]
        fig_heat = px.imshow(
            heat_data, text_auto=".1f", aspect="auto",
            color_continuous_scale=[[0,"#f85149"],[0.5,"#21262d"],[1,"#3fb950"]],
            title="Sector Rotation Heatmap",
            color_continuous_midpoint=0,
        )
        fig_heat.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e6edf3", height=380,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Raw table ─────────────────────────────────────────────────────
        st.markdown("#### Sector Detail Table")
        def _color_ret(v):
            try: return "color:#3fb950;font-weight:600" if float(v)>0 else "color:#f85149;font-weight:600"
            except: return ""

        styled_s = sector_df.style \
            .map(_color_ret, subset=["1W %","1M %","3M %"]) \
            .format({"1W %": "{:+.2f}%","1M %": "{:+.2f}%","3M %": "{:+.2f}%",
                     "Price": "${:.2f}"}, na_rep="–")
        st.dataframe(styled_s, use_container_width=True, hide_index=True)

        # ── Rotation insight ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💡 Rotation Insight")
        top3 = sector_df.nlargest(3,"1W %")["Sector"].tolist()
        bot3 = sector_df.nsmallest(3,"1W %")["Sector"].tolist()
        st.markdown(
            f'<div class="alert-box">🟢 <b>Money flowing INTO:</b> {", ".join(top3)}</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="danger-box">🔴 <b>Money flowing OUT OF:</b> {", ".join(bot3)}</div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — STOCK DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown("### 🔍 Stock Deep Dive")
    st.caption("Analyse any US stock — same scoring engine as the live scan.")

    d1, d2 = st.columns([2, 1])
    with d1:
        dive_ticker = st.text_input(
            "Enter any ticker", placeholder="e.g. HIMS, ARM, CELH, HOOD",
            key="dive_input"
        ).upper().strip()
    with d2:
        dive_btn = st.button("🔍 Analyse", use_container_width=True)

    if dive_btn and dive_ticker:
        with st.spinner(f"Analysing {dive_ticker}…"):
            cfg_d = load_config("config.yaml")
            result = None
            try:
                from scanner import analyze_stock
                result = analyze_stock(dive_ticker, cfg_d)
            except Exception as e:
                st.error(f"Error: {e}")

        if result is None:
            st.error(f"Could not analyse **{dive_ticker}**. Check the ticker is correct and has enough price history.")
        else:
            # ── Score banner ──────────────────────────────────────────────
            score = result["apex_score"]
            score_color = "#3fb950" if score >= 70 else ("#d29922" if score >= 40 else "#f85149")
            verdict = "Strong Setup 🟢" if score >= 70 else ("Watch List 🟡" if score >= 40 else "Skip / Weak 🔴")

            st.markdown(f"""
            <div style="background:#161b22;border:1px solid {score_color};border-radius:12px;
                        padding:20px 28px;margin-bottom:20px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <div style="font-size:2rem;font-weight:800;color:#e6edf3;">{dive_ticker}</div>
                  <div style="color:#8b949e;font-size:0.9rem;">
                    Market: {result['market']} &nbsp;|&nbsp; Theme: {result['theme']} &nbsp;|&nbsp;
                    Stage: {result.get('stage','–')}
                  </div>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:3rem;font-weight:900;color:{score_color};">{score:.0f}</div>
                  <div style="color:{score_color};font-size:0.9rem;font-weight:600;">
                    Apex Score — {verdict}
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── KPI row 1: Price & Performance ───────────────────────────
            st.markdown("#### 📊 Price & Performance")
            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("Price",        f"${result['price']:.2f}")
            k2.metric("1M Return",    pct_fmt(result['perf_1m_%']))
            k3.metric("3M Return",    pct_fmt(result['perf_3m_%']))
            k4.metric("6M Return",    pct_fmt(result['perf_6m_%']))
            k5.metric("% Off 52W High", pct_fmt(result['pct_off_high_%']))

            st.markdown("---")

            # ── KPI row 2: Strength & Structure ──────────────────────────
            st.markdown("#### 💪 Relative Strength & Structure")
            k6,k7,k8,k9,k10 = st.columns(5)
            rs3  = result.get("rs_3m", 0)
            rs6  = result.get("rs_6m", 0)
            k6.metric("RS vs Benchmark (3m)",  f"{rs3:.0f}" if rs3 else "–")
            k7.metric("RS vs Benchmark (6m)",  f"{rs6:.0f}" if rs6 else "–")
            k8.metric("vs 50MA",               pct_fmt(result.get("vs_50ma_%", 0)))
            k9.metric("vs 200MA",              pct_fmt(result.get("vs_200ma_%", 0)))
            k10.metric("ADR %",                f"{result.get('adr_%', 0):.1f}%")

            st.markdown("---")

            # ── KPI row 3: Stage & Volume ─────────────────────────────────
            st.markdown("#### 📐 Stage Analysis & Volume")
            k11,k12,k13,k14,k15 = st.columns(5)
            k11.metric("Stage",          result.get("stage","–"))
            k12.metric("Above 50MA",     "✅ Yes" if result.get("above_50ma") else "❌ No")
            k13.metric("Above 200MA",    "✅ Yes" if result.get("above_200ma") else "❌ No")
            k14.metric("50MA > 200MA",   "✅ Yes" if result.get("ma50_gt_ma200") else "❌ No")
            k15.metric("Vol Surge",      f"{result.get('vol_surge_x',1):.1f}x")

            st.markdown("---")

            # ── KPI row 4: Order Flow Persistence (NEW) ───────────────────
            st.markdown("#### 🌊 Order Flow Persistence")
            o1,o2,o3,o4,o5 = st.columns(5)
            o1.metric("Directional Bias",   result.get("of_bias","–"))
            o2.metric("Up Vol Ratio",       f"{result.get('of_up_vol_ratio',1):.2f}x")
            o3.metric("Bullish Days (10d)", f"{result.get('of_bullish_days',0):.0f}%")
            o4.metric("Consec Up Closes",   str(result.get("of_consec_up",0)))
            o5.metric("OF Score",           f"{result.get('of_score',0)}/8")
            st.caption("Up Vol Ratio > 1.5x = sustained institutional buying pressure across multiple sessions")

            st.markdown("---")

            # ── KPI row 5: VWAP / Auction Market Theory (NEW) ────────────
            st.markdown("#### 💧 VWAP & Auction Market Theory")
            v1,v2,v3,v4,v5 = st.columns(5)
            vwap_val = result.get("vwap")
            v1.metric("VWAP (20d)",      f"${vwap_val:.2f}" if vwap_val else "–")
            v2.metric("vs VWAP",         pct_fmt(result.get("vs_vwap_%",0)))
            v3.metric("VWAP Position",   result.get("vwap_position","–"))
            v4.metric("VWAP Slope",      result.get("vwap_slope","–"))
            v5.metric("VWAP Score",      f"{result.get('vwap_score',0)}/4")
            vwap_pos = result.get("vwap_position","")
            if "Above" in vwap_pos and "Extended" not in vwap_pos:
                st.markdown('<div class="alert-box">✅ Price above VWAP — buyers in control, value accepted higher</div>', unsafe_allow_html=True)
            elif "Extended Above" in vwap_pos:
                st.markdown('<div class="warn-box">⚡ Extended above VWAP — momentum strong but watch for mean reversion</div>', unsafe_allow_html=True)
            elif "Extended Below" in vwap_pos:
                st.markdown('<div class="danger-box">🔴 Extended below VWAP — sellers in control, avoid new longs</div>', unsafe_allow_html=True)

            st.markdown("---")

            # ── KPI row 6: Market Structure (NEW) ────────────────────────
            st.markdown("#### 🏗 Market Structure")
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Structure",          result.get("ms_structure","–"))
            m2.metric("HH / HL",            "✅ Yes" if result.get("ms_hh_hl") else "❌ No")
            m3.metric("Break of Structure",  "🚨 Yes" if result.get("ms_bos") else "No")
            m4.metric("Last Swing High",     f"${result.get('ms_swing_high',0):.2f}" if result.get("ms_swing_high") else "–")
            m5.metric("Last Swing Low",      f"${result.get('ms_swing_low',0):.2f}"  if result.get("ms_swing_low")  else "–")

            st.markdown("---")

            # ── KPI row 7: Price Action Patterns (NEW) ───────────────────
            st.markdown("#### 🕯 Price Action Patterns")
            pa_patterns = result.get("pa_patterns","None")
            if pa_patterns and pa_patterns != "None":
                pa_list = [p.strip() for p in pa_patterns.split("|")]
                pa_cols = st.columns(min(len(pa_list),4))
                pa_colors = {
                    "Bullish SFP (Bear Trap)":  "#3fb950",
                    "Bullish Engulfing":         "#3fb950",
                    "PA Confluence":             "#d29922",
                    "Inside Day (Compression)":  "#388bfd",
                    "Bullish Context Candle":    "#3fb950",
                    "Bearish SFP (Bull Trap)":   "#f85149",
                    "Bearish Engulfing":         "#f85149",
                    "Bearish Context Candle":    "#f85149",
                }
                for i, p in enumerate(pa_list[:4]):
                    col_c = pa_colors.get(p, "#8b949e")
                    pa_cols[i].markdown(
                        f'<div style="background:#161b22;border:1px solid {col_c};'
                        f'border-radius:8px;padding:12px;text-align:center;">'
                        f'<div style="color:{col_c};font-weight:700;font-size:0.9rem;">{p}</div>'
                        f'</div>', unsafe_allow_html=True)
                if "Bullish SFP" in pa_patterns:
                    st.markdown('<div class="alert-box">🎯 <b>Bullish SFP:</b> Price wicked below a swing low trapping shorts, then closed above — bear trap, smart money reversal signal.</div>', unsafe_allow_html=True)
                elif "Bearish SFP" in pa_patterns:
                    st.markdown('<div class="danger-box">🎯 <b>Bearish SFP:</b> Price wicked above a swing high trapping longs, then closed below — bull trap, potential reversal lower.</div>', unsafe_allow_html=True)
                if "PA Confluence" in pa_patterns:
                    st.markdown('<div class="warn-box">⚡ <b>PA Confluence:</b> Multiple signals aligning — higher probability setup.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;color:#8b949e;text-align:center;">No significant PA patterns on last candle</div>', unsafe_allow_html=True)
            st.metric("PA Score", f"{result.get('pa_score',0)}/5")

            st.markdown("---")

            # ── KPI row 8: Alpha Vantage Fundamentals (NEW) ─────────────
            st.markdown("#### 📊 Real Earnings & Fundamentals (Alpha Vantage)")
            eps_growth  = result.get("eps_growth_%")
            eps_surp    = result.get("eps_surprise_%")
            eps_accel   = result.get("eps_accel")
            beats       = result.get("consec_beats")
            rev_g       = result.get("rev_growth_%")
            eps_sc      = result.get("eps_score", 0)
            analyst_t   = result.get("analyst_target")
            pe          = result.get("pe_ratio")
            peg         = result.get("peg_ratio")
            eps_det     = result.get("eps_details","–")
            eps_trend   = result.get("eps_trend",[])

            if eps_growth is None and eps_surp is None:
                st.markdown('<div class="warn-box">⚠️ Alpha Vantage key not configured or quota reached. '
                            'Add your key to config.yaml under <code>alpha_vantage_key</code>.</div>',
                            unsafe_allow_html=True)
            else:
                av1,av2,av3,av4,av5 = st.columns(5)
                eps_c = "#3fb950" if (eps_growth or 0) > 15 else ("#d29922" if (eps_growth or 0) > 0 else "#f85149")
                av1.metric("EPS Growth YoY",   f"{eps_growth:+.1f}%" if eps_growth is not None else "–")
                av2.metric("Last Surprise",     f"{eps_surp:+.1f}%"  if eps_surp  is not None else "–")
                av3.metric("Accelerating",      "✅ Yes" if eps_accel else "❌ No")
                av4.metric("Consec Beats",      f"{beats}Q" if beats is not None else "–")
                av5.metric("Revenue Growth",    f"{rev_g:+.1f}%"     if rev_g     is not None else "–")

                av6,av7,av8,av9,av10 = st.columns(5)
                av6.metric("EPS Score",  f"{eps_sc}/15")
                av7.metric("Analyst Target", f"${analyst_t:.2f}" if analyst_t else "–")
                av8.metric("PE Ratio",   f"{pe:.1f}x" if pe else "–")
                av9.metric("PEG Ratio",  f"{peg:.2f}"  if peg else "–")
                av10.metric("Earn Momentum", result.get("earn_momentum","–"))

                # EPS trend sparkline
                if eps_trend:
                    pass  # go already imported at top
                    fig_eps = go.Figure(go.Scatter(
                        x=[f"Q-{len(eps_trend)-i}" for i in range(len(eps_trend))],
                        y=list(reversed(eps_trend)),
                        mode="lines+markers+text",
                        text=[f"${v:.2f}" for v in reversed(eps_trend)],
                        textposition="top center",
                        line=dict(color="#3fb950", width=2),
                        marker=dict(size=8, color="#3fb950"),
                    ))
                    fig_eps.update_layout(
                        title="EPS Trend (last 4 quarters)",
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        font_color="#e6edf3", height=200,
                        xaxis=dict(gridcolor="#21262d"),
                        yaxis=dict(gridcolor="#21262d", tickprefix="$"),
                        margin=dict(l=10,r=10,t=40,b=20),
                    )
                    st.plotly_chart(fig_eps, use_container_width=True)

                st.caption(f"Details: {eps_det}")

                # EPS quality signals
                if eps_accel and (beats or 0) >= 3 and (eps_growth or 0) > 25:
                    st.markdown('<div class="alert-box">🏆 <b>Exceptional EPS quality:</b> '
                                'Accelerating growth + 3+ consecutive beats + >25% YoY growth. '
                                'This is the kind of fundamental profile that precedes major price moves.</div>',
                                unsafe_allow_html=True)
                elif (eps_surp or 0) > 20:
                    st.markdown(f'<div class="alert-box">⚡ <b>Large earnings beat:</b> '
                                f'Beat estimates by {eps_surp:.1f}% last quarter — '
                                f'institutional re-rating often follows large positive surprises.</div>',
                                unsafe_allow_html=True)
                elif (eps_growth or 0) < 0:
                    st.markdown('<div class="danger-box">🔴 <b>Declining EPS:</b> '
                                'Earnings are shrinking year-over-year. '
                                'Strong price momentum with declining fundamentals is a yellow flag.</div>',
                                unsafe_allow_html=True)

                # Upside to analyst target
                if analyst_t and result.get("price"):
                    upside = (analyst_t / result["price"] - 1) * 100
                    target_color = "alert-box" if upside > 10 else ("warn-box" if upside > 0 else "danger-box")
                    st.markdown(f'<div class="{target_color}">📍 <b>Analyst Target:</b> ${analyst_t:.2f} '
                                f'= <b>{upside:+.1f}% upside</b> from current price ${result["price"]:.2f}</div>',
                                unsafe_allow_html=True)

            st.markdown("---")

            # ── Pattern + Earnings ────────────────────────────────────────
            p1, p2 = st.columns(2)
            with p1:
                pattern = result.get("pattern","–")
                breaking = result.get("breaking_out", False)
                pat_color = "#3fb950" if breaking else "#d29922"
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid {pat_color};
                            border-radius:8px;padding:16px 20px;">
                  <div style="color:#8b949e;font-size:0.75rem;text-transform:uppercase;">
                    Chart Pattern
                  </div>
                  <div style="font-size:1.3rem;font-weight:700;color:{pat_color};margin-top:6px;">
                    {pattern}
                  </div>
                  <div style="color:#8b949e;font-size:0.8rem;margin-top:4px;">
                    {'Active breakout — watch closely' if breaking else 'Not yet breaking out'}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with p2:
                earn = result.get("earn_momentum","–")
                earn_color = "#3fb950" if earn=="Strong" else ("#d29922" if earn=="Moderate" else "#8b949e")
                near = "✅ Near 52W High" if result.get("near_52wh") else "❌ Below 52W High"
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #30363d;
                            border-radius:8px;padding:16px 20px;">
                  <div style="color:#8b949e;font-size:0.75rem;text-transform:uppercase;">
                    Earnings Momentum
                  </div>
                  <div style="font-size:1.3rem;font-weight:700;color:{earn_color};margin-top:6px;">
                    {earn}
                  </div>
                  <div style="color:#8b949e;font-size:0.8rem;margin-top:4px;">{near}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Chart ─────────────────────────────────────────────────────
            st.markdown("#### 📈 Price Chart")
            hist_d = fetch_hist(dive_ticker, "1y")
            if not hist_d.empty:
                fig_d = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                      row_heights=[0.72, 0.28], vertical_spacing=0.04)
                fig_d.add_trace(go.Candlestick(
                    x=hist_d.index,
                    open=hist_d["Open"], high=hist_d["High"],
                    low=hist_d["Low"],   close=hist_d["Close"],
                    name="Price",
                    increasing_line_color="#3fb950",
                    decreasing_line_color="#f85149",
                ), row=1, col=1)
                fig_d.add_trace(go.Scatter(
                    x=hist_d.index, y=hist_d["MA50"],
                    line=dict(color="#d29922", width=1.5), name="50 MA"
                ), row=1, col=1)
                fig_d.add_trace(go.Scatter(
                    x=hist_d.index, y=hist_d["MA200"],
                    line=dict(color="#388bfd", width=1.5, dash="dot"), name="200 MA"
                ), row=1, col=1)
                vol_c = ["#3fb950" if hist_d["Close"].iloc[i] >= hist_d["Open"].iloc[i]
                         else "#f85149" for i in range(len(hist_d))]
                fig_d.add_trace(go.Bar(
                    x=hist_d.index, y=hist_d["Volume"],
                    marker_color=vol_c, name="Volume", opacity=0.7
                ), row=2, col=1)
                fig_d.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font_color="#e6edf3", xaxis_rangeslider_visible=False,
                    height=550, margin=dict(l=10,r=10,t=20,b=20),
                    legend=dict(orientation="h", y=1.02),
                )
                fig_d.update_yaxes(gridcolor="#21262d")
                fig_d.update_xaxes(gridcolor="#21262d")
                st.plotly_chart(fig_d, use_container_width=True)

            # ── Plain English Summary ─────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 💬 Plain English Summary")

            bullish = []
            bearish = []

            if result.get("above_200ma"):   bullish.append("Trading above the 200-day moving average (long-term uptrend)")
            else:                           bearish.append("Below the 200-day moving average — not in a confirmed uptrend")
            if result.get("ma50_gt_ma200"): bullish.append("50MA is above 200MA — classic Stage 2 uptrend structure")
            else:                           bearish.append("50MA is below 200MA — not yet in Stage 2")
            if rs3 and rs3 > 100:           bullish.append(f"RS score of {rs3:.0f} — beating the benchmark strongly")
            elif rs3 and rs3 > 70:          bullish.append(f"RS score of {rs3:.0f} — keeping pace with the market")
            else:                           bearish.append(f"RS score of {rs3:.0f} — lagging behind the market")
            if result.get("near_52wh"):     bullish.append("Near its 52-week high — price strength confirmed")
            else:                           bearish.append(f"{abs(result.get('pct_off_high_%',0)):.0f}% off 52-week high — needs to reclaim highs")
            if result.get("breaking_out"):  bullish.append(f"Active breakout pattern: {pattern}")
            if result.get("vol_surge_x",1) >= 1.4: bullish.append(f"Volume surge of {result.get('vol_surge_x',1):.1f}x — institutional interest")
            if result.get("perf_3m_%",0) > 15: bullish.append(f"3-month return of +{result.get('perf_3m_%',0):.1f}% — strong momentum")
            else:                           bearish.append(f"3-month return of {result.get('perf_3m_%',0):.1f}% — below momentum threshold")

            if bullish:
                for b in bullish:
                    st.markdown(f'<div class="alert-box">✅ {b}</div>', unsafe_allow_html=True)
            if bearish:
                for b in bearish:
                    st.markdown(f'<div class="danger-box">❌ {b}</div>', unsafe_allow_html=True)

    elif not dive_btn:
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                    padding:32px;text-align:center;color:#8b949e;">
          <div style="font-size:2rem;margin-bottom:8px;">🔍</div>
          <div style="font-size:1rem;">Type any ticker above and click <b>Analyse</b></div>
          <div style="font-size:0.85rem;margin-top:8px;">
            Works for any US stock: HIMS, HOOD, CELH, SOFI, DUOL, ONON…
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — OPTIONS FLOW
# ══════════════════════════════════════════════════════════════════════════════

with tabs[7]:
    st.markdown("### 🎯 Options Flow Scanner")
    st.caption("Scans for unusual options activity — large bets placed by smart money via yfinance options chain.")


    opt_tickers_default = df["ticker"].tolist()[:8] if not df.empty else ["NVDA","AAPL","TSLA","AMZN"]
    cfg_o = load_config("config.yaml")
    all_tickers_o = list(set(t for theme in cfg_o["us_themes"].values() for t in theme))

    oc1, oc2, oc3 = st.columns([2, 1, 1])
    with oc1:
        opt_mode = st.radio("Scan mode", ["Single ticker", "Scan my watchlist"], horizontal=True)
    with oc2:
        opt_min_vol = st.number_input("Min contract volume", value=200, step=100)
    with oc3:
        opt_btn = st.button("🎯 Scan Options Flow", use_container_width=True)

    if opt_mode == "Single ticker":
        opt_single = st.text_input("Ticker", placeholder="e.g. NVDA", key="opt_single").upper().strip()
    else:
        opt_single = None

    if opt_btn:
        with st.spinner("Fetching options chain data…"):
            if opt_mode == "Single ticker" and opt_single:
                opt_df = scan_options_flow(opt_single, min_volume=int(opt_min_vol))
            else:
                scan_tix = opt_tickers_default[:6]
                opt_df = scan_multiple(scan_tix, min_volume=int(opt_min_vol))

        if opt_df is None or opt_df.empty:
            st.warning("No unusual options activity found. Try lowering Min Volume or choosing a different ticker.")
        else:
            unusual = opt_df[opt_df["unusual"] == True]
            normal  = opt_df[opt_df["unusual"] == False]

            # KPIs
            ok1, ok2, ok3, ok4 = st.columns(4)
            with ok1: st.markdown(f'<div class="metric-card"><h3>Total Contracts</h3><div class="value white">{len(opt_df)}</div></div>', unsafe_allow_html=True)
            with ok2: st.markdown(f'<div class="metric-card"><h3>Unusual Activity</h3><div class="value amber">{len(unusual)}</div></div>', unsafe_allow_html=True)
            with ok3:
                calls = int((opt_df["type"] == "CALL").sum())
                st.markdown(f'<div class="metric-card"><h3>Calls (Bullish)</h3><div class="value green">{calls}</div></div>', unsafe_allow_html=True)
            with ok4:
                puts = int((opt_df["type"] == "PUT").sum())
                st.markdown(f'<div class="metric-card"><h3>Puts (Bearish)</h3><div class="value red">{puts}</div></div>', unsafe_allow_html=True)

            st.markdown("---")

            if not unusual.empty:
                st.markdown("#### 🚨 Unusual Activity — Possible Smart Money Moves")
                for _, row in unusual.head(10).iterrows():
                    color = "#3fb950" if "Bullish" in str(row.get("sentiment","")) else "#f85149"
                    tk_label = f" [{row['ticker']}]" if "ticker" in row and row["ticker"] else ""
                    st.markdown(
                        f'<div style="background:#161b22;border:1px solid {color};border-radius:8px;'
                        f'padding:12px 16px;margin:4px 0;">'
                        f'<b>{row["type"]}{tk_label}</b> — Strike ${row["strike"]} | '
                        f'Exp: {row.get("expiry","?")} | '
                        f'Vol: {row["volume"]:,} | OI: {row["open_interest"]:,} | '
                        f'Vol/OI: {row["vol/OI"]}x | IV: {row["IV_%"]}% | '
                        f'Notional: ${row["notional_$"]:,.0f} — <b style="color:{color}">{row["sentiment"]}</b>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.markdown("---")

            st.markdown("#### Full Options Chain")
            display_cols = [c for c in ["ticker","type","expiry","strike","spot","moneyness_%",
                            "volume","open_interest","vol/OI","IV_%","last_price",
                            "notional_$","unusual","sentiment"] if c in opt_df.columns]
            st.dataframe(opt_df[display_cols], use_container_width=True, hide_index=True, height=400)
    else:
        st.info("Select a ticker or watchlist scan and click **🎯 Scan Options Flow**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — INSIDER TRACKER
# ══════════════════════════════════════════════════════════════════════════════

with tabs[8]:
    st.markdown("### 🕵️ Insider Trading Tracker")
    st.caption("SEC Form 4 filings via OpenInsider — when executives buy their own stock, pay attention.")


    ic1, ic2, ic3 = st.columns([2, 1, 1])
    with ic1:
        ins_mode = st.radio("Mode", ["Single ticker", "Scan watchlist"], horizontal=True, key="ins_mode")
    with ic2:
        ins_days = st.selectbox("Look back", [7, 14, 30, 60, 90], index=2, key="ins_days")
    with ic3:
        ins_btn = st.button("🕵️ Fetch Insider Data", use_container_width=True)

    if ins_mode == "Single ticker":
        ins_ticker = st.text_input("Ticker", placeholder="e.g. TSM", key="ins_ticker").upper().strip()
    else:
        ins_ticker = None

    if ins_btn:
        with st.spinner("Fetching SEC Form 4 data from OpenInsider…"):
            if ins_mode == "Single ticker" and ins_ticker:
                ins_df = fetch_insider_trades(ticker=ins_ticker, days_back=ins_days, trade_type="P")
                if not ins_df.empty:
                    st.success(f"Found {len(ins_df)} insider purchase(s) for {ins_ticker}")
            else:
                cfg_i = load_config("config.yaml")
                watch_i = list(set(t for theme in cfg_i["us_themes"].values() for t in theme))
                ins_df = get_insider_summary(watch_i[:20], days_back=ins_days)
                if not ins_df.empty:
                    st.success(f"Insider summary across {len(ins_df)} tickers")

        if ins_df is None or ins_df.empty:
            st.warning("No insider purchases found in this period. Try extending the look-back window.")
        else:
            # Highlight cluster buys
            if "cluster_buy" in ins_df.columns:
                clusters = ins_df[ins_df["cluster_buy"] == True]
                if not clusters.empty:
                    st.markdown("#### 🔥 Cluster Buys — Multiple Insiders Buying Same Stock")
                    for _, row in clusters.iterrows():
                        tk = row.get("ticker","")
                        val = row.get("total_$", row.get("value", 0))
                        n   = row.get("insiders", row.get("insider_count", "?"))
                        st.markdown(
                            f'<div class="alert-box">🔥 <b>{tk}</b> — '
                            f'{n} insiders bought | Total: ${val:,.0f}</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown("---")

            st.markdown("#### All Insider Purchases")
            st.dataframe(ins_df, use_container_width=True, hide_index=True, height=450)

            st.markdown("""
            ---
            **How to use this:**
            - 🔥 **Cluster buys** (2+ insiders) = strongest possible signal — they rarely all buy at the same time unless they see value
            - Single executive buys = worth noting, especially C-suite (CEO, CFO, COO)
            - Insider *sales* are less meaningful — they sell for many reasons (taxes, diversification)
            - Combine with a high Apex Score for maximum conviction
            """)
    else:
        st.info("Select a mode and click **🕵️ Fetch Insider Data**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — DIVIDEND REINVESTMENT CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

with tabs[9]:
    st.markdown("### 📊 Dividend Reinvestment Calculator (DRIP)")
    st.caption("Shows the compounding power of reinvesting dividends over time.")

    @st.cache_data(ttl=3600)
    def get_dividend_info(ticker: str) -> dict:
        try:
            info = yf.Ticker(ticker).info
            return {
                "div_yield":    info.get("dividendYield", 0) or 0,
                "div_annual":   info.get("dividendRate", 0) or 0,
                "payout_ratio": info.get("payoutRatio", 0) or 0,
                "name":         info.get("longName", ticker),
                "sector":       info.get("sector", "–"),
            }
        except:
            return {}

    da1, da2 = st.columns(2)
    with da1:
        drip_ticker  = st.text_input("Ticker (optional — for live yield)", placeholder="e.g. AAPL, JNJ, KO", key="drip_tk").upper().strip()
        initial_inv  = st.number_input("Initial Investment ($)", min_value=100.0, value=10000.0, step=500.0)
        monthly_add  = st.number_input("Monthly Contribution ($)", min_value=0.0, value=200.0, step=50.0)
        years        = st.slider("Investment Period (Years)", 1, 40, 10)
    with da2:
        if drip_ticker:
            dinfo = get_dividend_info(drip_ticker)
            default_yield = round(dinfo.get("div_yield", 0.03) * 100, 2)
            default_growth = 5.0
            if dinfo.get("name"):
                st.info(f"**{dinfo['name']}** | Sector: {dinfo.get('sector','–')} | "
                        f"Div Yield: {default_yield:.2f}% | Payout: {dinfo.get('payout_ratio',0)*100:.0f}%")
        else:
            default_yield  = 3.0
            default_growth = 5.0

        div_yield_pct  = st.number_input("Annual Dividend Yield (%)", min_value=0.0, max_value=20.0,
                                          value=float(default_yield), step=0.1)
        price_growth   = st.number_input("Expected Annual Price Growth (%)", min_value=-5.0, max_value=30.0,
                                          value=float(default_growth), step=0.5)
        drip_on        = st.checkbox("Reinvest dividends (DRIP)", value=True)
        tax_rate       = st.slider("Dividend Tax Rate (%)", 0, 40, 15)

    calc_btn = st.button("📊 Calculate", use_container_width=True)

    if calc_btn:
        # ── Simulation ────────────────────────────────────────────────────
        div_rate     = div_yield_pct / 100
        growth_rate  = price_growth / 100
        tax          = tax_rate / 100
        months       = years * 12
        monthly_rate = growth_rate / 12
        monthly_div  = div_rate / 12

        records    = []
        value      = initial_inv
        shares     = initial_inv  # treat as $ units for simplicity
        total_divs = 0
        total_cont = initial_inv

        for m in range(1, months + 1):
            # Price appreciation
            value *= (1 + monthly_rate)
            # Dividend
            div_income = value * monthly_div
            after_tax  = div_income * (1 - tax)
            total_divs += after_tax
            if drip_on:
                value += after_tax
            # Monthly contribution
            value      += monthly_add
            total_cont += monthly_add

            if m % 12 == 0:
                records.append({
                    "Year":            m // 12,
                    "Portfolio Value": round(value, 2),
                    "Total Invested":  round(total_cont, 2),
                    "Total Dividends": round(total_divs, 2),
                    "Gain":            round(value - total_cont, 2),
                    "Gain %":          round((value / total_cont - 1) * 100, 1),
                })

        result_df = pd.DataFrame(records)

        # KPIs
        final_val  = result_df["Portfolio Value"].iloc[-1]
        total_inv  = result_df["Total Invested"].iloc[-1]
        total_div  = result_df["Total Dividends"].iloc[-1]
        total_gain = final_val - total_inv

        k1,k2,k3,k4 = st.columns(4)
        with k1: st.markdown(f'<div class="metric-card"><h3>Final Value</h3><div class="value green">${final_val:,.0f}</div></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="metric-card"><h3>Total Invested</h3><div class="value white">${total_inv:,.0f}</div></div>', unsafe_allow_html=True)
        with k3: st.markdown(f'<div class="metric-card"><h3>Total Dividends</h3><div class="value amber">${total_div:,.0f}</div></div>', unsafe_allow_html=True)
        with k4: st.markdown(f'<div class="metric-card"><h3>Total Gain</h3><div class="value green">${total_gain:,.0f} ({(final_val/total_inv-1)*100:.0f}%)</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Chart
        fig_drip = go.Figure()
        fig_drip.add_trace(go.Scatter(x=result_df["Year"], y=result_df["Portfolio Value"],
            name="Portfolio Value", line=dict(color="#3fb950", width=2.5), fill="tozeroy",
            fillcolor="rgba(63,185,80,0.08)"))
        fig_drip.add_trace(go.Scatter(x=result_df["Year"], y=result_df["Total Invested"],
            name="Total Invested", line=dict(color="#388bfd", width=2, dash="dot")))
        fig_drip.add_trace(go.Bar(x=result_df["Year"], y=result_df["Total Dividends"],
            name="Cumulative Dividends", marker_color="#d29922", opacity=0.5))
        fig_drip.update_layout(
            title=f"{'DRIP' if drip_on else 'No DRIP'} — {years}-Year Projection",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3",
            xaxis=dict(title="Year", gridcolor="#21262d"),
            yaxis=dict(title="Value ($)", gridcolor="#21262d", tickprefix="$"),
            height=400, legend=dict(orientation="h", y=1.05),
            margin=dict(l=10,r=10,t=50,b=20),
        )
        st.plotly_chart(fig_drip, use_container_width=True)

        st.markdown("#### Year-by-Year Breakdown")
        styled_d = result_df.style.format({
            "Portfolio Value": "${:,.0f}",
            "Total Invested":  "${:,.0f}",
            "Total Dividends": "${:,.0f}",
            "Gain":            "${:,.0f}",
            "Gain %":          "{:.1f}%",
        })
        st.dataframe(styled_d, use_container_width=True, hide_index=True)

        # DRIP vs No-DRIP comparison
        if drip_on:
            st.info(f"💡 By reinvesting dividends, your portfolio reaches **${final_val:,.0f}** — "
                    f"dividends contributed **${total_div:,.0f}** to your final wealth through compounding.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

with tabs[10]:
    st.markdown("### ⏱ Strategy Backtester")
    st.caption("Test the Apex Score strategy on historical data. Did it work? By how much?")


    ba1, ba2 = st.columns(2)
    with ba1:
        bt_mode       = st.radio("Mode", ["Single ticker", "Full watchlist"], horizontal=True)
        bt_ticker_in  = st.text_input("Ticker (single mode)", value="NVDA", key="bt_tk").upper().strip()
        bt_start      = st.date_input("Start Date", value=datetime(2021,1,1))
        bt_end        = st.date_input("End Date",   value=datetime(2024,12,31))
    with ba2:
        bt_entry_score = st.slider("Entry Score Threshold", 20, 80, 40, 5)
        bt_hold_days   = st.slider("Max Hold Days", 10, 120, 60, 5)
        st.markdown("""
        **Entry:** Apex Score ≥ threshold + Stage 2 confirmed
        **Exit:** Price closes below 50 MA or max hold days reached
        """)

    st.markdown("##### 🔬 Optional New Signal Filters")
    bf1, bf2, bf3, bf4 = st.columns(4)
    with bf1: bt_req_of    = st.checkbox("Require Persistent OF Bias",   value=False)
    with bf2: bt_req_vwap  = st.checkbox("Require Price Above VWAP",     value=False)
    with bf3: bt_req_hh_hl = st.checkbox("Require HH/HL Structure",      value=False)
    with bf4: bt_req_pa    = st.checkbox("Require PA Pattern (Engulf/Context)", value=False)
    st.caption("Tick these to test how much each new signal improves results vs base strategy alone.")

    bt_btn = st.button("⏱ Run Backtest", use_container_width=True)

    if bt_btn:
        start_str = bt_start.strftime("%Y-%m-%d")
        end_str   = bt_end.strftime("%Y-%m-%d")

        with st.spinner("Running backtest… (may take 1–2 min for full watchlist)"):
            if bt_mode == "Single ticker":
                bt_result = backtest_ticker(bt_ticker_in, start_str, end_str,
                                             bt_entry_score, True, bt_hold_days,
                                             bt_req_of, bt_req_vwap, bt_req_hh_hl, bt_req_pa)
                if "error" in bt_result:
                    st.error(bt_result["error"])
                elif not bt_result.get("trades"):
                    st.warning("No trades triggered. Try lowering the entry score or widening the date range.")
                else:
                    s = bt_result["summary"]
                    trades_df = pd.DataFrame(bt_result["trades"])

                    bk1,bk2,bk3,bk4,bk5 = st.columns(5)
                    with bk1: st.markdown(f'<div class="metric-card"><h3>Total Trades</h3><div class="value white">{s["total_trades"]}</div></div>', unsafe_allow_html=True)
                    wc = "green" if s["win_rate_%"] >= 50 else "red"
                    with bk2: st.markdown(f'<div class="metric-card"><h3>Win Rate</h3><div class="value {wc}">{s["win_rate_%"]}%</div></div>', unsafe_allow_html=True)
                    ac = "green" if s["avg_return_%"] >= 0 else "red"
                    with bk3: st.markdown(f'<div class="metric-card"><h3>Avg Return</h3><div class="value {ac}">{s["avg_return_%"]:+.1f}%</div></div>', unsafe_allow_html=True)
                    with bk4: st.markdown(f'<div class="metric-card"><h3>Best Trade</h3><div class="value green">{s["best_trade_%"]:+.1f}%</div></div>', unsafe_allow_html=True)
                    with bk5: st.markdown(f'<div class="metric-card"><h3>Worst Trade</h3><div class="value red">{s["worst_trade_%"]:+.1f}%</div></div>', unsafe_allow_html=True)

                    st.markdown("---")

                    # Equity curve
                    trades_df["cumulative_return_%"] = (1 + trades_df["return_%"] / 100).cumprod() * 100 - 100
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(
                        x=trades_df["exit_date"], y=trades_df["cumulative_return_%"],
                        name="Cumulative Return", line=dict(color="#3fb950", width=2),
                        fill="tozeroy", fillcolor="rgba(63,185,80,0.08)"
                    ))
                    fig_bt.add_hline(y=0, line_color="#30363d")
                    fig_bt.update_layout(
                        title=f"{bt_ticker_in} — Venu Strategy Backtest ({start_str} to {end_str})",
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3",
                        xaxis=dict(gridcolor="#21262d"),
                        yaxis=dict(gridcolor="#21262d", ticksuffix="%"),
                        height=380, margin=dict(l=10,r=10,t=40,b=20),
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)

                    # Individual trades
                    st.markdown("#### Trade Log")
                    def _color_ret_bt(v):
                        try: return "color:#3fb950" if float(v)>0 else "color:#f85149"
                        except: return ""
                    styled_bt = trades_df.style \
                        .map(_color_ret_bt, subset=["return_%"]) \
                        .format({"entry_price":"${:.2f}","exit_price":"${:.2f}",
                                 "return_%":"{:+.1f}%"})
                    st.dataframe(styled_bt, use_container_width=True, hide_index=True)

            else:
                cfg_bt = load_config("config.yaml")
                tix_bt = list(set(t for th in cfg_bt["us_themes"].values() for t in th))[:15]
                trades_df, agg = backtest_portfolio(tix_bt, start_str, end_str,
                                                     bt_entry_score, bt_hold_days,
                                                     bt_req_of, bt_req_vwap,
                                                     bt_req_hh_hl, bt_req_pa)
                if trades_df.empty:
                    st.warning("No trades triggered across the watchlist.")
                else:
                    bk1,bk2,bk3,bk4 = st.columns(4)
                    with bk1: st.markdown(f'<div class="metric-card"><h3>Total Trades</h3><div class="value white">{agg["total_trades"]}</div></div>', unsafe_allow_html=True)
                    wc = "green" if agg["win_rate_%"] >= 50 else "red"
                    with bk2: st.markdown(f'<div class="metric-card"><h3>Win Rate</h3><div class="value {wc}">{agg["win_rate_%"]}%</div></div>', unsafe_allow_html=True)
                    ac = "green" if agg["avg_return_%"] >= 0 else "red"
                    with bk3: st.markdown(f'<div class="metric-card"><h3>Avg Return/Trade</h3><div class="value {ac}">{agg["avg_return_%"]:+.1f}%</div></div>', unsafe_allow_html=True)
                    with bk4: st.markdown(f'<div class="metric-card"><h3>Best Trade</h3><div class="value green">{agg["best_trade"]}</div></div>', unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("#### All Trades")
                    st.dataframe(trades_df, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("Configure your backtest above and click **⏱ Run Backtest**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 12 — RISK CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

with tabs[11]:
    st.markdown("### ⚖️ Position Size & Risk Calculator")
    st.caption("Never risk more than 1–2% of your account on any single trade. This calculator tells you exactly how many shares to buy.")


    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("#### Trade Parameters")
        rc_account  = st.number_input("Account Size ($)", min_value=100.0, value=10000.0, step=500.0)
        rc_risk_pct = st.slider("Max Risk per Trade (%)", 0.25, 5.0, 1.0, 0.25)
        rc_entry    = st.number_input("Entry Price ($)", min_value=0.01, value=100.0, step=0.01)
        rc_stop     = st.number_input("Stop Loss Price ($)", min_value=0.01, value=93.0, step=0.01)
        rc_target   = st.number_input("Profit Target ($ — optional, 0 to skip)", min_value=0.0, value=115.0, step=0.01)
        rc_comm     = st.number_input("Commission per trade ($)", min_value=0.0, value=0.0, step=0.5)

    with rc2:
        st.markdown("#### What is this?")
        st.markdown("""
        **The #1 rule of trading:** Control how much you lose, not just how much you gain.

        This calculator uses the formula:
        > **Shares = (Account × Risk%) ÷ (Entry − Stop)**

        **Example:** $10,000 account, 1% risk, entry $100, stop $95:
        > Shares = ($10,000 × 1%) ÷ ($100 − $95) = $100 ÷ $5 = **20 shares**

        If the trade hits your stop, you lose exactly $100 — 1% of your account.

        **Reward:Risk (R:R):**
        - Minimum acceptable: **2:1** (risk $1 to make $2)
        - Ideal: **3:1 or higher**
        """)

    calc_r_btn = st.button("⚖️ Calculate Position", use_container_width=True)

    if calc_r_btn:
        if rc_stop >= rc_entry:
            st.error("Stop loss must be BELOW entry price for a long trade.")
        else:
            try:
                setup = TradeSetup(
                    account_size  = rc_account,
                    risk_pct      = rc_risk_pct,
                    entry_price   = rc_entry,
                    stop_price    = rc_stop,
                    target_price  = rc_target if rc_target > 0 else None,
                    commission    = rc_comm,
                )
                result = calculate_position(setup)

                # Warning banner
                if result.warning:
                    st.markdown(f'<div class="warn-box">⚠️ {result.warning}</div>', unsafe_allow_html=True)
                    st.markdown("")

                # Main result
                rr_color = "#3fb950" if (result.reward_risk or 0) >= 2 else "#f85149"
                st.markdown(f"""
                <div style="background:#161b22;border:2px solid #3fb950;border-radius:12px;
                            padding:24px;margin-bottom:20px;text-align:center;">
                  <div style="font-size:0.85rem;color:#8b949e;text-transform:uppercase;letter-spacing:.1em;">
                    Buy Exactly
                  </div>
                  <div style="font-size:4rem;font-weight:900;color:#3fb950;line-height:1.1;">
                    {result.shares} shares
                  </div>
                  <div style="font-size:1rem;color:#e6edf3;">
                    @ ${rc_entry:.2f} = <b>${result.position_size:,.2f}</b> total position
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # KPI row
                k1,k2,k3,k4,k5 = st.columns(5)
                with k1: st.markdown(f'<div class="metric-card"><h3>Max Loss $</h3><div class="value red">${result.risk_per_trade:,.2f}</div></div>', unsafe_allow_html=True)
                with k2: st.markdown(f'<div class="metric-card"><h3>Risk % of Account</h3><div class="value amber">{result.risk_pct_actual:.2f}%</div></div>', unsafe_allow_html=True)
                with k3: st.markdown(f'<div class="metric-card"><h3>Stop Distance</h3><div class="value white">${result.stop_distance:.2f} ({result.stop_pct:.1f}%)</div></div>', unsafe_allow_html=True)
                with k4:
                    rr_val = f"{result.reward_risk:.1f}:1" if result.reward_risk else "–"
                    st.markdown(f'<div class="metric-card"><h3>Reward:Risk</h3><div class="value" style="color:{rr_color}">{rr_val}</div></div>', unsafe_allow_html=True)
                with k5:
                    gain_val = f"+${result.target_gain:,.0f} ({result.target_gain_pct:.1f}%)" if result.target_gain else "–"
                    st.markdown(f'<div class="metric-card"><h3>Target Gain</h3><div class="value green">{gain_val}</div></div>', unsafe_allow_html=True)

                st.markdown("---")

                # Pyramiding plan
                st.markdown("#### 📐 Pyramiding Plan (Adding to a Winner)")
                st.caption("How to add to your position as the stock moves in your favour.")
                pyr_add_pct = st.slider("Add when price rises by (%)", 3, 15, 5, 1)
                pyr_adds    = st.slider("Number of add-ons", 1, 3, 2, 1)

                pyr_plan = pyramiding_plan(setup, add_pct=pyr_add_pct, n_adds=pyr_adds)
                pyr_df   = pd.DataFrame(pyr_plan)
                styled_pyr = pyr_df.style.format({
                    "price":         "${:.2f}",
                    "position_$":    "${:,.2f}",
                    "cumulative_$":  "${:,.2f}",
                })
                st.dataframe(styled_pyr, use_container_width=True, hide_index=True)

                st.markdown(f"""
                ---
                💡 **Breakeven price** (after commissions): **${result.breakeven_price:.4f}**
                
                Set your stop at **${rc_stop:.2f}** immediately after your order fills.
                If {rc_ticker_in if 'rc_ticker_in' in dir() else 'the stock'} hits that level, exit without hesitation — the math protects your account.
                """)

            except ValueError as e:
                st.error(str(e))

    else:
        # Show example
        st.markdown("""
        ---
        #### 💡 Quick Example
        | Parameter | Value |
        |---|---|
        | Account | $10,000 |
        | Risk per trade | 1% = $100 max loss |
        | Entry price | $50.00 |
        | Stop loss | $47.50 (5% below entry) |
        | **Shares to buy** | **$100 ÷ $2.50 = 40 shares** |
        | Position size | 40 × $50 = $2,000 (20% of account) |

        If stop is hit → lose $100 (1%). If target at $57 is hit → gain $280 (2.8:1 R:R) ✅
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 13 — AI DAILY BRIEFING
# ══════════════════════════════════════════════════════════════════════════════

with tabs[12]:
    st.markdown("### 🤖 AI Daily Briefing")
    st.caption("Claude reads your scan results and writes a plain-English morning briefing — like having a research analyst on call.")


    ab1, ab2 = st.columns([3, 1])
    with ab1:
        st.markdown("""
        The AI briefing analyses your latest scan results and tells you:
        - Which setups matter most today and why
        - Which themes have momentum
        - Active breakouts and what to do
        - A specific risk reminder based on today's data
        """)
    with ab2:
        gen_btn  = st.button("🤖 Generate Briefing", use_container_width=True)
        load_btn_ai = st.button("📂 Load Last Briefing", use_container_width=True)

    st.markdown("---")

    briefing_text = ""

    if gen_btn:
        if df.empty:
            st.warning("No scan data loaded. Run a Live Scan or Load Last Report first.")
        else:
            with st.spinner("Claude is reading your scan results and writing the briefing…"):
                try:
                    sec_df = sector_performance()
                except:
                    sec_df = pd.DataFrame()
                if generate_briefing is None:
                    st.error("AI Briefing module not loaded. Check that modules/ai_briefing.py exists in your repo.")
                    briefing_text = ""
                else:
                    try:
                        briefing_text = generate_briefing(df, sec_df, save=True)
                    except TypeError:
                        try:
                            briefing_text = generate_briefing(df, save=True)
                        except Exception as _be:
                            st.error(f"Briefing error: {_be}")
                            briefing_text = ""
                    except Exception as _be:
                        st.error(f"Briefing error: {_be}")
                        briefing_text = ""
            st.success("Briefing generated!")

    elif load_btn_ai:
        briefing_text = load_latest_briefing()
        if not briefing_text:
            st.info("No saved briefings yet. Generate one first.")

    if briefing_text:
        # Display as a styled card
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                    padding:28px 32px;line-height:1.8;font-size:0.97rem;">
          <div style="color:#8b949e;font-size:0.75rem;text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:16px;">
            📡 ApexScan AI — {datetime.now().strftime("%A, %B %d %Y")}
          </div>
          {briefing_text.replace(chr(10), '<br>').replace('**','<b>').replace('**','</b>')}
        </div>
        """, unsafe_allow_html=True)

        # Send to Telegram option
        st.markdown("---")
        if st.button("📲 Send to Telegram"):
            settings = load_alert_settings()
            if not settings.get("telegram_token"):
                st.warning("Configure Telegram in the 🔔 Alert Settings tab first.")
            else:
                msg = build_daily_briefing_alert(briefing_text)
                result = dispatch_alert(settings, msg, "ApexScan Daily Briefing")
                if result.get("telegram"):
                    st.success("Briefing sent to Telegram!")
                else:
                    st.error("Failed to send. Check your Telegram settings.")

        # Download
        st.download_button(
            "⬇ Download Briefing",
            briefing_text,
            file_name=f"apexscan_briefing_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
        )

    elif not gen_btn and not load_btn_ai:
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                    padding:32px;text-align:center;color:#8b949e;">
          <div style="font-size:2rem;margin-bottom:8px;">🤖</div>
          <div>Run a <b>Live Scan</b> first, then click <b>Generate Briefing</b></div>
          <div style="font-size:0.85rem;margin-top:8px;">
            The AI reads your results and writes a personalised morning briefing in seconds
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 14 — WATCHLIST MANAGER
# ══════════════════════════════════════════════════════════════════════════════

with tabs[13]:
    st.markdown("### 📋 Watchlist Manager")
    st.caption("Save named watchlists and scan each one independently. Your lists persist between sessions.")

    from scanner import analyze_stock as _analyze_stock

    wls = load_watchlists()

    wl1, wl2 = st.columns([1, 2])

    with wl1:
        st.markdown("#### Your Watchlists")

        # Create new list
        with st.expander("➕ Create New List"):
            new_list_name = st.text_input("List name", placeholder="e.g. Earnings Plays", key="new_wl_name")
            if st.button("Create") and new_list_name:
                wls = create_list(wls, new_list_name)
                save_watchlists(wls)
                st.success(f"Created '{new_list_name}'")
                st.rerun()

        # Select active list
        list_names = list(wls.keys())
        active_list = st.selectbox("Select watchlist", list_names, key="active_wl")

        if active_list:
            tickers_in_list = wls.get(active_list, [])

            # Add ticker
            wl_add_col, wl_btn_col = st.columns([3, 1])
            with wl_add_col:
                new_tk_wl = st.text_input("Add ticker", placeholder="e.g. HIMS", key="wl_add_tk").upper().strip()
            with wl_btn_col:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Add", key="wl_add_btn") and new_tk_wl:
                    wls = add_ticker(wls, active_list, new_tk_wl)
                    save_watchlists(wls)
                    st.rerun()

            # Import from comma list
            with st.expander("📥 Import tickers (comma separated)"):
                import_str = st.text_area("Paste tickers", placeholder="NVDA, AAPL, TSLA, MSFT", key="wl_import")
                if st.button("Import") and import_str:
                    wls = import_tickers(wls, active_list, import_str)
                    save_watchlists(wls)
                    st.success(f"Imported to '{active_list}'")
                    st.rerun()

            st.markdown(f"**{active_list}** — {len(tickers_in_list)} tickers")

            # Show tickers with remove buttons
            if tickers_in_list:
                for tk in tickers_in_list:
                    tc1, tc2 = st.columns([4, 1])
                    with tc1: st.markdown(f"• **{tk}**")
                    with tc2:
                        if st.button("✕", key=f"rm_{tk}_{active_list}"):
                            wls = remove_ticker(wls, active_list, tk)
                            save_watchlists(wls)
                            st.rerun()

                # Export
                st.markdown("---")
                export_str = export_watchlist(wls, active_list)
                st.text_area("Export (copy this)", value=export_str, height=60, key="wl_export")

                # Delete list (not default ones)
                if active_list not in ["High Conviction", "Monitoring", "Earnings Soon", "Swing Trades"]:
                    if st.button(f"🗑 Delete '{active_list}'", type="secondary"):
                        wls = delete_list(wls, active_list)
                        save_watchlists(wls)
                        st.rerun()
            else:
                st.info("No tickers yet. Add some above or import a comma-separated list.")

    with wl2:
        st.markdown("#### Scan This Watchlist")

        if active_list and wls.get(active_list):
            scan_wl_btn = st.button(f"🚀 Scan '{active_list}'", use_container_width=True)

            if scan_wl_btn:
                tix = wls[active_list]
                cfg_wl = load_config("config.yaml")
                with st.spinner(f"Scanning {len(tix)} tickers in '{active_list}'…"):
                    wl_df = scan_watchlist(active_list, tix, cfg_wl, _analyze_stock)

                if wl_df.empty:
                    st.warning("No results. Tickers may not have enough history or didn't pass filters.")
                else:
                    st.success(f"{len(wl_df)} results from '{active_list}'")

                    # Quick KPIs
                    wk1, wk2, wk3 = st.columns(3)
                    with wk1:
                        top_s = pd.to_numeric(wl_df["apex_score"], errors="coerce").max()
                        st.markdown(f'<div class="metric-card"><h3>Top Score</h3><div class="value green">{top_s:.0f}</div></div>', unsafe_allow_html=True)
                    with wk2:
                        bo_wl = int(wl_df.get("breaking_out", pd.Series([False]*len(wl_df))).sum()) if "breaking_out" in wl_df.columns else 0
                        st.markdown(f'<div class="metric-card"><h3>Breakouts</h3><div class="value amber">{bo_wl}</div></div>', unsafe_allow_html=True)
                    with wk3:
                        s2_wl = int(wl_df.get("stage", pd.Series([""]*len(wl_df))).str.contains("2 ✅", na=False).sum()) if "stage" in wl_df.columns else 0
                        st.markdown(f'<div class="metric-card"><h3>Stage 2</h3><div class="value blue">{s2_wl}</div></div>', unsafe_allow_html=True)

                    st.markdown("---")

                    # Results table
                    wl_show = [c for c in ["ticker","price","stage","perf_3m_%","rs_3m",
                                           "vol_surge_x","pattern","apex_score"] if c in wl_df.columns]
                    wl_disp = wl_df[wl_show].copy()
                    for col in ["apex_score","perf_3m_%","rs_3m"]:
                        if col in wl_disp.columns:
                            wl_disp[col] = pd.to_numeric(wl_disp[col], errors="coerce")

                    styled_wl = wl_disp.style \
                        .map(color_score, subset=["apex_score"]) \
                        .map(color_perf,  subset=["perf_3m_%"] if "perf_3m_%" in wl_disp.columns else []) \
                        .map(color_rs,    subset=["rs_3m"] if "rs_3m" in wl_disp.columns else []) \
                        .format({
                            "price":       "${:.2f}",
                            "perf_3m_%":   pct_fmt,
                            "rs_3m":       lambda v: f"{v:.0f}" if pd.notna(v) and v != 0 else "–",
                            "vol_surge_x": "{:.1f}x",
                            "apex_score":  "{:.0f}",
                        }, na_rep="–")
                    st.dataframe(styled_wl, use_container_width=True, height=420)

                    # Add top results to another watchlist
                    st.markdown("---")
                    st.markdown("**Promote top results to another list:**")
                    promote_target = st.selectbox("Target list", [l for l in list_names if l != active_list], key="wl_promote_target")
                    promote_n = st.slider("Promote top N", 1, min(10, len(wl_df)), 3, key="wl_promote_n")
                    if st.button("Promote"):
                        top_tix = wl_df["ticker"].head(promote_n).tolist()
                        for t in top_tix:
                            wls = add_ticker(wls, promote_target, t)
                        save_watchlists(wls)
                        st.success(f"Added {top_tix} → '{promote_target}'")
        else:
            st.info("Add tickers to your watchlist on the left, then scan here.")

        # ── Summary across all watchlists ─────────────────────────────────
        st.markdown("---")
        st.markdown("#### All Watchlists Overview")
        summary_rows = []
        for name, tix in wls.items():
            summary_rows.append({"Watchlist": name, "Tickers": len(tix),
                                  "Contents": ", ".join(tix[:5]) + ("…" if len(tix) > 5 else "")})
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 15 — ALERT SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[14]:
    st.markdown("### 🔔 Alert Settings")
    st.caption("Configure Telegram and Email alerts. Both work simultaneously — alerts fire automatically after every scan.")

    settings = load_alert_settings()

    # ── TELEGRAM ──────────────────────────────────────────────────────────
    st.markdown("#### 📲 Telegram Alerts")
    tg_col1, tg_col2 = st.columns(2)

    with tg_col1:
        tg_token = st.text_input(
            "Bot Token",
            value=settings.get("telegram_token", ""),
            type="password",
            placeholder="8972969518:AAGAmdRD...",
            key="tg_token"
        )
        tg_chat_id = st.text_input(
            "Chat ID",
            value=settings.get("telegram_chat_id", ""),
            placeholder="Find this at step 4 below",
            key="tg_chat"
        )

    with tg_col2:
        st.markdown("""
        **Your bot (ApexV8bot) is already created ✅**

        Now get your Chat ID:
        1. Open Telegram → find **ApexV8bot**
        2. Send it any message (e.g. "hello")
        3. Visit this URL in your browser:
           `https://api.telegram.org/bot8972969518:AAGAmdRDtyyXdv4pzqJc6og4YqyCsnBxXUg/getUpdates`
        4. Look for `"chat":{"id":` — the number after it is your Chat ID
        5. Paste that number above and save
        """)

    tc1, tc2 = st.columns(2)
    with tc1:
        if st.button("📲 Test Telegram", use_container_width=True):
            if tg_token and tg_chat_id:
                ok = test_telegram(tg_token, tg_chat_id)
                st.success("✅ Telegram test sent! Check your bot.") if ok else st.error("❌ Failed — double check your Chat ID.")
            else:
                st.warning("Enter both Bot Token and Chat ID first.")

    st.markdown("---")

    # ── EMAIL ─────────────────────────────────────────────────────────────
    st.markdown("#### 📧 Email Alerts (Gmail)")
    em_col1, em_col2 = st.columns(2)

    with em_col1:
        em_from = st.text_input(
            "Your Gmail address",
            value=settings.get("email_from", ""),
            placeholder="yourname@gmail.com",
            key="em_from"
        )
        em_pass = st.text_input(
            "Gmail App Password",
            value=settings.get("email_password", ""),
            type="password",
            placeholder="xxxx xxxx xxxx xxxx",
            key="em_pass"
        )
        em_to = st.text_input(
            "Send alerts to (can be same address)",
            value=settings.get("email_to", ""),
            placeholder="yourname@gmail.com",
            key="em_to"
        )

    with em_col2:
        st.markdown("""
        **How to get a Gmail App Password:**
        1. Go to **myaccount.google.com**
        2. Click **Security** on the left
        3. Enable **2-Step Verification** if not already on
        4. Search for **App Passwords** in the search bar
        5. Select **Mail** → **Generate**
        6. Copy the 16-character password (e.g. `abcd efgh ijkl mnop`)
        7. Paste it above — spaces are fine

        ⚠️ Use an App Password, NOT your regular Gmail password
        """)

    ec1, ec2 = st.columns(2)
    with ec1:
        if st.button("📧 Test Email", use_container_width=True):
            if em_from and em_pass and em_to:
                ok = send_email(
                    em_from, em_pass, em_to,
                    "ApexScan — Test Alert",
                    "✅ Your ApexScan email alerts are working correctly!"
                )
                st.success("✅ Test email sent! Check your inbox.") if ok else st.error("❌ Failed — check your Gmail and App Password.")
            else:
                st.warning("Fill in all three email fields first.")

    st.markdown("---")

    # ── PREFERENCES ───────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Alert Preferences")
    pr1, pr2, pr3, pr4 = st.columns(4)
    with pr1:
        alerts_on      = st.toggle("Enable All Alerts", value=settings.get("alerts_enabled", False))
        alert_breakout = st.checkbox("Breakout Alerts",      value=settings.get("alert_breakouts", True))
        alert_stop     = st.checkbox("Stop Loss Breach",     value=settings.get("alert_stop_breach", True))
    with pr2:
        alert_earn     = st.checkbox("Earnings Warnings",    value=settings.get("alert_earnings", True))
        alert_sfp      = st.checkbox("SFP Setup Alerts",     value=settings.get("alert_sfp_setup", True),
                                      help="Alert when a Swing Failure Pattern is detected — trap setup")
    with pr3:
        alert_of       = st.checkbox("Persistent Flow Alert",value=settings.get("alert_persistent_flow", True),
                                      help="Alert when Strong Bullish order flow persistence detected")
        alert_vwap_imb = st.checkbox("VWAP Imbalance Alert", value=settings.get("alert_vwap_imbalance", True),
                                      help="Alert when price is extended above or below VWAP")
    with pr4:
        min_score_alert = st.slider("Min Score to Alert", 30, 90,
                                     settings.get("min_score_alert", 60), 5)
        st.caption("Only alert on setups scoring above this threshold")

    st.markdown("---")

    if st.button("💾 Save All Settings", use_container_width=True, type="primary"):
        new_settings = {
            "telegram_token":        tg_token,
            "telegram_chat_id":      tg_chat_id,
            "email_from":            em_from,
            "email_password":        em_pass,
            "email_to":              em_to,
            "alerts_enabled":        alerts_on,
            "alert_breakouts":       alert_breakout,
            "alert_stop_breach":     alert_stop,
            "alert_earnings":        alert_earn,
            "alert_sfp_setup":       alert_sfp,
            "alert_persistent_flow": alert_of,
            "alert_vwap_imbalance":  alert_vwap_imb,
            "min_score_alert":       min_score_alert,
        }
        save_alert_settings(new_settings)
        st.success("✅ Settings saved! Both Telegram and Email alerts are configured.")

    # ── MANUAL TEST ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📨 Send a Manual Test Alert")
    st.caption("Tests both Telegram and Email simultaneously")
    test_msg = st.text_area(
        "Message",
        value="🔔 ApexScan Alert — Your breakout scanner is live and watching the market for you.",
        key="manual_alert_msg"
    )
    if st.button("Send to All Channels", use_container_width=True):
        cur = load_alert_settings()
        result = dispatch_alert(cur, test_msg, "ApexScan Alert")
        sent = []
        if result.get("telegram"): sent.append("Telegram ✅")
        if result.get("email"):    sent.append("Email ✅")
        if sent:
            st.success(f"Sent via: {', '.join(sent)}")
        elif not cur.get("alerts_enabled"):
            st.warning("Alerts are disabled — toggle 'Enable All Alerts' on and save first.")
        else:
            st.error("Nothing sent. Check your credentials and make sure settings are saved.")

    st.markdown("""
    ---
    **When alerts fire automatically:**
    - 🚀 **Breakout** — after every Live Scan, for high-scoring breakout setups
    - 🎯 **SFP Setup** — when a Swing Failure Pattern trap is detected
    - 📈 **Persistent Flow** — when Strong Bullish order flow is confirmed
    - 💧 **VWAP Imbalance** — when price is extended far from VWAP fair value
    - 💼 **Stop Breach** — after Portfolio Tracker loads, if holding drops below 50MA
    - 📅 **Earnings** — when you fetch the Earnings Calendar for upcoming reports
    - 🤖 **AI Briefing** — option to send morning briefing to Telegram after generation

    Telegram and Email both fire simultaneously when configured.
    Min Score filter applies to all scan-based alerts.
    """)




# ══════════════════════════════════════════════════════════════════════════════
# TAB 16 — INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════

with tabs[15]:
    st.markdown("### 🧠 Scan Interpretation")
    st.caption("Plain-English breakdown of your scan results across all four signal views.")

    if df.empty:
        st.info("Run a Live Scan or Load Last Report first, then come here for the interpretation.")
    else:
        # ── Ticker selector ───────────────────────────────────────────────
        interp_mode = st.radio(
            "Interpret",
            ["Full Scan Summary", "Single Ticker Deep Read"],
            horizontal=True
        )

        if interp_mode == "Single Ticker Deep Read":
            interp_ticker = st.selectbox("Choose ticker", df["ticker"].tolist(), key="interp_tk")
            interp_df = df[df["ticker"] == interp_ticker]
        else:
            interp_ticker = None
            interp_df = df.copy()

        st.markdown("---")

        # ════════════════════════════════════════════════════════════════
        # HELPER: colour-coded signal pill
        # ════════════════════════════════════════════════════════════════
        def pill(text, color="#3fb950"):
            return (f'<span style="background:{color}22;color:{color};'
                    f'border:1px solid {color};border-radius:4px;'
                    f'padding:2px 8px;font-size:0.8rem;font-weight:600;">'
                    f'{text}</span>')

        def green(t):  return pill(t, "#3fb950")
        def amber(t):  return pill(t, "#d29922")
        def red(t):    return pill(t, "#f85149")
        def blue(t):   return pill(t, "#388bfd")
        def purple(t): return pill(t, "#c084fc")

        # ════════════════════════════════════════════════════════════════
        # VIEW 1 — STANDARD
        # ════════════════════════════════════════════════════════════════
        with st.expander("📋 Standard View — Momentum & Stage", expanded=True):
            if interp_mode == "Full Scan Summary":
                total      = len(interp_df)
                stage2     = interp_df["stage"].str.contains("2 ✅", na=False).sum() if "stage" in interp_df.columns else 0
                stage4     = interp_df["stage"].str.contains("4 🔴", na=False).sum() if "stage" in interp_df.columns else 0
                avg_3m     = pd.to_numeric(interp_df.get("perf_3m_%", pd.Series()), errors="coerce").mean()
                avg_rs     = pd.to_numeric(interp_df.get("rs_3m", pd.Series()), errors="coerce").mean()
                breakouts  = interp_df.get("breaking_out", pd.Series([False]*total)).sum()
                top3       = interp_df.head(3)["ticker"].tolist()
                near_high  = interp_df.get("near_52wh", pd.Series([False]*total)).sum()

                # Market health verdict
                if stage2 / max(total,1) >= 0.6:
                    health = green("HEALTHY MARKET")
                    health_msg = "Most setups are in Stage 2 uptrends — conditions favour the bull side."
                elif stage2 / max(total,1) >= 0.4:
                    health = amber("MIXED CONDITIONS")
                    health_msg = "Market is split — be selective, stick to Stage 2 stocks only."
                else:
                    health = red("WEAK BREADTH")
                    health_msg = "Few Stage 2 setups — reduce position size and wait for clarity."

                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px 24px;margin-bottom:16px;">
                  <div style="font-size:1.1rem;font-weight:700;margin-bottom:12px;">
                    Market Condition: {health}
                  </div>
                  <p style="color:#e6edf3;margin:0 0 12px 0;">{health_msg}</p>
                  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                    <div><span style="color:#8b949e;font-size:0.75rem;">SETUPS PASSING</span><br>
                         <span style="font-size:1.4rem;font-weight:700;">{total}</span></div>
                    <div><span style="color:#8b949e;font-size:0.75rem;">STAGE 2 (BUY ZONE)</span><br>
                         <span style="font-size:1.4rem;font-weight:700;color:#3fb950;">{stage2}</span></div>
                    <div><span style="color:#8b949e;font-size:0.75rem;">AVG 3M RETURN</span><br>
                         <span style="font-size:1.4rem;font-weight:700;color:{'#3fb950' if avg_3m>0 else '#f85149'};">{avg_3m:+.1f}%</span></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Per-stock standard interpretation
                st.markdown("#### Stock-by-Stock Reading")
                for _, row in interp_df.head(15).iterrows():
                    tk    = row["ticker"]
                    score = float(row.get("apex_score", 0))
                    stage = str(row.get("stage","?"))
                    p3m   = float(row.get("perf_3m_%", 0) or 0)
                    rs    = float(row.get("rs_3m", 0) or 0)
                    patt  = str(row.get("pattern","–"))
                    brk   = bool(row.get("breaking_out", False))

                    # Score badge
                    score_badge = green(f"Score {score:.0f}") if score>=70 else (amber(f"Score {score:.0f}") if score>=40 else red(f"Score {score:.0f}"))
                    stage_badge = green(stage) if "2 ✅" in stage else (amber(stage) if "1 ⏳" in stage else red(stage))

                    # Reading sentences
                    sentences = []
                    if "2 ✅" in stage:
                        sentences.append(f"In a confirmed Stage 2 uptrend — price is above both moving averages with the 50MA rising above the 200MA.")
                    elif "1 ⏳" in stage:
                        sentences.append(f"Still building a base (Stage 1) — not yet buyable, needs to clear the 50MA.")
                    elif "4 🔴" in stage:
                        sentences.append(f"In a Stage 4 downtrend — avoid entirely until structure repairs.")
                    else:
                        sentences.append(f"Mixed stage — structure is unclear, wait for cleaner setup.")

                    if p3m > 30:
                        sentences.append(f"Exceptional 3-month momentum of {p3m:+.1f}% — this is a leading stock in its theme.")
                    elif p3m > 15:
                        sentences.append(f"Solid 3-month return of {p3m:+.1f}% — above the minimum momentum threshold.")
                    else:
                        sentences.append(f"3-month return of {p3m:+.1f}% is below the momentum threshold — needs to accelerate.")

                    if rs > 150:
                        sentences.append(f"RS of {rs:.0f} means it's massively outperforming the S&P 500 — buy leaders like this.")
                    elif rs > 100:
                        sentences.append(f"RS of {rs:.0f} — beating the S&P 500, in line with what you want to own.")
                    elif rs > 70:
                        sentences.append(f"RS of {rs:.0f} — keeping pace with the market but not leading.")
                    else:
                        sentences.append(f"RS of {rs:.0f} is below the market — lagging stocks rarely lead the next move.")

                    if brk:
                        sentences.append(f"Active breakout in progress ({patt}) — this is the highest-priority actionable setup.")
                    elif "Handle" in patt or "Tight" in patt:
                        sentences.append(f"Pattern ({patt}) suggests it's coiling for a move — watch the pivot point.")
                    elif "Deep Correction" in patt:
                        sentences.append(f"Currently in a deep correction ({patt}) — let it base and tighten before considering entry.")

                    st.markdown(
                        f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
                        f'padding:14px 18px;margin:6px 0;">'
                        f'<div style="margin-bottom:8px;">'
                        f'<span style="font-size:1rem;font-weight:700;color:#e6edf3;">{tk}</span>'
                        f'&nbsp;&nbsp;{score_badge}&nbsp;{stage_badge}</div>'
                        f'<ul style="margin:0;padding-left:18px;color:#c9d1d9;font-size:0.88rem;line-height:1.8;">'
                        + "".join(f"<li>{s}</li>" for s in sentences) +
                        f'</ul></div>',
                        unsafe_allow_html=True
                    )

            else:
                # Single ticker standard
                row   = interp_df.iloc[0]
                score = float(row.get("apex_score",0))
                stage = str(row.get("stage","?"))
                p3m   = float(row.get("perf_3m_%",0) or 0)
                p6m   = float(row.get("perf_6m_%",0) or 0)
                rs3   = float(row.get("rs_3m",0) or 0)
                patt  = str(row.get("pattern","–"))
                brk   = bool(row.get("breaking_out",False))
                near  = bool(row.get("near_52wh",False))
                off_h = float(row.get("pct_off_high_%",0) or 0)

                verdict = "Strong Setup" if score>=70 else ("Watch List" if score>=40 else "Not Ready")
                vc      = "#3fb950" if score>=70 else ("#d29922" if score>=40 else "#f85149")

                st.markdown(f"""
                <div style="background:#161b22;border:2px solid {vc};border-radius:12px;padding:24px;margin-bottom:20px;">
                  <div style="font-size:2rem;font-weight:800;color:#e6edf3;">{interp_ticker}</div>
                  <div style="font-size:0.9rem;color:#8b949e;margin:4px 0 16px 0;">Standard Signal Reading</div>
                  <div style="font-size:0.95rem;color:#c9d1d9;line-height:1.9;">
                    {'✅' if '2 ✅' in stage else '❌'} <b>Stage:</b> {stage} —
                    {'Price is above both moving averages and the 50MA is rising above the 200MA. This is the only stage worth holding or buying.' if '2 ✅' in stage else 'Not in a confirmed uptrend. Stage 2 is required before considering entry.'}<br>
                    {'✅' if p3m>15 else '⚠️'} <b>3M Return:</b> {p3m:+.1f}% —
                    {'Strong momentum. The stock has outperformed most of the market over the past quarter.' if p3m>15 else 'Below the 15% momentum threshold. Needs to accelerate before it qualifies as a leading stock.'}<br>
                    {'✅' if p6m>20 else '⚠️'} <b>6M Return:</b> {p6m:+.1f}% —
                    {'Sustained trend over 6 months confirms this is not a one-month wonder.' if p6m>20 else 'Six-month performance is muted — the trend may be early or stalling.'}<br>
                    {'✅' if rs3>100 else '⚠️'} <b>RS Score:</b> {rs3:.0f} —
                    {'Outperforming the S&P 500. You want to own the leaders, not keep up with the index.' if rs3>100 else 'Underperforming or matching the index. Leaders should have RS well above 100.'}<br>
                    {'✅' if near else '⚠️'} <b>52-Week High:</b> {'Within 15% of highs — near the top of its range, which is where breakouts happen.' if near else f'{abs(off_h):.1f}% below its 52-week high — needs to reclaim ground before a breakout is possible.'}<br>
                    {'🚀' if brk else '⏳'} <b>Pattern:</b> {patt} —
                    {'Active breakout in progress. Highest priority actionable setup.' if brk else 'No active breakout yet. Monitor for the pivot point trigger.'}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        # VIEW 2 — ORDER FLOW
        # ════════════════════════════════════════════════════════════════
        with st.expander("🌊 Order Flow Interpretation", expanded=True):
            if "of_bias" not in interp_df.columns:
                st.info("Run a fresh scan to get Order Flow data.")
            else:
                if interp_mode == "Full Scan Summary":
                    of_strong  = interp_df["of_bias"].str.contains("Strong Bullish", na=False).sum()
                    of_bull    = interp_df["of_bias"].str.contains("Bullish", na=False).sum()
                    of_bear    = interp_df["of_bias"].str.contains("Bearish", na=False).sum()
                    avg_ratio  = pd.to_numeric(interp_df.get("of_up_vol_ratio", pd.Series()), errors="coerce").mean()
                    max_consec = pd.to_numeric(interp_df.get("of_consec_up", pd.Series()), errors="coerce").max()

                    of_verdict = green("STRONG INSTITUTIONAL BUYING") if of_strong >= 3 else \
                                 amber("MODERATE BUYING PRESSURE") if of_bull > of_bear else \
                                 red("SELLING PRESSURE DOMINATES")

                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px 24px;margin-bottom:16px;">
                      <div style="font-size:1rem;font-weight:700;margin-bottom:10px;">
                        Order Flow Picture: {of_verdict}
                      </div>
                      <p style="color:#c9d1d9;font-size:0.9rem;margin:0 0 8px 0;">
                        Order flow persistence measures whether institutional money is consistently active on the buy side
                        over multiple sessions — the signature of TWAP/VWAP algorithms splitting large orders to avoid moving the market.
                      </p>
                      <p style="color:#c9d1d9;font-size:0.9rem;margin:0;">
                        <b>{of_strong}</b> stocks show Strong Bullish flow (the highest conviction signal) &nbsp;|&nbsp;
                        <b>{of_bull}</b> total with bullish bias &nbsp;|&nbsp;
                        <b>{of_bear}</b> with bearish bias &nbsp;|&nbsp;
                        Avg up/down vol ratio: <b>{avg_ratio:.2f}x</b> &nbsp;|&nbsp;
                        Longest consecutive up-close streak: <b>{int(max_consec) if pd.notna(max_consec) else 0} days</b>
                      </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Order Flow Reading Per Stock")
                    for _, row in interp_df.head(15).iterrows():
                        tk       = row["ticker"]
                        bias     = str(row.get("of_bias","–"))
                        ratio    = float(row.get("of_up_vol_ratio",1) or 1)
                        bull_pct = float(row.get("of_bullish_days",50) or 50)
                        consec   = int(row.get("of_consec_up",0) or 0)
                        of_sc    = int(row.get("of_score",0) or 0)

                        bias_badge = green(bias) if "Strong Bullish" in bias else \
                                     amber(bias) if "Bullish" in bias else \
                                     red(bias) if "Bearish" in bias else blue(bias)

                        if "Strong Bullish" in bias:
                            reading = (f"Institutional buying is persistent and heavy. "
                                      f"{bull_pct:.0f}% of the last 10 sessions closed up, "
                                      f"with {ratio:.2f}x more volume on up days than down days. "
                                      f"{'Consecutive up closes of ' + str(consec) + ' sessions suggests an active algorithm is working a large buy order.' if consec >= 3 else ''} "
                                      f"This is the pattern left behind when a fund is accumulating a position quietly over time.")
                        elif "Bullish" in bias:
                            reading = (f"Moderate buying pressure detected. {bull_pct:.0f}% of sessions closed up "
                                      f"with {ratio:.2f}x up-volume ratio. "
                                      f"Directional bias is positive but not yet showing the heavy conviction of institutional accumulation.")
                        elif "Bearish" in bias:
                            reading = (f"Selling pressure is present. Only {bull_pct:.0f}% of sessions closed up "
                                      f"and down-day volume is outpacing up-day volume ({ratio:.2f}x). "
                                      f"This pattern can indicate distribution — institutions quietly reducing positions.")
                        else:
                            reading = (f"Flow is neutral — no clear directional conviction from either side over the last 10 sessions. "
                                      f"Wait for a clearer signal before committing capital.")

                        st.markdown(
                            f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
                            f'padding:14px 18px;margin:6px 0;">'
                            f'<div style="margin-bottom:6px;">'
                            f'<span style="font-weight:700;color:#e6edf3;">{tk}</span>&nbsp;&nbsp;'
                            f'{bias_badge}&nbsp;{purple(f"OF Score {of_sc}/8")}</div>'
                            f'<p style="margin:0;color:#c9d1d9;font-size:0.88rem;line-height:1.7;">{reading}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                else:
                    row      = interp_df.iloc[0]
                    bias     = str(row.get("of_bias","–"))
                    ratio    = float(row.get("of_up_vol_ratio",1) or 1)
                    bull_pct = float(row.get("of_bullish_days",50) or 50)
                    consec   = int(row.get("of_consec_up",0) or 0)
                    of_sc    = int(row.get("of_score",0) or 0)

                    bias_color = "#3fb950" if "Bullish" in bias else ("#f85149" if "Bearish" in bias else "#d29922")

                    if "Strong Bullish" in bias:
                        of_summary = (
                            f"The order flow picture for {interp_ticker} is strongly bullish. "
                            f"Over the last 10 sessions, {bull_pct:.0f}% of days closed higher, "
                            f"and volume on up-days was {ratio:.2f}x the volume on down-days. "
                            f"{'A streak of ' + str(consec) + ' consecutive up-closes is a strong sign of an active institutional buyer systematically accumulating shares — typical of a fund using TWAP/VWAP execution.' if consec >= 3 else 'The volume-weighted bias confirms buyers are more active than sellers.'} "
                            f"This is the kind of persistent, quiet accumulation that often precedes a significant price move."
                        )
                    elif "Bullish" in bias:
                        of_summary = (
                            f"{interp_ticker} shows moderate bullish order flow. "
                            f"{bull_pct:.0f}% of the last 10 sessions closed up with a {ratio:.2f}x up-volume ratio. "
                            f"Buying pressure is real but not yet at the level that suggests heavy institutional accumulation. "
                            f"Monitor for the ratio to increase above 1.5x as confirmation."
                        )
                    elif "Bearish" in bias:
                        of_summary = (
                            f"{interp_ticker} is showing bearish order flow. "
                            f"Only {bull_pct:.0f}% of sessions closed up, and selling volume is dominating. "
                            f"This is a warning sign — even if the stock looks good technically, "
                            f"persistent down-day volume can signal that larger players are quietly exiting positions."
                        )
                    else:
                        of_summary = (
                            f"Order flow for {interp_ticker} is neutral — no clear directional bias from either side. "
                            f"The market is in equilibrium on this name. Wait for a directional signal before acting."
                        )

                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid {bias_color};border-radius:10px;padding:20px 24px;">
                      <div style="color:{bias_color};font-weight:700;font-size:1rem;margin-bottom:10px;">
                        {interp_ticker} — Order Flow: {bias} ({of_sc}/8 pts)
                      </div>
                      <p style="color:#c9d1d9;font-size:0.92rem;line-height:1.8;margin:0;">{of_summary}</p>
                      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px;">
                        <div style="background:#0d1117;border-radius:6px;padding:10px;">
                          <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">Up/Down Vol Ratio</div>
                          <div style="font-size:1.3rem;font-weight:700;color:{bias_color};">{ratio:.2f}x</div>
                          <div style="color:#8b949e;font-size:0.75rem;">{'✅ Institutional level' if ratio>=1.5 else '⚠️ Below threshold (1.5x)'}</div>
                        </div>
                        <div style="background:#0d1117;border-radius:6px;padding:10px;">
                          <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">Bullish Sessions (10d)</div>
                          <div style="font-size:1.3rem;font-weight:700;color:{bias_color};">{bull_pct:.0f}%</div>
                          <div style="color:#8b949e;font-size:0.75rem;">{'✅ Persistent buying' if bull_pct>=60 else '⚠️ Needs >60% to qualify'}</div>
                        </div>
                        <div style="background:#0d1117;border-radius:6px;padding:10px;">
                          <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">Consecutive Up Closes</div>
                          <div style="font-size:1.3rem;font-weight:700;color:{bias_color};">{consec} days</div>
                          <div style="color:#8b949e;font-size:0.75rem;">{'🔥 Active algo detected' if consec>=4 else '📊 Normal range'}</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        # VIEW 3 — VWAP & STRUCTURE
        # ════════════════════════════════════════════════════════════════
        with st.expander("💧 VWAP & Market Structure Interpretation", expanded=True):
            if "vwap_position" not in interp_df.columns:
                st.info("Run a fresh scan to get VWAP data.")
            else:
                if interp_mode == "Full Scan Summary":
                    above_vwap = interp_df["vwap_position"].str.contains("Above", na=False).sum()
                    ext_above  = interp_df["vwap_position"].str.contains("Extended Above", na=False).sum()
                    ext_below  = interp_df["vwap_position"].str.contains("Extended Below", na=False).sum()
                    hh_hl_ct   = interp_df.get("ms_hh_hl", pd.Series([False]*len(interp_df))).sum()
                    bos_ct     = interp_df.get("ms_bos", pd.Series([False]*len(interp_df))).sum()
                    rising_vwap= interp_df.get("vwap_slope", pd.Series()).str.contains("Rising", na=False).sum()

                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid #c084fc;border-radius:10px;padding:20px 24px;margin-bottom:16px;">
                      <div style="color:#c084fc;font-weight:700;font-size:1rem;margin-bottom:12px;">
                        VWAP & Auction Market Picture
                      </div>
                      <p style="color:#c9d1d9;font-size:0.9rem;line-height:1.7;margin:0 0 12px 0;">
                        VWAP (Volume Weighted Average Price) is the fairest measure of where the market agreed to transact.
                        Stocks above a rising VWAP have buyers in control and value is being accepted higher.
                        Stocks extended far above VWAP are at risk of mean reversion back to fair value.
                      </p>
                      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                        <div><span style="color:#8b949e;font-size:0.75rem;">ABOVE VWAP</span><br>
                             <span style="font-size:1.4rem;font-weight:700;color:#3fb950;">{above_vwap}</span>
                             <span style="color:#8b949e;font-size:0.8rem;"> of {len(interp_df)}</span></div>
                        <div><span style="color:#8b949e;font-size:0.75rem;">EXTENDED (RISKY)</span><br>
                             <span style="font-size:1.4rem;font-weight:700;color:#d29922;">{ext_above}</span></div>
                        <div><span style="color:#8b949e;font-size:0.75rem;">HH/HL STRUCTURE</span><br>
                             <span style="font-size:1.4rem;font-weight:700;color:#3fb950;">{hh_hl_ct}</span></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### VWAP & Structure Per Stock")
                    for _, row in interp_df.head(15).iterrows():
                        tk       = row["ticker"]
                        vwap_pos = str(row.get("vwap_position","–"))
                        vs_vwap  = float(row.get("vs_vwap_%",0) or 0)
                        slope    = str(row.get("vwap_slope","–"))
                        hh_hl    = bool(row.get("ms_hh_hl",False))
                        bos      = bool(row.get("ms_bos",False))
                        ms       = str(row.get("ms_structure","–"))
                        sh       = row.get("ms_swing_high")
                        sl       = row.get("ms_swing_low")

                        if "Extended Above" in vwap_pos:
                            vwap_read = (f"Trading {vs_vwap:+.1f}% above VWAP fair value — extended. "
                                        f"While momentum is strong, entering here means chasing. "
                                        f"Better to wait for a pullback toward VWAP before adding or initiating.")
                            vc = "#d29922"
                        elif "Above" in vwap_pos and slope == "Rising":
                            vwap_read = (f"Trading {vs_vwap:+.1f}% above a rising VWAP — ideal zone. "
                                        f"Buyers are in control, fair value is moving higher, "
                                        f"and this is where high-quality momentum entries are found.")
                            vc = "#3fb950"
                        elif "Above" in vwap_pos:
                            vwap_read = (f"Above VWAP by {vs_vwap:+.1f}% but VWAP slope is {slope.lower()}. "
                                        f"Position is acceptable but watch for VWAP to start declining, "
                                        f"which would signal a shift in auction control.")
                            vc = "#3fb950"
                        elif "Extended Below" in vwap_pos:
                            vwap_read = (f"Extended {vs_vwap:.1f}% below VWAP — sellers in full control. "
                                        f"Avoid new longs. Only consider watching for a VWAP reclaim on volume "
                                        f"as a potential reversal signal.")
                            vc = "#f85149"
                        else:
                            vwap_read = (f"Below VWAP by {abs(vs_vwap):.1f}%. The auction has not accepted value higher. "
                                        f"Wait for a confirmed close above VWAP before considering a long position.")
                            vc = "#f85149"

                        struct_read = ""
                        if hh_hl:
                            struct_read = f" Market structure confirms Higher Highs and Higher Lows — the uptrend is intact."
                            if sh: struct_read += f" Key resistance to watch: ${sh:.2f}."
                        elif "Bearish" in ms:
                            struct_read = f" Structure shows Lower Highs and Lower Lows — technically in a downtrend. Avoid."
                        else:
                            struct_read = f" Structure is transitional — no clear trend. Wait for HH/HL to establish."

                        if bos and hh_hl:
                            struct_read += f" A recent Break of Structure in an uptrend confirms the move is accelerating."

                        vwap_badge = green("Above VWAP ↑") if "Above" in vwap_pos and "Extended" not in vwap_pos \
                                else amber("Extended ↑") if "Extended Above" in vwap_pos \
                                else red("Below VWAP ↓")
                        ms_badge = green("HH/HL ✅") if hh_hl else red("No HH/HL")

                        st.markdown(
                            f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
                            f'padding:14px 18px;margin:6px 0;">'
                            f'<div style="margin-bottom:6px;">'
                            f'<span style="font-weight:700;color:#e6edf3;">{tk}</span>&nbsp;&nbsp;'
                            f'{vwap_badge}&nbsp;{ms_badge}</div>'
                            f'<p style="margin:0;color:#c9d1d9;font-size:0.88rem;line-height:1.7;">'
                            f'{vwap_read}{struct_read}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                else:
                    row      = interp_df.iloc[0]
                    vwap_val = row.get("vwap")
                    vs_vwap  = float(row.get("vs_vwap_%",0) or 0)
                    vwap_pos = str(row.get("vwap_position","–"))
                    slope    = str(row.get("vwap_slope","–"))
                    vwap_u   = row.get("vwap_upper")
                    vwap_l   = row.get("vwap_lower")
                    hh_hl    = bool(row.get("ms_hh_hl",False))
                    bos      = bool(row.get("ms_bos",False))
                    ms       = str(row.get("ms_structure","–"))
                    sh       = row.get("ms_swing_high")
                    sl       = row.get("ms_swing_low")
                    price    = float(row.get("price",0))

                    vc = "#3fb950" if "Above" in vwap_pos and "Extended" not in vwap_pos \
                         else "#d29922" if "Extended Above" in vwap_pos else "#f85149"

                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid {vc};border-radius:10px;padding:20px 24px;margin-bottom:16px;">
                      <div style="color:{vc};font-weight:700;font-size:1rem;margin-bottom:14px;">
                        {interp_ticker} — VWAP & Market Structure
                      </div>
                      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:20px;">
                        <div>
                          <div style="color:#8b949e;font-size:0.75rem;text-transform:uppercase;margin-bottom:8px;">VWAP Analysis</div>
                          <p style="color:#c9d1d9;font-size:0.9rem;line-height:1.8;margin:0;">
                            Current price <b>${price:.2f}</b> sits <b style="color:{vc};">{vs_vwap:+.1f}%</b> {"above" if vs_vwap>0 else "below"} the 20-day VWAP of <b>${vwap_val:.2f}</b>.<br>
                            {'The VWAP is rising, meaning the institutional fair value reference is moving higher — this is the setup you want.' if slope=='Rising' else
                             'The VWAP is flat, meaning value is not yet being accepted higher — the auction is in balance.' if slope=='Flat' else
                             'The VWAP is declining — sellers are pushing the fair value reference lower, a bearish signal.'}<br>
                            {'Upper band at $' + f'{vwap_u:.2f}' + ' — this is resistance where price may pause or pull back.' if vwap_u else ''}
                            {'Lower band at $' + f'{vwap_l:.2f}' + ' — this is VWAP support where buyers typically step in.' if vwap_l else ''}
                          </p>
                        </div>
                        <div>
                          <div style="color:#8b949e;font-size:0.75rem;text-transform:uppercase;margin-bottom:8px;">Market Structure</div>
                          <p style="color:#c9d1d9;font-size:0.9rem;line-height:1.8;margin:0;">
                            Structure: <b style="color:{'#3fb950' if 'Bullish' in ms else '#f85149'};">{ms}</b><br>
                            {'✅ Higher Highs and Higher Lows confirmed — the stock is making progress and the uptrend is structurally sound.' if hh_hl else '❌ No HH/HL structure yet — the stock has not proven its uptrend with progressively higher pivot points.'}<br>
                            {'🚨 Recent Break of Structure detected — price has moved beyond a key swing level, confirming the directional move.' if bos else ''}<br>
                            {'Key swing high to watch: $' + f'{sh:.2f}' + ' — a close above this on volume would be a structural confirmation.' if sh else ''}
                            {'Key swing low to defend: $' + f'{sl:.2f}' + ' — a close below this would signal trend deterioration.' if sl else ''}
                          </p>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        # VIEW 4 — PRICE ACTION
        # ════════════════════════════════════════════════════════════════
        with st.expander("🕯 Price Action Interpretation", expanded=True):
            if "pa_patterns" not in interp_df.columns:
                st.info("Run a fresh scan to get Price Action data.")
            else:
                pa_explanations = {
                    "Bullish SFP (Bear Trap)": (
                        "#3fb950",
                        "Swing Failure Pattern — Bullish",
                        "Price temporarily pierced below a recent swing low, triggering stop-losses and luring in short sellers, "
                        "then reversed and closed back above that level. The shorts are now trapped and forced to cover, "
                        "which fuels buying pressure. This is one of the highest-probability reversal setups in price action. "
                        "Entry is typically on the next candle's open if the close holds above the swing low."
                    ),
                    "Bearish SFP (Bull Trap)": (
                        "#f85149",
                        "Swing Failure Pattern — Bearish",
                        "Price spiked above a recent swing high, triggering breakout buyers and stop-losses on short positions, "
                        "then reversed and closed back below. The longs who chased the breakout are now trapped. "
                        "This signals the prior high is strong resistance and a move lower is likely."
                    ),
                    "Bullish Engulfing": (
                        "#3fb950",
                        "Bullish Engulfing Candle",
                        "A large bullish candle whose body completely engulfs the prior bearish candle. "
                        "This shows buyers decisively overwhelmed sellers in a single session. "
                        "Most powerful when occurring at a support level, above VWAP, or after a pullback in an uptrend."
                    ),
                    "Bearish Engulfing": (
                        "#f85149",
                        "Bearish Engulfing Candle",
                        "A large bearish candle engulfs the prior bullish candle — sellers took complete control. "
                        "Most significant at resistance levels or after an extended run-up."
                    ),
                    "Inside Day (Compression)": (
                        "#388bfd",
                        "Inside Day — Compression",
                        "Today's entire range fits within yesterday's high-low range. "
                        "The market is in equilibrium, with neither buyers nor sellers willing to extend the range. "
                        "Inside days signal compression before expansion — a directional move is building. "
                        "The breakout direction from the inside day often sets the short-term trend."
                    ),
                    "Bullish Context Candle": (
                        "#3fb950",
                        "Bullish Context Candle",
                        "A high-volume candle that closed in the top 25% of its range. "
                        "In auction market terms, the market tested lower prices but buyers rejected them decisively — "
                        "closing near the high shows the auction outcome was bullish. "
                        "Volume above average amplifies the significance of this signal."
                    ),
                    "Bearish Context Candle": (
                        "#f85149",
                        "Bearish Context Candle",
                        "High-volume candle closing in the bottom 25% of its range — sellers dominated the auction. "
                        "The market tested higher prices but rejected them, closing near the low."
                    ),
                    "PA Confluence": (
                        "#d29922",
                        "Price Action Confluence",
                        "Multiple price action signals are aligned on the same candle or within a close cluster of candles. "
                        "Confluence dramatically increases the probability of a signal — "
                        "when an SFP, engulfing candle, and bullish context candle all appear together, "
                        "it indicates the market has made a very decisive statement about direction."
                    ),
                }

                if interp_mode == "Full Scan Summary":
                    # Count PA pattern types
                    all_pa   = interp_df["pa_patterns"].dropna()
                    sfp_bull = all_pa.str.contains("Bullish SFP", na=False).sum()
                    sfp_bear = all_pa.str.contains("Bearish SFP", na=False).sum()
                    engulf   = all_pa.str.contains("Engulfing",   na=False).sum()
                    inside   = all_pa.str.contains("Inside Day",  na=False).sum()
                    context  = all_pa.str.contains("Context",     na=False).sum()
                    conflu   = all_pa.str.contains("Confluence",  na=False).sum()
                    any_pa   = (all_pa != "None").sum()

                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px 24px;margin-bottom:16px;">
                      <div style="font-weight:700;font-size:1rem;margin-bottom:12px;">Price Action Picture</div>
                      <p style="color:#c9d1d9;font-size:0.9rem;line-height:1.7;margin:0 0 12px 0;">
                        Price action patterns reveal what happened in the most recent session — the actual decisions made
                        by buyers and sellers. They complement technical indicators by showing real-time sentiment.
                      </p>
                      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                        <div><span style="color:#8b949e;font-size:0.75rem;">BULLISH SFP (TRAPS)</span><br>
                             <span style="font-size:1.4rem;font-weight:700;color:#3fb950;">{sfp_bull}</span></div>
                        <div><span style="color:#8b949e;font-size:0.75rem;">ENGULFING CANDLES</span><br>
                             <span style="font-size:1.4rem;font-weight:700;color:#3fb950;">{engulf}</span></div>
                        <div><span style="color:#8b949e;font-size:0.75rem;">PA CONFLUENCE</span><br>
                             <span style="font-size:1.4rem;font-weight:700;color:#d29922;">{conflu}</span></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Price Action Per Stock")
                    for _, row in interp_df.head(15).iterrows():
                        tk       = row["ticker"]
                        pa_str   = str(row.get("pa_patterns","None"))
                        pa_sc    = int(row.get("pa_score",0) or 0)
                        price_v  = float(row.get("price",0))

                        if pa_str == "None" or not pa_str:
                            pa_html = (f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
                                      f'padding:14px 18px;margin:6px 0;">'
                                      f'<span style="font-weight:700;color:#e6edf3;">{tk}</span>&nbsp;&nbsp;'
                                      f'<span style="color:#8b949e;font-size:0.85rem;">No significant PA pattern on last candle — '
                                      f'check again after tomorrow\'s session.</span></div>')
                        else:
                            patterns_found = [p.strip() for p in pa_str.split("|")]
                            readings = []
                            for p in patterns_found:
                                if p in pa_explanations:
                                    c, name, expl = pa_explanations[p]
                                    readings.append(f'<span style="color:{c};font-weight:600;">{name}:</span> {expl}')
                            pa_html = (f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
                                      f'padding:14px 18px;margin:6px 0;">'
                                      f'<div style="margin-bottom:8px;">'
                                      f'<span style="font-weight:700;color:#e6edf3;">{tk}</span>&nbsp;&nbsp;'
                                      f'{amber(f"PA Score {pa_sc}/5")}</div>'
                                      f'<ul style="margin:0;padding-left:18px;color:#c9d1d9;font-size:0.87rem;line-height:1.8;">'
                                      + "".join(f"<li>{r}</li>" for r in readings) +
                                      f'</ul></div>')
                        st.markdown(pa_html, unsafe_allow_html=True)

                else:
                    row    = interp_df.iloc[0]
                    pa_str = str(row.get("pa_patterns","None"))
                    pa_sc  = int(row.get("pa_score",0) or 0)

                    if pa_str == "None" or not pa_str:
                        st.markdown(f"""
                        <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px 24px;">
                          <div style="color:#8b949e;font-size:0.9rem;">
                            No significant price action pattern detected on {interp_ticker}'s last candle.
                            This is not necessarily negative — it simply means the most recent session
                            did not produce a definitive signal. Check again after tomorrow's close.
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        patterns_found = [p.strip() for p in pa_str.split("|")]
                        for p in patterns_found:
                            if p in pa_explanations:
                                c, name, expl = pa_explanations[p]
                                st.markdown(f"""
                                <div style="background:#161b22;border:1px solid {c};border-radius:10px;
                                            padding:20px 24px;margin-bottom:12px;">
                                  <div style="color:{c};font-weight:700;font-size:1rem;margin-bottom:10px;">
                                    {interp_ticker} — {name} (PA Score: {pa_sc}/5)
                                  </div>
                                  <p style="color:#c9d1d9;font-size:0.92rem;line-height:1.85;margin:0;">{expl}</p>
                                </div>
                                """, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        # VIEW 5 — FUNDAMENTALS (Alpha Vantage)
        # ════════════════════════════════════════════════════════════════
        with st.expander("📊 Fundamentals Interpretation (Alpha Vantage)", expanded=True):
            if "eps_growth_%" not in interp_df.columns or interp_df["eps_growth_%"].isna().all():
                st.markdown("""
                <div style="background:#161b22;border:1px solid #d29922;border-radius:8px;padding:16px 20px;">
                  <b style="color:#d29922;">⚠️ Alpha Vantage key not yet active</b><br>
                  <span style="color:#8b949e;font-size:0.9rem;">
                  Add your key to <code>config.yaml</code> under <code>alpha_vantage_key</code>
                  and run a fresh scan to unlock real EPS data, earnings surprises,
                  revenue growth, analyst targets and PE/PEG ratios for every stock.
                  </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                if interp_mode == "Full Scan Summary":
                    strong_eps = interp_df[
                        pd.to_numeric(interp_df.get("eps_growth_%", pd.Series()), errors="coerce") >= 25
                    ]
                    accel_eps  = interp_df[interp_df.get("eps_accel", pd.Series([False]*len(interp_df))) == True]
                    beat_3plus = interp_df[
                        pd.to_numeric(interp_df.get("consec_beats", pd.Series()), errors="coerce") >= 3
                    ]

                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                                padding:20px 24px;margin-bottom:16px;">
                      <div style="font-weight:700;font-size:1rem;margin-bottom:12px;">
                        Real Earnings Picture (Alpha Vantage Data)
                      </div>
                      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                        <div>
                          <span style="color:#8b949e;font-size:0.75rem;">EPS GROWTH ≥25%</span><br>
                          <span style="font-size:1.4rem;font-weight:700;color:#3fb950;">
                            {len(strong_eps)}
                          </span>
                          <div style="color:#8b949e;font-size:0.8rem;">
                            {", ".join(strong_eps["ticker"].head(5).tolist()) or "None"}
                          </div>
                        </div>
                        <div>
                          <span style="color:#8b949e;font-size:0.75rem;">ACCELERATING EPS</span><br>
                          <span style="font-size:1.4rem;font-weight:700;color:#d29922;">
                            {len(accel_eps)}
                          </span>
                          <div style="color:#8b949e;font-size:0.8rem;">
                            {", ".join(accel_eps["ticker"].head(5).tolist()) or "None"}
                          </div>
                        </div>
                        <div>
                          <span style="color:#8b949e;font-size:0.75rem;">3+ CONSEC BEATS</span><br>
                          <span style="font-size:1.4rem;font-weight:700;color:#388bfd;">
                            {len(beat_3plus)}
                          </span>
                          <div style="color:#8b949e;font-size:0.8rem;">
                            {", ".join(beat_3plus["ticker"].head(5).tolist()) or "None"}
                          </div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### EPS Quality Per Stock")
                    for _, row in interp_df.head(15).iterrows():
                        tk      = row["ticker"]
                        eg      = row.get("eps_growth_%")
                        es      = row.get("eps_surprise_%")
                        accel   = bool(row.get("eps_accel", False))
                        beats   = row.get("consec_beats")
                        rev_g   = row.get("rev_growth_%")
                        earn_m  = str(row.get("earn_momentum","–"))
                        eps_sc  = row.get("eps_score", 0)
                        tgt     = row.get("analyst_target")
                        price_v = row.get("price",0)

                        sentences = []

                        if eg is not None:
                            if eg >= 50:
                                sentences.append(f"EPS growing at {eg:+.1f}% YoY — exceptional growth rate. This is the kind of fundamental acceleration that institutional investors pay premium multiples for.")
                            elif eg >= 25:
                                sentences.append(f"EPS growth of {eg:+.1f}% YoY is strong and above the 25% threshold that marks a leading growth stock.")
                            elif eg >= 0:
                                sentences.append(f"EPS growing {eg:+.1f}% YoY — positive but moderate. Look for acceleration in coming quarters.")
                            else:
                                sentences.append(f"EPS declining {eg:.1f}% YoY — earnings are contracting. Strong price momentum with declining earnings is a yellow flag.")

                        if es is not None:
                            if es >= 20:
                                sentences.append(f"Beat estimates by {es:.1f}% last quarter — a massive positive surprise. Institutional re-rating often follows large beats.")
                            elif es >= 5:
                                sentences.append(f"Beat estimates by {es:.1f}% — consistent with a company executing above expectations.")
                            elif es < 0:
                                sentences.append(f"Missed estimates by {abs(es):.1f}% last quarter — a miss can create selling pressure even in uptrending stocks.")

                        if accel:
                            sentences.append("EPS growth is accelerating quarter-over-quarter — the business is gaining momentum, not slowing down.")

                        if beats and beats >= 3:
                            sentences.append(f"{beats} consecutive quarterly beats. Management has a track record of under-promising and over-delivering — a quality signal.")

                        if rev_g is not None and rev_g >= 20:
                            sentences.append(f"Revenue growing {rev_g:+.1f}% YoY — top-line growth is real and substantial, not just cost-cutting driven.")

                        if tgt and price_v:
                            upside = (tgt / price_v - 1) * 100
                            sentences.append(f"Analyst consensus target ${tgt:.2f} implies {upside:+.1f}% upside from current price.")

                        if not sentences:
                            sentences.append("Fundamental data not available for this ticker from Alpha Vantage.")

                        badge_c = "#3fb950" if "Strong" in earn_m else ("#d29922" if "Moderate" in earn_m else "#8b949e")
                        earn_badge = f'<span style="background:{badge_c}22;color:{badge_c};border:1px solid {badge_c};border-radius:4px;padding:2px 8px;font-size:0.75rem;">{earn_m}</span>'

                        st.markdown(
                            f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
                            f'padding:14px 18px;margin:6px 0;">'
                            f'<div style="margin-bottom:8px;">'
                            f'<span style="font-weight:700;color:#e6edf3;">{tk}</span>'
                            f'&nbsp;&nbsp;{earn_badge}&nbsp;'
                            f'<span style="color:#8b949e;font-size:0.8rem;">EPS Score: {eps_sc}/15</span></div>'
                            f'<ul style="margin:0;padding-left:18px;color:#c9d1d9;font-size:0.87rem;line-height:1.8;">'
                            + "".join(f"<li>{s}</li>" for s in sentences) +
                            f'</ul></div>',
                            unsafe_allow_html=True
                        )

                else:
                    # Single ticker
                    row     = interp_df.iloc[0]
                    eg      = row.get("eps_growth_%")
                    es      = row.get("eps_surprise_%")
                    accel   = bool(row.get("eps_accel", False))
                    beats   = row.get("consec_beats")
                    rev_g   = row.get("rev_growth_%")
                    earn_m  = str(row.get("earn_momentum","–"))
                    eps_sc  = row.get("eps_score", 0)
                    tgt     = row.get("analyst_target")
                    pe      = row.get("pe_ratio")
                    peg     = row.get("peg_ratio")
                    eps_det = row.get("eps_details","–")
                    price_v = float(row.get("price",0))

                    fund_color = "#3fb950" if "Strong" in earn_m else ("#d29922" if "Moderate" in earn_m else "#f85149")

                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid {fund_color};
                                border-radius:10px;padding:20px 24px;">
                      <div style="color:{fund_color};font-weight:700;font-size:1rem;margin-bottom:14px;">
                        {interp_ticker} — Fundamental Profile (EPS Score: {eps_sc}/15)
                      </div>
                      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:20px;">
                        <div>
                          <div style="color:#8b949e;font-size:0.75rem;text-transform:uppercase;margin-bottom:8px;">
                            Earnings Quality
                          </div>
                          <p style="color:#c9d1d9;font-size:0.9rem;line-height:1.85;margin:0;">
                            {"✅" if (eg or 0)>=15 else "⚠️"} <b>EPS Growth YoY:</b>
                            {f"{eg:+.1f}%" if eg is not None else "No data"} —
                            {f"Exceptional growth at {eg:.0f}% — this stock is earning far more than it did a year ago." if (eg or 0)>=50
                             else f"Strong growth at {eg:.0f}% — above the 25% threshold for leading growth stocks." if (eg or 0)>=25
                             else f"Moderate growth of {eg:.0f}%." if (eg or 0)>=0
                             else f"Earnings declining {eg:.0f}% — watch this carefully alongside price action." if eg is not None
                             else "Not available."}<br><br>
                            {"✅" if (es or 0)>=5 else "⚠️"} <b>Last Surprise:</b>
                            {f"{es:+.1f}%" if es is not None else "No data"} —
                            {f"Beat by {es:.1f}% — institutional buyers often accumulate after large beats." if (es or 0)>=20
                             else f"Beat by {es:.1f}% — consistent execution." if (es or 0)>=5
                             else f"Missed by {abs(es or 0):.1f}% — misses can weigh on price even in uptrends." if es is not None
                             else "Not available."}<br><br>
                            {"✅" if accel else "–"} <b>Accelerating:</b>
                            {"Yes — growth rate is increasing quarter over quarter. This is what drives institutional re-rating." if accel
                             else "No acceleration yet. Steady growth but not yet building momentum in the earnings line."}<br><br>
                            {"✅" if (beats or 0)>=3 else "–"} <b>Consecutive Beats:</b>
                            {f"{beats} quarters" if beats else "No data"} —
                            {f"{beats} consecutive beats shows management consistently sets achievable targets and exceeds them — a trust signal for institutional investors." if (beats or 0)>=3
                             else "Less than 3 consecutive beats." if beats is not None else "Not available."}
                          </p>
                        </div>
                        <div>
                          <div style="color:#8b949e;font-size:0.75rem;text-transform:uppercase;margin-bottom:8px;">
                            Valuation & Revenue
                          </div>
                          <p style="color:#c9d1d9;font-size:0.9rem;line-height:1.85;margin:0;">
                            📈 <b>Revenue Growth:</b>
                            {f"{rev_g:+.1f}% YoY — top-line growth confirms the earnings improvement is driven by real business expansion, not just cost cuts." if rev_g is not None else "Not available."}<br><br>
                            💰 <b>PE Ratio:</b> {f"{pe:.1f}x — " if pe else "Not available — "}
                            {f"premium valuation reflecting growth expectations." if (pe or 0)>40
                             else f"reasonable for a growth company." if (pe or 0)>20
                             else f"relatively cheap valuation." if (pe or 0)>0
                             else ""}<br><br>
                            📐 <b>PEG Ratio:</b> {f"{peg:.2f} — " if peg else "Not available — "}
                            {f"below 1.0 = potentially undervalued relative to growth." if (peg or 0)<1 and (peg or 0)>0
                             else f"above 2.0 = growth is priced in." if (peg or 0)>2
                             else ""}<br><br>
                            🎯 <b>Analyst Target:</b>
                            {f"${tgt:.2f} = {(tgt/price_v-1)*100:+.1f}% from current ${price_v:.2f}" if tgt and price_v else "Not available."}
                          </p>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Combined summary for single ticker ─────────────────────────
        if interp_mode == "Single Ticker Deep Read" and not interp_df.empty:
            st.markdown("---")
            st.markdown("#### 🎯 Combined Signal Summary")
            row    = interp_df.iloc[0]
            score  = float(row.get("apex_score",0))
            stage  = str(row.get("stage","?"))
            bias   = str(row.get("of_bias","–"))
            vwap_p = str(row.get("vwap_position","–"))
            ms_s   = str(row.get("ms_structure","–"))
            pa_s   = str(row.get("pa_patterns","None"))

            signals_green = sum([
                "2 ✅" in stage,
                "Bullish" in bias,
                "Above VWAP" in vwap_p and "Extended" not in vwap_p,
                "Bullish" in ms_s,
                "Bullish" in pa_s,
            ])

            if signals_green >= 4:
                verdict_html = f'<span style="color:#3fb950;font-weight:700;font-size:1.1rem;">HIGH CONVICTION SETUP ({signals_green}/5 signals aligned)</span>'
                verdict_text = f"All major signal categories are aligned bullishly for {interp_ticker}. This is the type of setup where multiple independent frameworks agree — the highest conviction scenario."
            elif signals_green >= 3:
                verdict_html = f'<span style="color:#d29922;font-weight:700;font-size:1.1rem;">MODERATE CONVICTION ({signals_green}/5 signals aligned)</span>'
                verdict_text = f"Most signals are positive for {interp_ticker} but some are missing. Worth monitoring closely. Wait for the remaining signals to confirm before committing full position size."
            else:
                verdict_html = f'<span style="color:#f85149;font-weight:700;font-size:1.1rem;">LOW CONVICTION — WAIT ({signals_green}/5 signals aligned)</span>'
                verdict_text = f"Too many signals are not yet aligned for {interp_ticker}. The setup needs more time to develop. Patience here protects capital for when conditions improve."

            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px;margin-top:8px;">
              <div style="margin-bottom:12px;">{verdict_html}</div>
              <p style="color:#c9d1d9;font-size:0.92rem;line-height:1.8;margin:0 0 16px 0;">{verdict_text}</p>
              <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">
                <div style="background:#0d1117;border-radius:6px;padding:10px;text-align:center;">
                  <div style="font-size:1.2rem;">{'✅' if '2 ✅' in stage else '❌'}</div>
                  <div style="color:#8b949e;font-size:0.7rem;margin-top:4px;">STAGE 2</div>
                </div>
                <div style="background:#0d1117;border-radius:6px;padding:10px;text-align:center;">
                  <div style="font-size:1.2rem;">{'✅' if 'Bullish' in bias else '❌'}</div>
                  <div style="color:#8b949e;font-size:0.7rem;margin-top:4px;">ORDER FLOW</div>
                </div>
                <div style="background:#0d1117;border-radius:6px;padding:10px;text-align:center;">
                  <div style="font-size:1.2rem;">{'✅' if 'Above VWAP' in vwap_p and 'Extended' not in vwap_p else '❌'}</div>
                  <div style="color:#8b949e;font-size:0.7rem;margin-top:4px;">ABOVE VWAP</div>
                </div>
                <div style="background:#0d1117;border-radius:6px;padding:10px;text-align:center;">
                  <div style="font-size:1.2rem;">{'✅' if 'Bullish' in ms_s else '❌'}</div>
                  <div style="color:#8b949e;font-size:0.7rem;margin-top:4px;">HH/HL</div>
                </div>
                <div style="background:#0d1117;border-radius:6px;padding:10px;text-align:center;">
                  <div style="font-size:1.2rem;">{'✅' if 'Bullish' in pa_s else '❌'}</div>
                  <div style="color:#8b949e;font-size:0.7rem;margin-top:4px;">PRICE ACTION</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 17 — GUIDE
# ══════════════════════════════════════════════════════════════════════════════

with tabs[16]:
    st.markdown("""
### 📖 How to Use ApexScan — Complete Guide

---

#### 🤖 AI Daily Briefing
- Run a **Live Scan** first, then click **Generate Briefing**
- Claude reads your results and writes a plain-English market briefing
- Covers top setups, active breakouts, theme rotation and a risk reminder
- Send directly to Telegram or download as text
- Takes about 10 seconds to generate

#### 📋 Watchlist Manager
- Create named lists: *High Conviction*, *Monitoring*, *Earnings Soon* etc.
- Add tickers manually or paste a comma-separated list to import
- **Scan any watchlist independently** — see scores for just those stocks
- Promote top results from one list to another with one click
- Lists persist between sessions

#### 🔔 Alert Settings
- **Telegram** (recommended): Free, instant, works on any phone
  - Follow the 5-step setup in the tab — takes 5 minutes
  - Alerts fire automatically after every scan
- **Email**: Works with Gmail app passwords
- Alert types: Breakouts, Stop Loss breach, Earnings warnings
- Set minimum score threshold so you only get alerts that matter

---

#### 🏆 Apex Score (0–100)
| Points | Signal |
|---|---|
| 0–40 | 3-month momentum |
| 25 | RS > benchmark |
| 15 | Stage 2 uptrend |
| 10 | Near 52-week high |
| 10 | Active breakout |

#### 📊 RS Score vs S&P 500
- **> 100** 🟢 Beating the market — buy leaders, not laggards
- **70–100** 🟡 Keeping pace
- **< 70** 🔴 Lagging — avoid

#### 📐 Stage Guide
- **Stage 2 ✅** — Only stage worth buying
- **Stage 1 ⏳** — Building base, not ready
- **Stage 3 ⚠️** — Rolling over, be careful
- **Stage 4 🔴** — Downtrend, avoid

#### 🎯 Options Flow
- **Vol/OI > 3x** = unusual — someone is placing a big bet
- Calls = bullish bet, Puts = bearish/hedge
- High notional ($) = more conviction behind the trade
- Combine with high Apex Score for maximum signal

#### 🕵️ Insider Tracker
- 🔥 **Cluster buy** = 2+ insiders buying = strongest possible signal
- C-suite buys (CEO, CFO) matter most
- Insider sells = less meaningful, they sell for many reasons

#### ⚖️ Risk Calculator — The Golden Rule
> **Never risk more than 1–2% of your account on any single trade**

Formula: `Shares = (Account × Risk%) ÷ (Entry − Stop)`

Always set a **minimum 2:1 reward:risk** before entering any trade.

#### ⏱ Backtester
- Entry: Apex Score ≥ threshold + Stage 2
- Exit: Price closes below 50MA or max hold days
- Use 2021–2024 for a full market cycle test (bull + bear + recovery)

---
> ⚠️ ApexScan is for research and education only — not financial advice.

---

#### 🔑 API Key Setup & Verification

**Alpha Vantage (for real EPS data):**
1. Go to **alphavantage.co** → click **Get Free API Key**
2. Open `config.yaml` in VS Code
3. Find the line: `alpha_vantage_key: "YOUR_ALPHA_VANTAGE_KEY_HERE"`
4. Replace the placeholder with your actual key (keep the quote marks)
5. Save the file (Ctrl+S)
6. **Restart the dashboard** — close CMD, reopen it, run the command again
7. The sidebar will show 🟢 Alpha Vantage ✓ Active when it's working

**Important:** After editing config.yaml you MUST restart the dashboard.
Just saving the file is not enough — Streamlit needs to reload.

**Finnhub (for news sentiment):**
Same process — paste your key at `finnhub_key:` in config.yaml.

**Free tier limits:**
- Alpha Vantage free: 25 API calls/day, 5/minute
- Each stock in the scan uses 3 AV calls (earnings + income + overview)
- Cached for 24h so repeated scans don't burn your quota
- With 45 tickers: first scan uses ~45 calls (exceeds daily free limit)
- **Solution:** Add `cache_hours: 168` (1 week) in config.yaml under `alpha_vantage:` to cache aggressively and stay within quota
    """)
