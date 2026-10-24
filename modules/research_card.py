"""
modules/research_card.py — Research card (Change 3, Stage 1)

A transparent, DESCRIPTIVE research card for one US stock. Pure functions: no network,
no storage, no UI, and no dependency on scanner.py, dashboard.py or any scoring code.
The caller supplies daily bars; nothing here fetches or writes anything.

What this is NOT: a score, a rating, a probability or a recommendation. Setup labels are
descriptions of the chart, and the evidence so far (Alpha Lab, matched-control evaluation)
is that no ApexScan setup type has been shown to predict returns. Every card says so.

What it adds over the existing Stock Deep Dive:
  * swings with DATES and CONFIRMATION bars, filtered so minor wiggles are not labelled;
  * a STRUCTURAL invalidation level (nearest confirmed swing low) instead of the 200-day
    average, which sat a median of about 12% below entry and more than 20% away for 27% of
    logged discoveries — too far to size a position against;
  * a target (next resistance zone) or the explicit reason none is defensible;
  * reward-to-risk only when both a target and a positive risk exist;
  * a sizing illustration, liquidity as a share of average volume, earnings proximity;
  * the evidence for AND against, and an explicit list of what is missing.

Point-in-time discipline. A swing at bar i needs `SWING_N` bars on each side and a reaction
measured within the `SWING_N` bars after it, so it is decided at bar i+SWING_N and NEVER
changes afterwards. Every indicator is a forward recursion. `build_card(..., asof=d)` cuts
the history at `asof` before anything else runs, so no code path can read a later bar.

Pre-specified parameters (set before any evaluation, NOT tuned): see the constants below.
Limits (also printed on every card): levels come from dividend-adjusted history and drift
slightly over time; there is no bid/ask feed so spread is unavailable; weekly bars exclude an
unfinished week; NGX is not validated and returns no card.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

CARD_VERSION = "CARD-1.0"

# ── pre-specified parameters (not tuned) ──────────────────────────────────
SWING_N = 5                    # bars each side that must not exceed the pivot
SWING_REACTION_ATR = 1.0       # price must react >= this many ATRs within SWING_N bars after the pivot
ATR_PERIOD = 14
RSI_PERIOD = 14
STOP_BUFFER_ATR = 0.25         # stop sits this far beyond the swing low
ZONE_ATR = 0.5                 # swing points within this many ATRs share a zone
BREAKOUT_NEAR_ATR = 1.5        # "breakout candidate": nearest resistance within this many ATRs above price
PULLBACK_ATR = 1.0             # "pullback": price at least this many ATRs below the last swing high
RVOL_CONFIRM = 1.5             # relative volume that counts as volume-confirmed
BREAKOUT_LOOKBACK_BARS = 5     # a breakout counts as "recent" for this many bars
WIDE_STOP_PCT = 15.0           # informational flag only
ADV_PARTICIPATION_WARN_PCT = 1.0   # informational flag only
EARNINGS_WARN_DAYS = 7
MIN_BARS = 60                  # below this no structural claim is made
SESSION_SETTLE_MINUTES = 15    # a US daily bar is treated as final this long after the 16:00 ET close
DEFAULT_ACCOUNT = 10_000.0     # illustration only, labelled as such
DEFAULT_RISK_PCT = 1.0

PRICE_BASIS = "yfinance daily history, dividend/split-adjusted; levels drift slightly as later dividends adjust it"
DISCLAIMER = ("Descriptive research only. Setup types have not been shown to predict returns; "
              "nothing here is a trade instruction.")


# ── small helpers ─────────────────────────────────────────────────────────
def _f(x, nd: int = 4):
    """JSON-safe float: None if not finite."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ── history preparation ───────────────────────────────────────────────────
def _session_incomplete(last_bar_date: date, now: Optional[datetime]) -> bool:
    """True if the last daily bar may still be forming (today's US session not yet settled)."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        et = now.astimezone(ZoneInfo("America/New_York"))
        if last_bar_date > et.date():
            return True
        if last_bar_date < et.date():
            return False
        settled = datetime.combine(et.date(), time(16, 0), tzinfo=et.tzinfo) + timedelta(minutes=SESSION_SETTLE_MINUTES)
        return et < settled
    except Exception:
        # No timezone database: be conservative. Treat today's bar as incomplete until 21:15 UTC.
        u = now.astimezone(timezone.utc)
        if last_bar_date > u.date():
            return True
        if last_bar_date < u.date():
            return False
        return (u.hour, u.minute) < (21, SESSION_SETTLE_MINUTES)


def prepare_history(hist: pd.DataFrame, now: Optional[datetime] = None,
                    drop_incomplete: bool = True) -> tuple:
    """Returns (clean_hist, notes). Sorts, de-duplicates, removes unusable rows, strips a timezone,
    and drops today's bar while its US session may still be forming — comparing a partial day's
    volume with full days' volume is a measurement error, not a signal. Never mutates the input."""
    notes = []
    need = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in need if c not in hist.columns]
    if missing:
        raise ValueError(f"history is missing columns: {missing}")
    h = hist[need].copy()
    idx = pd.DatetimeIndex(h.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    h.index = idx.normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    before = len(h)
    h = h.replace([np.inf, -np.inf], np.nan).dropna(subset=["High", "Low", "Close"])
    h["Volume"] = h["Volume"].fillna(0.0)
    h = h[(h["High"] >= h["Low"]) & (h["Close"] > 0)]
    if len(h) < before:
        notes.append(f"{before - len(h)} unusable bar(s) removed")
    if drop_incomplete and len(h) and _session_incomplete(h.index[-1].date(), now):
        notes.append(f"last bar ({_d(h.index[-1])}) excluded: US session may be incomplete")
        h = h.iloc[:-1]
    return h, notes


# ── causal indicators (forward recursions; each bar uses only earlier bars) ──
def sma(values, window: int) -> np.ndarray:
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    if len(v) >= window:
        c = np.cumsum(np.insert(v, 0, 0.0))
        out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def ema(values, span: int) -> np.ndarray:
    """EMA seeded with the SMA of the first `span` bars (standard), then alpha = 2/(span+1)."""
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    if len(v) < span:
        return out
    out[span - 1] = v[:span].mean()
    a = 2.0 / (span + 1)
    for i in range(span, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return out


def atr_wilder(high, low, close, period: int = ATR_PERIOD) -> np.ndarray:
    h, l, c = (np.asarray(x, float) for x in (high, low, close))
    n = len(c)
    out = np.full(n, np.nan)
    if n < period:
        return out
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    out[period - 1] = tr[:period].mean()
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def rsi_wilder(close, period: int = RSI_PERIOD) -> np.ndarray:
    """Wilder RSI. NaN until `period` changes exist, and NaN where both averages are zero (undefined)."""
    c = np.asarray(close, float)
    n = len(c)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    d = np.diff(c)
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag, al = gain[:period].mean(), loss[:period].mean()

    def _val(ag_, al_):
        if ag_ == 0 and al_ == 0:
            return np.nan
        if al_ == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + ag_ / al_)

    out[period] = _val(ag, al)
    for i in range(period + 1, n):
        ag = (ag * (period - 1) + gain[i - 1]) / period
        al = (al * (period - 1) + loss[i - 1]) / period
        out[i] = _val(ag, al)
    return out


# ── swings ────────────────────────────────────────────────────────────────
def find_swings(hist: pd.DataFrame, n: int = SWING_N, reaction_atr: float = SWING_REACTION_ATR) -> list:
    """Confirmed swing highs/lows, each decided at bar i+n and never revised.

    Pivot high at i:  High[i] >= every High in the n bars before and > every High in the n bars after
    (so a flat top yields exactly one swing: its LAST bar), AND price reacted: High[i] minus the
    lowest Low in the n bars after is at least `reaction_atr` x ATR(i). Pivot lows mirror this.
    The reaction filter is what keeps minor wiggles from being called swing points, and because it
    only uses the n confirmation bars the swing list on any prefix of the history equals the full
    list filtered by confirmation bar (prefix-stability)."""
    H, L = hist["High"].to_numpy(float), hist["Low"].to_numpy(float)
    C = hist["Close"].to_numpy(float)
    atr = atr_wilder(H, L, C)
    N = len(hist)
    swings = []
    for i in range(n, N - n):
        a = atr[i]
        if not math.isfinite(a) or a <= 0:
            continue
        if H[i] >= H[i - n:i].max() and H[i] > H[i + 1:i + n + 1].max():
            react = H[i] - L[i + 1:i + n + 1].min()
            if react >= reaction_atr * a:
                swings.append({"type": "high", "idx": i, "price": float(H[i]), "atr": float(a),
                               "reaction": float(react), "confirmed_idx": i + n})
        if L[i] <= L[i - n:i].min() and L[i] < L[i + 1:i + n + 1].min():
            react = H[i + 1:i + n + 1].max() - L[i]
            if react >= reaction_atr * a:
                swings.append({"type": "low", "idx": i, "price": float(L[i]), "atr": float(a),
                               "reaction": float(react), "confirmed_idx": i + n})
    swings.sort(key=lambda s: (s["idx"], s["type"]))
    dates = hist.index
    for s in swings:
        s["date"] = _d(dates[s["idx"]])
        s["confirmed_date"] = _d(dates[s["confirmed_idx"]])
    return swings


def classify_structure(swings: list, close: float) -> dict:
    """HH/HL, LH/LL or mixed from the last two confirmed swing highs and lows, plus a break flag:
    a close beyond the most recent opposing swing level. Descriptive only."""
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    out = {"structure": "insufficient", "higher_highs": None, "higher_lows": None,
           "lower_highs": None, "lower_lows": None, "break_of_structure": None,
           "last_swing_high": highs[-1] if highs else None, "last_swing_low": lows[-1] if lows else None}
    if len(highs) < 2 or len(lows) < 2:
        return out
    hh, hl = highs[-1]["price"] > highs[-2]["price"], lows[-1]["price"] > lows[-2]["price"]
    lh, ll = highs[-1]["price"] < highs[-2]["price"], lows[-1]["price"] < lows[-2]["price"]
    out.update({"higher_highs": hh, "higher_lows": hl, "lower_highs": lh, "lower_lows": ll})
    out["structure"] = "uptrend" if (hh and hl) else "downtrend" if (lh and ll) else "mixed"
    if out["structure"] == "uptrend" and close < lows[-1]["price"]:
        out["break_of_structure"] = "bearish: close below the last swing low"
    elif out["structure"] == "downtrend" and close > highs[-1]["price"]:
        out["break_of_structure"] = "bullish: close above the last swing high"
    return out


# ── zones ─────────────────────────────────────────────────────────────────
def cluster_zones(swings: list, atr: float, tol_mult: float = ZONE_ATR) -> list:
    """Group swing points (highs and lows together) whose prices sit within tol_mult x ATR of the
    cluster's lowest price. `touches` is the number of confirmed swing points — repeated reactions."""
    if not swings or not (atr and math.isfinite(atr) and atr > 0):
        return []
    pts = sorted(swings, key=lambda s: s["price"])
    clusters, cur = [], [pts[0]]
    for s in pts[1:]:
        if s["price"] - cur[0]["price"] <= tol_mult * atr:
            cur.append(s)
        else:
            clusters.append(cur)
            cur = [s]
    clusters.append(cur)
    zones = []
    for c in clusters:
        prices = [s["price"] for s in c]
        zones.append({"low": min(prices), "high": max(prices), "touches": len(c),
                      "n_highs": sum(1 for s in c if s["type"] == "high"),
                      "n_lows": sum(1 for s in c if s["type"] == "low"),
                      "first_date": min(s["date"] for s in c), "last_date": max(s["date"] for s in c),
                      "kind": "tested" if len(c) >= 2 else "single swing"})
    return zones


def _recent_breakout(hist: pd.DataFrame, swings: list, atr: np.ndarray, rvol_window: int = 50) -> Optional[dict]:
    """Most recent bar (within BREAKOUT_LOOKBACK_BARS) that closed above a zone high which the previous
    close had not exceeded. Zones are rebuilt CAUSALLY at each bar from swings confirmed before it."""
    C, V = hist["Close"].to_numpy(float), hist["Volume"].to_numpy(float)
    N = len(hist)
    for j in range(N - 1, max(N - 1 - BREAKOUT_LOOKBACK_BARS, 1) - 1, -1):
        known = [s for s in swings if s["confirmed_idx"] < j]
        a = atr[j - 1] if j - 1 < len(atr) else np.nan
        broken = [z for z in cluster_zones(known, a) if C[j - 1] <= z["high"] < C[j]]
        if broken:
            z = max(broken, key=lambda zz: zz["high"])
            base = V[max(0, j - rvol_window):j]
            rvol = (V[j] / base.mean()) if len(base) >= rvol_window and base.mean() > 0 else None
            return {"zone_low": z["low"], "zone_high": z["high"], "zone_touches": z["touches"],
                    "bars_ago": N - 1 - j, "date": _d(hist.index[j]), "rvol": _f(rvol, 2),
                    "volume_confirmed": bool(rvol is not None and rvol >= RVOL_CONFIRM)}
    return None


# ── volume, momentum, trend ───────────────────────────────────────────────
def volume_context(hist: pd.DataFrame) -> dict:
    V, C = hist["Volume"].to_numpy(float), hist["Close"].to_numpy(float)
    out = {"rvol_last": None, "avg_volume_20": None, "avg_dollar_volume_20": None,
           "up_down_volume_ratio_20": None, "basis": "complete sessions only; the last bar is excluded from its own baseline"}
    if len(V) >= 51 and V[-51:-1].mean() > 0:
        out["rvol_last"] = _f(V[-1] / V[-51:-1].mean(), 2)
    if len(V) >= 20:
        out["avg_volume_20"] = _f(V[-20:].mean(), 0)
        out["avg_dollar_volume_20"] = _f((V[-20:] * C[-20:]).mean(), 0)
    if len(V) >= 21:
        up = [V[i] for i in range(len(V) - 20, len(V)) if C[i] > C[i - 1]]
        dn = [V[i] for i in range(len(V) - 20, len(V)) if C[i] < C[i - 1]]
        if len(up) >= 3 and len(dn) >= 3 and np.mean(dn) > 0:
            out["up_down_volume_ratio_20"] = _f(np.mean(up) / np.mean(dn), 2)
    return out


def weekly_context(hist: pd.DataFrame) -> dict:
    """Weekly closes from COMPLETED weeks only (an unfinished week is dropped), vs 10- and 40-week SMAs."""
    out = {"available": False, "reason": None, "close": None, "sma10w": None, "sma40w": None,
           "above_sma10w": None, "above_sma40w": None}
    h = hist
    if len(h) and h.index[-1].weekday() < 4:                    # last bar before Friday: week unfinished
        h = h[h.index < (h.index[-1] - pd.Timedelta(days=int(h.index[-1].weekday())))]
    if not len(h):
        out["reason"] = "no completed week"
        return out
    wk = h["Close"].resample("W-FRI").last().dropna()
    if len(wk) < 40:
        out["reason"] = f"needs 40 completed weeks, have {len(wk)}"
        return out
    s10, s40 = wk.rolling(10).mean().iloc[-1], wk.rolling(40).mean().iloc[-1]
    out.update({"available": True, "close": _f(wk.iloc[-1]), "sma10w": _f(s10), "sma40w": _f(s40),
                "above_sma10w": bool(wk.iloc[-1] > s10), "above_sma40w": bool(wk.iloc[-1] > s40)})
    return out


def _slope(arr: np.ndarray, bars: int = 20) -> Optional[float]:
    if len(arr) > bars and math.isfinite(arr[-1]) and math.isfinite(arr[-1 - bars]) and arr[-1 - bars] != 0:
        return _f((arr[-1] / arr[-1 - bars] - 1) * 100, 2)
    return None


def trend_context(hist: pd.DataFrame) -> dict:
    C = hist["Close"].to_numpy(float)
    close = C[-1]
    e20, e50, s200 = ema(C, 20), ema(C, 50), sma(C, 200)

    def row(arr, label):
        v = arr[-1] if len(arr) else np.nan
        if not math.isfinite(v):
            return {"label": label, "value": None, "distance_pct": None, "slope_20bar_pct": None,
                    "reason": f"needs more bars ({len(C)} available)"}
        return {"label": label, "value": _f(v), "distance_pct": _f((close / v - 1) * 100, 2),
                "slope_20bar_pct": _slope(arr), "reason": None}
    r20, r50, r200 = row(e20, "20-day EMA"), row(e50, "50-day EMA"), row(s200, "200-day SMA")
    align = None
    if all(r["value"] is not None for r in (r20, r50, r200)):
        a, b, c = r20["value"], r50["value"], r200["value"]
        align = "bullish (20 > 50 > 200)" if a > b > c else "bearish (20 < 50 < 200)" if a < b < c else "mixed"
    return {"ema20": r20, "ema50": r50, "sma200": r200, "alignment": align, "weekly": weekly_context(hist),
            "note": "Moving averages lag price; they are context, not entry or exit signals."}


def momentum_context(hist: pd.DataFrame, swings: list) -> dict:
    C = hist["Close"].to_numpy(float)
    r = rsi_wilder(C)
    cur = r[-1] if len(r) else np.nan
    out = {"rsi14": _f(cur, 1), "rsi_change_10bar": None, "context": None,
           "divergence": {"bearish": "not assessable", "bullish": "not assessable"},
           "note": "RSI above 70 or below 30 is a reading about momentum, not a trade signal."}
    if not math.isfinite(cur):
        out["context"] = "RSI undefined (insufficient or flat data)"
        return out
    if len(r) > 10 and math.isfinite(r[-11]):
        out["rsi_change_10bar"] = _f(cur - r[-11], 1)
    out["context"] = ("RSI above 50" if cur > 50 else "RSI at or below 50") + \
                     ("; in the 70+ zone" if cur >= 70 else "; in the 30- zone" if cur <= 30 else "")
    N = len(C)
    for kind, key in (("high", "bearish"), ("low", "bullish")):
        pts = [s for s in swings if s["type"] == kind and s["idx"] >= N - 60]
        if len(pts) < 2:
            continue
        a, b = pts[-2], pts[-1]
        if not (math.isfinite(r[a["idx"]]) and math.isfinite(r[b["idx"]])):
            continue
        if kind == "high":
            hit = b["price"] > a["price"] and r[b["idx"]] < r[a["idx"]]
        else:
            hit = b["price"] < a["price"] and r[b["idx"]] > r[a["idx"]]
        out["divergence"][key] = (f"candidate: {kind}s on {a['date']} and {b['date']} "
                                  f"with RSI {r[a['idx']]:.1f} then {r[b['idx']]:.1f}") if hit else "none between the last two swings"
    return out


# ── plan: invalidation, target, risk, sizing ──────────────────────────────
def structural_invalidation(swings: list, entry_ref: float, atr_last: float) -> Optional[dict]:
    """The most recent confirmed swing low BELOW the reference entry, minus a small ATR buffer. If the
    most recent swing low is already above the entry the structure is broken and the next older low
    below the entry is used. None when no confirmed swing low lies below the entry."""
    lows = [s for s in swings if s["type"] == "low" and s["price"] < entry_ref]
    if not lows or not (atr_last and math.isfinite(atr_last)):
        return None
    s = lows[-1]
    stop = s["price"] - STOP_BUFFER_ATR * atr_last
    if stop <= 0 or stop >= entry_ref:
        return None
    return {"stop": stop, "swing_low": s["price"], "swing_date": s["date"], "swing_confirmed": s["confirmed_date"],
            "buffer_atr": STOP_BUFFER_ATR, "risk_per_share": entry_ref - stop,
            "risk_pct": (entry_ref - stop) / entry_ref * 100.0}


def pick_target(zones: list, entry_ref: float, min_low: float) -> Optional[dict]:
    """The nearest confirmed resistance zone whose low is above both the entry and `min_low`
    (the top of a zone being broken). Reported at the zone's LOW — the first touch, not the top."""
    above = sorted([z for z in zones if z["low"] > max(entry_ref, min_low)], key=lambda z: z["low"])
    if not above:
        return None
    z = above[0]
    return {"price": z["low"], "zone_low": z["low"], "zone_high": z["high"], "touches": z["touches"],
            "last_date": z["last_date"]}


def sizing_illustration(entry_ref: float, stop: float, account: float, risk_pct: float,
                        adv20_shares: Optional[float], adv20_dollar: Optional[float]) -> dict:
    risk_dollars = account * risk_pct / 100.0
    rps = entry_ref - stop
    shares = int(math.floor(risk_dollars / rps))
    capped = False
    if shares * entry_ref > account:
        shares, capped = int(math.floor(account / entry_ref)), True
    value = shares * entry_ref
    return {"account_size_illustration": _f(account, 2), "risk_pct": _f(risk_pct, 2),
            "risk_dollars": _f(risk_dollars, 2), "risk_per_share": _f(rps),
            "shares": shares, "position_value": _f(value, 2), "capped_by_account": capped,
            "actual_risk_dollars": _f(shares * rps, 2),
            "pct_of_adv20_shares": _f(shares / adv20_shares * 100, 3) if adv20_shares else None,
            "pct_of_adv20_dollar_volume": _f(value / adv20_dollar * 100, 3) if adv20_dollar else None,
            "note": "Illustration only: the account size is a placeholder; costs and slippage are not included."}


# ── setup interpretation (descriptive) ────────────────────────────────────
LABEL_TEXT = {
    "weak_or_deteriorating": "Weak or deteriorating structure",
    "trend_pullback": "Pullback within an uptrend structure",
    "breakout_candidate": "Breakout candidate: price just below a tested resistance zone",
    "breakout_occurred": "Recent breakout above a resistance zone",
    "trend_extended": "Uptrend structure with no nearby resistance and no pullback",
    "unclassified": "Mixed structure: no description applies",
}


def interpret_setup(structure: dict, close: float, atr_last: float, zones: list, sma200: Optional[float],
                    ema50: Optional[float], breakout: Optional[dict]) -> dict:
    st = structure["structure"]
    if st == "downtrend" or (sma200 is not None and ema50 is not None and close < sma200 and ema50 < sma200):
        return {"label": "weak_or_deteriorating"}
    if st == "uptrend":
        if breakout is not None:
            return {"label": "breakout_occurred", "breakout": breakout}
        res = sorted([z for z in zones if z["low"] >= close], key=lambda z: z["low"])
        if res and (res[0]["low"] - close) <= BREAKOUT_NEAR_ATR * atr_last:
            return {"label": "breakout_candidate", "zone": res[0]}
        sh, sl = structure["last_swing_high"], structure["last_swing_low"]
        if sh and sl and close <= sh["price"] - PULLBACK_ATR * atr_last and close > sl["price"]:
            return {"label": "trend_pullback", "swing_high": sh}
        return {"label": "trend_extended"}
    return {"label": "unclassified"}


# ── evidence lists ────────────────────────────────────────────────────────
def _evidence(card: dict) -> tuple:
    fo, ag = [], []
    st, tr, vol = card["structure"], card["trend"], card["volume"]
    if st and st.get("structure") == "uptrend":
        fo.append("Last two swing highs and lows are both higher (HH/HL).")
    if st and st.get("structure") == "downtrend":
        ag.append("Last two swing highs and lows are both lower (LH/LL).")
    if st and st.get("break_of_structure"):
        (ag if st["break_of_structure"].startswith("bearish") else fo).append(
            "Structure break — " + st["break_of_structure"] + ".")
    for key, nm in (("ema20", "20-day EMA"), ("ema50", "50-day EMA"), ("sma200", "200-day SMA")):
        r = tr[key]
        if r["value"] is not None:
            (fo if r["distance_pct"] > 0 else ag).append(
                f"Close is {abs(r['distance_pct']):.1f}% {'above' if r['distance_pct'] > 0 else 'below'} the {nm}.")
    if tr["alignment"] and tr["alignment"].startswith("bullish"):
        fo.append("Moving averages are aligned bullishly (20 > 50 > 200).")
    if tr["alignment"] and tr["alignment"].startswith("bearish"):
        ag.append("Moving averages are aligned bearishly (20 < 50 < 200).")
    wk = tr["weekly"]
    if wk["available"]:
        (fo if wk["above_sma40w"] else ag).append(
            f"Weekly close is {'above' if wk['above_sma40w'] else 'below'} the 40-week SMA.")
    if vol["rvol_last"] is not None:
        if vol["rvol_last"] >= RVOL_CONFIRM:
            fo.append(f"Last session volume is {vol['rvol_last']:.1f}x its 50-session average.")
        elif vol["rvol_last"] < 0.7:
            ag.append(f"Last session volume is only {vol['rvol_last']:.1f}x its 50-session average.")
    udr = vol["up_down_volume_ratio_20"]
    if udr is not None and udr > 1.2:
        fo.append(f"Up-day volume averages {udr:.2f}x down-day volume over 20 sessions.")
    elif udr is not None and udr < 0.8:
        ag.append(f"Up-day volume averages only {udr:.2f}x down-day volume over 20 sessions.")
    mo = card["momentum"]
    if mo["divergence"]["bearish"].startswith("candidate"):
        ag.append("Bearish RSI divergence candidate between the last two swing highs.")
    if mo["divergence"]["bullish"].startswith("candidate"):
        fo.append("Bullish RSI divergence candidate between the last two swing lows.")
    plan, rc = card["plan"], card["risk_context"]
    inv = plan.get("invalidation") if plan else None
    if inv and inv["risk_pct"] > WIDE_STOP_PCT:
        ag.append(f"Structural stop is {inv['risk_pct']:.1f}% from entry (wide).")
    if rc and rc["earnings"]["flag"]:
        ag.append(f"Earnings in {rc['earnings']['days_until']} day(s): binary event risk.")
    if rc and rc["liquidity"].get("participation_flag"):
        ag.append("Illustrated position exceeds 1% of average daily volume.")
    return fo, ag


# ── the card ──────────────────────────────────────────────────────────────
def _blank(ticker, market, reason, status, notes=None, observed_at=None) -> dict:
    return {"card_version": CARD_VERSION, "ticker": ticker, "market": market, "observed_at": observed_at,
            "observation_date": None, "price": None, "market_context": None, "trend": None, "structure": None,
            "levels": None, "volume": None, "momentum": None, "setup": None,
            "plan": {"status": status, "status_text": reason, "entry_condition": None, "invalidation": None,
                     "target": None, "target_reason": "Insufficient data", "reward_risk": None, "sizing": None},
            "risk_context": None, "evidence_for": [], "evidence_against": [],
            "missing_data": [reason], "confidence_limits": [PRICE_BASIS], "data": {"notes": notes or []},
            "disclaimer": DISCLAIMER}


def build_card(ticker: str, hist: pd.DataFrame, market: str = "US", asof=None,
               observed_at: Optional[str] = None, now: Optional[datetime] = None,
               drop_incomplete: bool = True, account_size: Optional[float] = None,
               risk_pct: float = DEFAULT_RISK_PCT, earnings_date: Optional[str] = None,
               market_context: Optional[dict] = None) -> dict:
    """Builds the research card. `asof` cuts the history FIRST so nothing later can influence the card."""
    if str(market).upper() != "US":
        return _blank(ticker, market, "Unvalidated market: US rules (calendar, liquidity, costs) do not transfer "
                                      "to NGX or other markets; no card is produced.", "unvalidated_market",
                      observed_at=observed_at)
    h, notes = prepare_history(hist, now=now, drop_incomplete=drop_incomplete)
    if asof is not None:
        h = h.loc[:pd.Timestamp(asof)]
    if len(h) < MIN_BARS:
        return _blank(ticker, market, f"Insufficient data: {len(h)} complete daily bars, {MIN_BARS} needed.",
                      "insufficient_data", notes, observed_at)

    close = float(h["Close"].iloc[-1])
    H, L, C = h["High"].to_numpy(float), h["Low"].to_numpy(float), h["Close"].to_numpy(float)
    atr = atr_wilder(H, L, C)
    atr_last = float(atr[-1])
    swings = find_swings(h)
    structure = classify_structure(swings, close)
    zones = cluster_zones(swings, atr_last)
    supports = sorted([z for z in zones if z["high"] <= close], key=lambda z: -z["high"])
    resistances = sorted([z for z in zones if z["low"] >= close], key=lambda z: z["low"])
    in_zone = next((z for z in zones if z["low"] < close < z["high"]), None)
    breakout = _recent_breakout(h, swings, atr)
    trend = trend_context(h)
    volume = volume_context(h)
    momentum = momentum_context(h, swings)
    sma200 = trend["sma200"]["value"]
    ema50 = trend["ema50"]["value"]
    setup = interpret_setup(structure, close, atr_last, zones, sma200, ema50, breakout)
    label = setup["label"]

    # entry reference and condition (a plan to evaluate, not an instruction)
    entry_ref, entry_basis, entry_condition, min_target_low = None, None, None, close
    if label == "trend_pullback":
        entry_ref, entry_basis = setup["swing_high"]["price"], "last swing high"
        entry_condition = (f"Close above {entry_ref:.2f} (the last swing high, {setup['swing_high']['date']}): "
                           "structure resumption after the pullback.")
    elif label == "breakout_candidate":
        z = setup["zone"]
        entry_ref, entry_basis, min_target_low = z["high"], "top of resistance zone", z["high"]
        where = (f"the swing high {z['high']:.2f}" if z["low"] == z["high"]
                 else f"top of the {z['touches']}-touch zone {z['low']:.2f}–{z['high']:.2f}")
        entry_condition = (f"Close above {z['high']:.2f} ({where}) with volume at least "
                           f"{RVOL_CONFIRM:.1f}x the 50-session average.")
    elif label == "breakout_occurred":
        b = setup["breakout"]
        entry_ref, entry_basis, min_target_low = close, "last complete close", b["zone_high"]
        entry_condition = (f"Breakout above {b['zone_high']:.2f} closed on {b['date']} "
                           f"({b['bars_ago']} bar(s) ago); volume "
                           + ("confirmed" if b["volume_confirmed"] else "NOT confirmed") + ".")
    inval = structural_invalidation(swings, entry_ref, atr_last) if entry_ref else None
    target = pick_target(zones, entry_ref, min_target_low) if entry_ref else None
    if entry_ref is None:
        target_reason = "No entry condition is defined for this description."
    elif target is None:
        target_reason = ("No defensible target: no confirmed swing-high zone lies above the entry "
                         "within the history, so any target would be a guess.")
    else:
        target_reason = None
    rr = None
    if inval and target:
        reward = target["price"] - entry_ref
        if reward > 0 and inval["risk_per_share"] > 0:
            rr = reward / inval["risk_per_share"]

    if label == "weak_or_deteriorating":
        status, status_text = "no_long_setup", "No long setup: structure is weak or deteriorating."
    elif label in ("unclassified", "trend_extended"):
        status, status_text = "no_long_setup", "No entry condition is defined for the current structure."
    elif label == "breakout_occurred" and setup["breakout"]["volume_confirmed"] and inval:
        status, status_text = "trigger_met", "Trigger conditions met on the last complete bar (not a validated setup)."
    else:
        status, status_text = "research_candidate", "Research candidate: the entry condition has not been met."

    # event / liquidity context
    earn = {"date": earnings_date, "days_until": None, "flag": False}
    if earnings_date:
        try:
            du = (pd.Timestamp(earnings_date).date() - h.index[-1].date()).days
            earn.update({"days_until": du, "flag": 0 <= du <= EARNINGS_WARN_DAYS})
        except Exception:
            earn["date"] = None
    adv_sh, adv_usd = volume["avg_volume_20"], volume["avg_dollar_volume_20"]
    sizing = None
    if inval and entry_ref:
        sizing = sizing_illustration(entry_ref, inval["stop"], DEFAULT_ACCOUNT if account_size is None else account_size,
                                     risk_pct, adv_sh, adv_usd)
    liq = {"avg_volume_20": adv_sh, "avg_dollar_volume_20": adv_usd, "spread": "unavailable (no bid/ask feed)",
           "participation_flag": bool(sizing and sizing["pct_of_adv20_shares"] is not None
                                      and sizing["pct_of_adv20_shares"] > ADV_PARTICIPATION_WARN_PCT)}

    ma200_stop_dist = _f((close / sma200 - 1) * 100, 2) if sma200 else None
    plan = {"status": status, "status_text": status_text, "entry_condition": entry_condition,
            "entry_ref": _f(entry_ref), "entry_ref_basis": entry_basis,
            "invalidation": None if not inval else {
                "stop": _f(inval["stop"]), "swing_low": _f(inval["swing_low"]), "swing_date": inval["swing_date"],
                "swing_confirmed": inval["swing_confirmed"], "buffer_atr": inval["buffer_atr"],
                "risk_per_share": _f(inval["risk_per_share"]), "risk_pct": _f(inval["risk_pct"], 2),
                "ma200_distance_pct_for_comparison": ma200_stop_dist},
            "target": None if not target else {k: (_f(v) if isinstance(v, float) else v) for k, v in target.items()},
            "target_reason": target_reason, "reward_risk": _f(rr, 3), "sizing": sizing}
    if entry_ref and not inval:
        plan["invalidation_reason"] = "No confirmed swing low below the entry: no structural invalidation level."

    missing = []
    if sma200 is None:
        missing.append("200-day SMA unavailable (fewer than 200 bars)")
    if not trend["weekly"]["available"]:
        missing.append("Weekly context unavailable: " + str(trend["weekly"]["reason"]))
    if volume["rvol_last"] is None:
        missing.append("Relative volume unavailable (needs 51 bars)")
    if not earnings_date:
        missing.append("Earnings date unavailable")
    if market_context is None:
        missing.append("Broader market context not supplied")
    if entry_ref and not inval:
        missing.append("No confirmed swing low below the entry")
    missing.append("Bid/ask spread unavailable")

    card = {"card_version": CARD_VERSION, "ticker": ticker, "market": "US", "observed_at": observed_at,
            "observation_date": _d(h.index[-1]), "price": {"close": _f(close), "date": _d(h.index[-1])},
            "market_context": market_context,
            "trend": trend,
            "structure": {"structure": structure["structure"], "higher_highs": structure["higher_highs"],
                          "higher_lows": structure["higher_lows"], "lower_highs": structure["lower_highs"],
                          "lower_lows": structure["lower_lows"], "break_of_structure": structure["break_of_structure"],
                          "swings": [{k: (_f(v) if isinstance(v, float) else v) for k, v in s.items()
                                      if k in ("type", "date", "price", "confirmed_date")} for s in swings[-8:]],
                          "method": (f"Fractal pivots, {SWING_N} bars each side, confirmed {SWING_N} bars later; a pivot counts "
                                     f"only if price reacted at least {SWING_REACTION_ATR:g} ATR within those {SWING_N} bars. "
                                     "Once confirmed a swing never changes.")},
            "levels": {"supports": [{k: (_f(v) if isinstance(v, float) else v) for k, v in z.items()} for z in supports[:3]],
                       "resistances": [{k: (_f(v) if isinstance(v, float) else v) for k, v in z.items()} for z in resistances[:3]],
                       "in_zone": None if not in_zone else {k: (_f(v) if isinstance(v, float) else v) for k, v in in_zone.items()},
                       "recent_breakout": breakout,
                       "note": "Zones are candidate areas from past swing points; price can pass straight through them."},
            "volume": volume, "momentum": momentum,
            "setup": {"label": label, "text": LABEL_TEXT[label],
                      "note": "A description of the chart. Setup types have not been shown to predict returns."},
            "plan": plan,
            "risk_context": {"earnings": earn, "liquidity": liq, "wide_stop": bool(inval and inval["risk_pct"] > WIDE_STOP_PCT)},
            "missing_data": missing,
            "confidence_limits": [PRICE_BASIS, "Swing levels are decided with a 5-bar lag by construction.",
                                  "No costs, spreads or slippage are modelled.",
                                  "One market episode of evidence so far; see the Alpha Lab independence check."],
            "data": {"bars": int(len(h)), "first_date": _d(h.index[0]), "last_date": _d(h.index[-1]),
                     "price_basis": PRICE_BASIS, "notes": notes},
            "disclaimer": DISCLAIMER}
    card["evidence_for"], card["evidence_against"] = _evidence(card)
    return card
