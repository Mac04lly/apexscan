"""
execution.py - trade construction rules for the registered audit (Amendments 1-7). Nothing is filled or defaulted:
any trade that cannot be built from real, present data is EXCLUDED and the reason is returned.

Sessions come from the reference calendar (^GSPC dates), never from the stock's own rows, so a missing stock row cannot
silently shift "63 sessions later".
"""
import numpy as np, pandas as pd

TIERS = ((20e6, 0.0010), (2e6, 0.0030), (0.0, 0.0075))      # registered: median 20-day dollar volume -> cost per side
H_DEFAULT = 63

def cost_per_side(median_dollar_volume: float, mult: float = 1.0) -> float:
    for floor, c in TIERS:
        if median_dollar_volume >= floor: return c * mult
    raise ValueError("unreachable")

def _excl(reason, **kw): return {"status": "excluded", "reason": reason, **kw}

def _open_on(hist, d):
    """(value|None, reason|None) for the Open on session d; absent row and NaN are different reasons."""
    if d not in hist.index: return None, "session_missing"
    v = hist.at[d, "Open"]
    if pd.isna(v): return None, "open_missing"
    if v <= 0: return None, "nonpositive_open"
    return float(v), None

def liquidity(hist, cal, t):
    """median Close*Volume over the 20 sessions ending at t; all 20 must be present and non-missing."""
    pos = cal.get_loc(t); ss = cal[max(0, pos - 19): pos + 1]
    if len(ss) < 20 or not set(ss) <= set(hist.index): return None
    w = hist.loc[ss, ["Close", "Volume"]]
    if w.isna().any().any(): return None
    return float((w["Close"] * w["Volume"]).median())

def plan_trade(hist: pd.DataFrame, cal: pd.DatetimeIndex, t, H: int = H_DEFAULT, cost_mult: float = 1.0,
               exit_pos_override: int = None, entry_pos_override: int = None):
    """Signal on the close of session t (data <= t only). Entry = Open of the next session; exit = Open H sessions after entry."""
    hist = hist.sort_index(); t = pd.Timestamp(t)
    if t not in cal: return _excl("signal_date_not_a_session", signal_date=t)
    p = cal.get_loc(t); pe = p + 1 if entry_pos_override is None else entry_pos_override
    px = pe + H if exit_pos_override is None else exit_pos_override
    base = {"signal_date": t}
    if pe >= len(cal): return _excl("entry_session_not_yet_available", **base)
    if px >= len(cal): return _excl("horizon_incomplete", entry_date=cal[pe], **base)
    de, dx = cal[pe], cal[px]; base.update(entry_date=de, exit_date=dx, holding_sessions=px - pe)
    dv = liquidity(hist, cal, t)
    if dv is None: return _excl("cost_tier_unavailable", **base)
    eo, r = _open_on(hist, de)
    if r: return _excl(f"entry_{r}", **base)
    xo, r = _open_on(hist, dx)
    if r:
        stopped = hist.index.max() < dx
        return _excl("no_exit_session_stock_stopped_trading" if (stopped and r == "session_missing") else f"exit_{r}", **base)
    c = cost_per_side(dv, cost_mult)
    net = xo * (1 - c) / (eo * (1 + c)) - 1
    # excursions use High/Low of sessions entry..exit-1 plus the exit open; any gap -> excursion unavailable (trade kept)
    ss = cal[pe:px]
    hl = hist.reindex(ss)[["High", "Low"]]
    if hl.isna().any().any():
        mfe = mae = np.nan; ex = "excursion_unavailable_missing_high_low"
    else:
        mfe = max(hl["High"].max(), xo) / eo - 1; mae = min(hl["Low"].min(), xo) / eo - 1; ex = "ok"
    return {"status": "ok", "reason": "", **base, "entry_open": eo, "exit_open": xo, "cost_side": c, "median_dollar_volume": dv,
            "cost_side_1x": cost_per_side(dv, 1.0), "net_return": net, "mfe": mfe, "mae": mae, "excursion": ex}

def net_at(entry_open, exit_open, cost_side_1x, mult):
    """Net return of a trade at a cost multiplier (0.5, 1, 2). Gross prices are unchanged; only the per-side cost scales."""
    c = cost_side_1x * mult
    return exit_open * (1 - c) / (entry_open * (1 + c)) - 1

def paired_e2_vs_e1(hist, cal, t_e2, t_e1, H=H_DEFAULT, cost_mult=1.0, max_gap=60):
    """Amendment 5. E2 enters at the open after its signal; E1 enters at the open after its signal; BOTH exit at the open
    H sessions after the E1 entry. Each pays its own cost tier on both sides. The E2 leg is held longer; both holding
    periods are returned so the difference is always visible."""
    cal = pd.DatetimeIndex(cal); t_e2, t_e1 = pd.Timestamp(t_e2), pd.Timestamp(t_e1)
    if t_e1 <= t_e2: return _excl("e1_not_after_e2")
    gap = cal.get_loc(t_e1) - cal.get_loc(t_e2)
    if gap > max_gap: return _excl("e1_more_than_60_sessions_after_e2", gap_sessions=gap)
    one = plan_trade(hist, cal, t_e1, H, cost_mult)
    if one["status"] != "ok": return _excl("e1_leg_" + one["reason"], gap_sessions=gap)
    xpos = cal.get_loc(one["exit_date"])
    two = plan_trade(hist, cal, t_e2, H, cost_mult, exit_pos_override=xpos)
    if two["status"] != "ok": return _excl("e2_leg_" + two["reason"], gap_sessions=gap)
    return {"status": "ok", "reason": "", "gap_sessions": gap, "common_exit_date": one["exit_date"],
            "e1_net": one["net_return"], "e2_net": two["net_return"], "diff_e2_minus_e1": two["net_return"] - one["net_return"],
            "e1_holding_sessions": one["holding_sessions"], "e2_holding_sessions": two["holding_sessions"],
            "e1_entry_open": one["entry_open"], "e2_entry_open": two["entry_open"], "exit_open": one["exit_open"],
            "e1_cost_side": one["cost_side"], "e2_cost_side": two["cost_side"]}

def exclusion_table(records):
    """records: iterable of dicts with ticker, approach and a plan_trade/paired result. Counts by ticker x approach x reason."""
    df = pd.DataFrame([{"ticker": r["ticker"], "approach": r["approach"], "reason": r["reason"]} for r in records if r["status"] != "ok"])
    if df.empty: return pd.DataFrame(columns=["ticker", "approach", "reason", "n"])
    return df.groupby(["ticker", "approach", "reason"]).size().reset_index(name="n")
