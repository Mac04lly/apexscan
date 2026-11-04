"""
audit/replay.py - Test A (historical rule replay) for ApexScan, using the scanner's OWN functions (imported from this repo).

What is replayed for each date t (using only bars <= t, a trailing 365-calendar-day window like yf period='1y'):
  above_200ma, perf_3m (63 bars), rs_3m (63 bars, vs ^GSPC), vol_filter, stage, fresh 200/50-day crosses, base breakout.
What is NOT replayed: the Apex Score (needs order flow, earnings, weekly and fundamental fields) -> reported as 'not replayed'.
This is Test A. It is NOT evidence of what the live app displayed on any day (that is Test B: the saved reports).
Read-only: nothing here writes to data/, calls run_scan, or changes any setting.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, yaml
import scanner as S

CFG = yaml.safe_load(open(REPO / "config.yaml", encoding="utf-8"))
TH = CFG["thresholds"]["us"]
MIN_3M, MIN_VOL = TH["min_3m_perf"], TH["min_volume"]
RS_FLOOR = -30.0                          # run_scan: `data["rs_3m"] < _min_rs` with _min_rs = -30 (non-gem)
MIN_BARS = CFG["scan"]["min_history_bars"]
WINDOW_DAYS = 365                         # yfinance period="1y"

def window(df, t, days=WINDOW_DAYS):
    return df[(df.index > t - pd.Timedelta(days=days)) & (df.index <= t)]

MIN_BENCH_BARS = 50                       # scanner.get_benchmark: `if len(data) > 50`
LOOKBACK_FOR_MA200 = 200
BENCH_ROLES = {"us_market": ("^GSPC", None), "russell_2500": ("^R25I", "SMMD"), "russell_3000_growth": ("^RAG", "IWZ")}

def select_benchmark(primary, fallback, t):
    """ApexScan's get_benchmark rule, applied to the data visible at t: primary if it has > 50 usable bars, else the
    fallback if IT has > 50, else UNAVAILABLE.  `primary`/`fallback` are (symbol, Series|None). Never substitutes anything else.
    Returns (series|None, symbol_used|'UNAVAILABLE', note)."""
    for sym, ser, is_fb in ((primary[0], primary[1], False), (fallback[0], fallback[1], True)):
        if sym is None or ser is None: continue
        w = window(ser.dropna(), t)
        if len(w) > MIN_BENCH_BARS:
            return w, sym, ("fallback in use (primary has <= 50 bars)" if is_fb else "primary")
    return None, "UNAVAILABLE", "no benchmark with > 50 bars"

def _missing_reason(h, cal_win, need_hl=False):
    """Why a signal day cannot be evaluated (None if it can). Nothing is ever filled."""
    last200 = h.iloc[-LOOKBACK_FOR_MA200:]
    if last200["Close"].isna().any():  return "missing_close_in_lookback"
    first = h.index[0] if len(h) else None
    if cal_win is not None and first is not None:
        need = [d for d in cal_win[-LOOKBACK_FOR_MA200:] if d >= first]
        if len(need) > len(set(need) & set(h.index)): return "missing_session_in_lookback"
    if h["Volume"].iloc[-20:].isna().any(): return "missing_volume_in_lookback"
    if need_hl and (h[["High", "Low", "Volume"]].iloc[-41:].isna().any().any()): return "missing_high_low_volume_for_breakout"
    return None

def replay_ticker(hist: pd.DataFrame, bench, dates=None, calendar=None, bench_r2500=(None, None, None), bench_r3000g=(None, None, None),
                  emulate_silent_failure=False):
    """hist: date-indexed adjusted OHLCV. bench: ^GSPC close Series, or {'^GSPC': Series}.
    bench_r2500 / bench_r3000g: (primary_symbol, primary_series, fallback_symbol, fallback_series) or ignored.
    calendar: session dates (the ^GSPC dates); used to detect absent sessions. Returns one row per date, including
    which benchmark series was actually used and why a date could not be evaluated.
    emulate_silent_failure=True reproduces the LIVE scanner's behaviour (relative strength 0.0 when the benchmark is
    unavailable); it is for Test B diagnostics only and the row is labelled. The default is to mark RS not evaluable."""
    hist = hist.sort_index()
    us = bench if isinstance(bench, dict) else {"^GSPC": bench}
    us_ser = us.get("^GSPC")
    cal = pd.DatetimeIndex(calendar) if calendar is not None else (pd.DatetimeIndex(us_ser.dropna().index) if us_ser is not None else None)
    dates = hist.index if dates is None else pd.DatetimeIndex(dates)
    out = []
    for t in dates:
        h = window(hist, t); n = len(h)
        row = {"date": t, "bars": n}
        if n < MIN_BARS:
            row.update(eligible_history=False, evaluable=False, exclusion_reason="insufficient_history"); out.append(row); continue
        row["eligible_history"] = True
        cal_win = list(window(pd.Series(0, index=cal), t).index) if cal is not None else None
        reason = _missing_reason(h, cal_win)
        # benchmark actually used, by role, recorded on every row
        b, bsym, bnote = select_benchmark(("^GSPC", us_ser), (None, None), t)
        row.update(bench_us_used=bsym, bench_us_note=bnote)
        for role, spec in (("r2500", bench_r2500), ("r3000g", bench_r3000g)):
            if spec and spec[0]:
                _, sym, note = select_benchmark((spec[0], spec[1]), (spec[2], spec[3]), t)
                row.update({f"bench_{role}_used": sym, f"bench_{role}_note": note})
        if reason is None and b is not None:        # RS uses the scanner's lookback on the stock's own sessions
            last64 = h.index[-64:]
            if not set(last64) <= set(b.index): reason = "benchmark_missing_sessions"
        if reason is not None:
            row.update(evaluable=False, exclusion_reason=reason); out.append(row); continue
        c = h["Close"]; px = float(c.iloc[-1])
        ma50 = c.rolling(50).mean().iloc[-1]; ma200 = c.rolling(200).mean().iloc[-1]
        above200 = bool(px > ma200)                      # <200 bars -> NaN average -> False, exactly as in the scanner
        perf3 = S.performance_pct(c, 63)
        if b is None:
            rs = 0.0 if emulate_silent_failure else None
            row["rs_note"] = "scanner would silently use 0.0" if emulate_silent_failure else "benchmark unavailable: RS not evaluable"
        else:
            rs = S.compute_rs(c, b, 63)
        volf = max(int(h["Volume"].iloc[-1]), int(h["Volume"].rolling(20).mean().iloc[-1]))
        ee = S.detect_early_entry(c, ma50, ma200, h)
        brk_reason = _missing_reason(h, cal_win, need_hl=True)
        brk, pat = (None, "not evaluable: " + brk_reason) if brk_reason else S.detect_base_breakout(h)
        g = {"g1_above_200ma": above200, "g2_perf3m_ok": perf3 >= MIN_3M,
             "g3_rs_ok": (rs >= RS_FLOOR) if rs is not None else None, "g4_volume_ok": volf >= MIN_VOL}
        row.update(evaluable=True, exclusion_reason="", close=px, ma50=float(ma50) if pd.notna(ma50) else np.nan,
                   ma200=float(ma200) if pd.notna(ma200) else np.nan, perf_3m=perf3, rs_3m=rs, vol_filter=volf,
                   stage=S.detect_stage(px, ma50, ma200) if pd.notna(ma200) else "n/a (<200 bars)",
                   fresh_200ma_cross=ee["fresh_200ma_cross"], days_since_200ma_cross=ee["days_since_200ma_cross"],
                   fresh_50ma_cross=ee["fresh_50ma_cross"], breakout=brk, pattern=pat, **g)
        # all four gates must be evaluable; an unavailable RS makes the day NOT evaluable for E1-E4, never a silent pass
        row["passes_replayed_gates"] = (all([g["g1_above_200ma"], g["g2_perf3m_ok"], g["g3_rs_ok"], g["g4_volume_ok"]])
                                        if rs is not None else None)
        out.append(row)
    return pd.DataFrame(out).set_index("date")

GATES = ["g1_above_200ma", "g2_perf3m_ok", "g3_rs_ok", "g4_volume_ok", "passes_replayed_gates"]


def first_pass_detail(r: pd.DataFrame, since=None, calendar=None) -> dict:
    """For each gate (and all four together): the start of the VERIFIED unbroken passing run that ends at the last replayed
    session, with its status. Passing sessions continue a run; a failed session, an unevaluable session, and (when `calendar` is
    given) a session with no replay row all break it. Nothing is bridged. Shares audit.cohorts.session_states/verified_run with
    cohorts.four_gate_lower_bound so the two cannot disagree.
    Returns {gate: {"date": ISO date or None, "status": ..., "note": ...}}."""
    from audit.cohorts import session_states, verified_run
    if since is not None:
        r = r[r.index >= pd.Timestamp(since)]
    res = {}
    for g in GATES:
        v = verified_run(session_states(r, g, calendar))
        res[g] = {"date": v["start"].date().isoformat() if v["start"] is not None else None, "status": v["status"], "note": v["note"]}
    return res


def first_pass_dates(r: pd.DataFrame, since=None, calendar=None):
    """Date-only view of first_pass_detail (the earlier return format): the start of the verified run, None when nothing was
    evaluable, 'not passing on last date' when the final session fails, 'last session not evaluable' when it cannot be assessed.
    A date may carry an uncertain start: see first_pass_detail for the status."""
    labels = {"gates_not_met_on_recorded_date": "not passing on last date", "recorded_date_not_evaluable": "last session not evaluable"}
    return {g: (d["date"] if d["date"] else labels.get(d["status"])) for g, d in first_pass_detail(r, since, calendar).items()}
