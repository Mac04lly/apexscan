"""audit/maturity.py - which recorded discoveries have completed a holding period (Amendment 2, item 2).

Entry = open of the first session AFTER the scan date. A discovery is matured at horizon H only if the open H sessions
after its entry exists, i.e. entry_position + H <= position of the last completed session. Nothing is projected."""
import datetime as dt
import pandas as pd

# NYSE full-day closures in 2026 (the only weekday gaps in the calendar used when no price calendar is available).
NYSE_HOLIDAYS_2026 = [dt.date(2026, 1, 1), dt.date(2026, 1, 19), dt.date(2026, 2, 16), dt.date(2026, 4, 3), dt.date(2026, 5, 25),
                      dt.date(2026, 6, 19), dt.date(2026, 7, 3), dt.date(2026, 9, 7), dt.date(2026, 11, 26), dt.date(2026, 12, 25)]
HORIZONS = (21, 63, 126)
BANDS = [(-10**6, -1, "no entry session yet"), (0, 0, "0 sessions"), (1, 20, "1-20 sessions"), (21, 41, "21-41 sessions"),
         (42, 62, "42-62 sessions"), (63, 10**6, "63+ sessions")]

def sessions_2026() -> pd.DatetimeIndex:
    days = pd.bdate_range("2026-01-01", "2026-12-31")
    return pd.DatetimeIndex([d for d in days if d.date() not in set(NYSE_HOLIDAYS_2026)])

def last_completed_session(today: dt.date, cal: pd.DatetimeIndex = None) -> pd.Timestamp:
    """The latest session strictly before `today` (a session is treated as complete only on a later calendar day)."""
    cal = sessions_2026() if cal is None else cal
    prior = cal[cal < pd.Timestamp(today)]
    return prior[-1]

def table(discoveries, last_session, cal: pd.DatetimeIndex = None, horizons=HORIZONS) -> pd.DataFrame:
    cal = sessions_2026() if cal is None else pd.DatetimeIndex(cal)
    last = cal.get_loc(pd.Timestamp(last_session))
    rows = []
    for d in discoveries:
        sd = pd.Timestamp(str(d["discovered_at"])[:10])
        pe = int(cal.searchsorted(sd, side="right"))
        has_entry = pe <= last
        age = last - pe if has_entry else -1
        r = {"ticker": d["ticker"], "discovered_at": sd.date().isoformat(), "entry_session": cal[pe].date().isoformat() if pe < len(cal) else "",
             "has_entry": has_entry, "sessions_since_entry": age}
        for h in horizons: r[f"matured_{h}"] = bool(has_entry and pe + h <= last)
        rows.append(r)
    return pd.DataFrame(rows)

def summary(df: pd.DataFrame, horizons=HORIZONS) -> dict:
    s = {"discoveries": len(df), "no_entry_yet": int((~df.has_entry).sum())}
    for h in horizons: s[f"matured_{h}"] = int(df[f"matured_{h}"].sum()); s[f"immature_{h}"] = len(df) - s[f"matured_{h}"]
    s["bands"] = {name: int(((df.sessions_since_entry >= lo) & (df.sessions_since_entry <= hi)).sum()) for lo, hi, name in BANDS}
    return s

def earliest_maturity(df: pd.DataFrame, cal: pd.DatetimeIndex, h: int):
    """Date of the open that completes horizon h for the earliest-entered discovery, or None if beyond the calendar."""
    e = df[df.entry_session != ""]
    if e.empty: return None
    pe = cal.get_loc(pd.Timestamp(min(e.entry_session)))
    return cal[pe + h].date() if pe + h < len(cal) else None
