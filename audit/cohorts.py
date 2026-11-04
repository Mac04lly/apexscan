"""audit/cohorts.py - Amendment 3: cohort labels, date concepts and dependence reporting (read-only helpers).

A1 = initial persistence/seeding cohort (recorded discovery date 2026-08-06); A2 = post-seeding first-recorded discovery
cohort. Neither is described as confirmed first-time qualification. Three date concepts are kept apart: the recorded
discovery date, the reconstructed four-gate date (a LOWER BOUND, not an ApexScan discovery date) and the actual full scanner
qualification date (not established for any record)."""
import pandas as pd

SEEDING_DATE = "2026-08-06"
A1 = "A1 - Initial persistence/seeding cohort"
A2 = "A2 - Post-seeding first-recorded discovery cohort"
FULL_QUALIFICATION_DATE = "not established"          # the historical Apex Score cannot be replayed
LOWER_BOUND_NOTE = ("Earliest date the four replayable gates (200-day average, 3-month return, relative strength, volume) were "
                    "all met. A lower bound on full scanner qualification, not an ApexScan discovery date.")
LIMITATION = ("The original qualification dates of the A1 records cannot be established from saved reports or Git history. "
              "A2 dates are first-recorded dates; some A2 stocks may have qualified earlier.")


def cohort_of(discovered_at) -> str:
    return A1 if str(discovered_at)[:10] == SEEDING_DATE else A2


def add_cohort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); out["cohort"] = [cohort_of(x) for x in out["discovered_at"]]; return out


def _windows(entry_pos: pd.Series, horizon: int):
    """Holding window of a trade = entry position .. entry position + horizon, inclusive."""
    return list(zip(entry_pos.astype(int), entry_pos.astype(int) + horizon))


def max_simultaneous(windows) -> int:
    ev = {}
    for a, b in windows:
        ev[a] = ev.get(a, 0) + 1; ev[b + 1] = ev.get(b + 1, 0) - 1
    cur = best = 0
    for k in sorted(ev): cur += ev[k]; best = max(best, cur)
    return best


def overlapping_pairs(windows) -> int:
    """Number of pairs of holding windows that share at least one session."""
    w = sorted(windows); n = 0
    for i, (a, b) in enumerate(w):
        for c, d in w[i + 1:]:
            if c > b: break
            n += 1
    return n


def independent_trades(trades: pd.DataFrame, horizon: int, ticker_col="ticker", entry_col="entry_pos") -> pd.DataFrame:
    """Amendment 3 item 6. Per ticker: order by entry session, accept the first, accept a later one only if its entry session
    is STRICTLY AFTER the exit session (entry + horizon) of the last accepted trade. Returns the accepted rows."""
    keep = []
    for _, g in trades.sort_values([ticker_col, entry_col], kind="stable").groupby(ticker_col, sort=False):
        last_exit = None
        for idx, pos in zip(g.index, g[entry_col]):
            if last_exit is None or pos > last_exit:
                keep.append(idx); last_exit = pos + horizon
    return trades.loc[keep]


def dependence_table(df: pd.DataFrame, horizon: int, date_col="discovered_at", entry_col="entry_pos", ticker_col="ticker") -> dict:
    """Counts that show how many independent observations stand behind a set of records."""
    if df.empty:
        return {"records": 0, "unique_tickers": 0, "distinct_recorded_dates": 0, "distinct_entry_dates": 0,
                "largest_single_date_share": 0.0, "max_simultaneous_open_windows": 0, "overlapping_window_pairs": 0,
                "non_overlapping_trades": 0}
    w = _windows(df[entry_col], horizon)
    share = float(df.groupby(date_col).size().max() / len(df))
    return {"records": int(len(df)), "unique_tickers": int(df[ticker_col].nunique()),
            "distinct_recorded_dates": int(df[date_col].nunique()), "distinct_entry_dates": int(df[entry_col].nunique()),
            "largest_single_date_share": round(share, 4), "max_simultaneous_open_windows": max_simultaneous(w),
            "overlapping_window_pairs": overlapping_pairs(w),
            "non_overlapping_trades": int(len(independent_trades(df, horizon, ticker_col, entry_col)))}


def population_a_tables(maturity_df: pd.DataFrame, cal: pd.DatetimeIndex, horizon: int = 63) -> pd.DataFrame:
    """One row per cohort (A1, A2). Deliberately no pooled row."""
    d = add_cohort(maturity_df[maturity_df.entry_session != ""].copy())
    d["entry_pos"] = [int(cal.get_loc(pd.Timestamp(x))) for x in d.entry_session]
    rows = []
    for name in (A1, A2):
        r = dependence_table(d[d.cohort == name], horizon); r["cohort"] = name
        sub = maturity_df.merge(d[["ticker", "cohort"]], on="ticker")
        r["matured_21"] = int(sub[sub.cohort == name]["matured_21"].sum()); r["matured_63"] = int(sub[sub.cohort == name]["matured_63"].sum())
        rows.append(r)
    return pd.DataFrame(rows).set_index("cohort")


def four_gate_lower_bound(replay: pd.DataFrame, recorded_date) -> dict:
    """Reconstructed four-gate date for one ticker (from audit.replay output indexed by date).
    status: 'ok', 'left_censored', 'gates_not_met_on_recorded_date' or 'no_evaluable_data'."""
    r = replay[replay.eligible_history.fillna(False) & replay.get("evaluable", True)] if len(replay) else replay
    out = {"recorded_discovery_date": str(recorded_date)[:10], "reconstructed_four_gate_date": None,
           "status": "no_evaluable_data", "full_scanner_qualification_date": FULL_QUALIFICATION_DATE}
    if r is None or r.empty: return out
    s = r["passes_replayed_gates"].dropna().astype(bool); s = s[s.index <= pd.Timestamp(recorded_date)]
    if s.empty: return out
    if not s.iloc[-1]:
        out["status"] = "gates_not_met_on_recorded_date"; return out
    broken = s[~s]
    start = s.index[0] if broken.empty else s.index[s.index > broken.index[-1]][0]
    out["reconstructed_four_gate_date"] = start.date().isoformat()
    out["status"] = "left_censored" if broken.empty else "ok"
    return out


def monthly_descriptive(trades: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Description only (Amendment 3 item 7): count, mean and median by calendar month of entry. Feeds no test or decision."""
    t = trades.copy(); t["month"] = pd.to_datetime(t[date_col]).dt.strftime("%Y-%m")
    return t.groupby("month")[value_col].agg(trades="count", mean="mean", median="median").reset_index()
