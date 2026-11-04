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


LEFT_CENSORED_NOTE = "Left-censored at the beginning of available evaluable history; the true qualification start may be earlier."
GAP_NOTE = ("The run begins after a session that could not be evaluated; the true start may be earlier. "
            "The date shown is the start of the verified unbroken run only.")


def session_states(r: pd.DataFrame, pass_col: str, calendar=None, upto=None) -> pd.Series:
    """One state per session: 'pass', 'fail' or 'unev' (not evaluable). The single place where this is decided, shared by
    four_gate_lower_bound and replay.first_pass_dates. A session is evaluable only if the replay marked it eligible and
    evaluable AND the gate value is present. If `calendar` is given, a session with no replay row is 'unev' as well."""
    r = r.copy()
    if calendar is not None: r = r.reindex(pd.DatetimeIndex(calendar).union(r.index))
    if upto is not None: r = r[r.index <= pd.Timestamp(upto)]
    if r.empty: return pd.Series(dtype=object)
    elig = r["eligible_history"].fillna(False).astype(bool) if "eligible_history" in r else pd.Series(True, index=r.index)
    evl = r["evaluable"].fillna(False).astype(bool) if "evaluable" in r else pd.Series(True, index=r.index)
    val = r[pass_col] if pass_col in r else pd.Series(index=r.index, dtype=object)
    ok = elig & evl & val.notna()
    passed = ok & val.fillna(False).astype(bool)
    state = pd.Series("unev", index=r.index, dtype=object); state[ok & ~passed] = "fail"; state[passed] = "pass"
    return state


def verified_run(state: pd.Series) -> dict:
    """The unbroken run of 'pass' sessions that ENDS at the last session. Only consecutive passing sessions belong to it:
    a fail or an unevaluable session breaks it. Returns {"status", "start" (Timestamp or None), "note", "first_evaluable"}.
    status: ok | left_censored | start_uncertain_unevaluable_gap | gates_not_met_on_recorded_date |
            recorded_date_not_evaluable | no_evaluable_data"""
    out = {"status": "no_evaluable_data", "start": None, "note": "", "first_evaluable": None}
    if state is None or len(state) == 0 or not (state != "unev").any(): return out
    out["first_evaluable"] = state[state != "unev"].index[0]
    last = state.iloc[-1]
    if last == "unev": out["status"] = "recorded_date_not_evaluable"; return out
    if last == "fail": out["status"] = "gates_not_met_on_recorded_date"; return out
    k = len(state) - 1
    while k > 0 and state.iloc[k - 1] == "pass": k -= 1
    out["start"] = state.index[k]
    before = state.iloc[:k]
    if before.empty or not (before != "unev").any(): out["status"] = "left_censored"; out["note"] = LEFT_CENSORED_NOTE
    elif before.iloc[-1] == "unev": out["status"] = "start_uncertain_unevaluable_gap"; out["note"] = GAP_NOTE
    else: out["status"] = "ok"
    return out


def four_gate_lower_bound(replay: pd.DataFrame, recorded_date, calendar=None) -> dict:
    """Reconstructed four-gate date for one ticker (from audit.replay output indexed by date). It is a LOWER BOUND on full
    scanner qualification and never an ApexScan discovery date. Every session counts: an unevaluable or (with a calendar)
    missing session is neither pass nor fail and CANNOT be skipped, so a run never bridges one (see verified_run for statuses)."""
    out = {"recorded_discovery_date": str(recorded_date)[:10], "reconstructed_four_gate_date": None, "status": "no_evaluable_data",
           "first_evaluable_date": None, "first_replayed_date": None, "note": "",
           "full_scanner_qualification_date": FULL_QUALIFICATION_DATE}
    if replay is None or len(replay) == 0: return out
    state = session_states(replay, "passes_replayed_gates", calendar, upto=recorded_date)
    if len(state) == 0: return out
    out["first_replayed_date"] = state.index[0].date().isoformat()
    v = verified_run(state)
    out["status"] = v["status"]; out["note"] = v["note"]
    if v["first_evaluable"] is not None: out["first_evaluable_date"] = v["first_evaluable"].date().isoformat()
    if v["start"] is not None: out["reconstructed_four_gate_date"] = v["start"].date().isoformat()
    return out


def monthly_descriptive(trades: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Description only (Amendment 3 item 7): count, mean and median by calendar month of entry. Feeds no test or decision."""
    t = trades.copy(); t["month"] = pd.to_datetime(t[date_col]).dt.strftime("%Y-%m")
    return t.groupby("month")[value_col].agg(trades="count", mean="mean", median="median").reset_index()
