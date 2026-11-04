#!/usr/bin/env python3
"""
validate_prices.py - data-quality report for the ApexScan audit export.   READ-ONLY, writes only into the data folder.

    python validate_prices.py --data apexscan_audit_data

Writes DATA_QUALITY_REPORT.md and dq_flags.csv. It never repairs, fills or drops data; it only reports.
Reference calendar = the dates present for ^GSPC (the scanner's own benchmark).
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

# Benchmark ROLES, as in ApexScan config.yaml + scanner.get_benchmark: the primary is used if it returns more than 50 bars,
# otherwise the fallback is tried. A role is satisfied by either symbol; ^GSPC has no fallback and is mandatory.
ROLES = {"us_market": ("^GSPC", None, True),
         "russell_2500": ("^R25I", "SMMD", False),
         "russell_3000_growth": ("^RAG", "IWZ", False)}
BENCH = [s for p, f, _ in ROLES.values() for s in (p, f) if s]
MIN_BENCH_BARS = 50                       # scanner.get_benchmark: `if len(data) > 50`
OHLCV = ["open", "high", "low", "close", "volume"]
RET_FLAG, RET_SEVERE = 0.50, 1.00          # |1-day return| thresholds
OHLC_TOL = 0.001                           # 0.1 % tolerance for open/close outside high-low
STALE_RUN = 5                              # identical consecutive closes
LATE_START_SESSIONS = 5                    # ticker starts more than this many sessions after the window start (info only)
EARLY_END_SESSIONS = 5                     # ticker ends more than this many sessions before the last calendar date


def longest_run(a: np.ndarray) -> int:
    best = cur = 1 if len(a) else 0
    for i in range(1, len(a)):
        cur = cur + 1 if a[i] == a[i - 1] else 1
        best = max(best, cur)
    return best


def validate(prices: pd.DataFrame, log: pd.DataFrame, window=("2019-01-01", "2026-10-30")):
    prices = prices.copy()
    n_raw = len(prices)
    flags, per = [], []
    ref = sorted(set(prices.loc[prices.ticker == "^GSPC", "date"]))
    ref_set = set(ref)
    ref_last = ref[-1] if ref else None
    ref_idx = {d: i for i, d in enumerate(ref)}

    def flag(t, kind, detail, sev="warn"):
        flags.append({"ticker": t, "check": kind, "severity": sev, "detail": detail})

    # --- requested vs delivered ---------------------------------------------------------------------------------
    delivered = set(prices.ticker.unique())
    for _, r in log.iterrows():
        if r["ticker"] not in delivered:
            flag(r["ticker"], "no_data", f"group={r['group']} status={r['status']} error={r.get('error', '')}", "error")
    nclose = prices.dropna(subset=["close"]).groupby("ticker").size() if len(prices) else pd.Series(dtype=int)
    roles = []
    for role, (prim, fb, mandatory) in ROLES.items():
        np_, nf = int(nclose.get(prim, 0)), int(nclose.get(fb, 0)) if fb else 0
        if np_ > MIN_BENCH_BARS:   used, why = prim, "primary available"
        elif fb and nf > MIN_BENCH_BARS: used, why = fb, f"FALLBACK in use: primary {prim} has {np_} bars"
        else:                      used, why = None, f"UNAVAILABLE: {prim} has {np_} bars" + (f", {fb} has {nf}" if fb else "")
        roles.append({"role": role, "primary": prim, "fallback": fb, "primary_bars": np_, "fallback_bars": nf,
                      "used": used, "note": why, "mandatory": mandatory})
        if used is None:
            flag(role, "benchmark_role_unavailable", why, "error" if mandatory else "warn")
        elif used != prim:
            flag(role, "benchmark_fallback_in_use", why, "info")          # recorded, never silent
    if not ref:
        flag("^GSPC", "no_reference_calendar", "cannot check missing sessions without ^GSPC", "error")

    for t, g in prices.groupby("ticker", sort=True):
        g = g.reset_index(drop=True)
        d = g["date"].astype(str)
        row = {"ticker": t, "rows": len(g), "first": d.min(), "last": d.max()}
        # duplicates / order
        dup = int(d.duplicated().sum()); row["duplicate_dates"] = dup
        if dup: flag(t, "duplicate_dates", f"{dup} duplicated dates", "error")
        if not (d.values[:-1] <= d.values[1:]).all():
            flag(t, "unsorted", "dates not in ascending order", "warn")
        # dates outside the reference calendar (weekends/holidays/bad dates)
        if ref:
            off = [x for x in d.unique() if x not in ref_set and t != "^GSPC"]
            row["dates_not_in_calendar"] = len(off)
            if off: flag(t, "dates_not_in_calendar", f"{len(off)} e.g. {off[:3]}", "warn")
            # missing sessions inside the ticker's own span
            lo, hi = d.min(), d.max()
            span = [x for x in ref if lo <= x <= hi]
            dset = set(d)                                                       # built once, not once per session
            miss = [x for x in span if x not in dset]
            row["missing_sessions"] = len(miss)
            if miss: flag(t, "missing_sessions", f"{len(miss)} of {len(span)} sessions missing, e.g. {miss[:3]}",
                          "error" if len(miss) > 0.02 * max(len(span), 1) else "warn")
            # early end / late start
            if ref_last and hi in ref_idx and ref_idx[ref_last] - ref_idx[hi] > EARLY_END_SESSIONS:
                flag(t, "ends_early", f"last bar {hi}, calendar ends {ref_last} (delisting, halt or data gap)", "info")
            if lo > window[0] and lo in ref_idx and ref_idx[lo] > LATE_START_SESSIONS:
                flag(t, "starts_late", f"first bar {lo} (listing after window start or data gap)", "info")
        # missing / invalid values (NOT filled)
        for c in OHLCV:
            n = int(g[c].isna().sum()); row[f"nan_{c}"] = n
            if n: flag(t, f"missing_{c}", f"{n} rows with {c} missing", "error" if c == "close" else "warn")
        px = g[["open", "high", "low", "close"]]
        nonpos = int((px <= 0).any(axis=1).sum()); row["nonpositive_price_rows"] = nonpos
        if nonpos: flag(t, "nonpositive_price", f"{nonpos} rows with a price <= 0", "error")
        negv = int((g["volume"] < 0).sum()); row["negative_volume_rows"] = negv
        if negv: flag(t, "negative_volume", f"{negv} rows", "error")
        zerov = int((g["volume"] == 0).sum()); row["zero_volume_rows"] = zerov
        if zerov and not t.startswith("^"): flag(t, "zero_volume", f"{zerov} rows with volume exactly 0 (kept as 0, not filled)", "info")
        hl = int((g["high"] < g["low"]).sum()); row["high_below_low"] = hl
        if hl: flag(t, "high_below_low", f"{hl} rows", "error")
        out = ((g["close"] > g["high"] * (1 + OHLC_TOL)) | (g["close"] < g["low"] * (1 - OHLC_TOL)) |
               (g["open"] > g["high"] * (1 + OHLC_TOL)) | (g["open"] < g["low"] * (1 - OHLC_TOL)))
        out = int(out.sum()); row["ohlc_inconsistent"] = out
        if out: flag(t, "ohlc_inconsistent", f"{out} rows with open/close outside high-low by >{OHLC_TOL:.1%}", "warn")
        # jumps and stale runs
        c = g["close"].astype(float)
        r = c.pct_change().abs()
        big = int((r > RET_FLAG).sum()); row["moves_over_50pct"] = big
        if big: flag(t, "large_1d_move", f"{big} daily moves >50% (max {r.max():.0%}) - check splits/bad ticks",
                     "error" if (r > RET_SEVERE).any() else "warn")
        run = longest_run(c.dropna().values) if c.notna().any() else 0; row["longest_flat_close_run"] = run
        if run >= STALE_RUN and not t.startswith("^"): flag(t, "stale_close_run", f"{run} identical consecutive closes", "warn")
        per.append(row)

    per = pd.DataFrame(per)
    flags = pd.DataFrame(flags, columns=["ticker", "check", "severity", "detail"])
    return {"rows_total": n_raw, "per_ticker": per, "flags": flags, "ref_first": ref[0] if ref else None, "ref_last": ref_last,
            "ref_sessions": len(ref), "benchmark_roles": pd.DataFrame(roles)}


def report(res, log, meta, out: Path):
    f, per = res["flags"], res["per_ticker"]
    L = ["# ApexScan audit - data-quality report", ""]
    L.append(f"Reference calendar: ^GSPC, {res['ref_first']} to {res['ref_last']}, {res['ref_sessions']} sessions.")
    L.append(f"Rows: {res['rows_total']:,}. Tickers with data: {len(per)}. Tickers requested: {len(log)}.")
    if meta: L.append(f"yfinance {meta.get('yfinance')} - run {meta.get('started_utc')}. Frame: {meta.get('frame_counts')}.")
    L += ["", "No value was filled, interpolated or set to zero by the exporter or this report.", ""]
    L += ["## Requested tickers by group and outcome", "", "| group | requested | with data | no data |", "| --- | --- | --- | --- |"]
    for g, x in log.groupby("group"):
        ok = int((x.status == "ok").sum()); L.append(f"| {g} | {len(x)} | {ok} | {len(x) - ok} |")
    L += ["", "## Flags by check", "", "| check | severity | tickers | occurrences |", "| --- | --- | --- | --- |"]
    if len(f):
        for (c, s), x in f.groupby(["check", "severity"]):
            L.append(f"| {c} | {s} | {x.ticker.nunique()} | {len(x)} |")
    else:
        L.append("| (none) | | 0 | 0 |")
    err = set(f.loc[f.severity == "error", "ticker"]) if len(f) else set()
    flagged = set(f.ticker) if len(f) else set()
    clean = [t for t in per.ticker if t not in flagged]
    L += ["", f"Tickers with at least one **error** flag: {len(err)}. Tickers with no flag at all: {len(clean)} of {len(per)}.", ""]
    if len(f):
        L += ["## Error-level detail (first 60)", "", "| ticker | check | detail |", "| --- | --- | --- |"]
        for r in f[f.severity == "error"].head(60).itertuples():
            L.append(f"| {r.ticker} | {r.check} | {str(r.detail)[:140]} |")
        L += ["", "Full list: dq_flags.csv. Per-ticker statistics: dq_per_ticker.csv."]
    (out / "DATA_QUALITY_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    f.to_csv(out / "dq_flags.csv", index=False); per.to_csv(out / "dq_per_ticker.csv", index=False)


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--data", default="apexscan_audit_data")
    a = p.parse_args(argv); d = Path(a.data)
    prices = pd.read_csv(d / "prices_long.csv.gz", dtype={"ticker": str, "date": str})
    log = pd.read_csv(d / "download_log.csv", dtype={"ticker": str}, keep_default_na=False)
    meta = json.loads((d / "run_metadata.json").read_text()) if (d / "run_metadata.json").exists() else {}
    res = validate(prices, log); report(res, log, meta, d)
    print((d / "DATA_QUALITY_REPORT.md").read_text()[:1500])
    return res


if __name__ == "__main__":
    main()
