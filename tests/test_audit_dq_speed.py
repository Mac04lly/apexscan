"""The data-quality check must stay fast on a full download (it once rebuilt a set inside a loop: ~2 s per ticker, about an hour
for 1,500 tickers) and must report exactly the same findings as before the speed fix."""
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd
from audit import data_quality as DQ


def _frame(t, idx, c):
    return pd.DataFrame({"ticker": t, "date": list(idx), "open": c, "high": c * 1.01, "low": c * .99, "close": c, "volume": 1e6})


def _golden():
    rng = np.random.default_rng(7); idx = pd.bdate_range(end="2026-10-30", periods=400).strftime("%Y-%m-%d"); fr = []
    for t in ["^GSPC", "^R25I", "^RAG", "A", "B", "C", "D", "E"]:
        d = _frame(t, idx, 50 * np.cumprod(1 + rng.normal(0, .01, len(idx))))
        if t == "B": d = d.drop(d.index[[50, 51, 52, 200]])                         # four missing sessions
        if t == "C": d = pd.concat([d, d.iloc[[10]]])                               # a duplicated date
        if t == "D": d.loc[d.index[100], "close"] = np.nan; d.loc[d.index[120], "close"] = d.close.iloc[119] * 2.5
        if t == "E": d = d.iloc[30:350]                                              # late start, early end
        fr.append(d)
    p = pd.concat(fr, ignore_index=True)
    log = pd.DataFrame({"ticker": p.ticker.unique().tolist() + ["ZZZ"], "group": "s", "status": ["ok"] * 8 + ["no_data"], "error": ""})
    return p, log


def test_findings_on_a_known_dataset_are_exactly_these():
    p, log = _golden(); f = DQ.validate(p, log)["flags"]
    got = sorted(zip(f.ticker, f.check, f.severity))
    assert got == sorted([("ZZZ", "no_data", "error"), ("B", "missing_sessions", "warn"), ("C", "duplicate_dates", "error"), ("C", "unsorted", "warn"),
                          ("D", "missing_close", "error"), ("D", "ohlc_inconsistent", "warn"), ("D", "large_1d_move", "error"),
                          ("E", "ends_early", "info"), ("E", "starts_late", "info")])
    assert "4 of 400 sessions missing" in f[f.check == "missing_sessions"].detail.iloc[0]


def test_a_full_size_download_is_checked_in_seconds_not_hours():
    idx = pd.bdate_range(end="2026-10-30", periods=1900).strftime("%Y-%m-%d"); rng = np.random.default_rng(1); fr = []
    for i in range(150):
        fr.append(_frame("^GSPC" if i == 0 else f"T{i}", idx, 50 * np.cumprod(1 + rng.normal(0, .01, 1900))))
    p = pd.concat(fr, ignore_index=True); log = pd.DataFrame({"ticker": p.ticker.unique(), "group": "s", "status": "ok", "error": ""})
    t0 = time.time(); DQ.validate(p, log); dt = time.time() - t0
    assert dt < 8, f"150 tickers took {dt:.1f}s; the old loop needed about 330 s"     # 1,500 tickers scale linearly: about 1 minute at this ceiling


def test_the_report_counts_tickers_with_no_flag_correctly():
    p, log = _golden(); p = p[~p.ticker.isin(["D", "E"])]; log = log[~log.ticker.isin(["D", "E"])]    # 6 tickers with data: B and C are flagged
    res = DQ.validate(p, log)
    import tempfile
    d = Path(tempfile.mkdtemp()); DQ.report(res, log, {}, d); md = (d / "DATA_QUALITY_REPORT.md").read_text(encoding="utf-8")
    assert "Tickers with no flag at all: 4 of 6" in md          # ^GSPC, ^R25I, ^RAG and A carry no flag
