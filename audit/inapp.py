"""audit/inapp.py - helpers used by pages/Audit.py. Read-only: nothing here writes to the repo or to data/."""
import io, tempfile, zipfile
from pathlib import Path
import pandas as pd

def prices_from_file(f) -> pd.DataFrame:
    return pd.read_csv(f, dtype={"ticker": str, "date": str})

def log_from_file(f) -> pd.DataFrame:
    return pd.read_csv(f, dtype={"ticker": str}, keep_default_na=False)

def dq(prices: pd.DataFrame, log: pd.DataFrame, meta=None):
    """(result dict, markdown report). Uses the exporter's validator unchanged; the report file is written to a temp dir."""
    from audit import data_quality as DQ
    res = DQ.validate(prices, log)
    with tempfile.TemporaryDirectory() as td:
        DQ.report(res, log, meta or {}, Path(td))
        md = (Path(td) / "DATA_QUALITY_REPORT.md").read_text(encoding="utf-8")
    return res, md

def has_blocking_errors(res) -> list:
    f = res["flags"]
    if not len(f): return []
    e = f[(f.severity == "error") & f.check.isin(["benchmark_role_unavailable", "no_reference_calendar"]) & ((f.ticker == "us_market") | (f.ticker == "^GSPC"))]
    return list(e.detail)

def ticker_hist(prices: pd.DataFrame, t: str) -> pd.DataFrame:
    g = prices[prices.ticker == t].copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.set_index("date").sort_index()
    return g.rename(columns=str.capitalize)[["Open", "High", "Low", "Close", "Volume"]]

def series(prices, sym):
    h = ticker_hist(prices, sym) if (prices.ticker == sym).any() else None
    return None if h is None else h["Close"]

def replay_for(prices: pd.DataFrame, ticker: str, start: str = "2026-07-01"):
    """Test A replay for one ticker with ApexScan's own functions. Returns (daily frame, per-gate verified-run detail:
    {gate: {date, status, note}}). A session with no row for this ticker breaks a run (the ^GSPC calendar is supplied). Never fills data."""
    from audit import replay as R
    gspc = series(prices, "^GSPC")
    if gspc is None or gspc.dropna().empty: raise ValueError("^GSPC has no data: relative strength cannot be replayed")
    hist = ticker_hist(prices, ticker)
    if hist.empty: raise ValueError(f"{ticker} has no rows in the price data")
    spec = lambda p, f: (p, series(prices, p), f, series(prices, f))
    cal = gspc.dropna().index
    dates = hist.index[hist.index >= pd.Timestamp(start)]
    r = R.replay_ticker(hist, {"^GSPC": gspc}, dates=dates, calendar=cal,
                        bench_r2500=spec("^R25I", "SMMD"), bench_r3000g=spec("^RAG", "IWZ"))
    return r, R.first_pass_detail(r, calendar=cal[cal >= pd.Timestamp(start)])

def zip_dir(d: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file(): z.write(p, p.relative_to(d).as_posix())
    return buf.getvalue()

def ticker_file(discoveries, path: Path, extra=("CHDN",)) -> Path:
    tk = sorted({d["ticker"] for d in discoveries} | set(extra))
    Path(path).write_text("\n".join(tk) + "\n", encoding="utf-8"); return Path(path)
