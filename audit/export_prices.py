#!/usr/bin/env python3
"""
export_prices.py  -  READ-ONLY data export for the ApexScan audit (v2, validated).

Run on any computer with internet access. It never touches the ApexScan repository or settings.

    pip install yfinance pandas
    python export_prices.py                      # default: stratified random sample + discovered tickers + benchmarks
    python export_prices.py --full-frame         # optional: download EVERY listed common stock (slow, large)
    python export_prices.py --listings-dir DIR   # use nasdaqlisted.txt / otherlisted.txt you saved yourself

Then run   python validate_prices.py --data apexscan_audit_data   and send back the whole output folder.

What it writes into --out (default ./apexscan_audit_data):
    listings_raw/*.txt        raw symbol directories exactly as downloaded (+ sha256 in run_metadata.json)
    frame.csv                 every eligible listing: symbol, exchange, name           (the sampling frame)
    sample.csv                the sampled symbols, stratum, seed                       (population C)
    benchmarks / discovered   included in the download list with their own 'group' value
    prices_long.csv.gz        ticker,date,open,high,low,close,volume  (adjusted, like the scanner)
    download_log.csv          one row per requested ticker: status, source, rows, first/last date, error text
    run_metadata.json         versions, parameters, timings, counts

Rules this script follows:
  * Prices match the scanner: Yahoo daily bars, auto_adjust=True (scanner.batch_fetch_history).
  * A row is dropped only if ALL of open/high/low/close/volume are missing (same as the scanner's
    dropna(how="all")). Partly-missing rows are KEPT as NaN. Nothing is ever filled, interpolated or set to 0.
  * A ticker that returns nothing is logged as 'empty', retried one-by-one, and reported - never skipped silently.
  * Benchmarks are exactly those in ApexScan config.yaml (see BENCHMARKS below).
"""
import argparse, gzip, hashlib, io, json, random, re, sys, time, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ---- exact symbols from ApexScan config.yaml `benchmarks:` and scanner._BENCH_FALLBACKS -------------------------
# us: ^GSPC (used for rs_3m / rs_6m).  russell_2500: ^R25I (fallback SMMD).  russell_3000_growth: ^RAG (fallback IWZ).
# NGX (^NGXASI, fallback FM) is deliberately NOT part of this US sample.
BENCHMARKS = ["^GSPC", "^R25I", "SMMD", "^RAG", "IWZ"]
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
URL_NASDAQ = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
URL_OTHER = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
NOT_COMMON = re.compile(r"\b(?:warrants?|rights?|units?|preferred|pfd|depositary|notes?|debentures?|"
                        r"etf|etn|fund|trust\s+preferred|subordinated|due\s+20\d\d)\b", re.I)


# ------------------------------------------------------------------------------------------------------------
# listings -> frame
# ------------------------------------------------------------------------------------------------------------
def _read_pipe(text: str, required: list) -> pd.DataFrame:
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("File Creation Time")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Symbol directory format changed: missing columns {missing}; got {list(df.columns)}")
    return df


def build_frame(nasdaq_text: str, other_text: str) -> pd.DataFrame:
    a = _read_pipe(nasdaq_text, ["Symbol", "Security Name", "Test Issue", "ETF"])
    b = _read_pipe(other_text, ["ACT Symbol", "Security Name", "Exchange", "ETF", "Test Issue"])
    rows = []
    for r in a.itertuples(index=False):
        rows.append((r[a.columns.get_loc("Symbol")], "NASDAQ", r[a.columns.get_loc("Security Name")],
                     r[a.columns.get_loc("Test Issue")], r[a.columns.get_loc("ETF")]))
    emap = {"N": "NYSE", "A": "NYSE American"}
    for r in b.itertuples(index=False):
        ex = r[b.columns.get_loc("Exchange")]
        if ex in emap:
            rows.append((r[b.columns.get_loc("ACT Symbol")], emap[ex], r[b.columns.get_loc("Security Name")],
                         r[b.columns.get_loc("Test Issue")], r[b.columns.get_loc("ETF")]))
    f = pd.DataFrame(rows, columns=["raw_symbol", "exchange", "name", "test", "etf"])
    f = f[(f.test != "Y") & (f.etf != "Y")]
    f = f[~f.raw_symbol.str.contains(r"[\$\^\s]", regex=True)]
    f = f[~f.name.str.contains(NOT_COMMON)]
    f["symbol"] = f.raw_symbol.str.replace(".", "-", regex=False)      # Yahoo uses BRK-B, not BRK.B
    f = f.drop_duplicates("symbol").sort_values(["exchange", "symbol"]).reset_index(drop=True)
    return f[["symbol", "exchange", "name", "raw_symbol"]]


CASE_STUDY = "CHDN"
EXCLUDE_FROM_SAMPLE = frozenset({CASE_STUDY})      # Amendment 1: removed from the pool BEFORE drawing, so n stays 250/250/120


def draw_sample(frame: pd.DataFrame, per_exchange: dict, seed: int, exclude=EXCLUDE_FROM_SAMPLE) -> pd.DataFrame:
    out = []
    for ex, n in per_exchange.items():
        pool = sorted(s for s in frame.loc[frame.exchange == ex, "symbol"] if s not in exclude)
        rng = random.Random(f"{seed}:{ex}")                # registered seed and per-exchange stream unchanged
        pick = sorted(rng.sample(pool, min(n, len(pool))))
        out += [(s, ex, seed, len(pool), len(pick)) for s in pick]
    res = pd.DataFrame(out, columns=["symbol", "exchange", "seed", "frame_size_for_exchange", "n_drawn"])
    bad = set(res.symbol) & set(exclude)
    if bad:                                                # belt and braces: never write an excluded symbol
        raise RuntimeError(f"Excluded symbols appeared in the sample: {sorted(bad)}")
    return res


# ------------------------------------------------------------------------------------------------------------
# yfinance output -> tidy rows.  Handles every layout yfinance has produced.
# ------------------------------------------------------------------------------------------------------------
def _clean_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def normalise(df, batch: list) -> dict:
    """Return {ticker: DataFrame[Open,High,Low,Close,Volume] indexed by date, NaN preserved}.
    Layouts handled: MultiIndex (Ticker, Price) [group_by='ticker'], MultiIndex (Price, Ticker) [group_by='column'],
    flat columns (old single-ticker behaviour). A ticker absent from the result simply has no key."""
    out = {}
    if df is None or len(df) == 0:
        return out
    df = df.copy()
    df.index = _clean_index(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        lv0, lv1 = set(map(str, df.columns.get_level_values(0))), set(map(str, df.columns.get_level_values(1)))
        if lv0 & set(FIELDS) and not lv1 & set(FIELDS):
            ticker_level, field_level = 1, 0
        elif lv1 & set(FIELDS) and not lv0 & set(FIELDS):
            ticker_level, field_level = 0, 1
        else:
            raise RuntimeError(f"Cannot tell ticker level from columns {list(df.columns)[:6]}")
        for t in sorted(set(df.columns.get_level_values(ticker_level))):
            sub = df.xs(t, axis=1, level=ticker_level)
            sub = sub[[c for c in FIELDS if c in sub.columns]]
            if len(sub.columns) == len(FIELDS):
                out[str(t)] = sub
    else:
        if len(batch) != 1:
            raise RuntimeError("Flat columns returned for a multi-ticker batch; cannot attribute rows to tickers")
        sub = df[[c for c in FIELDS if c in df.columns]]
        if len(sub.columns) == len(FIELDS):
            out[batch[0]] = sub
    return out


def tidy(t: str, sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.dropna(how="all")                      # same rule as scanner: drop only fully-empty rows
    # duplicate dates are deliberately KEPT here so the validator can report them
    x = sub.reset_index()
    x.columns = ["date"] + [c.lower() for c in sub.columns]
    x.insert(0, "ticker", t)
    x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")
    return x[["ticker", "date", "open", "high", "low", "close", "volume"]]


# ------------------------------------------------------------------------------------------------------------
# download
# ------------------------------------------------------------------------------------------------------------
def download_batch(yf, batch, start, end):
    kw = dict(start=start, end=end, auto_adjust=True, group_by="ticker", progress=False, threads=True)
    for attempt in range(3):
        try:
            try:
                return yf.download(batch, multi_level_index=True, **kw)
            except TypeError:                         # very old yfinance has no multi_level_index argument
                return yf.download(batch, **kw)
        except Exception as e:                       # network / rate limit
            err = f"{type(e).__name__}: {e}"
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(err)


def download_single(yf, t, start, end):
    h = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
    if h is None or len(h) == 0:
        return None
    h = h.copy()
    h.index = _clean_index(h.index)
    return h[[c for c in FIELDS if c in h.columns]]


def run(args, yf=None, fetch_text=None, progress=None):
    out = Path(args.out); (out / "listings_raw").mkdir(parents=True, exist_ok=True)
    meta = {"started_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args), "python": sys.version.split()[0],
            "pandas": pd.__version__}
    if yf is None:
        import yfinance as yf
    meta["yfinance"] = getattr(yf, "__version__", "unknown")

    # 1. listings
    texts = {}
    for name, url in (("nasdaqlisted.txt", URL_NASDAQ), ("otherlisted.txt", URL_OTHER)):
        if args.listings_dir:
            txt = (Path(args.listings_dir) / name).read_text(encoding="utf-8", errors="replace")
        elif fetch_text:
            txt = fetch_text(url)
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            txt = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="replace")
        (out / "listings_raw" / name).write_text(txt, encoding="utf-8")
        texts[name] = txt
        meta[f"sha256_{name}"] = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    frame = build_frame(texts["nasdaqlisted.txt"], texts["otherlisted.txt"])
    frame.assign(excluded_from_sample=frame.symbol.isin(EXCLUDE_FROM_SAMPLE)).to_csv(out / "frame.csv", index=False)
    meta["frame_counts"] = frame.exchange.value_counts().to_dict()

    # 2. sample
    sample = draw_sample(frame, {"NASDAQ": args.n_nasdaq, "NYSE": args.n_nyse, "NYSE American": args.n_amex}, args.seed)
    sample.to_csv(out / "sample.csv", index=False)

    # 3. request list with a group label (a ticker can carry several; first label wins in this column)
    req = {}
    for t in BENCHMARKS: req[t] = "benchmark"
    req[CASE_STUDY] = "case_study"                         # always downloaded, never part of population C
    disc = Path(args.tickers_file)
    if disc.exists():
        for t in disc.read_text().split():
            req.setdefault(t.strip(), "discovered")
    pool = frame.symbol if args.full_frame else sample.symbol
    for t in pool: req.setdefault(t, "frame" if args.full_frame else "sample")
    tickers = list(req)
    meta["requested"] = {g: sum(1 for v in req.values() if v == g) for g in set(req.values())}

    # 4. download
    frames, log = [], {}
    in_sample = set(sample.symbol)                         # a ticker can be a discovery AND in population C
    def note(t, **k): log.setdefault(t, {"ticker": t, "group": req[t], "in_sample": t in in_sample}).update(k)
    for i in range(0, len(tickers), args.batch_size):
        batch = tickers[i:i + args.batch_size]
        try:
            raw = download_batch(yf, batch, args.start, args.end); berr = ""
        except Exception as e:
            raw, berr = None, str(e)
        got = normalise(raw, batch) if raw is not None else {}
        for t in batch:
            sub = got.get(t)
            if sub is not None:
                x = tidy(t, sub)
                if len(x): frames.append(x); note(t, status="ok", source="batch", rows=len(x), error=""); continue
            note(t, status="empty_in_batch", source="batch", rows=0, error=berr)
        nb, tb = i // args.batch_size + 1, -(-len(tickers) // args.batch_size)
        print(f"batch {nb}/{tb}", flush=True)
        if progress: progress(nb, tb, f"batch {nb} of {tb}")
    for t in [t for t, v in log.items() if v["status"] != "ok"]:      # one-by-one retry, like the scanner's fallback
        try:
            sub = download_single(yf, t, args.start, args.end)
            if sub is not None:
                x = tidy(t, sub)
                if len(x): frames.append(x); note(t, status="ok", source="single_retry", rows=len(x), error=""); continue
            note(t, status="empty", source="single_retry", rows=0, error="no rows returned")
        except Exception as e:
            note(t, status="failed", source="single_retry", rows=0, error=f"{type(e).__name__}: {e}")

    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ticker", "date", "open", "high", "low", "close", "volume"])
    prices.to_csv(out / "prices_long.csv.gz", index=False, compression="gzip")
    for t, g in prices.groupby("ticker"):
        log[t].update(first_date=g.date.min(), last_date=g.date.max())
    pd.DataFrame(log.values()).to_csv(out / "download_log.csv", index=False)
    meta.update(rows=len(prices), tickers_with_data=int(prices.ticker.nunique()), tickers_requested=len(tickers),
                tickers_without_data=sum(1 for v in log.values() if v["status"] != "ok"),
                finished_utc=datetime.now(timezone.utc).isoformat())
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ("rows", "tickers_requested", "tickers_with_data", "tickers_without_data")}))
    return meta


def default_end():
    """Exclusive end date = tomorrow (UTC), so the download includes every completed session up to today.
    Replaces the fixed 2026-11-02 (which stopped at the 30 Oct 2026 session) so later price refreshes get later sessions."""
    return (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="apexscan_audit_data")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=default_end(), help="exclusive; default is tomorrow (UTC), i.e. up to the latest completed session")
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--n-nasdaq", type=int, default=250)
    p.add_argument("--n-nyse", type=int, default=250)
    p.add_argument("--n-amex", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--full-frame", action="store_true")
    p.add_argument("--listings-dir", default=None)
    p.add_argument("--tickers-file", default=str(Path(__file__).resolve().parent / "tickers.txt"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
