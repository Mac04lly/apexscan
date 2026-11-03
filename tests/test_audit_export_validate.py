import sys, types, json
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, pytest
from audit.tools import export_prices as X
from audit import data_quality as V

NASDAQ_TXT = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAA|Alpha Inc. - Common Stock|Q|N|N|100|N|N
BBB|Beta Corp. - Common Stock|G|N|N|100|N|N
QQQX|Some ETF Trust|G|N|N|100|Y|N
TST|Test Co - Common Stock|G|Y|N|100|N|N
WARR|Gamma Acquisition Corp - Warrants|G|N|N|100|N|N
File Creation Time: 1031202618:00|||||||
"""
OTHER_TXT = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
CCC|Charlie Corp. Common Stock|N|CCC|N|100|N|CCC
BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK.B
DDD|Delta Mining Inc. Common Stock|A|DDD|N|100|N|DDD
EEE|Echo ETF|P|EEE|Y|100|N|EEE
FFF|Foxtrot Corp Preferred Stock|N|FFF|N|100|N|FFF
GGG$A|Golf Pfd|N|GGG$A|N|100|N|GGG$A
File Creation Time: 1031202618:00|||||||
"""
FRAME = X.build_frame(NASDAQ_TXT, OTHER_TXT)

def test_frame_keeps_only_listed_common_stock_on_three_exchanges():
    got = {(r.symbol, r.exchange) for r in FRAME.itertuples()}
    assert got == {("AAA","NASDAQ"),("BBB","NASDAQ"),("CCC","NYSE"),("BRK-B","NYSE"),("DDD","NYSE American")}

def test_frame_fails_loudly_if_directory_format_changes():
    with pytest.raises(RuntimeError, match="format changed"):
        X.build_frame(NASDAQ_TXT.replace("Test Issue","Tst"), OTHER_TXT)

def test_sample_is_deterministic_and_seed_sensitive():
    big = pd.DataFrame({"symbol":[f"S{i}" for i in range(500)],"exchange":"NYSE","name":"x","raw_symbol":"x"})
    a=X.draw_sample(big,{"NYSE":50},1); b=X.draw_sample(big,{"NYSE":50},1); c=X.draw_sample(big,{"NYSE":50},2)
    assert list(a.symbol)==list(b.symbol) and list(a.symbol)!=list(c.symbol)
    assert len(X.draw_sample(big,{"NYSE":900},1))==500          # takes all when the pool is smaller

def _idx(n=5, tz=None):
    return pd.date_range("2026-01-05", periods=n, freq="B", tz=tz)
def _ohlcv(n=5, base=10.0):
    v=np.arange(n)+base
    return pd.DataFrame({"Open":v,"High":v+1,"Low":v-1,"Close":v+0.5,"Volume":1000+np.arange(n)})

def _multi(tickers, ticker_first=True, n=5, tz=None):
    parts={t:_ohlcv(n,10+i) for i,t in enumerate(tickers)}
    df=pd.concat(parts,axis=1)                                   # (ticker, field)
    df.index=_idx(n,tz)
    if not ticker_first: df=df.swaplevel(axis=1).sort_index(axis=1)
    return df

@pytest.mark.parametrize("ticker_first",[True,False])
@pytest.mark.parametrize("tickers",[["A","B","C"],["A"]])        # single ticker is MULTI-index in yfinance>=0.2.48
def test_normalise_multiindex_both_orientations_single_and_multi(ticker_first,tickers):
    got=X.normalise(_multi(tickers,ticker_first),tickers)
    assert set(got)==set(tickers)
    for t in tickers: assert list(got[t].columns)==X.FIELDS and len(got[t])==5

def test_normalise_flat_single_ticker_old_yfinance():
    df=_ohlcv(); df.index=_idx(); df["Dividends"]=0.0
    assert list(X.normalise(df,["A"])["A"].columns)==X.FIELDS

def test_normalise_flat_multi_ticker_is_refused_not_guessed():
    df=_ohlcv(); df.index=_idx()
    with pytest.raises(RuntimeError): X.normalise(df,["A","B"])

def test_normalise_strips_timezone_to_plain_dates():
    got=X.normalise(_multi(["A","B"],tz="America/New_York"),["A","B"])
    assert got["A"].index.tz is None and got["A"].index[0]==pd.Timestamp("2026-01-05")

def test_missing_ticker_is_all_nan_and_becomes_zero_rows_not_zeros():
    df=_multi(["A","B"]); df[("B","Close")]=np.nan; 
    for c in X.FIELDS: df[("B",c)]=np.nan
    got=X.normalise(df,["A","B"])
    assert len(X.tidy("B",got["B"]))==0 and len(X.tidy("A",got["A"]))==5

def test_partly_missing_rows_are_kept_as_nan_never_zero():
    df=_multi(["A"]); df.loc[df.index[2],("A","Volume")]=np.nan
    x=X.tidy("A",X.normalise(df,["A"])["A"])
    assert len(x)==5 and x.volume.isna().sum()==1 and (x.volume.dropna()>0).all()

# ---------- full pipeline with a fake yfinance ----------
class FakeYF:
    __version__="fake-1.7"
    def __init__(self, flat_single=False, no_data=(), n=40): self.flat_single, self.no_data, self.n, self.calls=flat_single,set(no_data),n,[]
    def _frame(self,t,i):
        d=_ohlcv(self.n,10+i); d.index=pd.date_range("2026-01-05",periods=self.n,freq="B",tz="America/New_York"); return d
    def download(self,batch,multi_level_index=True,**kw):
        self.calls.append(("download",tuple(batch)))
        parts={t:(self._frame(t,i) if t not in self.no_data else self._frame(t,i)*np.nan) for i,t in enumerate(batch)}
        df=pd.concat(parts,axis=1)
        if len(batch)==1 and not multi_level_index: df=df[batch[0]]
        return df
    def Ticker(self,t):
        outer=self
        class T:
            def history(s,**kw):
                outer.calls.append(("single",t))
                return None if t in outer.no_data else outer._frame(t,0)
        return T()

def _args(tmp,**kw):
    d=dict(out=str(tmp/"o"),start="2026-01-01",end="2026-12-31",seed=7,n_nasdaq=1,n_nyse=1,n_amex=1,batch_size=4,
           full_frame=False,listings_dir=str(tmp/"l"),tickers_file=str(tmp/"t.txt"))
    d.update(kw); return types.SimpleNamespace(**d)

def _setup(tmp):
    (tmp/"l").mkdir(); (tmp/"l"/"nasdaqlisted.txt").write_text(NASDAQ_TXT); (tmp/"l"/"otherlisted.txt").write_text(OTHER_TXT)
    (tmp/"t.txt").write_text("AAA\nZZZ\n")

def test_pipeline_end_to_end_logs_every_requested_ticker_and_reports_failures(tmp_path):
    _setup(tmp_path); yf=FakeYF(no_data={"ZZZ","^RAG"})
    meta=X.run(_args(tmp_path),yf=yf)
    log=pd.read_csv(tmp_path/"o"/"download_log.csv",keep_default_na=False)
    assert set(X.BENCHMARKS)<=set(log.ticker)                                # exact config benchmarks requested
    assert log.set_index("ticker").loc["ZZZ","status"]=="empty" and log.set_index("ticker").loc["^RAG","status"]=="empty"
    assert ("single","ZZZ") in yf.calls                                       # failed ones retried one by one
    assert meta["tickers_without_data"]==2 and (log.status=="ok").sum()==len(log)-2
    p=pd.read_csv(tmp_path/"o"/"prices_long.csv.gz")
    assert "ZZZ" not in set(p.ticker) and (p.groupby("ticker").size()==40).all()
    assert (tmp_path/"o"/"sample.csv").exists() and (tmp_path/"o"/"frame.csv").exists()

def test_benchmark_symbols_match_apexscan_config_exactly():
    import yaml
    cfg=yaml.safe_load(open(REPO / "config.yaml", encoding="utf-8"))["benchmarks"]
    assert X.BENCHMARKS==[cfg["us"],cfg["russell_2500"],cfg["russell_2500_fallback"],cfg["russell_3000_growth"],cfg["russell_3000_growth_fallback"]]

def test_pipeline_works_with_old_flat_single_ticker_output(tmp_path):
    _setup(tmp_path)
    class OldYF(FakeYF):
        def download(self,batch,**kw):
            if len(batch)==1:
                self.calls.append(("download",tuple(batch))); return self._frame(batch[0],0)  # old yfinance: flat
            return super().download(batch)
    # force a final batch of exactly one ticker
    yf=OldYF(); X.run(_args(tmp_path,batch_size=1),yf=yf)
    assert any(c[0]=="download" and len(c[1])==1 for c in yf.calls), "test must exercise a one-ticker batch"
    p=pd.read_csv(tmp_path/"o"/"prices_long.csv.gz"); assert p.ticker.nunique()>=5

# ---------- validator: clean data passes, every injected defect is caught ----------
def _cal(n=60): return pd.bdate_range("2026-01-05",periods=n).strftime("%Y-%m-%d")
def _tk(t,n=60,seed=0,start=0):
    rng=np.random.default_rng(seed); c=50*np.cumprod(1+rng.normal(0,0.01,n))
    d=pd.DataFrame({"ticker":t,"date":_cal(n+start)[start:],"open":c,"high":c*1.01,"low":c*0.99,"close":c,"volume":rng.integers(1e5,1e6,n).astype(float)})
    return d
def _log(ts): return pd.DataFrame({"ticker":ts,"group":"sample","status":"ok","error":""})
def _run(frames):
    p=pd.concat(frames,ignore_index=True)
    allbench=[_tk(b,seed=i+90) for i,b in enumerate(V.BENCH) if b not in set(p.ticker)]
    p=pd.concat([p]+allbench,ignore_index=True)
    return V.validate(p,_log(list(p.ticker.unique())))
def _checks(res,t): f=res["flags"]; return set(f.loc[f.ticker==t,"check"])

def test_validator_control_clean_data_has_no_flags():
    res=_run([_tk("OK")]); assert len(res["flags"])==0

def test_validator_catches_each_injected_defect():
    c={}
    a=_tk("DUP"); c["DUP"]=pd.concat([a,a.iloc[[5]]],ignore_index=True)
    a=_tk("NANC"); a.loc[10,"close"]=np.nan; c["NANC"]=a
    a=_tk("NANV"); a.loc[10,"volume"]=np.nan; c["NANV"]=a
    a=_tk("ZERO"); a.loc[10,["open","high","low","close"]]=0.0; c["ZERO"]=a
    a=_tk("GAP"); c["GAP"]=a.drop(index=[20,21,22,23]).reset_index(drop=True)
    a=_tk("HL"); a.loc[10,"high"]=a.loc[10,"low"]*0.9; c["HL"]=a
    a=_tk("JUMP"); a.loc[30:,["open","high","low","close"]]*=3; c["JUMP"]=a
    a=_tk("STALE"); a.loc[20:27,["open","high","low","close"]]=a.loc[20,"close"]; c["STALE"]=a
    c["END"]=_tk("END",n=40)
    a=_tk("WKND"); a.loc[10,"date"]="2026-02-07"; c["WKND"]=a
    a=_tk("NEGV"); a.loc[10,"volume"]=-5; c["NEGV"]=a
    res=_run(list(c.values()))
    exp={"DUP":"duplicate_dates","NANC":"missing_close","NANV":"missing_volume","ZERO":"nonpositive_price","GAP":"missing_sessions",
         "HL":"high_below_low","JUMP":"large_1d_move","STALE":"stale_close_run","END":"ends_early","WKND":"dates_not_in_calendar","NEGV":"negative_volume"}
    for t,k in exp.items(): assert k in _checks(res,t),(t,k,_checks(res,t))

def test_validator_reports_missing_benchmark_and_requested_but_absent_tickers():
    p=_tk("OK"); log=_log(["OK","LOST"])
    res=V.validate(p,log); chk=set(res["flags"].check)
    assert {"no_data","benchmark_role_unavailable","no_reference_calendar"}<=chk

def test_validator_never_alters_values(tmp_path):
    p=_tk("OK"); p.loc[3,"volume"]=np.nan; before=p.copy()
    V.validate(p,_log(["OK"])); pd.testing.assert_frame_equal(p,before)


# ---------- Amendment 1: CHDN exclusion ----------
def _frame_with_chdn(n=40):
    syms=[f"N{i:03d}" for i in range(n)]+["CHDN"]
    return pd.DataFrame({"symbol":syms,"exchange":"NASDAQ","name":"x","raw_symbol":syms})

@pytest.mark.parametrize("seed",[1,7,20261031,99])
def test_chdn_is_never_in_the_sample_even_when_the_draw_takes_the_whole_pool(seed):
    fr=_frame_with_chdn(40)
    s=X.draw_sample(fr,{"NASDAQ":41},seed)                 # asks for more than the pool: would include CHDN if it were allowed
    assert "CHDN" not in set(s.symbol) and len(s)==40

def test_exclusion_happens_before_drawing_so_sample_size_is_preserved():
    fr=_frame_with_chdn(300)
    s=X.draw_sample(fr,{"NASDAQ":250},20261031)
    assert len(s)==250 and "CHDN" not in set(s.symbol) and s.frame_size_for_exchange.iloc[0]==300

def test_registered_seed_still_reproduces_the_same_draw():
    fr=_frame_with_chdn(300)
    a=X.draw_sample(fr,{"NASDAQ":250},20261031); b=X.draw_sample(fr,{"NASDAQ":250},20261031)
    assert list(a.symbol)==list(b.symbol)

def test_sample_csv_written_by_the_pipeline_never_contains_chdn(tmp_path):
    _setup(tmp_path)
    (tmp_path/"l"/"nasdaqlisted.txt").write_text(NASDAQ_TXT.replace("File Creation","CHDN|Churchill Downs Inc. - Common Stock|Q|N|N|100|N|N\nFile Creation"))
    X.run(_args(tmp_path,n_nasdaq=50,n_nyse=50,n_amex=50),yf=FakeYF())      # sizes larger than the pools: everything eligible is drawn
    sample=pd.read_csv(tmp_path/"o"/"sample.csv")
    assert "CHDN" not in set(sample.symbol)
    log=pd.read_csv(tmp_path/"o"/"download_log.csv",keep_default_na=False).set_index("ticker")
    assert log.loc["CHDN","group"]=="case_study" and str(log.loc["CHDN","in_sample"])=="False"     # still downloaded for the replay
    fr=pd.read_csv(tmp_path/"o"/"frame.csv"); assert bool(fr.set_index("symbol").loc["CHDN","excluded_from_sample"])

# ---------- Amendment 2: benchmark roles ----------
def _bench_run(present):
    fr=[_tk(t,seed=i) for i,t in enumerate(present)]
    return V.validate(pd.concat(fr+[_tk("OK",seed=50)],ignore_index=True),_log(present+["OK"]))
def _roles(res): return res["benchmark_roles"].set_index("role")

def test_a_role_is_satisfied_by_its_fallback_and_the_substitution_is_recorded():
    res=_bench_run(["^GSPC","SMMD","IWZ"])                  # no ^R25I, no ^RAG
    r=_roles(res)
    assert r.loc["russell_2500","used"]=="SMMD" and r.loc["russell_3000_growth","used"]=="IWZ" and r.loc["us_market","used"]=="^GSPC"
    f=res["flags"]; assert not (f.severity=="error").any()
    assert set(f[f.check=="benchmark_fallback_in_use"].ticker)=={"russell_2500","russell_3000_growth"}

def test_primary_is_preferred_when_both_exist():
    r=_roles(_bench_run(["^GSPC","^R25I","SMMD","^RAG","IWZ"])); assert r.loc["russell_2500","used"]=="^R25I" and r.loc["russell_3000_growth","used"]=="^RAG"

def test_a_primary_with_50_or_fewer_bars_falls_back_like_get_benchmark():
    short=_tk("^R25I",n=50,seed=3); full=[_tk(t,seed=i) for i,t in enumerate(["^GSPC","SMMD","^RAG"])]
    res=V.validate(pd.concat(full+[short],ignore_index=True),_log(["^GSPC","SMMD","^RAG","^R25I"]))
    assert _roles(res).loc["russell_2500","used"]=="SMMD"

def test_missing_gspc_is_an_error_and_missing_russell_roles_are_warnings_not_silent_substitutes():
    res=_bench_run(["SPY","^RAG"]); f=res["flags"]; r=_roles(res)
    assert pd.isna(r.loc["us_market","used"])                               # SPY is NOT silently substituted for ^GSPC
    e=f[f.check=="benchmark_role_unavailable"].set_index("ticker")
    assert e.loc["us_market","severity"]=="error" and e.loc["russell_2500","severity"]=="warn"

def test_validator_no_longer_demands_every_symbol():
    res=_bench_run(["^GSPC","^R25I","IWZ"]); assert not (res["flags"].severity=="error").any()


def test_progress_callback_is_called_once_per_batch_and_does_not_change_the_output(tmp_path):
    _setup(tmp_path); seen = []
    meta = X.run(_args(tmp_path), yf=FakeYF(), progress=lambda n, t, m: seen.append((n, t, m)))
    assert seen and seen[-1][0] == seen[-1][1] and all(t == seen[0][1] for _, t, _ in seen)
    meta2 = X.run(_args(tmp_path, out=str(tmp_path / "o2")), yf=FakeYF())
    assert meta["rows"] == meta2["rows"] and meta["tickers_with_data"] == meta2["tickers_with_data"]
