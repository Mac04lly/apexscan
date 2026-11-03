import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, pytest
from audit import inapp as IA


def synth(chdn=True, gspc=True, n=330, seed=1):
    idx = pd.bdate_range(end="2026-10-30", periods=n); rng = np.random.default_rng(seed); frames = []
    def mk(t, c, vol=1_500_000):
        c = np.asarray(c, float); return pd.DataFrame({"ticker": t, "date": idx.strftime("%Y-%m-%d"), "open": c, "high": c * 1.01, "low": c * .99, "close": c, "volume": float(vol)})
    bench = 4000 * np.cumprod(1 + rng.normal(0.0003, 0.004, n))
    shape = np.concatenate([np.linspace(90, 110, n // 3), np.linspace(110, 74, n // 3), np.linspace(74, 95, n - 2 * (n // 3))]) * (1 + rng.normal(0, .004, n))
    if gspc: frames.append(mk("^GSPC", bench))
    for b in ("^R25I", "SMMD", "^RAG", "IWZ"): frames.append(mk(b, bench * (1 + rng.normal(0, .001, n))))
    if chdn: frames.append(mk("CHDN", shape))
    frames.append(mk("OTHR", 50 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))))
    return pd.concat(frames, ignore_index=True)


def test_dq_runs_on_synthetic_data_and_produces_a_markdown_report():
    p = synth(); log = pd.DataFrame({"ticker": list(p.ticker.unique()), "group": "sample", "status": "ok", "error": ""})
    res, md = IA.dq(p, log); assert "Reference calendar: ^GSPC" in md and "Tickers with at least one **error** flag: 0" in md and len(res["per_ticker"]) == p.ticker.nunique()
    assert IA.has_blocking_errors(res) == []


def test_missing_gspc_is_reported_as_blocking():
    p = synth(gspc=False); log = pd.DataFrame({"ticker": list(p.ticker.unique()), "group": "sample", "status": "ok", "error": ""})
    res, _ = IA.dq(p, log); assert IA.has_blocking_errors(res)


def test_replay_for_finds_the_same_200_day_date_as_an_independent_calculation():
    p = synth(); r, fp = IA.replay_for(p, "CHDN", "2026-06-01")
    h = IA.ticker_hist(p, "CHDN")["Close"]; ok = {}
    for t in r.index:
        w = h[(h.index > t - pd.Timedelta(days=365)) & (h.index <= t)].values; ok[t] = len(w) >= 200 and w[-1] > w[-200:].mean()
    s = pd.Series(ok); f = s[~s]; first = (s.index[0] if f.empty else s.index[s.index > f.index[-1]][0]).date().isoformat() if s.iloc[-1] else "not passing on last date"
    assert fp["g1_above_200ma"] == first


def test_replay_refuses_without_gspc_and_for_an_unknown_ticker():
    with pytest.raises(ValueError, match="GSPC"): IA.replay_for(synth(gspc=False), "CHDN")
    with pytest.raises(ValueError, match="no rows"): IA.replay_for(synth(), "NOPE")


def test_ticker_file_contains_every_discovery_plus_chdn_once(tmp_path):
    p = IA.ticker_file([{"ticker": "A"}, {"ticker": "B"}, {"ticker": "CHDN"}], tmp_path / "t.txt")
    assert p.read_text().split() == ["A", "B", "CHDN"]
