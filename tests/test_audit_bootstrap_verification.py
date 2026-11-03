import sys, os, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pytest
stats = pytest.importorskip('scipy.stats')
smf = pytest.importorskip('statsmodels.formula.api')
from audit import stats_tools as ST
from audit.evidence import verify_bootstrap as VB

def _data(seed, effect=0.0, T=200):
    rng = np.random.default_rng(seed); b, v = VB.draw(rng, T, 0.08, "skew")
    return b, v.assign(net_return=v.net_return + effect)

def _scipy_ci(b, v, B=9999, seed=3):
    tk = sorted(set(b.ticker) | set(v.ticker))
    sb = b.groupby("ticker").net_return.agg(["sum", "count"]).reindex(tk).fillna(0)
    sv = v.groupby("ticker").net_return.agg(["sum", "count"]).reindex(tk).fillna(0)
    def stat(a, n1, c, n2, axis=-1):                 # per-cluster sums/counts of each arm, resampled together (paired=True)
        return c.sum(axis) / np.maximum(n2.sum(axis), 1) - a.sum(axis) / np.maximum(n1.sum(axis), 1)
    r = stats.bootstrap((sb["sum"].values, sb["count"].values, sv["sum"].values, sv["count"].values), lambda a, n1, c, n2, axis=-1: stat(a, n1, c, n2, axis),
                        paired=True, vectorized=True, n_resamples=B, method="percentile", random_state=seed, confidence_level=0.95)
    return r.confidence_interval.low, r.confidence_interval.high

def test_confidence_interval_matches_scipy_stats_bootstrap():
    b, v = _data(1, 0.02)
    mine = ST.cluster_bootstrap_diff(b, v, B=9999, seed=5)
    lo, hi = _scipy_ci(b, v)
    assert mine["ci_low"] == pytest.approx(lo, abs=0.004) and mine["ci_high"] == pytest.approx(hi, abs=0.004)

def test_point_estimate_is_the_plain_difference_of_means():
    b, v = _data(2); assert ST.cluster_bootstrap_diff(b, v, B=500)["diff"] == pytest.approx(v.net_return.mean() - b.net_return.mean())

def test_p_value_is_the_confidence_interval_inversion_and_agrees_with_the_null_imposed_p():
    for s in range(6):
        b, v = _data(10 + s, effect=0.01 * s)
        d0, d = ST._boot_dist(b, v, 4000, 7); p_reg = ST.cluster_bootstrap_diff(b, v, B=4000, seed=7)["p"]
        assert p_reg == pytest.approx(ST.centred_p(d0, d), abs=0.03)

def test_p_value_agrees_with_statsmodels_cluster_robust_regression():
    diffs = []
    for s in range(25):
        b, v = _data(100 + s, effect=0.006 * (s % 5))
        p = ST.cluster_bootstrap_diff(b, v, B=2000, seed=s)["p"]
        df = pd.concat([b.assign(arm=0), v.assign(arm=1)], ignore_index=True)
        pc = smf.ols("net_return ~ arm", df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.ticker)[0]}, use_t=True).pvalues["arm"]
        diffs.append(abs(p - pc))
    assert np.median(diffs) < 0.02 and max(diffs) < 0.08

def test_two_sided_p_is_symmetric_in_the_sign_of_the_difference():
    b, v = _data(5, 0.03); p1 = ST.cluster_bootstrap_diff(b, v, B=3000, seed=2)["p"]
    p2 = ST.cluster_bootstrap_diff(v, b, B=3000, seed=2)["p"]; assert abs(p1 - p2) < 0.01

def test_size_under_the_null_is_close_to_nominal():
    rng = np.random.default_rng(2026); rej = 0; N = 400
    for i in range(N):
        b, v = VB.draw(rng, 200, 0.08, "skew")
        rej += ST.cluster_bootstrap_diff(b, v, B=499, seed=i)["p"] < 0.05
    assert 0.025 < rej / N < 0.085          # 400 sims: ~ +/-2.2pp Monte Carlo noise around 5%

def test_a_one_sided_p_would_be_caught():
    b, v = _data(7, 0.0); d0, d = ST._boot_dist(b, v, 3000, 1)
    one_sided = (np.sum(d <= 0) + 1) / 3001
    two_sided = ST.cluster_bootstrap_diff(b, v, B=3000, seed=1)["p"]
    assert two_sided == pytest.approx(min(1.0, 2 * min(one_sided, (np.sum(d >= 0) + 1) / 3001)))
