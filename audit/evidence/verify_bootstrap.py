"""Independent verification of the registered ticker-cluster bootstrap p-value (Amendment 1, item 3).
All simulations are under the NULL (true mean difference exactly 0). A valid 5% test rejects about 5% of the time."""
import sys, os, warnings, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from audit import stats_tools as ST

def draw(rng, T, tau, kind, lam_b=1.5, lam_v=1.0):
    u = rng.normal(0, tau, T)                                    # shared ticker effect: hits both arms
    def arm(lam):
        n = rng.poisson(lam, T); tk = np.repeat(np.arange(T), n)
        if kind == "normal":  e = rng.normal(0, .15, len(tk))
        elif kind == "skew":  e = rng.lognormal(0, .8, len(tk)); e = (e - np.exp(.32)) * .12   # mean-zero, right-skewed
        else:                 e = rng.standard_t(3, len(tk)) * .08                              # heavy tails
        return pd.DataFrame({"ticker": [f"T{i}" for i in tk], "net_return": u[tk] + e})
    return arm(lam_b), arm(lam_v)

def pvals(b, v, B, seed):
    d0, d = ST._boot_dist(b, v, B, seed)
    pa = min(1.0, 2 * min((np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1)))
    pb = ST.centred_p(d0, d)
    df = pd.concat([b.assign(arm=0), v.assign(arm=1)], ignore_index=True)
    m = smf.ols("net_return ~ arm", df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.ticker)[0]}, use_t=True)
    pc = float(m.pvalues["arm"])
    pd_ = float(stats.ttest_ind(v.net_return, b.net_return, equal_var=False).pvalue)       # ignores clustering
    return pa, pb, pc, pd_

if __name__ == "__main__":
    N, B = int(sys.argv[1]), 999; rows = []
    t0 = time.time()
    for T in (150, 300, 600):
        for tau in (0.0, 0.08):
            for kind in ("normal", "skew", "heavy"):
                rng = np.random.default_rng(hash((T, tau, kind)) % 2**32)
                P = []
                for i in range(N):
                    b, v = draw(rng, T, tau, kind)
                    if len(b) < 5 or len(v) < 5: continue
                    P.append(pvals(b, v, B, i))
                P = np.array(P)
                rows.append({"tickers": T, "ticker_sd": tau, "returns": kind, "sims": len(P),
                             **{f"{n}@5%": float((P[:, j] < .05).mean()) for j, n in enumerate(["registered_percentile", "centred_null", "statsmodels_cluster_t", "naive_welch_ignores_clusters"])},
                             **{f"{n}@1%": float((P[:, j] < .01).mean()) for j, n in enumerate(["registered_percentile", "centred_null", "statsmodels_cluster_t", "naive_welch_ignores_clusters"])}})
                print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv("bootstrap_size_simulation.csv", index=False)
