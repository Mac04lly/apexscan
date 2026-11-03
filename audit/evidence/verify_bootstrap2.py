"""Supplementary null simulation where clustering MATTERS: each arm has its own ticker-level effect, trades repeat within a
ticker (3 per ticker per arm on average). Ignoring clusters should now over-reject; the cluster methods should not."""
import sys, os, warnings; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from audit.evidence import verify_bootstrap as VB
def draw2(rng, T, tau, kind, lam=3.0):
    def arm():
        u = rng.normal(0, tau, T); n = rng.poisson(lam, T); tk = np.repeat(np.arange(T), n)
        e = rng.normal(0, .15, len(tk)) if kind == "normal" else (rng.lognormal(0, .8, len(tk)) - np.exp(.32)) * .12
        return pd.DataFrame({"ticker": [f"T{i}" for i in tk], "net_return": u[tk] + e})
    return arm(), arm()
rows = []
for T in (150, 300):
    for kind in ("normal", "skew"):
        rng = np.random.default_rng(T * 7 + len(kind)); P = []
        for i in range(int(sys.argv[1])):
            b, v = draw2(rng, T, 0.10, kind); P.append(VB.pvals(b, v, 999, i))
        P = np.array(P); rows.append({"tickers": T, "returns": kind, "sims": len(P),
            **{f"{n}@5%": round(float((P[:, j] < .05).mean()), 4) for j, n in enumerate(["registered_percentile", "centred_null", "statsmodels_cluster_t", "naive_welch_ignores_clusters"])}})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("bootstrap_size_simulation_clustered.csv", index=False)
