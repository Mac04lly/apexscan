"""stats_tools.py - Holm correction, clustered bootstrap, evidence classification (Amendments 3, 6, 7)."""
import numpy as np, pandas as pd

N_MIN_TRADES, N_MIN_TICKERS = 100, 50       # per arm, per period (Amendment 7)
EX_MIN_TRADES, EX_MIN_TICKERS = 50, 20      # exchange-level claims (Amendment 7)
EQUIV_MARGIN = 0.01                         # 1.0 percentage point of 63-session net return
ALPHA, K = 0.05, 3

def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

def _boot_dist(base, var, B, seed):
    """Bootstrap distribution of (mean variant - mean base), resampling tickers as clusters (both arms move together)."""
    rng = np.random.default_rng(seed)
    tk = np.array(sorted(set(base.ticker) | set(var.ticker)))
    sb = base.groupby("ticker").net_return.agg(["sum", "count"]).reindex(tk).fillna(0)
    sv = var.groupby("ticker").net_return.agg(["sum", "count"]).reindex(tk).fillna(0)
    d0 = var.net_return.mean() - base.net_return.mean()
    idx = rng.integers(0, len(tk), size=(B, len(tk)))
    mb = sb["sum"].values[idx].sum(1) / np.maximum(sb["count"].values[idx].sum(1), 1)
    mv = sv["sum"].values[idx].sum(1) / np.maximum(sv["count"].values[idx].sum(1), 1)
    return d0, mv - mb

def cluster_bootstrap_diff(base: pd.DataFrame, var: pd.DataFrame, B=10000, seed=1, ci=0.95):
    """Registered method (Amendment 1): percentile bootstrap, tickers resampled as clusters."""
    d0, d = _boot_dist(base, var, B, seed)
    lo, hi = np.percentile(d, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    p = min(1.0, 2 * min((np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1)))
    return {"diff": float(d0), "ci_low": float(lo), "ci_high": float(hi), "p": float(p), "B": B}

def centred_p(d0, d):
    """Null-imposed bootstrap p: how often does a bootstrap difference, re-centred on its own mean (i.e. forced to a true
    difference of zero), land at least as far from zero as the observed difference?"""
    dc = d - d.mean()
    return float((np.sum(np.abs(dc) >= abs(d0)) + 1) / (len(d) + 1))

def classify(n_var, tk_var, n_base, tk_base, diff, p_adj, ci_high_bonf, n_min=N_MIN_TRADES, t_min=N_MIN_TICKERS):
    """Amendment 7. Insufficient evidence is a different outcome from evidence of no improvement."""
    if min(n_var, n_base) < n_min or min(tk_var, tk_base) < t_min:
        return "insufficient_sample"
    if p_adj < ALPHA and diff > 0: return "statistically_supported_improvement"
    if p_adj < ALPHA and diff < 0: return "evidence_of_worse"
    if ci_high_bonf <= EQUIV_MARGIN: return "evidence_of_no_improvement"
    return "inconclusive"

def compare_three(base, variants: dict, B=10000, seed=1):
    """E1 vs each of E2, E3, E4 on the validation period. Reports unadjusted AND Holm-adjusted p. No ranking, no 'winner'."""
    rows = []
    for name, v in variants.items():
        r = cluster_bootstrap_diff(base, v, B=B, seed=seed)
        rb = cluster_bootstrap_diff(base, v, B=B, seed=seed, ci=1 - ALPHA / K)           # Bonferroni-width interval for 'no improvement'
        rows.append({"variant": name, "n_variant": len(v), "tickers_variant": v.ticker.nunique(), "n_base": len(base),
                     "tickers_base": base.ticker.nunique(), "diff": r["diff"], "ci95_low": r["ci_low"], "ci95_high": r["ci_high"],
                     "p_unadjusted": r["p"], "ci_bonf_high": rb["ci_high"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_unadjusted.values)
    out["sig_unadjusted"] = out.p_unadjusted < ALPHA; out["sig_holm"] = out.p_holm < ALPHA
    out["evidence"] = [classify(r.n_variant, r.tickers_variant, r.n_base, r.tickers_base, r["diff"], r.p_holm, r.ci_bonf_high)
                       for _, r in out.iterrows()]
    return out.drop(columns="ci_bonf_high")

def assert_strategy_population(pop):
    """Amendment 6: population B (rally-selected) is descriptive only. It was chosen by its outcome, so any return
    statistic from it is biased by construction."""
    if str(pop).upper() == "B":
        raise ValueError("Population B is descriptive only: it must not be used for profitability or threshold tuning")


COST_MULTS = (0.5, 1.0, 2.0)                  # registered: half, registered, double

def with_cost(trades: pd.DataFrame, mult: float) -> pd.DataFrame:
    """Recompute net_return at a cost multiplier from the stored gross prices. Needs entry_open, exit_open, cost_side_1x."""
    c = trades["cost_side_1x"] * mult
    return trades.assign(net_return=trades["exit_open"] * (1 - c) / (trades["entry_open"] * (1 + c)) - 1)

def cost_sensitivity(base: pd.DataFrame, variants: dict, mults=COST_MULTS, B=10000, seed=1):
    """Every comparison at half, registered and double cost, with a flag for conclusions that depend on the cost assumption.
    cost_dependent = True when the evidence class, the Holm significance or the sign of the difference changes across the three."""
    parts = []
    for m in mults:
        t = compare_three(with_cost(base, m), {k: with_cost(v, m) for k, v in variants.items()}, B=B, seed=seed)
        t.insert(0, "cost_mult", m)
        t["mean_net_variant"] = [with_cost(variants[k], m).net_return.mean() for k in t.variant]
        t["mean_net_E1"] = with_cost(base, m).net_return.mean()
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    dep = out.groupby("variant").apply(lambda g: bool(g.evidence.nunique() > 1 or g.sig_holm.nunique() > 1 or (np.sign(g["diff"]).nunique() > 1)))
    out["cost_dependent"] = out.variant.map(dep)
    return out
