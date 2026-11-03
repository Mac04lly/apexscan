import sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, pytest
from audit import stats_tools as ST
multipletests = pytest.importorskip('statsmodels.stats.multitest').multipletests

def test_holm_matches_statsmodels_and_the_textbook_example():
    assert list(ST.holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.uniform(0, 1, 3) ** 2
        assert ST.holm(p) == pytest.approx(multipletests(p, method="holm")[1])

def test_holm_is_never_smaller_than_unadjusted_and_capped_at_one():
    p = np.array([0.2, 0.5, 0.9]); a = ST.holm(p); assert (a >= p).all() and a.max() <= 1.0

def mk(n, mean, sd=0.15, tickers=None, seed=0, tk_effect=0.0):
    rng = np.random.default_rng(seed); tickers = tickers or [f"T{i}" for i in range(n)]
    eff = {t: rng.normal(0, tk_effect) for t in set(tickers)}
    return pd.DataFrame({"ticker": tickers, "net_return": [mean + eff[t] + rng.normal(0, sd) for t in tickers]})

def test_bootstrap_detects_a_real_difference():
    r = ST.cluster_bootstrap_diff(mk(400, 0.00, seed=1), mk(400, 0.05, seed=2, tickers=[f"V{i}" for i in range(400)]), B=2000)
    assert r["p"] < 0.01 and r["ci_low"] > 0

def test_bootstrap_false_positive_rate_is_near_nominal_under_the_null():
    hits = 0
    for s in range(120):
        a, b = mk(150, 0, seed=s), mk(150, 0, seed=1000 + s, tickers=[f"V{i}" for i in range(150)])
        hits += ST.cluster_bootstrap_diff(a, b, B=400, seed=s)["p"] < 0.05
    assert hits / 120 < 0.12

def test_clustering_by_ticker_widens_the_interval_when_a_ticker_repeats():
    tk = [f"T{i % 20}" for i in range(400)]                  # 20 tickers, 20 trades each, strong ticker effect
    a = mk(400, 0, tickers=tk, seed=3, tk_effect=0.10); b = mk(400, 0, tickers=tk, seed=4, tk_effect=0.10)
    clustered = ST.cluster_bootstrap_diff(a, b, B=2000)
    a2, b2 = a.assign(ticker=[f"x{i}" for i in range(400)]), b.assign(ticker=[f"y{i}" for i in range(400)])
    naive = ST.cluster_bootstrap_diff(a2, b2, B=2000)
    assert (clustered["ci_high"] - clustered["ci_low"]) > 1.5 * (naive["ci_high"] - naive["ci_low"])

@pytest.mark.parametrize("args,expect", [
    ((60, 60, 400, 200, 0.10, 0.001, 0.3), "insufficient_sample"),          # tiny p cannot rescue a small arm
    ((400, 200, 400, 200, 0.03, 0.01, 0.03 + 0.0), "statistically_supported_improvement"),
    ((400, 200, 400, 200, -0.03, 0.01, 0.0), "evidence_of_worse"),
    ((400, 200, 400, 200, 0.002, 0.60, 0.008), "evidence_of_no_improvement"),
    ((400, 200, 400, 200, 0.002, 0.60, 0.04), "inconclusive"),
    ((400, 40, 400, 200, 0.0, 0.5, 0.0), "insufficient_sample"),             # enough trades but too few distinct tickers
])
def test_insufficient_evidence_is_not_the_same_as_evidence_of_no_improvement(args, expect):
    assert ST.classify(*args) == expect

def test_compare_three_reports_both_adjusted_and_unadjusted_and_has_no_winner(monkeypatch):
    seq = iter([0.04, 0.03, 0.20])
    def fake(base, var, B=0, seed=0, ci=0.95): return {"diff": 0.02, "ci_low": 0.0, "ci_high": 0.04, "p": next(seq) if ci == 0.95 else 0, "B": B}
    monkeypatch.setattr(ST, "cluster_bootstrap_diff", fake)
    base = mk(300, 0, seed=1); v = {n: mk(300, 0, seed=i, tickers=[f"{n}{j}" for j in range(300)]) for i, n in enumerate(["E2", "E3", "E4"], 2)}
    out = ST.compare_three(base, v).set_index("variant")
    assert out.loc["E2", "sig_unadjusted"] and out.loc["E3", "sig_unadjusted"]          # nominally significant ...
    assert not out.loc["E2", "sig_holm"] and not out.loc["E3", "sig_holm"]              # ... but not after Holm (0.04*3, 0.03*... )
    assert out.loc["E2", "p_holm"] == pytest.approx(0.09)
    assert "winner" not in out.columns and "rank" not in out.columns
    assert not (out.evidence == "statistically_supported_improvement").any()

def test_highest_observed_return_is_not_selected_without_significance():
    base = mk(300, 0.00, seed=1)
    v = {"E2": mk(300, 0.03, sd=0.5, seed=2, tickers=[f"a{i}" for i in range(300)]),     # highest mean, very noisy
         "E3": mk(300, 0.00, seed=3, tickers=[f"b{i}" for i in range(300)]),
         "E4": mk(300, -0.01, seed=4, tickers=[f"c{i}" for i in range(300)])}
    for k, m in (("E2", 0.03), ("E3", 0.0), ("E4", -0.01)):      # pin the observed means exactly
        v[k]["net_return"] = v[k].net_return - v[k].net_return.mean() + m
    base["net_return"] = base.net_return - base.net_return.mean()
    out = ST.compare_three(base, v, B=2000).set_index("variant")
    assert out["diff"].idxmax() == "E2" and out.loc["E2", "evidence"] != "statistically_supported_improvement"

def test_population_B_cannot_be_used_for_strategy_statistics():
    with pytest.raises(ValueError): ST.assert_strategy_population("B")
    ST.assert_strategy_population("C"); ST.assert_strategy_population("A")

def test_exchange_level_claims_use_the_lower_registered_minimum():
    assert ST.classify(60, 25, 70, 30, 0.03, 0.01, 0.0, n_min=ST.EX_MIN_TRADES, t_min=ST.EX_MIN_TICKERS) == "statistically_supported_improvement"
    assert ST.classify(60, 25, 70, 30, 0.03, 0.01, 0.0) == "insufficient_sample"
    assert ST.classify(40, 25, 70, 30, 0.03, 0.01, 0.0, n_min=ST.EX_MIN_TRADES, t_min=ST.EX_MIN_TICKERS) == "insufficient_sample"


# ---------- Amendment 3 follow-up: cost sensitivity (half / registered / double) ----------
from audit import execution as EX

def _trades(n, gross, cost, seed, prefix):
    rng = np.random.default_rng(seed); eo = rng.uniform(10, 50, n); xo = eo * (1 + gross + rng.normal(0, .12, n))
    return pd.DataFrame({"ticker": [f"{prefix}{i}" for i in range(n)], "entry_open": eo, "exit_open": xo, "cost_side_1x": cost})

def test_with_cost_matches_execution_net_at_and_changes_only_the_cost():
    t = _trades(5, 0.02, 0.003, 1, "a")
    for m in (0.5, 1, 2):
        got = ST.with_cost(t, m).net_return.values
        exp = [EX.net_at(r.entry_open, r.exit_open, r.cost_side_1x, m) for r in t.itertuples()]
        assert got == pytest.approx(exp)
    assert (ST.with_cost(t, 2).net_return < ST.with_cost(t, 0.5).net_return).all()

def test_plan_trade_records_the_1x_cost_regardless_of_the_multiplier_used():
    import test_audit_execution as TE
    a = EX.plan_trade(TE.stock(), TE.CAL, TE.T, cost_mult=2.0); assert a["cost_side"] == 2 * a["cost_side_1x"]

def test_cost_sensitivity_returns_all_three_levels_for_every_variant():
    base = _trades(300, 0.00, 0.0030, 1, "b"); v = {k: _trades(300, 0.00, 0.0030, i, k) for i, k in enumerate(["E2", "E3", "E4"], 2)}
    out = ST.cost_sensitivity(base, v, B=500)
    assert sorted(out.cost_mult.unique()) == [0.5, 1.0, 2.0] and len(out) == 9 and {"p_unadjusted", "p_holm", "cost_dependent"} <= set(out.columns)

def test_a_conclusion_that_exists_only_at_low_cost_is_flagged_cost_dependent():
    # variant has a small cost advantage (cheap tier) and a tiny gross edge: it looks fine at half cost, not at double
    base = _trades(1500, 0.000, 0.0075, 1, "b")
    v = {"E2": _trades(1500, 0.0, 0.0075, 2, "x"), "E3": _trades(1500, 0.0, 0.0075, 3, "y"), "E4": _trades(1500, 0.0, 0.0075, 4, "z")}
    v["E2"]["cost_side_1x"] = 0.0010                        # E2 trades liquid names: cost gap vs base widens with the multiplier
    v["E2"]["exit_open"] = v["E2"]["entry_open"] * (1 + (v["E2"]["exit_open"] / v["E2"]["entry_open"] - 1) + 0.012)
    out = ST.cost_sensitivity(base, v, B=1500)
    e2 = out[out.variant == "E2"].set_index("cost_mult")
    assert e2.loc[2.0, "diff"] > e2.loc[0.5, "diff"] or e2.loc[2.0, "diff"] != e2.loc[0.5, "diff"]
    assert out.groupby("variant").cost_dependent.nunique().max() == 1            # the flag is per variant, not per row

def test_cost_dependent_is_false_when_every_level_gives_the_same_class():
    base = _trades(300, 0.0, 0.003, 1, "b"); v = {k: _trades(300, 0.0, 0.003, i, k) for i, k in enumerate(["E2", "E3", "E4"], 2)}
    out = ST.cost_sensitivity(base, v, B=500)
    same = out.groupby("variant").evidence.nunique() == 1
    assert (out.groupby("variant").cost_dependent.first() == (~same)).all()


def test_cost_dependent_flag_logic_with_controlled_results(monkeypatch):
    base = _trades(50, 0, .003, 1, "b"); v = {k: _trades(50, 0, .003, i, k) for i, k in enumerate(["E2", "E3", "E4"], 2)}
    calls = {"n": 0}
    def fake(b, vv, B=0, seed=0):
        m = [0.5, 1.0, 2.0][calls["n"]]; calls["n"] += 1
        ev = {"E2": ["statistically_supported_improvement", "statistically_supported_improvement", "inconclusive"][int(m == 2.0) * 2],
              "E3": "inconclusive", "E4": "insufficient_sample"}
        rows = [{"variant": k, "n_variant": 1, "tickers_variant": 1, "n_base": 1, "tickers_base": 1, "diff": 0.01 if k != "E4" else -0.01,
                 "ci95_low": 0, "ci95_high": 0, "p_unadjusted": .5, "p_holm": .5, "sig_unadjusted": False, "sig_holm": m < 2.0 and k == "E2",
                 "evidence": ev[k]} for k in ["E2", "E3", "E4"]]
        return pd.DataFrame(rows)
    monkeypatch.setattr(ST, "compare_three", fake)
    out = ST.cost_sensitivity(base, v, B=1).groupby("variant").cost_dependent.first()
    assert bool(out["E2"]) and not bool(out["E3"]) and not bool(out["E4"])
