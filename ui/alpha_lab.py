"""
ui/alpha_lab.py — Apex Alpha Lab workspace (V9 Phases 4-7)

The research dashboard V9's spec calls for: "The user can investigate
the evidence without touching code." Everything here reads from
data/alpha_observations.json (Phase 1) with outcomes frozen by
modules/outcome_engine.py (Phase 2), and is pure display logic — every
number shown is computed by modules/alpha_metrics.py, nothing is
computed inline here, so this file can be tested/reasoned about
separately from the statistics themselves.

This is intentionally a standalone module rather than more lines added
to dashboard.py (which is already ~10,600 lines) — the V9 spec's own
§37 explicitly asks for exactly this: extract reusable UI components
into ui/, and keep dashboard.py as the orchestration layer that calls
them. render_alpha_lab() is the single entry point dashboard.py needs.

Never raises up into dashboard.py — every section is wrapped so one
broken chart never takes down the whole tab, matching the defensive
pattern already used for the AI layer and the rest of this app.
"""
from __future__ import annotations
import logging

import pandas as pd
import streamlit as st

log = logging.getLogger("apexscan.ui.alpha_lab")

HORIZONS = ["5D", "10D", "20D", "40D", "60D"]


def _fmt_pct(v, sign=True):
    if v is None:
        return "–"
    return f"{v:+.2f}%" if sign else f"{v:.2f}%"


def _fmt_ci(ci):
    if not ci:
        return "–"
    return f"[{ci[0]:+.1f}%, {ci[1]:+.1f}%]"


def _metrics_row_to_disp(prefix_key: str, r: dict) -> dict:
    return {
        prefix_key: r.get(prefix_key, r.get("label", "")),
        "N": r["n"],
        "Sample": r["sample_classification"].split(" — ")[0],
        "Win Rate": f"{r['win_rate_%']:.1f}%" if r["win_rate_%"] is not None else "–",
        "Expectancy": _fmt_pct(r["expectancy_%"]),
        "Median": _fmt_pct(r["median_return_%"]),
        "Profit Factor": (f"{r['profit_factor']:.2f}" if isinstance(r["profit_factor"], (int, float)) else "–"),
        "95% CI": _fmt_ci(r["confidence_interval_95"]),
        "Avg Excess vs Benchmark": _fmt_pct(r["avg_excess_return_%"]),
    }


def _render_overview(observations: list, horizon: str):
    from modules.alpha_metrics import compute_alpha_lab_overview

    ov = compute_alpha_lab_overview(observations, horizon)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observations", ov["total_observations"])
    c2.metric(f"Resolved @ {horizon}", ov["resolved_observations"])
    c3.metric("Unresolved (still pending)", ov["unresolved_observations"])
    overall = ov["overall"]
    c4.metric(f"Overall Win Rate @ {horizon}",
              f"{overall['win_rate_%']:.1f}%" if overall["win_rate_%"] is not None else "–")

    c5, c6, c7 = st.columns(3)
    c5.metric("Expectancy", _fmt_pct(overall["expectancy_%"]))
    c6.metric("Avg Excess vs Benchmark", _fmt_pct(overall["avg_excess_return_%"]))
    c7.metric("Profit Factor",
              f"{overall['profit_factor']:.2f}" if isinstance(overall["profit_factor"], (int, float)) else "–")

    st.caption(f"Sample: {overall['sample_classification']}")

    bs, bf, wf = ov["best_setup"], ov["best_feature"], ov["worst_feature"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Best-evidenced setup (n≥10)**")
        if bs:
            st.write(f"{bs['setup_label']} — {_fmt_pct(bs['expectancy_%'])} expectancy, n={bs['n']}")
        else:
            st.caption("No setup has 10+ resolved observations at this horizon yet.")
    with col2:
        st.markdown("**Widest feature spread (n≥10 per bucket)**")
        if bf:
            st.write(f"{bf['feature_label']}: {bf['best_bucket_label']} "
                     f"({_fmt_pct(bf['best_bucket_expectancy_%'])}) vs {bf['worst_bucket_label']} "
                     f"({_fmt_pct(bf['worst_bucket_expectancy_%'])})")
        else:
            st.caption("No feature has 2+ buckets with 10+ resolved observations yet.")

    if ov["unresolved_observations"] > 0:
        st.caption(
            f"{ov['unresolved_observations']} observation(s) haven't reached the {horizon} mark "
            "yet — they'll join these numbers automatically once they do. Nothing here is "
            "estimated ahead of time."
        )


def _render_score_validation(observations: list, horizon: str):
    from modules.alpha_metrics import compute_alpha_metrics_by_score_bucket

    st.caption(
        "Each stock's return is measured at a FIXED number of trading days after discovery, "
        "frozen permanently once computed — not 'whatever day you happened to check.' Includes "
        "a 95% confidence interval, so a bucket's apparent edge can be judged against real "
        "statistical uncertainty, not just a point estimate."
    )
    rows = [r for r in compute_alpha_metrics_by_score_bucket(observations, horizon) if r["n"] > 0]
    if not rows:
        st.info(f"No observations have reached the {horizon} horizon yet.")
        return
    disp = pd.DataFrame([_metrics_row_to_disp("Score Bucket", {**r, "label": r["score_bucket"]})
                         for r in rows])
    disp = disp.rename(columns={"Score Bucket": "Score Bucket"})
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(
        "A bucket whose 95% CI spans zero hasn't yet demonstrated a statistically real edge at "
        "this horizon, regardless of how good the point estimate looks — that's the honest read, "
        "not a discouraging one."
    )


def _render_setup_alpha(observations: list, horizon: str):
    from modules.alpha_metrics import compute_alpha_metrics_by_setup

    st.caption(
        "Groups every discovery by the setup type it was tagged with at the moment it qualified "
        "— e.g. pullback to the 50-day average vs. a fresh 200-day reclaim. Answers: which setup "
        "types actually work?"
    )
    rows = compute_alpha_metrics_by_setup(observations, horizon)
    if not rows:
        st.info("No setups logged yet.")
        return
    disp = pd.DataFrame([_metrics_row_to_disp("Setup", {**r, "label": r["setup_label"]}) for r in rows])
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(
        "Rows with n=0 are setup types that exist in the taxonomy but haven't produced a "
        "discovery yet — that's informative too, not a bug."
    )


def _render_feature_alpha(observations: list, horizon: str):
    from modules.alpha_metrics import FEATURE_REGISTRY, compute_alpha_metrics_by_feature, rank_feature_alpha

    st.caption(
        "Buckets every observation by ONE feature and measures outcomes per bucket — the same "
        "test applied to the Apex Score above, applied to every individual ingredient that goes "
        "into it. Results describe historical association within ApexScan's own observed "
        "universe, never a causal claim."
    )
    ranked = rank_feature_alpha(observations, horizon)
    conclusive = [r for r in ranked if r["is_conclusive"]]
    if conclusive:
        st.markdown("**Feature Alpha ranking** — spread between a feature's best- and "
                    "worst-performing bucket (larger spread = more separation observed so far)")
        rank_disp = pd.DataFrame([{
            "Feature": r["feature_label"],
            "Spread": _fmt_pct(r["spread_%"]),
            "Best Bucket": f"{r['best_bucket_label']} ({_fmt_pct(r['best_bucket_expectancy_%'])})",
            "Worst Bucket": f"{r['worst_bucket_label']} ({_fmt_pct(r['worst_bucket_expectancy_%'])})",
            "Qualifying Buckets": r["n_qualifying_buckets"],
            "Total N": r["total_n"],
        } for r in conclusive])
        st.dataframe(rank_disp, use_container_width=True, hide_index=True)
    else:
        st.info("No feature yet has 2+ buckets with 10+ resolved observations at this horizon.")

    st.markdown("---")
    feature_keys = list(FEATURE_REGISTRY.keys())
    labels = [FEATURE_REGISTRY[k]["label"] for k in feature_keys]
    sel_label = st.selectbox("Drill into one feature", labels, key="alpha_lab_feature_drill")
    sel_key = feature_keys[labels.index(sel_label)]
    detail_rows = compute_alpha_metrics_by_feature(observations, horizon, sel_key)
    detail_rows = [r for r in detail_rows if r["n"] > 0]
    if not detail_rows:
        st.caption("No resolved observations for this feature at this horizon yet.")
    else:
        detail_disp = pd.DataFrame([_metrics_row_to_disp("Bucket", {**r, "label": r["bucket_label"]})
                                    for r in detail_rows])
        st.dataframe(detail_disp, use_container_width=True, hide_index=True)


def _render_conditional_alpha(observations: list, horizon: str):
    from modules.alpha_metrics import FEATURE_REGISTRY, compute_conditional_alpha

    st.caption(
        "When do a feature's strongest signals actually work? Filters to observations in a "
        "feature's top bucket, then breaks that filtered set down by market regime, sector, or "
        "setup — e.g. 'high RS, by sector.'"
    )
    feature_keys = list(FEATURE_REGISTRY.keys())
    labels = [FEATURE_REGISTRY[k]["label"] for k in feature_keys]
    c1, c2 = st.columns([2, 1])
    with c1:
        sel_label = st.selectbox("Feature (top bucket)", labels, key="alpha_lab_cond_feature")
    with c2:
        condition_type = st.selectbox("Break down by", ["regime", "sector", "setup"],
                                       key="alpha_lab_cond_type")
    sel_key = feature_keys[labels.index(sel_label)]

    result = compute_conditional_alpha(observations, horizon, sel_key, condition_type)
    st.caption(f"Filter: {result['filter_description']}")
    rows = [r for r in result["rows"] if r["n"] > 0]
    if not rows:
        st.info("No qualifying observations have resolved at this horizon yet.")
        return
    disp = pd.DataFrame([_metrics_row_to_disp(condition_type.capitalize(),
                         {**r, "label": r["condition_value"]}) for r in rows])
    st.dataframe(disp, use_container_width=True, hide_index=True)
    if condition_type == "regime" and len({r["condition_value"] for r in rows}) <= 1:
        st.caption(
            "Only one market regime has been observed so far — regime-conditional differences "
            "won't be visible until ApexScan has scanned across more varied market conditions. "
            "This isn't a bug; it's an honest reflection of how much history exists yet."
        )


def _render_combinations_and_findings(observations: list, horizon: str):
    from modules.alpha_metrics import get_combination_alpha_table, generate_research_findings

    st.markdown("#### Combination Research")
    st.caption(
        "A deliberately SHORT, fixed list of sensible feature combinations — not an exhaustive "
        "search. Testing many combinations and reporting only the best-looking one is a classic "
        "way to manufacture a fake edge (multiple-testing bias); this list is fixed in code and "
        "every combination is shown, win or lose."
    )
    combo_rows = get_combination_alpha_table(observations, horizon)
    combo_disp = pd.DataFrame([_metrics_row_to_disp("Combination", {**r, "label": r["label"]})
                               for r in combo_rows])
    st.dataframe(combo_disp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Research Findings")
    st.caption(
        "Auto-generated from setups, feature buckets, and the combinations above that have "
        "reached at least 10 resolved observations. Every finding traces back to the exact "
        "setup/feature/combination that produced it — nothing here is a black-box claim."
    )
    min_n = st.slider("Minimum sample size per finding", 10, 100, 10, step=5,
                      key="alpha_lab_findings_min_n")
    findings = generate_research_findings(observations, horizon, min_n=min_n)
    if not findings:
        st.info(f"No setup, feature bucket, or combination has reached {min_n} resolved "
                f"observations at {horizon} yet.")
        return
    for i, f in enumerate(findings, 1):
        st.markdown(f"**Finding {i}.** {f['statement']}")
        st.caption(f"Evidence: {f['evidence']} · source: {f['source']['type']} "
                  f"({f['source'].get('key', '')})")


def render_alpha_lab():
    """Single entry point dashboard.py calls. Loads observations itself
    — the caller doesn't need to fetch anything first."""
    st.markdown("## 🧪 Apex Alpha Lab")
    st.caption(
        "Prove what works. Every table below is computed from immutable, timestamped "
        "observations — features captured at the exact moment a stock qualified, outcomes "
        "frozen once measured, never recomputed with hindsight."
    )

    try:
        from modules.alpha_validation import load_observations
        from modules.outcome_engine import compute_all_pending_outcomes

        c1, c2 = st.columns([1, 3])
        with c1:
            horizon = st.selectbox("Horizon", HORIZONS, index=2, key="alpha_lab_horizon")
        with c2:
            if st.button("🔄 Compute Pending Outcomes Now", key="alpha_lab_compute_btn"):
                with st.spinner("Computing any outcomes that have reached their horizon…"):
                    n_computed = compute_all_pending_outcomes()
                st.success(f"Updated {n_computed} observation(s)." if n_computed
                          else "Nothing new to compute yet.")

        observations = load_observations()
        if not observations:
            st.info(
                "No Alpha Observations logged yet — these accumulate automatically from your "
                "next scan onward."
            )
            return

        tabs = st.tabs(["Overview", "Score Validation", "Setup Alpha", "Feature Alpha",
                        "Conditional Alpha", "Combinations & Findings"])

        with tabs[0]:
            try:
                _render_overview(observations, horizon)
            except Exception as e:
                st.caption(f"Overview unavailable this session: {e}")
        with tabs[1]:
            try:
                _render_score_validation(observations, horizon)
            except Exception as e:
                st.caption(f"Score Validation unavailable this session: {e}")
        with tabs[2]:
            try:
                _render_setup_alpha(observations, horizon)
            except Exception as e:
                st.caption(f"Setup Alpha unavailable this session: {e}")
        with tabs[3]:
            try:
                _render_feature_alpha(observations, horizon)
            except Exception as e:
                st.caption(f"Feature Alpha unavailable this session: {e}")
        with tabs[4]:
            try:
                _render_conditional_alpha(observations, horizon)
            except Exception as e:
                st.caption(f"Conditional Alpha unavailable this session: {e}")
        with tabs[5]:
            try:
                _render_combinations_and_findings(observations, horizon)
            except Exception as e:
                st.caption(f"Combinations & Findings unavailable this session: {e}")

        st.markdown("---")
        st.caption(
            "**Model Comparison** — scaffolded for later use: every observation already carries "
            "a model_version tag (currently APEX-9.0 for all of them). Once a second model "
            "version exists (e.g. after a reviewed scoring change), this section will compare "
            "them side by side. Not built yet because there's only one version to compare."
        )
    except Exception as e:
        log.warning(f"Alpha Lab failed to render: {e}")
        st.caption(f"Alpha Lab unavailable this session: {e}")
