"""
pages/Audit.py - ApexScan evidence audit (read-only).

A separate page: it appears as "Audit" in the sidebar and changes nothing about scanning, scoring, stored data or any other
tab. It reads data/discoveries.json, never writes to data/, never calls the scanner's run_scan, and holds downloaded prices
only in this browser session (and in a download you choose to save).
"""
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ApexScan Audit", page_icon="🔬", layout="wide")

from audit import inapp, maturity, registry  # noqa: E402  (after set_page_config, which must be the first Streamlit call)


def _md(text: str):
    st.markdown(text.replace("$", "\\$"))          # a bare $ would be read as maths


def _show(df: pd.DataFrame, **kw):
    st.dataframe(df.astype(str), hide_index=True, use_container_width=True, **kw)


st.title("🔬 ApexScan Audit")
st.caption("Evidence only. This page does not change scanning, scores, settings or saved data.")

tabs = st.tabs(["Status", "Registration", "Live discoveries", "Get prices", "Data quality", "CHDN replay"])

# ── Status ────────────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Where the audit stands (1 Nov 2026)")
    _show(pd.DataFrame([
        ("Methodology, Amendment 1 and 2 registered", "Done", "Hash-checked on the Registration tab"),
        ("Live discoveries: matured vs immature", "Live on this page", "Live discoveries tab"),
        ("Price export tool", "Built and tested on synthetic data", "Not yet run against live Yahoo data"),
        ("Data-quality report", "Tool built", "Waits for real prices; must be reviewed before any analysis"),
        ("CHDN replay (Test A)", "Engine built and tested", "Waits for real prices"),
        ("Research-period results", "Not started", "Waits for reviewed data"),
        ("Validation-period results", "Not started", "Run once, after the data-quality review"),
        ("Final recommendation", "Not available", "No result has been examined, so none is made"),
    ], columns=["Item", "State", "Note"]))
    st.info("Test A (replay) shows what ApexScan's rules would have said. It is never evidence of what the live app displayed "
            "on a given day. That evidence (Test B) comes from the saved scan reports.")

# ── Registration ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Registered text and integrity check")
    rows = registry.verify()
    if rows and all(r["status"] == "ok" for r in rows):
        st.success("All registered documents match the hashes recorded on the day they were registered.")
    else:
        st.error("A registered document has changed or is missing. Compare the hashes below.")
    _show(pd.DataFrame(rows)[["document", "registered_utc", "status", "recorded"]])
    for name, label in (("methodology.md", "Methodology (registered 31 Oct 2026)"), ("amendments.md", "Amendment 1"),
                        ("amendments2.md", "Amendment 2")):
        with st.expander(label):
            _md(registry.text(name))

# ── Live discoveries ─────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Recorded discoveries: which have completed a holding period?")
    st.caption("Entry is the open of the first session after the scan date. A discovery counts as matured only if the exit open "
               "H sessions later already exists. Immature observations are never averaged with matured ones.")
    dpath = REPO / "data" / "discoveries.json"
    try:
        disc = json.loads(dpath.read_text(encoding="utf-8"))
        cal = maturity.sessions_2026()
        default_last = maturity.last_completed_session(dt.date.today(), cal).date()
        last = st.date_input("Data through (last completed session)", value=default_last, key="audit_data_through")
        last = pd.Timestamp(last)
        if last not in cal:
            st.warning("That date is not a trading session; pick a trading day.")
        else:
            df = maturity.table(disc, last, cal)
            s = maturity.summary(df)
            c = st.columns(5)
            c[0].metric("Discoveries", s["discoveries"])
            c[1].metric("No entry session yet", s["no_entry_yet"])
            for col, h in zip(c[2:], maturity.HORIZONS):
                col.metric(f"Matured at {h} sessions", s[f"matured_{h}"], help=f"{s[f'immature_{h}']} not yet matured")
            e63 = maturity.earliest_maturity(df, cal, 63)
            if s["matured_63"] == 0:
                st.warning("None of the discoveries has completed the registered 63-session holding period"
                           + (f"; the earliest does so on {e63}." if e63 else ".") + " No statement is made about the 63-session horizon until they do.")
            st.markdown("**Sessions of history since entry**")
            _show(pd.DataFrame({"band": list(s["bands"]), "discoveries": list(s["bands"].values())}))
            with st.expander("Per-discovery table"):
                _show(df)
            st.download_button("Download this table (CSV)", df.to_csv(index=False), "population_A_maturity.csv", "text/csv", key="audit_dl_maturity")
    except Exception as ex:
        st.error(f"Could not read the discovery log: {ex}")

# ── Get prices ───────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Download the price history needed for the replay")
    st.caption("Runs the registered export from this app's server: Yahoo daily prices (adjusted, like the scanner) for the 620-stock "
               "random sample, the recorded discoveries, CHDN and the benchmarks. Read-only. It takes several minutes; stay on this "
               "page until it finishes. Results are kept in this browser session and offered as a download.")
    full = st.checkbox("Download every listed common stock instead (very slow, large)", value=False, key="audit_full_frame")
    if st.button("Run the export", key="audit_run_export", type="primary"):
        try:
            from audit.tools import export_prices as EX
            disc = json.loads((REPO / "data" / "discoveries.json").read_text(encoding="utf-8"))
            out = Path(tempfile.mkdtemp(prefix="apexscan_audit_"))
            tf = inapp.ticker_file(disc, out / "tickers_live.txt")
            argv = ["--out", str(out / "apexscan_audit_data"), "--tickers-file", str(tf)] + (["--full-frame"] if full else [])
            bar = st.progress(0.0, text="Starting")
            meta = EX.run(EX.parse_args(argv), progress=lambda n, t, msg: bar.progress(n / t, text=f"Downloading {msg}"))
            d = out / "apexscan_audit_data"
            prices = inapp.prices_from_file(d / "prices_long.csv.gz"); log = inapp.log_from_file(d / "download_log.csv")
            res, md = inapp.dq(prices, log, meta)
            st.session_state.update(audit_prices=prices, audit_log=log, audit_meta=meta, audit_dq=(res, md), audit_zip=inapp.zip_dir(d))
            bar.progress(1.0, text="Done")
            st.success(f"Finished: {meta['tickers_with_data']} of {meta['tickers_requested']} tickers returned data. "
                       "Open the Data quality tab and review it before anything else.")
        except Exception as ex:
            st.error(f"The export did not finish: {ex}")
    if "audit_zip" in st.session_state:
        st.download_button("Download everything (zip)", st.session_state["audit_zip"], "apexscan_audit_data.zip", "application/zip", key="audit_dl_zip")
    st.markdown("Already have the files? Upload them on the **Data quality** tab instead.")

# ── Data quality ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Data-quality report")
    up1 = st.file_uploader("prices_long.csv.gz (optional, replaces the session download)", type=["gz", "csv"], key="audit_up_prices")
    up2 = st.file_uploader("download_log.csv (needed with an upload)", type=["csv"], key="audit_up_log")
    if up1 is not None and up2 is not None:
        try:
            prices = inapp.prices_from_file(up1); log = inapp.log_from_file(up2)
            st.session_state.update(audit_prices=prices, audit_log=log, audit_dq=inapp.dq(prices, log))
        except Exception as ex:
            st.error(f"Could not read the uploaded files: {ex}")
    if "audit_dq" not in st.session_state:
        st.info("No price data loaded yet. Use the Get prices tab, or upload the two files above.")
    else:
        res, md = st.session_state["audit_dq"]
        blockers = inapp.has_blocking_errors(res)
        if blockers:
            st.error("Blocking problem: " + "; ".join(blockers))
        flags = res["flags"]
        c = st.columns(3)
        c[0].metric("Tickers with data", len(res["per_ticker"]))
        c[1].metric("Error flags", int((flags.severity == "error").sum()) if len(flags) else 0)
        c[2].metric("Warnings", int((flags.severity == "warn").sum()) if len(flags) else 0)
        _md(md)
        st.download_button("Download the report (markdown)", md, "DATA_QUALITY_REPORT.md", "text/markdown", key="audit_dl_dq")
        if len(flags):
            with st.expander("All flags"):
                _show(flags)

# ── CHDN replay ──────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("CHDN: when did each ApexScan gate first pass? (Test A replay)")
    st.caption("Uses ApexScan's own functions on prices that were available on each date. The Apex Score itself is not replayed. "
               "This is a replay, not a record of what the live app showed.")
    if "audit_prices" not in st.session_state:
        st.info("Load price data first (Get prices or Data quality tab).")
    else:
        prices = st.session_state["audit_prices"]
        known = sorted(set(prices.ticker))
        t = st.selectbox("Ticker", known, index=known.index("CHDN") if "CHDN" in known else 0, key="audit_replay_ticker")
        start = st.text_input("Replay from (YYYY-MM-DD)", "2026-07-01", key="audit_replay_start")
        if st.button("Run the replay", key="audit_run_replay"):
            try:
                r, fp = inapp.replay_for(prices, t, start)
                st.session_state["audit_replay"] = (t, r, fp)
            except Exception as ex:
                st.error(f"Replay not possible: {ex}")
        if "audit_replay" in st.session_state:
            t, r, fp = st.session_state["audit_replay"]
            names = {"g1_above_200ma": "Close above 200-day average", "g2_perf3m_ok": "3-month return at least 0%",
                     "g3_rs_ok": "Relative strength at least -30", "g4_volume_ok": "Volume at least 100,000",
                     "passes_replayed_gates": "All four replayed gates together"}
            _show(pd.DataFrame([(names[k], v if v else "n/a") for k, v in fp.items()], columns=["Gate", "Passing continuously since"]))
            st.caption("The score gate (at least 20) is not replayed. A date shown as 'not passing on last date' means the gate is closed at the end of the data.")
            ev = r["evaluable"].fillna(False) if "evaluable" in r else pd.Series(dtype=bool)
            st.markdown(f"Days replayed: **{len(r)}**; days that could not be evaluated: **{int((~ev).sum())}** (reasons are in the table).")
            cols = [c for c in ["close", "ma200", "perf_3m", "rs_3m", "stage", "bench_us_used", "g1_above_200ma", "g2_perf3m_ok",
                                "g3_rs_ok", "g4_volume_ok", "passes_replayed_gates", "exclusion_reason"] if c in r.columns]
            _show(r[cols].tail(25).reset_index())
            st.download_button("Download the full daily replay (CSV)", r.reset_index().to_csv(index=False), f"{t}_replay.csv", "text/csv", key="audit_dl_replay")
