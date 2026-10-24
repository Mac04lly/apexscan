"""
ui/research_card.py — Research card panel for the Stock Deep Dive (Change 3, Stage 2)

Display only. All logic lives in modules/research_card.py (the card) and
modules/research_card_log.py (snapshots and their evaluation); this file renders them.

Why it keeps its own state: the Deep Dive result is drawn inside `if dive_btn and dive_ticker:`, and a
Streamlit button is true only on the run right after the click. Any widget inside that block (an input,
a Log button) triggers a rerun in which the whole result disappears. So the Deep Dive calls this panel
AFTER that block, passing the ticker box and button values; the panel remembers the last analysed ticker
itself and keeps working across reruns.

It never raises into the Deep Dive: any failure becomes a short warning and the rest of the tab is unaffected.
Descriptive only: no score, no rating, no recommendation wording.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:          # pragma: no cover
    yf = None

from modules import research_card as R
from modules import research_card_log as RL

log = logging.getLogger(__name__)
ACTIVE_KEY = "rc_active_ticker"
STATUS_LABEL = {"research_candidate": "RESEARCH CANDIDATE", "trigger_met": "TRIGGER CONDITIONS MET",
                "no_long_setup": "NO LONG SETUP DEFINED", "insufficient_data": "INSUFFICIENT DATA",
                "unvalidated_market": "UNVALIDATED MARKET"}


def _fetch_history_raw(ticker: str) -> pd.DataFrame:
    """Two years of daily bars (about 500), enough for the 200-day SMA and 40 weekly bars."""
    if yf is None:
        return pd.DataFrame()
    try:
        return yf.Ticker(ticker).history(period="2y", auto_adjust=True)
    except Exception as e:
        log.warning(f"Research card history fetch failed for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def _load_history(ticker: str) -> pd.DataFrame:
    return _fetch_history_raw(ticker)


def _market_of(ticker: str) -> str:
    """Yahoo marks non-US listings with a dot suffix (.LG for NGX, .L, .TO ...); US share classes use a hyphen."""
    t = ticker.upper()
    if t.endswith(".LG"):
        return "NG"
    return "OTHER" if "." in t else "US"


def _money(x) -> str:
    return "–" if x is None else f"${x:,.2f}"


def _pct(x, nd=1, sign=False) -> str:
    if x is None:
        return "–"
    return f"{x:+.{nd}f}%" if sign else f"{x:.{nd}f}%"


def render_research_card(ticker: str, analysed_now: bool) -> None:
    """Entry point called by the Deep Dive. `ticker` is the text box value, `analysed_now` the Analyse button."""
    try:
        if analysed_now and ticker:
            st.session_state[ACTIVE_KEY] = ticker
        t = st.session_state.get(ACTIVE_KEY)
        if not t:
            return
        _render(t)
    except Exception as e:                      # never take the Deep Dive down with the card
        log.exception("Research card failed")
        st.warning(f"The research card could not be built ({type(e).__name__}). "
                   "The rest of the Deep Dive is unaffected.")


def _render(t: str) -> None:
    market = _market_of(t)
    st.divider()
    st.markdown(f"### 🧾 Research card — {t}")
    st.caption("Built for the ticker you last analysed. " + R.DISCLAIMER)

    hist = pd.DataFrame()
    if market == "US":
        hist = _load_history(t)
        if hist is None or len(hist) == 0:
            st.info(f"No price history is available for {t}, so no card can be built.")
            return

    c1, c2 = st.columns(2)
    account = c1.number_input("Account size for the illustration ($)", min_value=100.0, value=10_000.0,
                              step=1000.0, key="rc_account")
    risk = c2.number_input("Risk per trade (% of account)", min_value=0.1, max_value=5.0, value=1.0,
                           step=0.25, key="rc_risk")
    observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    card = R.build_card(t, hist, market=market, observed_at=observed_at, account_size=account, risk_pct=risk)
    plan = card["plan"]
    st.info(f"**{STATUS_LABEL[plan['status']]}** — {plan['status_text']}")
    if plan["status"] in ("insufficient_data", "unvalidated_market"):
        for m in card["missing_data"]:
            st.caption(m)
        return

    d = card["data"]
    st.caption(f"Observation: {card['observation_date']} · close {_money(card['price']['close'])} · "
               f"{d['bars']} complete daily bars" + (" · " + "; ".join(d["notes"]) if d["notes"] else ""))
    st.markdown(f"**Description:** {card['setup']['text']}")
    st.caption(card["setup"]["note"])

    _render_plan(card, account, risk)
    _render_evidence(card)
    _render_structure(card)
    _render_context(card)
    with st.expander("Missing data and limits"):
        for m in card["missing_data"]:
            st.markdown(f"- {m}")
        for m in card["confidence_limits"]:
            st.markdown(f"- {m}")
    _render_log(t, card)


def _render_plan(card: dict, account: float, risk: float) -> None:
    p = card["plan"]
    st.markdown("#### Plan to evaluate")
    st.markdown("**Entry condition:** " + (p["entry_condition"] or "none defined for this description"))
    inv = p.get("invalidation")
    if inv:
        cmp_ = inv["ma200_distance_pct_for_comparison"]
        st.markdown(
            f"**Invalidation:** {_money(inv['stop'])}, just below the {inv['swing_date']} swing low "
            f"{_money(inv['swing_low'])} (confirmed {inv['swing_confirmed']}). Risk from the reference entry "
            f"{_money(p['entry_ref'])}: {_pct(inv['risk_pct'])} ({_money(inv['risk_per_share'])} per share)."
            + (f" For comparison, the 200-day SMA is {_pct(cmp_)} from the close." if cmp_ is not None else ""))
        if card["risk_context"]["wide_stop"]:
            st.caption(f"This structural stop is wider than {R.WIDE_STOP_PCT:.0f}%: size accordingly.")
    elif p["entry_condition"]:
        st.markdown("**Invalidation:** " + p.get("invalidation_reason", "not available"))
    if p["target"]:
        t = p["target"]
        what = (f"a single swing high at {_money(t['zone_low'])}" if t["zone_low"] == t["zone_high"] else
                f"the low of a {t['touches']}-touch resistance zone ({_money(t['zone_low'])}–{_money(t['zone_high'])})")
        st.markdown(f"**Target:** {_money(t['price'])}, {what}. "
                    + (f"**Reward-to-risk:** {p['reward_risk']:.2f}" if p["reward_risk"] is not None else ""))
    elif p["entry_condition"]:
        st.markdown("**Target:** " + (p["target_reason"] or "none"))
    s = p.get("sizing")
    if s:
        st.markdown(
            f"**Sizing illustration** (account {_money(account)}, risking {risk:g}%): "
            f"**{s['shares']:,} shares**, position {_money(s['position_value'])}, actual risk "
            f"{_money(s['actual_risk_dollars'])}"
            + (", capped by the account size" if s["capped_by_account"] else "")
            + (f"; {s['pct_of_adv20_shares']:.3g}% of 20-day average volume" if s["pct_of_adv20_shares"] is not None else "")
            + ".")
        st.caption(s["note"])
    elif p["entry_condition"]:
        st.caption("No sizing: it needs both an entry condition and a structural stop.")


def _render_evidence(card: dict) -> None:
    st.markdown("#### Evidence")
    a, b = st.columns(2)
    with a:
        st.markdown("**Supporting**")
        for e in card["evidence_for"] or ["Nothing in the data supports this description."]:
            st.markdown(f"- {e}")
    with b:
        st.markdown("**Contradicting**")
        for e in card["evidence_against"] or ["Nothing in the data contradicts it."]:
            st.markdown(f"- {e}")


def _render_structure(card: dict) -> None:
    s, lv = card["structure"], card["levels"]
    with st.expander("Structure, swings and levels"):
        st.markdown(f"**Structure:** {s['structure']}" + (f" — {s['break_of_structure']}" if s["break_of_structure"] else ""))
        st.caption(s["method"])
        if s["swings"]:
            st.dataframe(pd.DataFrame([{"Type": x["type"], "Swing date": x["date"], "Price": x["price"],
                                        "Confirmed": x["confirmed_date"]} for x in s["swings"]]),
                         use_container_width=True, hide_index=True)
        for title, zs in (("Resistance zones above price", lv["resistances"]), ("Support zones below price", lv["supports"])):
            st.markdown(f"**{title}**")
            if zs:
                st.dataframe(pd.DataFrame([{"Zone": f"{z['low']:.2f}–{z['high']:.2f}", "Touches": z["touches"],
                                            "Kind": z["kind"], "Last swing": z["last_date"]} for z in zs]),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("None in the available history.")
        if lv["recent_breakout"]:
            b = lv["recent_breakout"]
            st.markdown(f"**Recent breakout:** closed above {b['zone_high']:.2f} on {b['date']} ({b['bars_ago']} bar(s) ago); "
                        f"relative volume {b['rvol'] if b['rvol'] is not None else 'unavailable'}x "
                        f"({'confirmed' if b['volume_confirmed'] else 'not confirmed'}).")
        st.caption(lv["note"])


def _render_context(card: dict) -> None:
    tr, v, m, rc = card["trend"], card["volume"], card["momentum"], card["risk_context"]
    with st.expander("Trend, volume, momentum and risk context"):
        rows = []
        for k in ("ema20", "ema50", "sma200"):
            r = tr[k]
            rows.append({"Average": r["label"], "Value": r["value"], "Close vs it": _pct(r["distance_pct"], 1, True),
                         "20-bar slope": _pct(r["slope_20bar_pct"], 1, True)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        wk = tr["weekly"]
        st.markdown(f"**Alignment:** {tr['alignment'] or 'unavailable'} · **Weekly:** "
                    + (f"close {wk['close']} vs 10-week {wk['sma10w']} / 40-week {wk['sma40w']}" if wk["available"]
                       else f"unavailable ({wk['reason']})"))
        st.caption(tr["note"])
        st.markdown(f"**Volume:** last session {v['rvol_last'] if v['rvol_last'] is not None else '–'}x its 50-session "
                    f"average · up/down-day volume {v['up_down_volume_ratio_20'] if v['up_down_volume_ratio_20'] is not None else '–'}x · "
                    f"20-day average {v['avg_volume_20']:,.0f} shares" if v["avg_volume_20"] else "**Volume:** unavailable")
        st.caption(v["basis"])
        st.markdown(f"**RSI(14):** {m['rsi14'] if m['rsi14'] is not None else '–'} · {m['context']}")
        st.markdown(f"Divergence — bearish: {m['divergence']['bearish']}; bullish: {m['divergence']['bullish']}")
        st.caption(m["note"])
        e = rc["earnings"]
        st.markdown("**Earnings:** " + (f"{e['date']} ({e['days_until']} day(s))" if e["date"] else "date unavailable")
                    + f" · **Spread:** {rc['liquidity']['spread']}")


def _render_log(t: str, card: dict) -> None:
    st.markdown("#### 📌 Snapshot log")
    st.caption(RL.SELECTION_NOTE + " One card per ticker per day. Outcomes are measured from the logged close "
               "by the existing outcome engine.")
    if st.button("Log this card as a research snapshot", key=f"rc_log_{t}"):
        res = RL.append_and_save(card, logged_at=card["observed_at"])
        msg = {"added": ("success", f"Logged. The log now holds {res.get('n')} card(s)."),
               "duplicate": ("info", f"A card for {t} dated {card['observation_date']} is already logged; nothing changed."),
               "no_credentials": ("warning", "Storage credentials are not configured, so nothing was saved."),
               "read_failed": ("warning", "The card log could not be read, so nothing was written (the existing log is untouched)."),
               "save_failed": ("warning", "The card could not be saved; nothing was changed."),
               "not_loggable": ("info", "This card carries no setup description and is not logged.")}[res["status"]]
        getattr(st, msg[0])(msg[1])

    c1, c2 = st.columns(2)
    if c1.button("📊 Load card log summary", key="rc_summary_btn"):
        state, recs = RL.load_log_state()
        st.session_state["rc_summary"] = (state, RL.summarize_log(recs) if state in ("ok", "missing") else None)
    if c2.button("🔄 Update outcomes for logged cards", key="rc_outcomes_btn"):
        with st.spinner("Resolving due horizons with the existing outcome engine…"):
            res = RL.resolve_and_save()
        text = {"updated": f"Updated {res['updated']} card(s).", "nothing_due": "No horizon is due yet.",
                "empty": "The card log is empty.", "no_credentials": "Storage credentials are not configured.",
                "read_failed": "The card log could not be read; nothing was written.",
                "save_failed": "Outcomes were computed but could not be saved."}[res["status"]]
        st.info(text)
    if st.session_state.get("rc_summary"):
        _render_summary(*st.session_state["rc_summary"])


def _render_summary(state: str, s: dict) -> None:
    if s is None:
        st.warning({"no_credentials": "Storage credentials are not configured.",
                    "error": "The card log could not be read."}.get(state, "The card log is unavailable."))
        return
    st.markdown(f"**{s['n_cards']} card(s) logged.** Resolved: " + ", ".join(f"{k}: {v}" for k, v in s["resolved"].items()))
    if s["by_setup"]:
        st.dataframe(pd.DataFrame([{"Description": k, "Cards": v} for k, v in sorted(s["by_setup"].items(), key=lambda x: -x[1])]),
                     use_container_width=True, hide_index=True)
    ind = s["independence_20D"]
    st.caption(f"Independence @ 20D: {ind['n_rows']} resolved card(s) span {ind['n_windows']} independent window(s) → {ind['label']}. {ind['note']}")
    st.markdown("**Pre-registered hypotheses** (set before any card was logged)")
    st.dataframe(pd.DataFrame([{
        "ID": h["id"], "Hypothesis": h["text"], "Resolved": h["n"], "Windows": h["windows"],
        "Mean excess vs SPY": _pct(h["mean_excess_%"], 2, True),
        "95% CI (cluster-robust)": (f"[{h['ci'][0]:+.1f}%, {h['ci'][1]:+.1f}%]" if h["ci"] else "withheld"),
        "Verdict": h["verdict"]} for h in s["hypotheses"]]), use_container_width=True, hide_index=True)
    st.caption(s["control_note"] + " " + s["selection_note"] + " No performance claim is made before 12 independent windows.")
