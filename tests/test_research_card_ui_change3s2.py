"""
tests/test_research_card_ui_change3s2.py — Change 3, Stage 2 (panel and Deep Dive hook)

Drives the panel through Streamlit's AppTest with history, storage and the log injected (no network),
and drives the REAL dashboard through a click on Analyse. Checks: hidden until analysed, persistence across the
reruns that make the Deep Dive result vanish, sizing inputs, non-US and thin-data handling, error containment,
every log/summary/outcome message, no write without a click, the wording policy on everything rendered, and
that dashboard.py differs from baseline by exactly the two hook lines with its CRLF endings intact.
"""
import ast
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import test_research_card_change3 as T
import test_research_card_log_change3s2 as TL
from streamlit.testing.v1 import AppTest

from modules import research_card as R
from modules import research_card_log as RL
from ui import research_card as U

END = "2025-06-13"                                           # a Friday well in the past: every bar is a complete session


def _hist(points=None, lead=260):
    return T.make_hist(T.zigzag(points or (T.UP + [124.5]), lead=lead), end=END)


@pytest.fixture(autouse=True)
def _fresh_cache():
    U._load_history.clear()
    yield
    U._load_history.clear()


def _script():
    import streamlit as st
    from ui.research_card import render_research_card
    t = st.text_input("Enter any ticker", key="dive_input").upper().strip()
    b = st.button("Analyse", key="dive_btn")
    render_research_card(t, b)
    st.write("AFTER-PANEL")


def _app(monkeypatch, hist=None, history_by_ticker=None):
    monkeypatch.setattr(U, "_fetch_history_raw",
                        (lambda tk: history_by_ticker.get(tk, pd.DataFrame())) if history_by_ticker is not None
                        else (lambda tk: hist if hist is not None else _hist()))
    return AppTest.from_function(_script, default_timeout=60)


def _analyse(at, ticker="AAA"):
    at.run()
    at.text_input[0].set_value(ticker).run()
    at.button(key="dive_btn").click().run()
    return at


def _texts(at):
    out = []
    for coll in (at.markdown, at.caption, at.info, at.warning, at.success, at.error):
        out += [e.value for e in coll]
    for d in at.dataframe:
        out.append(d.value.to_string())
    return out


def _no_exc(at):
    assert not at.exception, [e.value for e in at.exception]


# ── market routing ────────────────────────────────────────────────────────
def test_market_routing():
    assert U._market_of("AAPL") == "US" and U._market_of("BRK-B") == "US" and U._market_of("brk-b") == "US"
    assert U._market_of("ZENITHBANK.LG") == "NG" and U._market_of("zenithbank.lg") == "NG"
    assert U._market_of("SHEL.L") == "OTHER" and U._market_of("SHOP.TO") == "OTHER"


# ── visibility and persistence ────────────────────────────────────────────
def test_nothing_renders_until_analyse_is_clicked(monkeypatch):
    at = _app(monkeypatch)
    at.run(); _no_exc(at)
    assert not any("Research card" in t for t in _texts(at))
    at.text_input[0].set_value("AAA").run(); _no_exc(at)
    assert not any("Research card" in t for t in _texts(at)), "typing a ticker is not an Analyse click"
    assert at.markdown[-1].value == "AFTER-PANEL"


def test_card_renders_all_sections_after_analyse(monkeypatch):
    at = _analyse(_app(monkeypatch)); _no_exc(at)
    t = " ".join(_texts(at))
    assert "Research card — AAA" in t and "RESEARCH CANDIDATE" in t and "Descriptive research only" in t
    for needle in ("Entry condition:", "Invalidation:", "Target:", "Sizing illustration", "Supporting", "Contradicting",
                   "Description:", "Snapshot log"):
        assert needle in t, needle
    assert "Breakout candidate" in t and "AFTER-PANEL" in t
    keys = {b.key for b in at.button}
    assert {"rc_log_AAA", "rc_summary_btn", "rc_outcomes_btn"} <= keys
    assert {n.key for n in at.number_input} == {"rc_account", "rc_risk"}


def test_card_survives_the_rerun_that_would_clear_the_deep_dive_result(monkeypatch):
    at = _analyse(_app(monkeypatch))
    assert at.session_state[U.ACTIVE_KEY] == "AAA"
    at.number_input(key="rc_risk").set_value(0.5).run(); _no_exc(at)          # a rerun in which Analyse is False
    assert any("Research card — AAA" in t for t in _texts(at))
    at.number_input(key="rc_account").set_value(50_000).run(); _no_exc(at)
    assert any("Research card — AAA" in t for t in _texts(at))


def test_a_new_analyse_replaces_the_card(monkeypatch):
    other = _hist(T.UP + [121])
    at = _app(monkeypatch, history_by_ticker={"AAA": _hist(), "BBB": other})
    _analyse(at, "AAA")
    assert any("Breakout candidate" in t for t in _texts(at))
    at.text_input[0].set_value("BBB").run()
    at.button(key="dive_btn").click().run(); _no_exc(at)
    t = " ".join(_texts(at))
    assert "Research card — BBB" in t and "Research card — AAA" not in t and "Pullback within an uptrend" in t


# ── sizing inputs ─────────────────────────────────────────────────────────
def _shares_in(at):
    m = re.search(r"\*\*([\d,]+) shares\*\*", " ".join(e.value for e in at.markdown))
    return int(m.group(1).replace(",", "")) if m else None


def test_sizing_responds_to_account_and_risk_inputs(monkeypatch):
    h = _hist()
    at = _analyse(_app(monkeypatch, h))
    base = R.build_card("AAA", h, now=T.NOW)["plan"]["sizing"]["shares"]
    assert _shares_in(at) == base
    at.number_input(key="rc_account").set_value(100_000).run()
    at.number_input(key="rc_risk").set_value(2.0).run(); _no_exc(at)
    expect = R.build_card("AAA", h, now=T.NOW, account_size=100_000, risk_pct=2.0)["plan"]["sizing"]["shares"]
    assert _shares_in(at) == expect and expect > base


# ── non-US, thin data, empty data ─────────────────────────────────────────
@pytest.mark.parametrize("ticker,market_word", [("ZENITHBANK.LG", "UNVALIDATED MARKET"), ("SHEL.L", "UNVALIDATED MARKET")])
def test_non_us_tickers_get_no_card_and_no_log_button(monkeypatch, ticker, market_word):
    at = _analyse(_app(monkeypatch), ticker); _no_exc(at)
    t = " ".join(_texts(at))
    assert market_word in t and "do not transfer" in t
    assert not any(b.key and b.key.startswith("rc_log_") for b in at.button)
    assert "Entry condition" not in t and "Sizing illustration" not in t


def test_thin_history_says_insufficient_and_offers_no_log(monkeypatch):
    at = _analyse(_app(monkeypatch, T.make_hist(np.linspace(100, 120, 40), end=END))); _no_exc(at)
    t = " ".join(_texts(at))
    assert "INSUFFICIENT DATA" in t and "Entry condition" not in t
    assert not any(b.key and b.key.startswith("rc_log_") for b in at.button)


def test_empty_history_is_reported_plainly(monkeypatch):
    at = _analyse(_app(monkeypatch, history_by_ticker={}), "NOPE"); _no_exc(at)
    assert any("No price history is available for NOPE" in t for t in _texts(at))
    assert at.markdown[-1].value == "AFTER-PANEL"


# ── error containment ─────────────────────────────────────────────────────
def test_a_failure_inside_the_card_never_reaches_the_rest_of_the_page(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(R, "build_card", boom)
    at = _analyse(_app(monkeypatch)); _no_exc(at)
    assert any("could not be built (RuntimeError)" in w.value for w in at.warning)
    assert any("rest of the Deep Dive is unaffected" in w.value for w in at.warning)
    assert at.markdown[-1].value == "AFTER-PANEL"


# ── log, summary and outcome buttons ──────────────────────────────────────
@pytest.mark.parametrize("status,kind,needle", [
    ("added", "success", "Logged. The log now holds 7"),
    ("duplicate", "info", "already logged"),
    ("no_credentials", "warning", "credentials are not configured"),
    ("read_failed", "warning", "existing log is untouched"),
    ("save_failed", "warning", "nothing was changed"),
    ("not_loggable", "info", "not logged"),
])
def test_log_button_reports_every_outcome_honestly(monkeypatch, status, kind, needle):
    calls = []
    monkeypatch.setattr(RL, "append_and_save", lambda card, logged_at=None: (calls.append(card["ticker"]), {"status": status, "n": 7})[1])
    at = _analyse(_app(monkeypatch))
    assert calls == [], "rendering the card must never write"
    at.button(key="rc_log_AAA").click().run(); _no_exc(at)
    assert calls == ["AAA"]
    shown = [e.value for e in getattr(at, kind)]
    assert any(needle in s for s in shown), (kind, shown)


def test_summary_button_loads_on_demand_and_renders_the_pre_registered_table(monkeypatch):
    loads = []
    recs = TL._rows("trend_pullback", 12, 4, 4.0, 2.0) + TL._rows("breakout_candidate", 3, 4, 1.0, 1.0)
    for r in recs:
        r["status"] = "research_candidate"
    monkeypatch.setattr(RL, "load_log_state", lambda: (loads.append(1), ("ok", recs))[1])
    at = _analyse(_app(monkeypatch))
    assert loads == [], "the log is only read when asked"
    at.button(key="rc_summary_btn").click().run(); _no_exc(at)
    assert loads == [1]
    t = " ".join(_texts(at))
    assert "card(s) logged" in t and "Pre-registered hypotheses" in t
    assert "Outperformed SPY (adequate, 12 windows)" in t and "Insufficient independent windows (3)" in t
    assert "withheld" in t and "not a random sample" in t and "No performance claim is made before 12 independent windows" in t
    at.number_input(key="rc_risk").set_value(1.5).run()                         # summary persists across reruns
    assert any("Pre-registered hypotheses" in x for x in _texts(at)) and loads == [1]


@pytest.mark.parametrize("state,needle", [("no_credentials", "credentials are not configured"), ("error", "could not be read")])
def test_summary_failure_states(monkeypatch, state, needle):
    monkeypatch.setattr(RL, "load_log_state", lambda: (state, []))
    at = _analyse(_app(monkeypatch))
    at.button(key="rc_summary_btn").click().run(); _no_exc(at)
    assert any(needle in w.value for w in at.warning)


def test_summary_of_a_missing_log_says_zero_cards(monkeypatch):
    monkeypatch.setattr(RL, "load_log_state", lambda: ("missing", []))
    at = _analyse(_app(monkeypatch))
    at.button(key="rc_summary_btn").click().run(); _no_exc(at)
    assert any("0 card(s) logged" in t for t in _texts(at))


@pytest.mark.parametrize("res,needle", [
    ({"status": "updated", "updated": 3}, "Updated 3 card(s)"), ({"status": "nothing_due", "updated": 0}, "No horizon is due yet"),
    ({"status": "empty", "updated": 0}, "log is empty"), ({"status": "no_credentials", "updated": 0}, "credentials are not configured"),
    ({"status": "read_failed", "updated": 0}, "nothing was written"), ({"status": "save_failed", "updated": 0}, "could not be saved")])
def test_outcome_button_messages(monkeypatch, res, needle):
    called = []
    monkeypatch.setattr(RL, "resolve_and_save", lambda *a, **k: (called.append(1), res)[1])
    at = _analyse(_app(monkeypatch))
    assert called == []
    at.button(key="rc_outcomes_btn").click().run(); _no_exc(at)
    assert called == [1] and any(needle in i.value for i in at.info)


# ── wording policy on everything rendered ─────────────────────────────────
def test_nothing_rendered_uses_recommendation_wording_or_a_score(monkeypatch):
    seen = []
    for hist_pts, tk in ((T.UP + [124.5], "AAA"), (T.UP + [121], "BBB"), ([140, 120, 132, 112, 124, 104, 112], "CCC")):
        monkeypatch.setattr(RL, "load_log_state", lambda: ("ok", TL._rows("trend_pullback", 12, 4, 4.0, 2.0)))
        at = _analyse(_app(monkeypatch, _hist(hist_pts)), tk)
        at.button(key="rc_summary_btn").click().run(); _no_exc(at)
        seen += _texts(at)
    assert len(seen) > 40
    for s in seen:
        assert not T.BANNED.search(s), s[:160]
    assert not re.search(r"\b(score|rating|probability)\b", " ".join(seen), re.I), "no score, rating or probability is shown"


# ── Deep Dive hook: exact, structural, and live ───────────────────────────
def test_dashboard_hook_is_exactly_two_lines_and_the_file_was_not_converted():
    raw = (REPO / "dashboard.py").read_bytes()
    # A whole-file CRLF->LF conversion (what a careless editor does) leaves thousands of bare LFs. The two inserted
    # hook lines may legitimately be LF if added in a web editor, so at most two are tolerated.
    bare_lf = raw.count(b"\n") - raw.count(b"\r\n")
    assert bare_lf <= 2, f"dashboard.py has {bare_lf} bare-LF lines: its line endings were converted"
    assert len(re.findall(rb"    from ui\.research_card import render_research_card\r?\n", raw)) == 1
    assert len(re.findall(rb"    render_research_card\(dive_ticker, dive_btn\)\r?\n", raw)) == 1
    assert raw.count(b"research_card") == 3


def test_dashboard_hook_sits_after_the_deep_dive_if_block_inside_its_tab():
    tree = ast.parse((REPO / "dashboard.py").read_text(encoding="utf-8"))
    tab7 = next(n for n in tree.body if isinstance(n, ast.With)
                and "tabs[6]" in ast.unparse(n.items[0].context_expr))
    body = tab7.body
    imp, call = body[-2], body[-1]
    assert isinstance(imp, ast.ImportFrom) and imp.module == "ui.research_card"
    assert isinstance(call, ast.Expr) and ast.unparse(call.value) == "render_research_card(dive_ticker, dive_btn)"
    deep_if = body[-3]
    assert isinstance(deep_if, ast.If) and "dive_btn" in ast.unparse(deep_if.test)
    # both names the hook passes are assigned (inside the nested column blocks) BEFORE the hook line
    assigned = {t.id for n in ast.walk(tab7) if isinstance(n, ast.Assign) and n.lineno < imp.lineno
                for t in n.targets if isinstance(t, ast.Name)}
    assert {"dive_ticker", "dive_btn"} <= assigned


def test_dashboard_differs_from_baseline_by_exactly_two_added_lines():
    r = subprocess.run(["git", "diff", "--numstat", "baseline-pre-change3s2", "--", "dashboard.py"],
                       cwd=REPO, capture_output=True, text=True)
    tag = subprocess.run(["git", "rev-parse", "--verify", "-q", "baseline-pre-change3s2"], cwd=REPO, capture_output=True)
    if tag.returncode != 0 or r.returncode != 0:
        pytest.skip("baseline tag not available in this checkout")
    committed = subprocess.run(["git", "diff", "--numstat", "baseline-pre-change3s2", "HEAD", "--", "dashboard.py"],
                               cwd=REPO, capture_output=True, text=True).stdout.split()
    working = r.stdout.split()
    stat = working or committed
    assert stat[:2] == ["2", "0"], stat


def test_the_real_deep_dive_shows_the_card_after_a_click_on_analyse(monkeypatch):
    import scanner
    monkeypatch.setattr(U, "_fetch_history_raw", lambda tk: _hist())
    monkeypatch.setattr(scanner, "analyze_stock", lambda *a, **k: None)          # the Deep Dive's own analysis: offline
    at = AppTest.from_file(str(REPO / "dashboard.py"), default_timeout=120)
    at.run(timeout=120); _no_exc(at)
    assert not any("Research card" in e.value for e in at.markdown), "no card before an Analyse click"
    box = next(t for t in at.text_input if t.key == "dive_input")
    box.set_value("AAA").run(timeout=120)
    next(b for b in at.button if b.label == "🔍 Analyse").click().run(timeout=120); _no_exc(at)
    assert any("Research card — AAA" in e.value for e in at.markdown)
    assert any("RESEARCH CANDIDATE" in i.value for i in at.info)
    at.number_input(key="rc_risk").set_value(0.5).run(timeout=120); _no_exc(at)
    assert any("Research card — AAA" in e.value for e in at.markdown), "the card persists although Analyse is now False"
