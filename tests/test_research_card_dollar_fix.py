"""
tests/test_research_card_dollar_fix.py — display bug: bare `$` in Streamlit markdown

Streamlit renders text between two unescaped `$` as a LaTeX formula, so a line holding a stop price and a swing-low
price (two dollar amounts) printed as garbled math in the live app. The earlier UI tests compared strings but never
asked how markdown would render them. These tests scan EVERYTHING the panel sends to markdown for an unescaped `$`.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import test_research_card_change3 as T
import test_research_card_ui_change3s2 as TU
from modules import research_card as R
from modules import research_card_log as RL
from ui import research_card as U

UNESCAPED = re.compile(r"(?<!\\)\$")


def _md(at):
    return [e.value for coll in (at.markdown, at.caption, at.info, at.warning, at.success, at.error) for e in coll]


def test_money_is_escaped_and_never_forms_a_formula_pair():
    assert U._money(63.233) == "\\$63.23" and U._money(None) == "–" and U._money(1234.5) == "\\$1,234.50"
    line = f"Stop {U._money(63.23)} just below the swing low {U._money(63.92)}; risk {U._money(18.81)} per share"
    assert not UNESCAPED.search(line) and line.count("\\$") == 3


def test_tiny_volume_percentages_are_not_shown_as_zero():
    assert U._pct_of_volume(0.0002) == "<0.001%" and U._pct_of_volume(0.001) == "0.001%" and U._pct_of_volume(1.2345) == "1.23%"


@pytest.mark.parametrize("points,ticker", [(T.UP + [124.5], "AAA"), (T.UP + [121], "BBB"),
                                           ([140, 120, 132, 112, 124, 104, 112], "CCC")])
def test_nothing_sent_to_markdown_contains_a_bare_dollar_sign(monkeypatch, points, ticker):
    monkeypatch.setattr(RL, "load_log_state", lambda: ("ok", []))
    at = TU._analyse(TU._app(monkeypatch, TU._hist(points)), ticker)
    at.button(key="rc_summary_btn").click().run()
    TU._no_exc(at)
    strings = _md(at)
    assert len(strings) > 15
    bad = [s for s in strings if UNESCAPED.search(s)]
    assert not bad, bad[:2]


def test_the_breakout_card_prints_its_prices_with_escaped_dollars(monkeypatch):
    at = TU._analyse(TU._app(monkeypatch, TU._hist(T.UP + [124.5])))
    inval = next(s for s in _md(at) if s.startswith("**Invalidation:**"))
    sizing = next(s for s in _md(at) if s.startswith("**Sizing illustration**"))
    assert inval.count("\\$") >= 3 and "just below the" in inval and not UNESCAPED.search(inval)
    assert sizing.count("\\$") >= 3 and "shares**" in sizing and not UNESCAPED.search(sizing)
    target = next(s for s in _md(at) if s.startswith("**Target:**"))
    assert target.count("\\$") >= 1 and not UNESCAPED.search(target)


def test_the_volume_percentage_in_the_sizing_line_is_never_a_bare_zero(monkeypatch):
    h = TU._hist(T.UP + [124.5])
    pct = R.build_card("AAA", h, now=T.NOW)["plan"]["sizing"]["pct_of_adv20_shares"]
    at = TU._analyse(TU._app(monkeypatch, h))
    sizing = next(s for s in _md(at) if s.startswith("**Sizing illustration**"))
    expected = "<0.001%" if pct < 0.001 else f"{pct:.3g}%"
    assert f"; {expected} of 20-day average volume" in sizing and "; 0% of" not in sizing


def test_no_string_in_the_panel_source_carries_a_bare_dollar_sign():
    tree = ast.parse((REPO / "ui" / "research_card.py").read_text(encoding="utf-8"))
    allowed_label = "Account size for the illustration ($)"       # a widget label with a single $, not markdown text
    offenders = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and "$" in n.value:
            if n.value == allowed_label or n.value.startswith("\\$") or "`$`" in n.value:
                continue
            offenders.append(n.value[:60])
    assert not offenders, offenders
