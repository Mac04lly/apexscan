"""
tests/test_research_card_history_retry_change5.py — Change 5: a failed price fetch must not be remembered

Before: `_load_history` was wrapped in st.cache_data(ttl=600) and returned an EMPTY frame when Yahoo failed or
returned nothing. Streamlit caches return values, so the empty frame was stored for ten minutes: the card said
"No price history is available" and pressing Analyse again did not even call Yahoo.
Now: an empty or failed fetch raises inside the cached function (exceptions are never cached) and the wrapper turns
it into an empty frame; a fetch that returned bars is still cached for ten minutes.

Display/loader only: no card, label, level or log logic is touched. These tests drive the REAL panel through
AppTest with the fetch injected, and count how many times "Yahoo" is called.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import test_research_card_change3 as T
import test_research_card_ui_change3s2 as TU
from ui import research_card as U


@pytest.fixture(autouse=True)
def _fresh_cache():
    """st.cache_data survives across AppTest instances; the public name must still clear it."""
    U._load_history.clear()
    yield
    U._load_history.clear()


class Feed:
    """A fake price service: scripted replies per call, records every call."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, ticker):
        self.calls.append(ticker)
        r = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return r() if callable(r) else r


GOOD = lambda: TU._hist()
EMPTY = pd.DataFrame()


def _panel(monkeypatch, feed):
    monkeypatch.setattr(U, "_fetch_history_raw", feed)
    return TU.AppTest.from_function(TU._script, default_timeout=60)


def _joined(at):
    return " ".join(TU._texts(at))


# ── the reported bug ──────────────────────────────────────────────────────
def test_a_failed_fetch_is_retried_on_the_next_analyse_and_the_card_appears(monkeypatch):
    feed = Feed(EMPTY, GOOD)
    at = TU._analyse(_panel(monkeypatch, feed), "DOCU")
    assert "No price history is available for DOCU" in _joined(at) and len(feed.calls) == 1
    at.button(key="dive_btn").click().run()
    TU._no_exc(at)
    assert len(feed.calls) == 2, "pressing Analyse again must reach the price service again"
    j = _joined(at)
    assert "No price history" not in j and "Plan to evaluate" in j and "Research card" in j


def test_a_none_reply_is_treated_like_an_empty_one_and_also_not_remembered(monkeypatch):
    feed = Feed(None, GOOD)
    at = TU._analyse(_panel(monkeypatch, feed), "DOCU")
    assert "No price history is available for DOCU" in _joined(at)
    at.button(key="dive_btn").click().run()
    assert len(feed.calls) == 2 and "Plan to evaluate" in _joined(at)


def test_an_always_failing_ticker_keeps_trying_and_keeps_explaining(monkeypatch):
    feed = Feed(EMPTY)
    at = TU._analyse(_panel(monkeypatch, feed), "NOPE")
    for _ in range(2):
        at.button(key="dive_btn").click().run()
        TU._no_exc(at)
    assert len(feed.calls) == 3
    assert at.markdown[-1].value == "AFTER-PANEL"                      # the rest of the Deep Dive is unaffected


def test_the_message_says_what_happened_and_what_to_do(monkeypatch):
    at = TU._analyse(_panel(monkeypatch, Feed(EMPTY)), "NOPE")
    msg = next(t for t in TU._texts(at) if t.startswith("No price history is available for NOPE"))
    assert "no card can be built" in msg and "temporarily unavailable" in msg and "may not exist" in msg
    assert "Nothing was saved" in msg and "Analyse again" in msg
    assert not T.BANNED.search(msg) and "$" not in msg


# ── what must still work ──────────────────────────────────────────────────
def test_a_successful_fetch_is_still_cached_for_the_ttl(monkeypatch):
    feed = Feed(GOOD)
    at = TU._analyse(_panel(monkeypatch, feed), "AAA")
    for _ in range(3):
        at.button(key="dive_btn").click().run()
    TU._no_exc(at)
    assert len(feed.calls) == 1, "good data must not be re-fetched on every rerun"
    assert "Plan to evaluate" in _joined(at)


def test_the_public_clear_really_drops_a_cached_good_result(monkeypatch):
    feed = Feed(GOOD)
    monkeypatch.setattr(U, "_fetch_history_raw", feed)
    assert len(U._load_history("AAA")) > 100 and len(U._load_history("AAA")) > 100 and len(feed.calls) == 1
    U._load_history.clear()
    U._load_history("AAA")
    assert len(feed.calls) == 2


def test_one_tickers_failure_does_not_affect_another(monkeypatch):
    by = {"BAD": EMPTY, "OK": GOOD}
    feed = lambda tk: (by[tk]() if callable(by[tk]) else by[tk])
    monkeypatch.setattr(U, "_fetch_history_raw", feed)
    assert len(U._load_history("BAD")) == 0
    assert len(U._load_history("OK")) > 100 and len(U._load_history("BAD")) == 0


def test_the_loader_returns_a_dataframe_never_raises_and_never_the_internal_exception(monkeypatch):
    for reply in (EMPTY, None, pd.DataFrame({"Close": []})):
        monkeypatch.setattr(U, "_fetch_history_raw", Feed(reply))
        out = U._load_history("ZZZ")
        assert isinstance(out, pd.DataFrame) and len(out) == 0


def test_a_yahoo_exception_is_contained_not_cached_and_recovers(monkeypatch):
    class Boom:
        def Ticker(self, t):
            raise RuntimeError("yahoo down")

    class Fine:
        def Ticker(self, t):
            return type("T", (), {"history": staticmethod(lambda **k: GOOD())})()

    monkeypatch.setattr(U, "yf", Boom())
    assert len(U._load_history("DOCU")) == 0
    monkeypatch.setattr(U, "yf", Fine())
    assert len(U._load_history("DOCU")) > 100, "the earlier exception must not have been remembered"


def test_non_us_tickers_never_reach_the_price_service(monkeypatch):
    feed = Feed(GOOD)
    at = TU._analyse(_panel(monkeypatch, feed), "ZENITHBANK.LG")
    TU._no_exc(at)
    assert feed.calls == [] and "UNVALIDATED MARKET" in _joined(at)


def test_the_panel_never_calls_the_raw_cached_function_directly():
    """The raising function is an implementation detail: only the wrapper may be used by the panel."""
    import ast
    tree = ast.parse((REPO / "ui" / "research_card.py").read_text(encoding="utf-8"))
    render = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_render")
    names = {n.id for n in ast.walk(render) if isinstance(n, ast.Name)}
    assert "_load_history" in names and "_load_history_cached" not in names
    loader = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_load_history")
    assert any(isinstance(n, ast.Name) and n.id == "_load_history_cached" for n in ast.walk(loader))
    assert any(isinstance(n, ast.ExceptHandler) and "_NoHistory" in ast.dump(n) for n in ast.walk(loader))
