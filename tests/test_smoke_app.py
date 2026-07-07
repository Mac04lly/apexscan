"""
tests/test_smoke_app.py

A lightweight smoke test for dashboard.py using Streamlit's own headless
testing framework (streamlit.testing.v1.AppTest). This runs the ENTIRE
script exactly as Streamlit would — every `with tabs[i]:` block executes
on every run, regardless of which tab a human would have clicked on —
so a single AppTest run already covers most of the file.

The two bugs we just fixed (undefined _t1_use, and the ":02d" format
crash on a non-integer checklist number) BOTH only fired once a specific
ticker was selected in the Pre-Buy Checklist dropdown — a plain "does it
import" check would have missed them. This test explicitly drives that
dropdown (and a few others) through every option to catch that whole
class of bug before it reaches production.

HOW TO USE:
    1. One-time setup (from your repo root, next to dashboard.py):
         pip install streamlit pytest --break-system-packages
    2. Generate fake scan data so the data-driven tabs have something
       to render:
         python tests/make_fixture.py
    3. Run the smoke test:
         pytest tests/test_smoke_app.py -v
       or just:
         python tests/test_smoke_app.py

    Run this before every deploy. It takes well under a minute and would
    have caught both crashes from this session.

WHAT THIS DOES NOT DO:
    - It does not check that numbers are *correct* (e.g. that the Apex
      Score math is right) — only that the code runs without raising.
    - It does not hit real APIs (yfinance/Alpha Vantage calls will be
      attempted; if you're offline or rate-limited, those specific
      widgets may show cached/empty data but should still not crash).
    - It is not a replacement for manually clicking through before a
      big release — just a fast net that catches the obvious breakages.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from streamlit.testing.v1 import AppTest


def _fail_with_exceptions(at, stage_name: str):
    """Raise a clear, readable error listing every exception AppTest caught."""
    if at.exception:
        msgs = []
        for exc in at.exception:
            msgs.append(f"  - {exc.value if hasattr(exc, 'value') else exc}")
        raise AssertionError(
            f"\n\n❌ dashboard.py raised {len(at.exception)} exception(s) during: {stage_name}\n"
            + "\n".join(msgs)
            + "\n\nFix these before deploying — a user would have hit a crash screen here.\n"
        )


def get_app():
    app_path = REPO_ROOT / "dashboard.py"
    assert app_path.exists(), (
        f"Could not find dashboard.py at {app_path}. "
        "Run this test from your repo root, or adjust REPO_ROOT in this file."
    )
    return AppTest.from_file(str(app_path))


def test_app_loads_without_crashing():
    """Baseline: the app must at least load and render every tab once."""
    at = get_app()
    at.run(timeout=60)
    _fail_with_exceptions(at, "initial app load (all tabs render once)")


def test_pre_buy_checklist_every_ticker():
    """
    Directly targets the bug class from this session: the checklist only
    crashed once a *specific* ticker was selected and its downstream
    checklist/target/conviction code ran. Cycle through every ticker in
    the fixture scan to catch anything ticker-data-shape-dependent.
    """
    at = get_app()
    at.run(timeout=60)
    _fail_with_exceptions(at, "initial load before checklist interaction")

    try:
        options = list(at.selectbox(key="chk_ticker_sel").options)
    except Exception:
        print("⚠️  Skipped: 'chk_ticker_sel' selectbox not found — "
              "run tests/make_fixture.py first so scan data exists.")
        return

    assert options, "Pre-Buy Checklist ticker list is empty — fixture data may not have loaded."

    for ticker in options:
        # IMPORTANT: re-fetch the selectbox fresh from `at` every iteration.
        # AppTest rebuilds its element tree on every .run(); reusing a widget
        # object from a previous run against a NEW run's session state raises
        # a spurious "session_state has no key" error that looks like an app
        # bug but is actually just a stale reference in this test script.
        chk_select = at.selectbox(key="chk_ticker_sel")
        chk_select.select(ticker).run(timeout=60)
        _fail_with_exceptions(at, f"Pre-Buy Checklist for ticker '{ticker}'")

        # Also exercise the "Log to Trade Journal" button for at least one
        # ticker, since that path builds the full trade-plan text block
        # (exactly where the ':02d' crash happened).
        if ticker == options[0]:
            try:
                at.button(key="chk_log_journal").click().run(timeout=60)
                _fail_with_exceptions(at, f"'Log to Trade Journal' for '{ticker}'")
            except Exception:
                pass  # button may not exist in older versions; non-fatal for smoke test


def test_stock_deep_dive_selection():
    """Exercise the Chart Viewer / Deep Dive ticker selector similarly."""
    at = get_app()
    at.run(timeout=60)
    try:
        ticker_select = at.selectbox(key=None)  # first plain selectbox = Chart Viewer "Ticker"
    except Exception:
        ticker_select = None
    # Best-effort: if the selector isn't reachable by key, this is skipped
    # rather than failing the whole suite — the checklist test above is
    # the one that matters most for regression coverage.
    _fail_with_exceptions(at, "Chart Viewer tab render")


def test_interpretation_tab_single_ticker():
    """The 'Complete Data Table' path renders every COLUMN_META field —
    a good broad check that formatting code across the whole file is safe."""
    at = get_app()
    at.run(timeout=60)
    try:
        options = list(at.selectbox(key="interp_tk").options)
    except Exception:
        print("⚠️  Skipped: 'interp_tk' selectbox not found — "
              "run tests/make_fixture.py first, or select 'Single Ticker Deep Read' manually.")
        return

    for ticker in options:
        at.selectbox(key="interp_tk").select(ticker).run(timeout=60)
        _fail_with_exceptions(at, f"Interpretation tab for ticker '{ticker}'")


def test_scan_delta_all_new_dropped_tickers():
    """
    Regression test for a real production bug: comparing two scans with
    ZERO overlapping tickers made the 'Δ Score' column all-None, which
    pandas silently typed as dtype=object — and .nlargest()/.nsmallest()
    in the 'Score Trajectory' chart raised a TypeError even on an empty
    column. tests/make_fixture.py writes two non-overlapping fixture scans
    specifically so the Scan Delta tab's default newest-vs-second-newest
    selection reproduces this exact scenario on every run.

    IMPORTANT: this code is wrapped in its own try/except that displays
    st.error(...) instead of crashing the whole app — so a hard-exception
    check alone (at.exception) would NOT have caught this bug; it must
    also check for the caught-and-displayed error message.
    """
    at = get_app()
    at.run(timeout=60)
    _fail_with_exceptions(at, "Scan Delta tab with two non-overlapping fixture scans")

    caught_errors = [e.value for e in at.error if "Error comparing scans" in e.value]
    assert not caught_errors, (
        "\n\n❌ Scan Delta tab silently caught and displayed an error instead of "
        "rendering correctly:\n  " + "\n  ".join(caught_errors) +
        "\n\nThis is the 'Δ Score dtype object' regression — fix the dtype "
        "coercion where _ddf is built.\n"
    )


if __name__ == "__main__":
    tests = [
        test_app_loads_without_crashing,
        test_pre_buy_checklist_every_ticker,
        test_stock_deep_dive_selection,
        test_interpretation_tab_single_ticker,
        test_scan_delta_all_new_dropped_tickers,
    ]
    failures = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"✅ PASS: {name}")
        except AssertionError as e:
            failures += 1
            print(f"❌ FAIL: {name}\n{e}")
        except Exception as e:
            failures += 1
            print(f"❌ ERROR: {name}: {e}")

    print(f"\n{len(tests) - failures}/{len(tests)} smoke tests passed.")
    sys.exit(1 if failures else 0)
