"""pages/Audit.py: appears as its own page, loads without error, is read-only, and does nothing heavy until a button is pressed."""
import hashlib, json, sys, types
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tests"))
import pandas as pd, pytest
from streamlit.testing.v1 import AppTest
import test_audit_inapp as TI

PAGE = str(REPO / "pages" / "Audit.py")


def _snapshot():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((REPO / "data").glob("*.json"))}


def _texts(at):
    out = [m.value for m in at.markdown] + [x.value for x in at.caption] + [x.value for x in at.subheader] + [x.value for x in at.success] + [x.value for x in at.error] + [x.value for x in at.warning] + [x.value for x in at.info]
    return " ".join(out)


def test_the_page_exists_in_the_pages_folder_so_streamlit_lists_it_in_the_sidebar():
    assert (REPO / "pages" / "Audit.py").exists() and [p.name for p in (REPO / "pages").glob("*.py")] == ["Audit.py"]


def test_the_page_loads_cleanly_with_six_tabs_and_an_intact_registration():
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception and at.title[0].value.endswith("ApexScan Audit") and len(at.tabs) == 6
    t = _texts(at); assert "match the hashes recorded" in t and "Where the audit stands" in t


def test_live_discoveries_tab_reports_zero_matured_at_63_sessions():
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.run(); labels = {m.label: m.value for m in at.metric}
    assert labels["Discoveries"] == str(len(json.loads((REPO / "data" / "discoveries.json").read_text(encoding="utf-8"))))
    assert labels["Matured at 63 sessions"] == "0" and "None of the discoveries has completed the registered 63-session" in _texts(at)


def test_opening_the_page_does_not_write_to_data_or_start_a_download(monkeypatch):
    from audit.tools import export_prices as EX
    called = []
    monkeypatch.setattr(EX, "run", lambda *a, **k: called.append(1))
    before = _snapshot(); at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception and called == [] and _snapshot() == before


def test_the_page_never_touches_the_scanner_the_stores_or_the_other_tabs():
    src = Path(PAGE).read_text(encoding="utf-8") + "".join(p.read_text(encoding="utf-8") for p in (REPO / "audit").glob("*.py"))
    for banned in ("run_scan(", "save_report", "gh_storage", "import dashboard", "from dashboard", "log_new_discoveries", "MODEL_VERSION"):
        assert banned not in src, banned


def _fake_export(args, progress=None, **kw):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = TI.synth(); p.to_csv(out / "prices_long.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"ticker": list(p.ticker.unique()), "group": "sample", "status": "ok", "source": "batch", "rows": 1, "error": ""}).to_csv(out / "download_log.csv", index=False)
    if progress: progress(1, 1, "batch 1 of 1")
    return {"tickers_with_data": p.ticker.nunique(), "tickers_requested": p.ticker.nunique(), "rows": len(p)}


def test_export_then_data_quality_then_chdn_replay_work_end_to_end_with_a_fake_download(monkeypatch):
    from audit.tools import export_prices as EX
    monkeypatch.setattr(EX, "run", _fake_export)
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    at.button(key="audit_run_export").click().run()
    assert not at.exception and "Finished:" in _texts(at) and "audit_dq" in at.session_state
    assert any(m.label == "Error flags" and m.value == "0" for m in at.metric)
    at.button(key="audit_run_replay").click().run()
    assert not at.exception and "Passing continuously since (reconstructed; lower bound)" in " ".join(str(df.value.columns.tolist()) for df in at.dataframe)


def test_a_failed_export_shows_a_plain_message_and_keeps_the_page_alive(monkeypatch):
    from audit.tools import export_prices as EX
    def boom(*a, **k): raise RuntimeError("yahoo down")
    monkeypatch.setattr(EX, "run", boom)
    at = AppTest.from_file(PAGE, default_timeout=60).run(); at.button(key="audit_run_export").click().run()
    assert not at.exception and "The export did not finish: yahoo down" in _texts(at)


def test_dollar_signs_in_the_registered_text_are_escaped_so_they_are_not_read_as_maths():
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    amend = [m.value for m in at.markdown if "Amendment 1" in m.value or "$20M" in m.value or "20M" in m.value]
    assert amend and all("$" not in v.replace("\\$", "") for v in amend)


def test_cohorts_are_shown_separately_with_the_agreed_names_and_no_pooled_row():
    from audit import cohorts as C
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception
    frames = [df.value for df in at.dataframe]
    ct = [f for f in frames if "Unique tickers" in f.columns and "Non-overlapping trades (per ticker)" in f.columns]
    assert len(ct) == 1 and list(ct[0]["cohort"]) == [C.A1, C.A2]                 # exactly the two cohorts, nothing pooled
    t = _texts(at); assert "Initial persistence/seeding cohort" in t and "Post-seeding first-recorded discovery cohort" in t
    assert "not confirmed first-time qualifications" in t and "cannot be established" in t


def test_three_date_concepts_are_kept_apart_on_the_page():
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    dates = [df.value for df in at.dataframe if "Meaning" in df.value.columns]
    d = dates[0]; assert list(d["Date"]) == ["Recorded discovery date", "Reconstructed four-gate date", "Actual full scanner qualification date"]
    assert "not established" in d["Available"].iloc[2] and "lower bound" in d["Meaning"].iloc[1].lower()


def test_amendment_3_is_shown_on_the_registration_tab():
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert any("Amendment 3" in e.label for e in at.expander) and any("Amendments 1, 2, 3 and 4 registered" in str(df.value) for df in at.dataframe)


def test_amendment_4_is_shown_on_the_registration_tab():
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert any("Amendment 4" in e.label for e in at.expander)
