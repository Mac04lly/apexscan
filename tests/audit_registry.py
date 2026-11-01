"""The registered methodology and amendments must not change after registration (hash check, line-ending safe)."""
import shutil, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from audit import registry as RG

REGISTERED_METHODOLOGY = "3ec2d04fe3ae95de1c7ae91f3b6d3b0a6e8cf5fe6adb20e2bee203274d4f5d77"   # recorded 31 Oct 2026, 21:54 UTC


def _copy(tmp_path):
    d = tmp_path / "reg"; shutil.copytree(RG.REG_DIR, d); return d


def test_all_registered_documents_are_intact_and_three_are_recorded():
    rows = RG.verify(); assert len(rows) == 3 and all(r["status"] == "ok" for r in rows) and RG.all_intact()


def test_the_methodology_hash_is_pinned_independently_of_the_record_file():
    r = {x["document"]: x for x in RG.verify()}
    assert r["methodology.md"]["current"] == REGISTERED_METHODOLOGY and r["methodology.md"]["recorded"] == REGISTERED_METHODOLOGY


def test_an_edit_to_any_registered_document_is_detected(tmp_path):
    for name in ("methodology.md", "amendments.md", "amendments2.md"):
        d = _copy(tmp_path / name); p = d / name; p.write_text(p.read_text(encoding="utf-8") + "\nquietly added sentence", encoding="utf-8")
        st = {x["document"]: x["status"] for x in RG.verify(d)}
        assert st[name] == "CHANGED" and not RG.all_intact(d)


def test_windows_line_endings_do_not_cause_a_false_alarm(tmp_path):
    d = _copy(tmp_path); p = d / "methodology.md"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n")); assert RG.all_intact(d)


def test_a_missing_document_is_reported_not_ignored(tmp_path):
    d = _copy(tmp_path); (d / "amendments.md").unlink()
    assert {x["document"]: x["status"] for x in RG.verify(d)}["amendments.md"] == "missing" and not RG.all_intact(d)


def test_changing_the_hash_record_to_match_an_edit_is_still_caught_for_the_methodology(tmp_path):
    d = _copy(tmp_path); p = d / "methodology.md"; p.write_text(p.read_text(encoding="utf-8") + "x", encoding="utf-8")
    rec = d / "PREREGISTRATION.txt"; new = RG.sha256_normalised(p.read_bytes())
    rec.write_text(rec.read_text(encoding="utf-8").replace(REGISTERED_METHODOLOGY, new), encoding="utf-8")
    assert RG.all_intact(d)                                                    # the record file alone cannot prove integrity ...
    assert RG.verify(d)[0]["current"] != REGISTERED_METHODOLOGY                # ... which is why the hash is also pinned in this test
