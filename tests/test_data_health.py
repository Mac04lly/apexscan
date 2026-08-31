"""
tests/test_data_health.py — Persistence Health Check tests

Mocks modules.gh_storage's functions as imported into modules.data_health
(not raw `requests` again — gh_storage itself already has its own
dedicated test suite; this tests data_health's own classification,
round-trip verification, and summarization logic).
"""
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules import data_health


# ── check_store ───────────────────────────────────────────────────────

def test_check_store_ok_for_small_healthy_store():
    with patch("modules.data_health.load_json_from_github", return_value=([{"a": 1}], "sha123")):
        result = data_health.check_store("tok", "owner/repo", "data/small.json")
    assert result["status"] == "ok"
    assert result["record_count"] == 1
    assert result["size_bytes"] is not None


def test_check_store_flags_large_files():
    big_payload = [{"note": "x" * 1000} for _ in range(5000)]  # comfortably over 3MB
    with patch("modules.data_health.load_json_from_github", return_value=(big_payload, "sha123")):
        result = data_health.check_store("tok", "owner/repo", "data/big.json")
    assert result["status"] == "large"


def test_check_store_missing_or_empty():
    with patch("modules.data_health.load_json_from_github", return_value=(None, None)):
        result = data_health.check_store("tok", "owner/repo", "data/gone.json")
    assert result["status"] == "missing_or_empty"


def test_check_store_exception_is_caught_not_raised():
    with patch("modules.data_health.load_json_from_github", side_effect=Exception("network blew up")):
        result = data_health.check_store("tok", "owner/repo", "data/x.json")
    assert result["status"] == "error"
    assert "network blew up" in result["detail"]


# ── check_all_stores ─────────────────────────────────────────────────

def test_check_all_stores_one_failure_does_not_stop_the_rest():
    def fake_load(token, repo, path):
        if path == "data/broken.json":
            raise Exception("simulated failure")
        return ([1, 2, 3], "sha")

    with patch("modules.data_health.load_json_from_github", side_effect=fake_load):
        results = data_health.check_all_stores("tok", "owner/repo",
                                               paths=["data/broken.json", "data/fine.json"])
    assert len(results) == 2
    statuses = {r["path"]: r["status"] for r in results}
    assert statuses["data/broken.json"] == "error"
    assert statuses["data/fine.json"] == "ok"


def test_check_all_stores_defaults_to_known_stores_list():
    with patch("modules.data_health.load_json_from_github", return_value=([1], "sha")):
        results = data_health.check_all_stores("tok", "owner/repo")
    assert len(results) == len(data_health.KNOWN_STORES)


# ── check_write_pipeline ─────────────────────────────────────────────

def test_write_pipeline_ok_when_roundtrip_matches():
    captured = {}

    def fake_save(token, repo, path, data, message=None):
        captured["written"] = data
        return True

    def fake_load(token, repo, path):
        return (captured["written"], "sha")

    with patch("modules.data_health.save_json_to_github", side_effect=fake_save), \
         patch("modules.data_health.load_json_from_github", side_effect=fake_load):
        result = data_health.check_write_pipeline("tok", "owner/repo")
    assert result["status"] == "ok"


def test_write_pipeline_write_failed():
    with patch("modules.data_health.save_json_to_github", return_value=False):
        result = data_health.check_write_pipeline("tok", "owner/repo")
    assert result["status"] == "write_failed"


def test_write_pipeline_roundtrip_mismatch_detected():
    with patch("modules.data_health.save_json_to_github", return_value=True), \
         patch("modules.data_health.load_json_from_github", return_value=({"wrong": "data"}, "sha")):
        result = data_health.check_write_pipeline("tok", "owner/repo")
    assert result["status"] == "roundtrip_mismatch"


def test_write_pipeline_exception_caught():
    with patch("modules.data_health.save_json_to_github", side_effect=Exception("boom")):
        result = data_health.check_write_pipeline("tok", "owner/repo")
    assert result["status"] == "error"


def test_write_pipeline_uses_dedicated_sentinel_path_never_a_real_store():
    captured_path = {}

    def fake_save(token, repo, path, data, message=None):
        captured_path["path"] = path
        return True

    with patch("modules.data_health.save_json_to_github", side_effect=fake_save), \
         patch("modules.data_health.load_json_from_github", return_value=({}, "sha")):
        data_health.check_write_pipeline("tok", "owner/repo")
    assert captured_path["path"] == data_health.SENTINEL_PATH
    assert captured_path["path"] not in data_health.KNOWN_STORES


# ── summarize ────────────────────────────────────────────────────────

def test_summarize_all_healthy_is_ok():
    stores = [{"path": "a", "status": "ok"}, {"path": "b", "status": "ok"}]
    write = {"status": "ok"}
    summary = data_health.summarize(stores, write)
    assert summary["overall_status"] == "ok"
    assert summary["stores_with_problems"] == 0


def test_summarize_store_problem_flags_attention_needed():
    stores = [{"path": "a", "status": "ok"}, {"path": "b", "status": "error"}]
    write = {"status": "ok"}
    summary = data_health.summarize(stores, write)
    assert summary["overall_status"] == "attention_needed"
    assert summary["problem_paths"] == ["b"]


def test_summarize_large_flag_alone_does_not_flip_overall_status():
    stores = [{"path": "a", "status": "large"}]
    write = {"status": "ok"}
    summary = data_health.summarize(stores, write)
    assert summary["overall_status"] == "ok"
    assert summary["stores_flagged_large"] == 1


def test_summarize_write_failure_flags_attention_needed_even_if_stores_fine():
    stores = [{"path": "a", "status": "ok"}]
    write = {"status": "write_failed"}
    summary = data_health.summarize(stores, write)
    assert summary["overall_status"] == "attention_needed"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
