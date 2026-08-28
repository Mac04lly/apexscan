"""
tests/test_model_governance_v9_phase8.py — V9 Phase 8 regression tests

Mocks modules.gh_storage so the governance WORKFLOW (propose -> record
walk-forward -> approve, and its guard rails) is tested independent of
live GitHub credentials — this repo has no test credentials, by design
(the security note in CHANGELOG.md about a leaked/rotated token is
exactly why none should ever be embedded in tests).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modules.model_registry as mr


class _FakeStorage:
    """In-memory stand-in for the GitHub-backed store, matching
    load_json_from_github / save_json_to_github's signatures exactly."""
    def __init__(self):
        self.data = None

    def load(self, token, repo, path, branch="main"):
        return (self.data, "fake-sha") if self.data is not None else (None, None)

    def save(self, token, repo, path, data, branch="main", message=None):
        self.data = data
        return True


def _patch_storage(monkeypatch):
    fake = _FakeStorage()
    monkeypatch.setattr(mr, "_load_gh", fake.load)
    monkeypatch.setattr(mr, "_save_gh", fake.save)
    monkeypatch.setattr(mr, "_get_gh_creds", lambda: ("fake-token", "fake/repo"))
    return fake


def test_ensure_production_entry_backfills(monkeypatch):
    _patch_storage(monkeypatch)
    registry = mr.ensure_production_entry_exists()
    assert mr.MODEL_VERSION in registry["versions"]
    assert registry["versions"][mr.MODEL_VERSION]["status"] == mr.STATUS_PRODUCTION


def test_propose_research_model_creates_research_entry(monkeypatch):
    _patch_storage(monkeypatch)
    record = mr.propose_research_model("APEX-9.1-TEST", "Increase RS weighting",
                                       source_finding="S2-HIGH-RS", proposed_by="tester")
    assert record["status"] == mr.STATUS_RESEARCH
    assert record["promoted_from"] == mr.MODEL_VERSION
    table = mr.get_registry_table()
    ids = {r["version_id"] for r in table}
    assert "APEX-9.1-TEST" in ids and mr.MODEL_VERSION in ids


def test_propose_research_model_rejects_missing_fields(monkeypatch):
    _patch_storage(monkeypatch)
    try:
        mr.propose_research_model("", "some description")
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        mr.propose_research_model("APEX-X", "")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_propose_duplicate_version_id_raises(monkeypatch):
    _patch_storage(monkeypatch)
    mr.propose_research_model("APEX-DUP", "first proposal")
    try:
        mr.propose_research_model("APEX-DUP", "second proposal")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_approve_promotion_requires_named_approver(monkeypatch):
    _patch_storage(monkeypatch)
    mr.propose_research_model("APEX-NOAPPROVER", "desc")
    mr.record_walk_forward_result("APEX-NOAPPROVER", {"passed": True, "pass_rate": 0.8})
    try:
        mr.approve_promotion("APEX-NOAPPROVER", "")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_approve_promotion_requires_passing_walk_forward(monkeypatch):
    _patch_storage(monkeypatch)
    mr.propose_research_model("APEX-NOWF", "desc")
    # No walk-forward result attached at all.
    try:
        mr.approve_promotion("APEX-NOWF", "reviewer")
        assert False, "expected ValueError (no walk-forward on file)"
    except ValueError:
        pass
    # A FAILING walk-forward result should also block approval.
    mr.record_walk_forward_result("APEX-NOWF", {"passed": False, "pass_rate": 0.2})
    try:
        mr.approve_promotion("APEX-NOWF", "reviewer")
        assert False, "expected ValueError (failing walk-forward)"
    except ValueError:
        pass


def test_approve_promotion_succeeds_with_passing_walk_forward(monkeypatch):
    _patch_storage(monkeypatch)
    mr.propose_research_model("APEX-GOOD", "desc")
    mr.record_walk_forward_result("APEX-GOOD", {"passed": True, "pass_rate": 0.75})
    record = mr.approve_promotion("APEX-GOOD", "reviewer-1")
    assert record["status"] == mr.STATUS_APPROVED_PENDING_ACTIVATION
    assert record["approved_by"] == "reviewer-1"


def test_approve_promotion_never_changes_module_constant(monkeypatch):
    """The whole point of Phase 8: approval records a decision but can
    NEVER make MODEL_VERSION itself change — that still requires a human
    to edit the source file."""
    _patch_storage(monkeypatch)
    before = mr.MODEL_VERSION
    mr.propose_research_model("APEX-SAFE", "desc")
    mr.record_walk_forward_result("APEX-SAFE", {"passed": True, "pass_rate": 0.9})
    mr.approve_promotion("APEX-SAFE", "reviewer")
    assert mr.MODEL_VERSION == before
    assert mr.get_model_version() == before


def test_reject_proposal(monkeypatch):
    _patch_storage(monkeypatch)
    mr.propose_research_model("APEX-REJECT", "desc")
    record = mr.reject_proposal("APEX-REJECT", "Didn't survive walk-forward", "reviewer")
    assert record["status"] == mr.STATUS_REJECTED
    assert record["rejected_reason"] == "Didn't survive walk-forward"


def test_compare_models_handles_zero_observations_on_either_side(monkeypatch):
    _patch_storage(monkeypatch)
    result = mr.compare_models("APEX-A", "APEX-B", [], "20D")
    assert result["metrics_a"]["n"] == 0
    assert result["metrics_b"]["n"] == 0


def test_compare_models_splits_by_tagged_version(monkeypatch):
    _patch_storage(monkeypatch)
    obs = [
        {"model_version": "APEX-A", "outcomes": {"20D": {"forward_return_%": 5.0,
         "excess_return_%": 3.0, "mfe_%": 6, "mae_%": -1, "benchmark_return_%": 1.0}}},
        {"model_version": "APEX-B", "outcomes": {"20D": {"forward_return_%": -2.0,
         "excess_return_%": -3.0, "mfe_%": 1, "mae_%": -3, "benchmark_return_%": 1.0}}},
    ]
    result = mr.compare_models("APEX-A", "APEX-B", obs, "20D")
    assert result["metrics_a"]["n"] == 1
    assert result["metrics_b"]["n"] == 1
    assert result["metrics_a"]["expectancy_%"] == 5.0
    assert result["metrics_b"]["expectancy_%"] == -2.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
