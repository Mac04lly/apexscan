"""
tests/test_gh_storage_large_files.py — gh_storage.py Git Data API fix

Verifies the fix for the ~1MB Contents API ceiling that broke
data/alpha_observations.json: load_json_from_github() no longer depends
on the size-limited `content` field, and save_json_to_github() no
longer depends on a (possibly-broken) prior read to get `sha` — it uses
the Git Data API's ref/commit/blob/tree/commit/ref sequence instead.

Uses a stateful fake-GitHub harness (records blobs/trees/commits/refs
in memory, keyed by sha) rather than ad-hoc per-call mocks, so a real
save-then-load round trip can be verified end-to-end without live
credentials — the strongest test available for a multi-step API
sequence like this.
"""
import sys
import json
import base64
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules import gh_storage


class _FakeGitHub:
    """Minimal in-memory model of the Git Data API surface gh_storage.py
    actually uses: refs, commits, trees, blobs — enough to support a
    real save -> load round trip against mocked `requests` calls."""

    def __init__(self):
        self.blobs = {}    # sha -> content string
        self.trees = {}    # sha -> {path: blob_sha}
        self.commits = {}  # sha -> {"tree": tree_sha, "parents": [...]}
        self.refs = {}      # "heads/branch" -> commit_sha
        self._counter = 0
        # Seed an initial empty commit/tree/ref so the first save has
        # something to build on, matching a real repo with history.
        empty_tree_sha = self._new_sha("tree0")
        self.trees[empty_tree_sha] = {}
        initial_commit_sha = self._new_sha("commit0")
        self.commits[initial_commit_sha] = {"tree": empty_tree_sha, "parents": []}
        self.refs["heads/main"] = initial_commit_sha

    def _new_sha(self, seed):
        self._counter += 1
        return hashlib.sha1(f"{seed}-{self._counter}".encode()).hexdigest()

    def get_file_content(self, path):
        """Test helper: resolves the CURRENT content at `path` by
        walking the current ref -> commit -> tree."""
        commit_sha = self.refs.get("heads/main")
        if not commit_sha:
            return None
        tree_sha = self.commits[commit_sha]["tree"]
        blob_sha = self.trees.get(tree_sha, {}).get(path)
        return self.blobs.get(blob_sha) if blob_sha else None


def _mock_requests_for(fake_gh: _FakeGitHub, existing_path: str = None):
    """Returns (mock_get, mock_post, mock_patch) wired against `fake_gh`,
    handling exactly the endpoints gh_storage.py calls."""

    def fake_get(url, headers=None, params=None, timeout=None):
        resp = MagicMock()
        if url.endswith("/git/ref/heads/main"):
            sha = fake_gh.refs.get("heads/main")
            resp.status_code = 200
            resp.json.return_value = {"object": {"sha": sha}}
            resp.raise_for_status = MagicMock()
            return resp
        if "/git/commits/" in url:
            commit_sha = url.rsplit("/", 1)[-1]
            resp.status_code = 200
            resp.json.return_value = {"tree": {"sha": fake_gh.commits[commit_sha]["tree"]}}
            resp.raise_for_status = MagicMock()
            return resp
        if "/contents/" in url:
            # Contents-API-shaped calls: metadata call (default Accept)
            # and raw-content call (Accept: application/vnd.github.raw).
            path = url.split("/contents/", 1)[1]
            content = fake_gh.get_file_content(path)
            if content is None:
                resp.status_code = 404
                resp.raise_for_status = MagicMock(side_effect=Exception("404"))
                return resp
            resp.status_code = 200
            accept = (headers or {}).get("Accept", "")
            if accept == "application/vnd.github.raw":
                resp.text = content
            else:
                # Metadata response — deliberately does NOT populate
                # `content`, matching real GitHub behavior for large
                # files. sha is still correct, which is the whole point.
                commit_sha = fake_gh.refs.get("heads/main")
                tree_sha = fake_gh.commits[commit_sha]["tree"]
                resp.json.return_value = {"sha": tree_sha and _blob_sha_for(fake_gh, path), "content": ""}
            resp.raise_for_status = MagicMock()
            return resp
        raise AssertionError(f"Unexpected GET url in test: {url}")

    def _blob_sha_for(fake_gh, path):
        commit_sha = fake_gh.refs.get("heads/main")
        tree_sha = fake_gh.commits[commit_sha]["tree"]
        return fake_gh.trees.get(tree_sha, {}).get(path)

    def fake_post(url, headers=None, json=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 201
        resp.raise_for_status = MagicMock()
        if url.endswith("/git/blobs"):
            raw = base64.b64decode(json["content"]).decode("utf-8")
            sha = fake_gh._new_sha(raw[:20])
            fake_gh.blobs[sha] = raw
            resp.json.return_value = {"sha": sha}
            return resp
        if url.endswith("/git/trees"):
            base_tree_sha = json["base_tree"]
            new_tree = dict(fake_gh.trees.get(base_tree_sha, {}))
            for entry in json["tree"]:
                new_tree[entry["path"]] = entry["sha"]
            sha = fake_gh._new_sha(str(new_tree))
            fake_gh.trees[sha] = new_tree
            resp.json.return_value = {"sha": sha}
            return resp
        if url.endswith("/git/commits"):
            sha = fake_gh._new_sha(json["message"])
            fake_gh.commits[sha] = {"tree": json["tree"], "parents": json["parents"]}
            resp.json.return_value = {"sha": sha}
            return resp
        raise AssertionError(f"Unexpected POST url in test: {url}")

    def fake_patch(url, headers=None, json=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        assert url.endswith("/git/refs/heads/main")
        fake_gh.refs["heads/main"] = json["sha"]
        return resp

    return fake_get, fake_post, fake_patch


# ── save_json_to_github: sequencing and correctness ────────────────────

def test_save_uses_git_data_api_sequence_not_contents_put():
    fake_gh = _FakeGitHub()
    fake_get, fake_post, fake_patch = _mock_requests_for(fake_gh)
    with patch("requests.get", side_effect=fake_get), \
         patch("requests.post", side_effect=fake_post), \
         patch("requests.patch", side_effect=fake_patch):
        ok = gh_storage.save_json_to_github("tok", "owner/repo", "data/test.json", {"a": 1})
    assert ok is True
    assert json.loads(fake_gh.get_file_content("data/test.json")) == {"a": 1}


def test_save_no_credentials_returns_false_and_makes_no_calls():
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        ok = gh_storage.save_json_to_github("", "", "data/test.json", {"a": 1})
    assert ok is False
    mock_get.assert_not_called()
    mock_post.assert_not_called()


def test_save_failure_midway_returns_false_not_raise():
    fake_gh = _FakeGitHub()
    fake_get, fake_post, fake_patch = _mock_requests_for(fake_gh)

    def failing_post(*args, **kwargs):
        raise Exception("simulated network failure creating blob")

    with patch("requests.get", side_effect=fake_get), \
         patch("requests.post", side_effect=failing_post), \
         patch("requests.patch", side_effect=fake_patch):
        ok = gh_storage.save_json_to_github("tok", "owner/repo", "data/test.json", {"a": 1})
    assert ok is False


# ── load_json_from_github: no longer depends on the size-limited field ─

def test_load_uses_raw_media_type_not_base64_content_field():
    fake_gh = _FakeGitHub()
    fake_get, fake_post, fake_patch = _mock_requests_for(fake_gh)
    with patch("requests.get", side_effect=fake_get), \
         patch("requests.post", side_effect=fake_post), \
         patch("requests.patch", side_effect=fake_patch):
        gh_storage.save_json_to_github("tok", "owner/repo", "data/big.json", {"big": True})
        data, sha = gh_storage.load_json_from_github("tok", "owner/repo", "data/big.json")
    assert data == {"big": True}
    assert sha is not None


def test_load_missing_file_returns_none_none():
    fake_gh = _FakeGitHub()
    fake_get, fake_post, fake_patch = _mock_requests_for(fake_gh)
    with patch("requests.get", side_effect=fake_get):
        data, sha = gh_storage.load_json_from_github("tok", "owner/repo", "data/nope.json")
    assert data is None and sha is None


def test_load_no_credentials_returns_none_none_and_makes_no_calls():
    with patch("requests.get") as mock_get:
        data, sha = gh_storage.load_json_from_github("", "", "data/test.json")
    assert data is None and sha is None
    mock_get.assert_not_called()


# ── The core regression: a "large" file (metadata content field empty,
# exactly like real GitHub for files >1MB) still loads correctly ───────

def test_load_succeeds_even_when_metadata_content_field_is_empty():
    """This is the exact failure mode that broke alpha_observations.json:
    the metadata call's `content` field is empty (as GitHub actually
    returns for large files) — the fixed implementation must not depend
    on that field at all, only on the separate raw-content call."""
    fake_gh = _FakeGitHub()
    fake_get, fake_post, fake_patch = _mock_requests_for(fake_gh)
    large_payload = {"observations": [{"ticker": f"T{i}", "note": "x" * 1000} for i in range(2000)]}
    with patch("requests.get", side_effect=fake_get), \
         patch("requests.post", side_effect=fake_post), \
         patch("requests.patch", side_effect=fake_patch):
        saved = gh_storage.save_json_to_github("tok", "owner/repo", "data/alpha_observations.json",
                                               large_payload)
        assert saved is True
        data, sha = gh_storage.load_json_from_github("tok", "owner/repo", "data/alpha_observations.json")
    assert data == large_payload
    assert sha is not None
    assert len(data["observations"]) == 2000


# ── True round trip: save, then a SEPARATE update, then load again ────

def test_save_then_update_then_load_round_trip_preserves_latest_content():
    fake_gh = _FakeGitHub()
    fake_get, fake_post, fake_patch = _mock_requests_for(fake_gh)
    with patch("requests.get", side_effect=fake_get), \
         patch("requests.post", side_effect=fake_post), \
         patch("requests.patch", side_effect=fake_patch):
        gh_storage.save_json_to_github("tok", "owner/repo", "data/x.json", {"version": 1})
        gh_storage.save_json_to_github("tok", "owner/repo", "data/x.json", {"version": 2})
        data, _ = gh_storage.load_json_from_github("tok", "owner/repo", "data/x.json")
    assert data == {"version": 2}


def test_saving_one_file_does_not_disturb_another():
    fake_gh = _FakeGitHub()
    fake_get, fake_post, fake_patch = _mock_requests_for(fake_gh)
    with patch("requests.get", side_effect=fake_get), \
         patch("requests.post", side_effect=fake_post), \
         patch("requests.patch", side_effect=fake_patch):
        gh_storage.save_json_to_github("tok", "owner/repo", "data/a.json", {"file": "a"})
        gh_storage.save_json_to_github("tok", "owner/repo", "data/b.json", {"file": "b"})
        data_a, _ = gh_storage.load_json_from_github("tok", "owner/repo", "data/a.json")
        data_b, _ = gh_storage.load_json_from_github("tok", "owner/repo", "data/b.json")
    assert data_a == {"file": "a"}
    assert data_b == {"file": "b"}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
