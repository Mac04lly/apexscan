"""modules/gh_storage.py — durable JSON storage backed by the GitHub repo,
so data survives Streamlit Cloud redeploys and idle-container wipes.

════════════════════════════════════════════════════════════════════════
FIX: the Contents API has a ~1MB practical ceiling — READS AND WRITES
════════════════════════════════════════════════════════════════════════
GitHub's Contents API (GET/PUT /repos/{repo}/contents/{path}) silently
breaks once a file crosses roughly 1MB:
  - READ: the JSON response's `content` field comes back empty for
    large files (the API still returns 200 with metadata, just no
    content) — decoding that empty string produced exactly the
    "Expecting value: line 1 column 1 (char 0)" error seen once
    data/alpha_observations.json grew past ~1.2MB.
  - WRITE: since the read above silently failed, the previous
    implementation got sha=None back and — believing the file didn't
    exist yet — omitted `sha` from the PUT body. GitHub then correctly
    rejects that as "you didn't supply the sha of the file you're
    overwriting" with a 422, since the file DOES already exist.

Both are fixed here by switching to the Git Data API (blob + tree +
commit + ref objects), which has no practical size ceiling (blobs up to
100MB) and is the standard, documented way to read/write large files via
the GitHub API without a full git clone:
  - load_json_from_github(): one metadata call (works at any size, gets
    `sha`) + one RAW-content call (Accept: application/vnd.github.raw,
    which streams the actual bytes instead of wrapping them in
    size-limited base64-in-JSON).
  - save_json_to_github(): the full low-level sequence — read branch
    head -> read its tree -> create a new blob with the new content ->
    create a new tree (one path changed, base_tree reused for
    everything else) -> create a new commit -> fast-forward the branch
    ref to it. This is more API calls per operation (about 2 for a
    read, 6 for a write) than the original single-call approach, but
    GitHub's authenticated rate limit is 5,000 requests/hour and this
    app makes at most a handful of these calls per scan — the extra
    calls are immaterial next to that budget.

Both public function signatures and return contracts are UNCHANGED —
every existing caller (modules/alpha_validation.py,
modules/model_registry.py, modules/apex10_baseline.py,
modules/apex10_precursor.py, and anywhere else in this app) needs no
changes at all.
"""
import base64, json, logging
import requests

log = logging.getLogger(__name__)
GITHUB_API = "https://api.github.com"
_TIMEOUT = 20


def _headers(token, accept="application/vnd.github+json"):
    return {"Authorization": f"token {token}", "Accept": accept}


def load_json_from_github(token, repo, path, branch="main"):
    """Returns (data, sha). (None, None) if missing, unconfigured, or failed.
    Works for files of any size — see module docstring."""
    if not token or not repo:
        return None, None
    try:
        url = f"{GITHUB_API}/repos/{repo}/contents/{path}"

        # Metadata call: returns `sha` correctly regardless of file size —
        # only the `content` field is unreliable for large files, not this.
        meta_r = requests.get(url, headers=_headers(token), params={"ref": branch}, timeout=10)
        if meta_r.status_code == 404:
            return None, None
        meta_r.raise_for_status()
        sha = meta_r.json().get("sha")

        # Raw-content call: the `application/vnd.github.raw` media type
        # streams the actual file bytes directly instead of wrapping them
        # in base64-inside-JSON — this is what sidesteps the ~1MB ceiling.
        raw_r = requests.get(url, headers=_headers(token, accept="application/vnd.github.raw"),
                             params={"ref": branch}, timeout=_TIMEOUT)
        if raw_r.status_code == 404:
            return None, None
        raw_r.raise_for_status()
        return json.loads(raw_r.text), sha
    except Exception as e:
        log.warning(f"GitHub load failed for {path}: {e}")
        return None, None


def _get_branch_head_sha(token, repo, branch):
    r = requests.get(f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}",
                     headers=_headers(token), timeout=10)
    r.raise_for_status()
    return r.json()["object"]["sha"]


def _get_commit_tree_sha(token, repo, commit_sha):
    r = requests.get(f"{GITHUB_API}/repos/{repo}/git/commits/{commit_sha}",
                     headers=_headers(token), timeout=10)
    r.raise_for_status()
    return r.json()["tree"]["sha"]


def _create_blob(token, repo, content_str):
    body = {"content": base64.b64encode(content_str.encode("utf-8")).decode("ascii"),
           "encoding": "base64"}
    r = requests.post(f"{GITHUB_API}/repos/{repo}/git/blobs", headers=_headers(token),
                      json=body, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()["sha"]


def _create_tree(token, repo, base_tree_sha, path, blob_sha):
    # Specifying the full nested path in one entry (e.g. "data/x.json") is
    # standard Trees API usage — GitHub creates any missing intermediate
    # directories implicitly; no special-casing needed for a brand-new
    # file or a brand-new subdirectory.
    body = {"base_tree": base_tree_sha,
           "tree": [{"path": path, "mode": "100644", "type": "blob", "sha": blob_sha}]}
    r = requests.post(f"{GITHUB_API}/repos/{repo}/git/trees", headers=_headers(token),
                      json=body, timeout=10)
    r.raise_for_status()
    return r.json()["sha"]


def _create_commit(token, repo, message, tree_sha, parent_sha):
    body = {"message": message, "tree": tree_sha, "parents": [parent_sha]}
    r = requests.post(f"{GITHUB_API}/repos/{repo}/git/commits", headers=_headers(token),
                      json=body, timeout=10)
    r.raise_for_status()
    return r.json()["sha"]


def _update_ref(token, repo, branch, commit_sha):
    body = {"sha": commit_sha}
    r = requests.patch(f"{GITHUB_API}/repos/{repo}/git/refs/heads/{branch}",
                       headers=_headers(token), json=body, timeout=10)
    r.raise_for_status()


def save_json_to_github(token, repo, path, data, branch="main", message=None):
    """Never raises — returns True/False so callers can fall back safely.
    Works for files of any size — see module docstring."""
    if not token or not repo:
        return False
    try:
        content_str = json.dumps(data, indent=2, default=str)
        head_sha = _get_branch_head_sha(token, repo, branch)
        base_tree_sha = _get_commit_tree_sha(token, repo, head_sha)
        blob_sha = _create_blob(token, repo, content_str)
        new_tree_sha = _create_tree(token, repo, base_tree_sha, path, blob_sha)
        new_commit_sha = _create_commit(token, repo, message or f"Update {path}",
                                        new_tree_sha, head_sha)
        _update_ref(token, repo, branch, new_commit_sha)
        return True
    except Exception as e:
        log.warning(f"GitHub save failed for {path}: {e}")
        return False
