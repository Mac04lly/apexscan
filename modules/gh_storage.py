"""modules/gh_storage.py — durable JSON storage backed by the GitHub repo,
so data survives Streamlit Cloud redeploys and idle-container wipes."""
import base64, json, logging
import requests

log = logging.getLogger(__name__)
GITHUB_API = "https://api.github.com"


def _headers(token):
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def load_json_from_github(token, repo, path, branch="main"):
    """Returns (data, sha). (None, None) if missing, unconfigured, or failed."""
    if not token or not repo:
        return None, None
    try:
        url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
        r = requests.get(url, headers=_headers(token), params={"ref": branch}, timeout=10)
        if r.status_code == 404:
            return None, None
        r.raise_for_status()
        content = r.json()
        raw = base64.b64decode(content["content"]).decode("utf-8")
        return json.loads(raw), content["sha"]
    except Exception as e:
        log.warning(f"GitHub load failed for {path}: {e}")
        return None, None


def save_json_to_github(token, repo, path, data, branch="main", message=None):
    """Never raises — returns True/False so callers can fall back safely."""
    if not token or not repo:
        return False
    try:
        url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
        _, sha = load_json_from_github(token, repo, path, branch)
        body = {
            "message": message or f"Update {path}",
            "content": base64.b64encode(json.dumps(data, indent=2, default=str).encode()).decode(),
            "branch": branch,
        }
        if sha:
            body["sha"] = sha
        r = requests.put(url, headers=_headers(token), json=body, timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"GitHub save failed for {path}: {e}")
        return False
