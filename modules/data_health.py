"""
modules/data_health.py — Persistence Health Check

Built directly in response to a real incident: data/alpha_observations.json
silently failed to load AND save for an unknown period once it crossed
GitHub's Contents API ~1MB ceiling (root-caused and fixed in
modules/gh_storage.py — see that module's docstring for the mechanism).
That failure sat undetected until write volume happened to be high
enough to surface it in logs, and confirming it required a manual
clone-and-inspect cycle outside the running app.

This module turns that manual diagnostic into a real, reusable,
on-demand tool: for every known persisted store, verify it actually
loads correctly, and separately verify the write pipeline itself works
— via a dedicated sentinel file, NEVER by touching real data.

Never runs automatically. Every check here makes real GitHub API calls
— manual/on-demand only, same principle as the Historical Precursor
Engine's manual trigger.
"""
from __future__ import annotations
from datetime import datetime
import json

from modules.gh_storage import load_json_from_github, save_json_to_github

# Every known persisted JSON store in this app, as of this check.
# Deliberately a maintained list, not auto-discovered (e.g. by grepping
# source for "data/*.json" strings) — a store that's silently broken
# might just as easily be silently missing from an auto-discovery pass
# that only looks at working code paths. If a new module adds a new
# data/*.json store, add its path here too.
KNOWN_STORES = [
    "data/alpha_observations.json",
    "data/apex10_baseline.json",
    "data/apex10_feature_history.json",
    "data/discoveries.json",
    "data/short_discoveries.json",
    "data/model_registry.json",
    "data/fundamentals_history.json",
    "data/chart_notes.json",
    "data/checklist_watchlist.json",
    "data/watchlists.json",
    "data/watchlists_backup.json",
    "data/portfolio.json",
    "data/trade_journal.json",
    "data/trade_log.json",
    "data/last_report.json",
    "data/previous_report.json",
    "data/alert_settings.json",
    "data/marketstack_quota.json",
    "data/twelve_data_quota.json",
]

SENTINEL_PATH = "data/_health_check_sentinel.json"

# Rough early-warning size threshold. Since the gh_storage.py fix moved
# to the Git Data API (blobs up to 100MB), this is no longer a hard
# ceiling like the ~1MB Contents API limit that caused the original
# incident — it's a proactive "keep an eye on this" flag. A large file
# is still slower to load, slower to diff in git history, and worth
# knowing about before it becomes a problem the way
# alpha_observations.json already did once.
SIZE_WARNING_BYTES = 3 * 1024 * 1024  # 3MB


def check_store(token: str, repo: str, path: str) -> dict:
    """Checks ONE store: does it load, and how big is it. Read-only —
    never writes anything, safe to run against real data at any time."""
    started = datetime.now()
    try:
        data, sha = load_json_from_github(token, repo, path)
    except Exception as e:
        return {"path": path, "status": "error", "detail": str(e),
               "size_bytes": None, "record_count": None, "checked_at": started.isoformat()}

    if data is None and sha is None:
        return {"path": path, "status": "missing_or_empty",
               "detail": "Not found, or failed to load — check app logs for the specific "
                        "gh_storage warning.",
               "size_bytes": None, "record_count": None, "checked_at": started.isoformat()}

    try:
        size_bytes = len(json.dumps(data).encode("utf-8"))
    except Exception:
        size_bytes = None

    record_count = len(data) if isinstance(data, (list, dict)) else None
    status = "large" if (size_bytes is not None and size_bytes >= SIZE_WARNING_BYTES) else "ok"

    return {
        "path": path, "status": status, "detail": None,
        "size_bytes": size_bytes, "record_count": record_count,
        "checked_at": started.isoformat(),
    }


def check_all_stores(token: str, repo: str, paths: list = None) -> list:
    """Runs check_store() over every known store. One store's failure
    never stops the rest from being checked."""
    paths = paths if paths is not None else KNOWN_STORES
    return [check_store(token, repo, p) for p in paths]


def check_write_pipeline(token: str, repo: str) -> dict:
    """The other half of the diagnostic: proves the FULL write sequence
    (auth, Git Data API blob/tree/commit/ref) actually works end to end
    — without ever touching a real data file. Writes a small,
    timestamped payload to a dedicated sentinel path, reads it back,
    confirms it matches exactly."""
    started = datetime.now()
    payload = {"health_check_run_at": started.isoformat(), "marker": "apexscan-data-health-check"}
    try:
        saved = save_json_to_github(token, repo, SENTINEL_PATH, payload,
                                    message="Data health check — write pipeline verification")
        if not saved:
            return {"status": "write_failed",
                   "detail": "save_json_to_github returned False — see app logs.",
                   "checked_at": started.isoformat()}
        readback, _ = load_json_from_github(token, repo, SENTINEL_PATH)
        if readback != payload:
            return {"status": "roundtrip_mismatch",
                   "detail": f"Wrote {payload!r} but read back {readback!r}.",
                   "checked_at": started.isoformat()}
        return {"status": "ok", "detail": "Write, then read-back, matched exactly.",
               "checked_at": started.isoformat()}
    except Exception as e:
        return {"status": "error", "detail": str(e), "checked_at": started.isoformat()}


def summarize(store_results: list, write_result: dict) -> dict:
    """Rolls everything into one headline verdict, for a simple
    pass/fail-style display. A 'large' flag alone does NOT flip the
    overall status — it's informational, not a failure."""
    problems = [r for r in store_results if r["status"] in ("error", "missing_or_empty")]
    large = [r for r in store_results if r["status"] == "large"]
    overall_ok = (not problems) and write_result.get("status") == "ok"
    return {
        "overall_status": "ok" if overall_ok else "attention_needed",
        "stores_checked": len(store_results),
        "stores_with_problems": len(problems),
        "stores_flagged_large": len(large),
        "write_pipeline_status": write_result.get("status"),
        "problem_paths": [r["path"] for r in problems],
        "large_paths": [r["path"] for r in large],
    }
