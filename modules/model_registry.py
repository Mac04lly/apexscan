"""
modules/model_registry.py — Model & Strategy Version Registry (V9 Phase 1)

Every AlphaObservation is tagged with the model/strategy version active
at the moment it was created. If scoring logic changes later, bump the
version here — old observations keep their original tag and are never
retroactively reassigned, so historical evidence is never corrupted by
a later change in how the score itself is computed.
"""

# Bump these whenever scoring/strategy logic actually changes.
# Never edit history to match a new version — old observations keep
# whatever version was active when they were created.
MODEL_VERSION = "APEX-9.0"

STRATEGY_VERSIONS = {
    "swing":     "SWING-2.0",
    "position":  "POSITION-1.0",
    "long_term": "LONGTERM-1.1",  # bumped for the smooth-scoring fix
    "dividend":  "DIVIDEND-1.0",
    "value":     "VALUE-1.0",
}

def get_model_version() -> str:
    return MODEL_VERSION

def get_strategy_version(strategy_name: str) -> str:
    return STRATEGY_VERSIONS.get((strategy_name or "swing").lower(), "UNKNOWN-1.0")


# ══════════════════════════════════════════════════════════════════════════
# V9 PHASE 8 — MODEL GOVERNANCE
#
# Everything above this line (MODEL_VERSION, STRATEGY_VERSIONS) is the
# actual production-activation mechanism, unchanged: a human edits those
# constants and redeploys, exactly as before. What follows is a
# PERSISTED AUDIT LEDGER + WORKFLOW GATE layered on top of that — it
# never changes what those constants return, and it never edits
# scanner.py/strategies.py itself. Its entire job is to make the
# Phase 8 promotion pipeline real and enforced:
#
#     Research Finding -> Human review -> Model proposal ->
#     Out-of-sample test -> Approval -> new version
#
# A research version can be proposed and walk-forward-tested here, but
# approve_promotion() only records that a named human signed off after
# seeing a passing walk-forward result — it deliberately CANNOT make a
# version live. Activating it still means a person edits MODEL_VERSION
# above and redeploys. That split is the whole point: "a model can
# improve without corrupting historical evidence" requires that no
# automated process can ever silently change what score existing (or
# future) observations get tagged with.
# ══════════════════════════════════════════════════════════════════════════
from datetime import datetime as _datetime
from modules.gh_storage import load_json_from_github as _load_gh, save_json_to_github as _save_gh

STATUS_PRODUCTION = "production"
STATUS_RESEARCH = "research"
STATUS_APPROVED_PENDING_ACTIVATION = "approved_pending_activation"
STATUS_REJECTED = "rejected"
STATUS_DEPRECATED = "deprecated"

REGISTRY_PATH = "data/model_registry.json"


def _get_gh_creds():
    """Same pattern as modules/alpha_validation.py — reuses whatever
    GitHub credentials are already configured for every other persistent
    store in this app."""
    try:
        import streamlit as st
        return st.secrets.get("github_token", ""), st.secrets.get("github_repo", "")
    except Exception:
        return "", ""


def load_registry() -> dict:
    token, repo = _get_gh_creds()
    if token and repo:
        data, _ = _load_gh(token, repo, REGISTRY_PATH)
        if isinstance(data, dict) and "versions" in data:
            return data
    return {"versions": {}, "history": []}


def save_registry(registry: dict, message: str = None):
    token, repo = _get_gh_creds()
    if token and repo:
        _save_gh(token, repo, REGISTRY_PATH, registry, message=message or "Update model_registry.json")


def _log_event(registry: dict, event: str, version_id: str, by: str = None, extra: dict = None):
    entry = {"event": event, "version_id": version_id, "at": _datetime.now().isoformat()}
    if by:
        entry["by"] = by
    if extra:
        entry.update(extra)
    registry["history"].append(entry)


def ensure_production_entry_exists(registry: dict = None) -> dict:
    """Backfills a registry record for the CURRENT live MODEL_VERSION if
    one doesn't exist yet — handles the case where governance tracking
    starts after production scoring already existed, without pretending
    to know exactly when that version first shipped."""
    registry = registry if registry is not None else load_registry()
    if MODEL_VERSION not in registry["versions"]:
        registry["versions"][MODEL_VERSION] = {
            "version_id": MODEL_VERSION, "status": STATUS_PRODUCTION,
            "description": "Production scoring (backfilled registry entry — created when "
                           "governance tracking began, not necessarily when this version "
                           "actually shipped).",
            "created_at": _datetime.now().isoformat(), "promoted_at": _datetime.now().isoformat(),
            "promoted_from": None, "source_finding": None, "proposed_by": None,
            "approved_by": None, "walk_forward_result": None,
        }
        _log_event(registry, "backfilled_production_entry", MODEL_VERSION)
        save_registry(registry)
    return registry


def propose_research_model(version_id: str, description: str, source_finding: str = None,
                           proposed_by: str = None) -> dict:
    """Registers a new RESEARCH-status model version. Purely a record —
    does not touch production scoring in any way."""
    if not version_id or not description:
        raise ValueError("version_id and description are both required.")
    registry = ensure_production_entry_exists()
    if version_id in registry["versions"]:
        raise ValueError(f"Version {version_id!r} already exists in the registry.")
    record = {
        "version_id": version_id, "status": STATUS_RESEARCH, "description": description,
        "created_at": _datetime.now().isoformat(), "promoted_at": None,
        "promoted_from": MODEL_VERSION, "source_finding": source_finding,
        "proposed_by": proposed_by, "approved_by": None, "walk_forward_result": None,
    }
    registry["versions"][version_id] = record
    _log_event(registry, "proposed", version_id, by=proposed_by,
              extra={"source_finding": source_finding})
    save_registry(registry, message=f"Propose research model {version_id}")
    return record


def record_walk_forward_result(version_id: str, result: dict) -> dict:
    """Attaches a walk-forward validation result (from
    modules/walk_forward.py) to a research version — required before
    that version can be approved."""
    registry = load_registry()
    if version_id not in registry["versions"]:
        raise ValueError(f"Unknown version_id: {version_id!r}")
    registry["versions"][version_id]["walk_forward_result"] = result
    _log_event(registry, "walk_forward_recorded", version_id,
              extra={"passed": result.get("passed")})
    save_registry(registry, message=f"Record walk-forward result for {version_id}")
    return registry["versions"][version_id]


def approve_promotion(version_id: str, approved_by: str) -> dict:
    """Marks a research version as APPROVED FOR ACTIVATION. Requires an
    explicit named human approver — never auto-approved — and requires a
    passing walk-forward result already on file. This still does NOT
    change MODEL_VERSION or make anything live; see module docstring."""
    if not approved_by or not str(approved_by).strip():
        raise ValueError("approve_promotion requires an explicit approved_by name — "
                         "nothing here can approve itself.")
    registry = load_registry()
    if version_id not in registry["versions"]:
        raise ValueError(f"Unknown version_id: {version_id!r}")
    record = registry["versions"][version_id]
    if record["status"] != STATUS_RESEARCH:
        raise ValueError(f"Only a 'research' status version can be approved "
                         f"(current status: {record['status']!r}).")
    wf = record.get("walk_forward_result")
    if not wf or not wf.get("passed"):
        raise ValueError("Cannot approve promotion without a passing walk-forward result on "
                         "file for this version. Run walk-forward validation first.")
    record["status"] = STATUS_APPROVED_PENDING_ACTIVATION
    record["approved_by"] = approved_by
    record["approved_at"] = _datetime.now().isoformat()
    _log_event(registry, "approved", version_id, by=approved_by)
    save_registry(registry, message=f"Approve promotion of {version_id} (pending manual activation)")
    return record


def reject_proposal(version_id: str, reason: str, rejected_by: str) -> dict:
    if not reason or not rejected_by:
        raise ValueError("reject_proposal requires both a reason and a rejected_by name.")
    registry = load_registry()
    if version_id not in registry["versions"]:
        raise ValueError(f"Unknown version_id: {version_id!r}")
    record = registry["versions"][version_id]
    record["status"] = STATUS_REJECTED
    record["rejected_by"] = rejected_by
    record["rejected_reason"] = reason
    record["rejected_at"] = _datetime.now().isoformat()
    _log_event(registry, "rejected", version_id, by=rejected_by, extra={"reason": reason})
    save_registry(registry, message=f"Reject proposal {version_id}")
    return record


def get_registry_table() -> list:
    """Flat, display-ready list of every version on record, oldest first."""
    registry = ensure_production_entry_exists()
    return sorted(registry["versions"].values(), key=lambda r: r.get("created_at") or "")


def compare_models(version_a: str, version_b: str, observations: list, horizon: str = "20D") -> dict:
    """Compares realized Alpha Metrics between two model_versions, using
    only observations actually tagged with each version — never a
    simulated or retroactively-relabeled comparison. Either side may
    legitimately be n=0 if that version hasn't accumulated observations
    yet; that's reported honestly, not estimated."""
    from modules.alpha_metrics import compute_alpha_metrics
    obs_a = [o for o in (observations or []) if o.get("model_version") == version_a]
    obs_b = [o for o in (observations or []) if o.get("model_version") == version_b]
    return {
        "version_a": version_a, "version_b": version_b,
        "metrics_a": compute_alpha_metrics(obs_a, horizon),
        "metrics_b": compute_alpha_metrics(obs_b, horizon),
    }
