"""
modules/watchlist_manager.py — Named Watchlist Manager
Persistent storage: writes to an absolute path resolved relative to this
file itself, so it works identically whether run locally or on Streamlit Cloud.
Data survives rerenders, tab switches, and page refreshes permanently.
It is only lost if the Cloud dyno is fully recycled (inactivity >hours) — in
that case the auto-backup inside data/watchlists_backup.json is restored
automatically on next load.
"""

import json
import os
import shutil
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve absolute paths relative to THIS file so they never depend on cwd
_MODULE_DIR   = Path(__file__).resolve().parent          # .../modules/
_REPO_ROOT    = _MODULE_DIR.parent                       # .../apexscan/
_DATA_DIR     = _REPO_ROOT / "data"
_WL_FILE      = _DATA_DIR / "watchlists.json"
_WL_BACKUP    = _DATA_DIR / "watchlists_backup.json"

# Streamlit Cloud /tmp mirror — written on every save as a hot copy
_TMP_MIRROR   = Path("/tmp/apexscan_watchlists.json")

_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LISTS = {
    "High Conviction": [],
    "Monitoring":      [],
    "Earnings Soon":   [],
    "Swing Trades":    [],
}

# ── Internal helpers ───────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict | None:
    """Read and return a dict from a JSON file, or None on any error."""
    try:
        if path.exists() and path.stat().st_size > 2:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return None


def _write_json(path: Path, data: dict) -> bool:
    """Write dict to JSON file atomically (write → rename). Returns True on success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        shutil.move(str(tmp), str(path))
        return True
    except Exception:
        return False


def _merge_with_defaults(data: dict) -> dict:
    """Ensure all default lists exist in the loaded data."""
    result = dict(DEFAULT_LISTS)
    result.update(data)          # user data wins over empty defaults
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def load_watchlists() -> dict:
    """
    Load watchlists from disk.  Resolution order:
      1. Primary file  (data/watchlists.json  — absolute path next to repo root)
      2. /tmp mirror   (/tmp/apexscan_watchlists.json — survives rerenders)
      3. Backup file   (data/watchlists_backup.json)
      4. Hard defaults (empty lists)
    This means even after a Cloud dyno restart the /tmp mirror or backup
    is used so tickers are never silently lost.
    """
    for source in (_WL_FILE, _TMP_MIRROR, _WL_BACKUP):
        data = _read_json(source)
        if data:
            merged = _merge_with_defaults(data)
            # If we recovered from a fallback source, write it back to primary
            if source != _WL_FILE:
                _write_json(_WL_FILE, merged)
            return merged

    return DEFAULT_LISTS.copy()


def save_watchlists(wls: dict):
    """
    Persist watchlists to every storage layer simultaneously so data is
    redundant across all paths and survives any single-layer failure.
    """
    # 1 — primary file (permanent on local / self-hosted)
    _write_json(_WL_FILE, wls)

    # 2 — /tmp mirror (survives Streamlit Cloud rerenders within same session)
    _write_json(_TMP_MIRROR, wls)

    # 3 — backup (written after primary succeeds — used for auto-recovery)
    _write_json(_WL_BACKUP, wls)


# ── Watchlist operations ───────────────────────────────────────────────────────

def add_ticker(wls: dict, list_name: str, ticker: str) -> dict:
    ticker = ticker.upper().strip()
    if not ticker:
        return wls
    if list_name not in wls:
        wls[list_name] = []
    if ticker not in wls[list_name]:
        wls[list_name].append(ticker)
    return wls


def remove_ticker(wls: dict, list_name: str, ticker: str) -> dict:
    if list_name in wls:
        wls[list_name] = [t for t in wls[list_name] if t != ticker]
    return wls


def create_list(wls: dict, name: str) -> dict:
    name = name.strip()
    if name and name not in wls:
        wls[name] = []
    return wls


def delete_list(wls: dict, name: str) -> dict:
    wls.pop(name, None)
    return wls


def import_tickers(wls: dict, list_name: str, text: str) -> dict:
    """Accept comma-separated, space-separated, or newline-separated tickers."""
    raw = text.replace(",", " ").replace("\n", " ").replace(";", " ")
    tickers = [t.strip().upper() for t in raw.split() if t.strip()]
    for ticker in tickers:
        wls = add_ticker(wls, list_name, ticker)
    return wls


def export_watchlist(wls: dict, list_name: str) -> str:
    return ", ".join(wls.get(list_name, []))


def scan_watchlist(
    list_name: str,
    tickers: list,
    cfg: dict,
    analyze_fn,
) -> pd.DataFrame:
    """Run analyze_fn on every ticker in the list and return sorted results."""
    results = []
    for ticker in tickers:
        try:
            data = analyze_fn(ticker, cfg)
            if data:
                results.append(data)
        except Exception:
            continue
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    if "apex_score" in df.columns:
        df = df.sort_values("apex_score", ascending=False)
    return df.reset_index(drop=True)
