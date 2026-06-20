"""
modules/watchlist_manager.py — Named Watchlist Manager
"""

import json
import pandas as pd
from pathlib import Path

WATCHLIST_FILE = "data/watchlists.json"
Path("data").mkdir(exist_ok=True)

DEFAULT_LISTS = {
    "High Conviction": [],
    "Monitoring": [],
    "Earnings Soon": [],
    "Swing Trades": [],
}


def load_watchlists() -> dict:
    if Path(WATCHLIST_FILE).exists():
        with open(WATCHLIST_FILE) as f:
            return json.load(f)
    return DEFAULT_LISTS.copy()


def save_watchlists(wls: dict):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(wls, f, indent=2)


def add_ticker(wls: dict, list_name: str, ticker: str) -> dict:
    ticker = ticker.upper().strip()
    if list_name not in wls:
        wls[list_name] = []
    if ticker and ticker not in wls[list_name]:
        wls[list_name].append(ticker)
    return wls


def remove_ticker(wls: dict, list_name: str, ticker: str) -> dict:
    if list_name in wls:
        wls[list_name] = [t for t in wls[list_name] if t != ticker]
    return wls


def create_list(wls: dict, name: str) -> dict:
    if name and name not in wls:
        wls[name] = []
    return wls


def delete_list(wls: dict, name: str) -> dict:
    wls.pop(name, None)
    return wls


def scan_watchlist(list_name: str, tickers: list, cfg: dict, analyze_fn) -> pd.DataFrame:
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
    return pd.DataFrame(results).sort_values("apex_score", ascending=False).reset_index(drop=True)


def import_tickers(wls: dict, list_name: str, text: str) -> dict:
    tickers = [t.strip().upper() for t in text.replace(",", " ").replace("\n", " ").split() if t.strip()]
    for ticker in tickers:
        wls = add_ticker(wls, list_name, ticker)
    return wls


def export_watchlist(wls: dict, list_name: str) -> str:
    return ", ".join(wls.get(list_name, []))
