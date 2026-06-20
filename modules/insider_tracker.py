"""
modules/insider_tracker.py — SEC Form 4 Insider Trade Tracker via OpenInsider
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup


def fetch_insider_trades(ticker: str = None, days_back: int = 30, trade_type: str = "P") -> pd.DataFrame:
    try:
        url = "http://openinsider.com/screener"
        params = {
            "s":    ticker or "",
            "o":    "",
            "pl":   "",
            "ph":   "",
            "ll":   "",
            "lh":   "",
            "fd":   days_back,
            "fdr":  "",
            "td":   0,
            "tdr":  "",
            "fdlyl":"",
            "fdlyh":"",
            "daysago": days_back,
            "xp":   1 if trade_type == "P" else 0,
            "xs":   0,
            "vl":   "",
            "vh":   "",
            "ocl":  "",
            "och":  "",
            "sic1": -1,
            "sicl": 100,
            "sich": 9999,
            "isofficer": 1,
            "iscob":     1,
            "isdirector":1,
            "istenpercent":1,
            "isother":   1,
            "grp":  0,
            "nfl":  "",
            "nfh":  "",
            "nil":  "",
            "nih":  "",
            "nol":  "",
            "noh":  "",
            "v2l":  "",
            "v2h":  "",
            "oc2l": "",
            "oc2h": "",
            "sortcol": 1,
            "cnt":  100,
            "page": 1,
        }
        resp = requests.get(url, params=params, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "tinytable"})
        if not table:
            return pd.DataFrame()

        rows = []
        headers = [th.text.strip() for th in table.find_all("th")]
        for tr in table.find_all("tr")[1:]:
            cells = [td.text.strip() for td in tr.find_all("td")]
            if cells:
                rows.append(dict(zip(headers, cells)))

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Rename columns for clarity
        col_map = {
            "X": "trade_type", "Filing Date": "filing_date",
            "Trade Date": "trade_date", "Ticker": "ticker",
            "Company Name": "company", "Insider Name": "insider",
            "Title": "title", "Trade Type": "trade_type",
            "Price": "price", "Qty": "shares", "Owned": "owned",
            "ΔOwn": "delta_own", "Value": "value",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        return df

    except Exception as e:
        return pd.DataFrame()


def get_insider_summary(tickers: list, days_back: int = 30) -> pd.DataFrame:
    all_rows = []
    for ticker in tickers[:20]:
        try:
            df = fetch_insider_trades(ticker=ticker, days_back=days_back, trade_type="P")
            if not df.empty:
                count = len(df)
                all_rows.append({
                    "ticker":        ticker,
                    "insider_count": count,
                    "cluster_buy":   count >= 2,
                    "total_$":       df["value"].str.replace("[$,+]", "", regex=True).astype(float).sum() if "value" in df.columns else 0,
                    "last_filing":   df["filing_date"].iloc[0] if "filing_date" in df.columns else "–",
                })
        except Exception:
            continue

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows).sort_values("insider_count", ascending=False)
