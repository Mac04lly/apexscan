"""
modules/alerts.py — Telegram + Email Alert System
"""

import json
import requests
import smtplib
import pandas as pd
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

ALERT_SETTINGS_FILE = "data/alert_settings.json"
Path("data").mkdir(exist_ok=True)


def load_alert_settings() -> dict:
    if Path(ALERT_SETTINGS_FILE).exists():
        with open(ALERT_SETTINGS_FILE) as f:
            return json.load(f)
    return {
        "telegram_token": "",
        "telegram_chat_id": "",
        "email_from": "",
        "email_password": "",
        "email_to": "",
        "alerts_enabled": False,
        "alert_breakouts": True,
        "alert_stop_breach": True,
        "alert_earnings": True,
        "alert_sfp_setup": True,
        "alert_persistent_flow": True,
        "alert_vwap_imbalance": True,
        "min_score_alert": 60,
    }


def save_alert_settings(settings: dict):
    with open(ALERT_SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def test_telegram(token: str, chat_id: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": "✅ ApexScan Telegram alert test — connection successful!",
            "parse_mode": "HTML",
        }, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def send_telegram(token: str, chat_id: str, message: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def send_email(from_addr: str, password: str, to_addr: str, subject: str, body: str) -> bool:
    try:
        msg = MIMEMultipart()
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_addr, password)
            server.sendmail(from_addr, to_addr, msg.as_string())
        return True
    except Exception:
        return False


def dispatch_alert(settings: dict, message: str, subject: str = "ApexScan Alert") -> dict:
    results = {"telegram": False, "email": False}
    if not settings.get("alerts_enabled"):
        return results
    if settings.get("telegram_token") and settings.get("telegram_chat_id"):
        results["telegram"] = send_telegram(
            settings["telegram_token"], settings["telegram_chat_id"], message)
    if settings.get("email_from") and settings.get("email_password") and settings.get("email_to"):
        results["email"] = send_email(
            settings["email_from"], settings["email_password"],
            settings["email_to"], subject, message)
    return results


def check_and_fire_alerts(df: pd.DataFrame, portfolio: list, settings: dict, fetch_price_fn) -> list:
    if not settings.get("alerts_enabled"):
        return []
    fired = []
    min_score = settings.get("min_score_alert", 60)

    # Breakout alerts
    if settings.get("alert_breakouts") and "breaking_out" in df.columns:
        breakouts = df[(df["breaking_out"] == True) &
                       (pd.to_numeric(df["apex_score"], errors="coerce") >= min_score)]
        for _, row in breakouts.iterrows():
            msg = (f"🚀 <b>BREAKOUT ALERT</b>\n"
                   f"<b>{row['ticker']}</b> — Score: {row.get('apex_score','?'):.0f}\n"
                   f"Pattern: {row.get('pattern','?')}\n"
                   f"Stage: {row.get('stage','?')} | OF: {row.get('of_bias','?')}")
            dispatch_alert(settings, msg, f"Breakout: {row['ticker']}")
            fired.append(("breakout", row["ticker"]))

    # SFP alerts
    if settings.get("alert_sfp_setup") and "pa_sfp" in df.columns:
        sfps = df[(df["pa_sfp"] == "Bullish SFP") &
                  (pd.to_numeric(df["apex_score"], errors="coerce") >= min_score)]
        for _, row in sfps.iterrows():
            msg = (f"🎯 <b>BULLISH SFP SETUP</b>\n"
                   f"<b>{row['ticker']}</b> — Bear trap detected\n"
                   f"Score: {row.get('apex_score','?'):.0f} | {row.get('stage','?')}")
            dispatch_alert(settings, msg, f"SFP: {row['ticker']}")
            fired.append(("sfp", row["ticker"]))

    # Persistent flow alerts
    if settings.get("alert_persistent_flow") and "of_bias" in df.columns:
        strong = df[(df["of_bias"] == "Strong Bullish") &
                    (pd.to_numeric(df["apex_score"], errors="coerce") >= min_score)]
        for _, row in strong.iterrows():
            msg = (f"📈 <b>STRONG BULLISH FLOW</b>\n"
                   f"<b>{row['ticker']}</b> — Institutional accumulation detected\n"
                   f"Up/Down Vol: {row.get('of_up_vol_ratio','?')}x | Score: {row.get('apex_score','?'):.0f}")
            dispatch_alert(settings, msg, f"Flow: {row['ticker']}")
            fired.append(("flow", row["ticker"]))

    # Portfolio stop breach alerts
    if settings.get("alert_stop_breach") and portfolio:
        for holding in portfolio:
            try:
                tk = holding["ticker"]
                cost = holding["buy_price"]
                live = fetch_price_fn(tk)
                if live and live.get("price") and live.get("ma50"):
                    if live["price"] < live["ma50"]:
                        msg = (f"💼 <b>STOP BREACH</b>\n"
                               f"<b>{tk}</b> dropped below 50MA\n"
                               f"Price: ${live['price']} | 50MA: ${live['ma50']}\n"
                               f"Buy price: ${cost}")
                        dispatch_alert(settings, msg, f"Stop Breach: {tk}")
                        fired.append(("stop", tk))
            except Exception:
                continue

    return fired


def build_daily_briefing_alert(briefing_text: str) -> str:
    lines = briefing_text.split("\n")[:15]
    return "📡 <b>ApexScan Daily Briefing</b>\n\n" + "\n".join(lines)

