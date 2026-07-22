"""
modules/alerts.py — ApexScan Alert System
Telegram + Email notifications for breakouts, SFP, flow, VWAP alerts.
"""

import requests
import json
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

log = logging.getLogger(__name__)

SETTINGS_FILE = Path("data/alert_settings.json")


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def load_alert_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    return {
        "alerts_enabled":        False,
        "telegram_token":        "",
        "telegram_chat_id":      "",
        "email_from":            "",
        "email_password":        "",
        "email_to":              "",
        "alert_breakouts":       True,
        "alert_stop_breach":     True,
        "alert_earnings":        True,
        "alert_sfp_setup":       True,
        "alert_persistent_flow": True,
        "alert_vwap_imbalance":  True,
        "min_score_alert":       60,
    }


def save_alert_settings(settings: dict):
    try:
        SETTINGS_FILE.parent.mkdir(exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
    except Exception as e:
        log.warning(f"Could not save alert settings: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram(token: str, chat_id: str, message: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    if not token or not chat_id:
        log.warning("Telegram: token or chat_id missing")
        return False
    try:
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    str(chat_id),
            "text":       message,
            "parse_mode": "Markdown",
        }, timeout=10)
        data = resp.json()
        if data.get("ok"):
            log.info("Telegram message sent successfully")
            return True
        else:
            log.warning(f"Telegram API error: {data}")
            return False
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
        return False


def test_telegram(token: str, chat_id: str) -> bool:
    """Send a test message to verify Telegram is configured correctly."""
    msg = (
        "✅ *ApexScan Alert Test*\n\n"
        "Your Telegram alerts are working correctly\\!\n\n"
        f"_Sent at {datetime.now().strftime('%Y\\-%m\\-%d %H:%M:%S')}_"
    )
    # Try MarkdownV2 first, fall back to plain text
    try:
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    str(chat_id),
            "text":       "✅ ApexScan Alert Test\n\nYour Telegram alerts are working correctly!\n\nSent at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "parse_mode": "Markdown",
        }, timeout=10)
        data = resp.json()
        if data.get("ok"):
            return True
        # If parse error, retry plain text
        if data.get("error_code") == 400:
            resp2 = requests.post(url, json={
                "chat_id": str(chat_id),
                "text": "ApexScan Test: Telegram alerts working! " + datetime.now().strftime('%H:%M:%S'),
            }, timeout=10)
            return resp2.json().get("ok", False)
        log.warning(f"Telegram test error: {data}")
        return False
    except Exception as e:
        log.warning(f"Telegram test failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL
# ══════════════════════════════════════════════════════════════════════════════

def send_email(from_addr: str, password: str, to_addr: str,
               subject: str, body: str) -> bool:
    if not all([from_addr, password, to_addr]):
        return False
    try:
        msg = MIMEMultipart()
        msg["From"]    = from_addr
        msg["To"]      = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_addr, password)
            server.sendmail(from_addr, to_addr, msg.as_string())
        return True
    except Exception as e:
        log.warning(f"Email send failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

def dispatch_alert(settings: dict, message: str,
                   subject: str = "ApexScan Alert") -> dict:
    """Send alert to all configured channels."""
    results = {"telegram": False, "email": False}
    if not settings.get("alerts_enabled"):
        return results

    tok = settings.get("telegram_token", "")
    cid = settings.get("telegram_chat_id", "")
    if tok and cid:
        results["telegram"] = send_telegram(tok, cid, message)

    ef  = settings.get("email_from", "")
    ep  = settings.get("email_password", "")
    et  = settings.get("email_to", "")
    if ef and ep and et:
        plain = message.replace("*","").replace("_","").replace("`","")
        results["email"] = send_email(ef, ep, et, subject, plain)

    return results


def build_daily_briefing_alert(briefing_text: str) -> tuple:
    subject = f"ApexScan Daily Briefing — {datetime.now().strftime('%b %d')}"
    preview = briefing_text[:900] + "…" if len(briefing_text) > 900 else briefing_text
    msg = (
        f"📡 *ApexScan Daily Briefing*\n"
        f"_{datetime.now().strftime('%A, %B %d %Y — %H:%M')}_\n\n"
        f"{preview}"
    )
    return subject, msg


def check_and_fire_alerts(scan_df: pd.DataFrame, portfolio: list,
                           settings: dict, price_fetcher) -> list:
    """Fire alerts after a scan. Returns list of sent messages."""
    if not settings.get("alerts_enabled") or scan_df.empty:
        return []

    sent    = []
    min_s   = settings.get("min_score_alert", 60)
    tok     = settings.get("telegram_token","")
    cid     = settings.get("telegram_chat_id","")
    scores  = pd.to_numeric(scan_df.get("apex_score", pd.Series()), errors="coerce")

    # Breakout alerts
    if settings.get("alert_breakouts") and "breaking_out" in scan_df.columns:
        bos = scan_df[(scan_df["breaking_out"]==True) & (scores >= min_s)]
        for _, row in bos.iterrows():
            msg = (f"🚀 *BREAKOUT — {row['ticker']}*\n"
                   f"Pattern: {row.get('pattern','–')}\n"
                   f"Price: ${row.get('price','–')} | Score: {row.get('apex_score','–')}\n"
                   f"3m Return: {row.get('perf_3m_%','–')}% | RS: {row.get('rs_3m','–')}\n"
                   f"OF: {row.get('of_bias','–')} | VWAP: {row.get('vwap_position','–')}\n"
                   f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_")
            if tok and cid: send_telegram(tok, cid, msg)
            sent.append(msg)

    # SFP alerts
    if settings.get("alert_sfp_setup") and "pa_sfp" in scan_df.columns:
        sfp = scan_df[scan_df["pa_sfp"].notna() & (scan_df["pa_sfp"]!="") & (scores >= min_s-10)]
        for _, row in sfp.iterrows():
            is_bull = "Bullish" in str(row.get("pa_sfp",""))
            msg = (f"{'🎯' if is_bull else '⚠️'} *SFP — {row['ticker']}*\n"
                   f"Type: {row.get('pa_sfp','–')}\n"
                   f"{'Bears trapped — potential reversal UP' if is_bull else 'Bulls trapped — potential reversal DOWN'}\n"
                   f"Score: {row.get('apex_score','–')} | OF: {row.get('of_bias','–')}\n"
                   f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_")
            if tok and cid: send_telegram(tok, cid, msg)
            sent.append(msg)

    # Persistent flow alerts
    if settings.get("alert_persistent_flow") and "of_bias" in scan_df.columns:
        fl = scan_df[scan_df["of_bias"].str.contains("Strong Bullish", na=False) & (scores >= min_s)]
        for _, row in fl.iterrows():
            msg = (f"📈 *PERSISTENT FLOW — {row['ticker']}*\n"
                   f"Bias: {row.get('of_bias','–')} | Up/Down Vol: {row.get('of_up_vol_ratio','–')}x\n"
                   f"Score: {row.get('apex_score','–')}\n"
                   f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_")
            if tok and cid: send_telegram(tok, cid, msg)
            sent.append(msg)

    return sent
