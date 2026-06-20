"""
scheduler.py — Automated daily scan scheduler
Run as a background process or via cron.
Usage:
    python scheduler.py                     # Run once immediately
    python scheduler.py --loop              # Loop daily at configured time
    python scheduler.py --time 16:05 --loop # Loop at 4:05 PM (after US close)
"""

import argparse
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

from scanner import load_config, run_scan, save_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/scheduler.log"),
    ],
)
log = logging.getLogger(__name__)


def send_alert(df, top_n: int = 5):
    """
    Simple console alert. Extend to send email / Telegram / Slack here.
    """
    if df.empty:
        return
    top = df.head(top_n)
    log.info("=" * 60)
    log.info(f"  🔥 DAILY ALERT — {datetime.now().strftime('%Y-%m-%d')}")
    log.info("=" * 60)
    for _, row in top.iterrows():
        log.info(
            f"  {row['ticker']:<18} [{row['market']}] "
            f"score={row['apex_score']:.0f}  3m={row['perf_3m_%']:+.1f}%  "
            f"pattern={row['pattern']}"
        )
    log.info("=" * 60)

    breakouts = df[df["breaking_out"] == True]
    if not breakouts.empty:
        log.info(f"  🚨 {len(breakouts)} ACTIVE BREAKOUT(S): {', '.join(breakouts['ticker'].tolist())}")


def run_once(markets=None):
    log.info(f"Starting scan | markets={markets}")
    cfg = load_config("config.yaml")
    df = run_scan(cfg, markets=markets or ["US", "NG"])
    if not df.empty:
        path = save_report(df)
        send_alert(df)
        return path
    else:
        log.warning("Scan returned no results.")
        return None


def next_run_time(target_time_str: str) -> datetime:
    """Calculate next datetime for the given HH:MM time."""
    now = datetime.now()
    h, m = map(int, target_time_str.split(":"))
    candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def loop_scheduler(run_time: str, markets=None):
    log.info(f"Scheduler started | daily run at {run_time} | markets={markets}")
    while True:
        next_dt = next_run_time(run_time)
        wait_secs = (next_dt - datetime.now()).total_seconds()
        log.info(f"Next scan at {next_dt.strftime('%Y-%m-%d %H:%M')} (in {wait_secs/3600:.1f}h)")
        time.sleep(max(0, wait_secs))
        try:
            run_once(markets)
        except Exception as e:
            log.error(f"Scan failed: {e}")
        time.sleep(60)  # prevent double-run at boundary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run on a daily schedule")
    parser.add_argument("--time", default="16:10", help="Time to run daily (HH:MM, 24h)")
    parser.add_argument("--markets", nargs="+", default=["US", "NG"])
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    if args.loop:
        loop_scheduler(args.time, args.markets)
    else:
        run_once(args.markets)
