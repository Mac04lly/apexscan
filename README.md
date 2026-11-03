# 📡 ApexScan — US & Nigeria (NGX) Stock Scanner

A production-grade, **100% free-tier** stock scanner and dashboard built on momentum principles, Stage Analysis, and theme rotation — for both US and Nigerian (NGX) markets.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Add your free Finnhub key in config.yaml
#    Get one at https://finnhub.io — improves US news signals

# 3. Run a scan from the CLI
python scanner.py --markets US NG

# 4. Launch the dashboard
streamlit run dashboard.py

# 5. Schedule daily automated scans
python scheduler.py --loop --time 16:10 --markets US NG
```

---

## 📁 Project Structure

```
stock_scanner/
├── config.yaml          # All themes, thresholds, market settings
├── scanner.py           # Core scan engine (analysis + scoring)
├── dashboard.py         # Streamlit visual dashboard
├── scheduler.py         # Daily automated scan scheduler
├── requirements.txt
├── reports/             # Auto-saved CSV reports (scan_YYYYMMDD_HHMM.csv)
└── logs/                # scanner.log, scheduler.log
```

---

## 🏆 The Apex Score (0–100)

| Signal | Points | Method |
|---|---|---|
| 3m performance > threshold | 0–40 | Price return vs N bars ago |
| RS vs benchmark > min | 0–25 | Stock 3m / Benchmark 3m × 100 |
| Stage 2 setup | 0–15 | Price > 200MA **and** 50MA > 200MA |
| Near 52-week high | 10 | Price ≥ 85% (US) / 80% (NG) of 52w high |
| Active breakout pattern | 10 | Tight base + price near high + volume surge |

---

## 🌍 Market Support

### United States (US)
- **Benchmark:** S&P 500 (`^GSPC`)
- **Data:** yfinance (price history) + Finnhub free tier (company news)
- **Volume filter:** ≥ 1,000,000 shares/day
- **Thresholds:** 3m return > 15%, RS > 70

### Nigeria (NGX)
- **Benchmark:** NGX All-Share Index (`^NGSE`)
- **Data:** yfinance only — Finnhub does not cover NGX tickers
- **Volume filter:** ≥ 50,000 (NGX is a much thinner market)
- **Thresholds:** 3m return > 10%, RS > 60
- **Ticker format:** Append `.NG` suffix (e.g., `GTCO.NG`, `DANGCEM.NG`)

---

## ⚠️ Known Data Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| `^NGSE` on yfinance is sparse / missing | NG RS scores may return 0 | Monitor trends within NG universe itself |
| Many NGX tickers have thin or missing yfinance history | Ticker skipped if < 100 bars | Lower `min_history_bars` in config.yaml |
| Finnhub free tier: no EPS data | Earnings momentum is a **proxy** only | Upgrade to paid Finnhub/Polygon for real EPS |
| yfinance data may lag 24h for some NG tickers | Use for swing/position trading, not scalping | — |
| No intraday data | Analysis is end-of-day only | — |

---

## 🔧 Configuration (`config.yaml`)

```yaml
# Add/remove tickers from any theme list
us_themes:
  ai_semis: [NVDA, MU, ARM, AVGO]
  cybersecurity: [CRWD, PANW, FTNT]

ng_themes:
  banking: [GTCO.NG, ZENITHBANK.NG, UBA.NG]
  oil_gas: [SEPLAT.NG, OANDO.NG]

thresholds:
  us:
    min_3m_perf: 15      # Lower for more results, raise for strict filter
    min_volume: 1000000
  ng:
    min_3m_perf: 10
    min_volume: 50000
```

---

## 📬 Alerts & Automation

The scheduler prints alerts to the console and log. To extend to real notifications:

```python
# In scheduler.py → send_alert()
# Add email:
import smtplib
# Add Telegram:
requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", ...)
# Add Slack:
requests.post(SLACK_WEBHOOK_URL, json={"text": message})
```

---

## 📊 Pattern Detection

| Pattern | Condition |
|---|---|
| **Flat Base Breakout** | Base depth < 15%, price near high, volume surge |
| **Cup Breakout** | Base depth 15–35%, price near high, volume surge |
| **Near High (No Vol)** | Price near high, volume not surging yet |
| **Basing (N% deep)** | Consolidation — watch for narrowing range |

---

## ⚖️ Disclaimer

This tool is for **research and educational purposes only**. It does not constitute financial advice. Past performance does not guarantee future results. Always do your own due diligence before investing.
