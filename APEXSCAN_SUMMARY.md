# ApexScan — Comprehensive Project Summary
## For Platform Migration & Further Development

---

## 1. What ApexScan Is

ApexScan is a **free-tier, self-hosted stock intelligence platform** built in Python and Streamlit. It is a full-featured trading research tool that covers the entire workflow from market screening → stock analysis → risk management → portfolio tracking → automated alerts.

It was built specifically to replicate — at zero cost — the functionality of paid platforms like Trade Ideas ($228/mo), Koyfin ($149/mo), and Finviz Elite ($39/mo), combined into a single interface.

The software runs locally on any Windows, Mac or Linux machine and launches via a web browser (localhost:8501).

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| UI Framework | Streamlit 1.35+ |
| Data (primary) | yfinance (Yahoo Finance wrapper) |
| Data (news) | Finnhub free tier API |
| Insider data | OpenInsider (public SEC scrape) |
| AI Briefing | Anthropic Claude API (claude-sonnet-4-6) |
| Charts | Plotly |
| Config | YAML |
| Storage | JSON files (local) |
| Alerts | Telegram Bot API + Gmail SMTP |
| No database | All data stored in CSV/JSON flat files |

**All data sources are free tier.** No paid APIs required to run the core product.

---

## 3. File Structure

```
stock_scanner/
├── dashboard.py          # Main Streamlit app — 2100+ lines, 16 tabs
├── scanner.py            # Core analysis engine — all technical logic
├── scheduler.py          # Daily automated scan scheduler
├── config.yaml           # All settings, themes, thresholds
├── requirements.txt      # Python dependencies
├── README.md             # Setup documentation
│
├── modules/
│   ├── __init__.py
│   ├── ai_briefing.py    # Claude API integration for morning briefing
│   ├── alerts.py         # Telegram + Email alert system
│   ├── backtester.py     # Historical strategy backtesting engine
│   ├── insider_tracker.py # SEC Form 4 insider trade fetcher
│   ├── options_flow.py   # Options chain unusual activity scanner
│   ├── risk_calc.py      # Position sizing calculator
│   └── watchlist_manager.py # Named watchlist persistence
│
├── data/
│   ├── portfolio.json    # User's stock holdings (auto-created)
│   ├── watchlists.json   # Named watchlists (auto-created)
│   ├── alert_settings.json # Alert config (auto-created)
│   └── briefings/        # Saved AI briefings (auto-created)
│
├── reports/              # Daily scan CSVs (auto-created)
└── logs/                 # scanner.log, scheduler.log (auto-created)
```

---

## 4. Core Scoring System — The Apex Score (0–100)

The Apex Score is the proprietary composite ranking system at the heart of ApexScan. It is calculated for every stock on every scan.

| Component | Max Points | Logic |
|---|---|---|
| 3-Month Price Momentum | 40 | If 3m return > 15%, score = min(40, return%) |
| Relative Strength vs S&P 500 | 25 | RS > 70 = 25pts, RS > 50 = 12pts |
| Stage 2 Uptrend | 15 | Price > MA50 > MA200 = 15pts, price > MA200 only = 7pts |
| Near 52-Week High | 10 | Price ≥ 85% of 52-week high |
| Active Breakout Pattern | 10 | Detected via base/breakout algorithm |
| **Maximum** | **100** | |

**Recommended thresholds:**
- Score ≥ 70 = Strong setup, worth researching
- Score 40–70 = Watchlist candidate
- Score < 40 = Skip

---

## 5. Technical Analysis Logic (scanner.py)

All of this runs in `scanner.py → analyze_stock()`:

### Relative Strength (RS Score)
- Compares stock 3-month return vs S&P 500 (^GSPC) 3-month return
- Formula: `(stock_return / abs(benchmark_return)) × 100`
- Score > 100 = outperforming the market
- Timezone-safe alignment: strips tz from both series before computing
- Falls back to independent lookback if date alignment fails

### Stage Detection (Weinstein-style)
- Stage 2 ✅ = price > MA50 > MA200 (only stage to buy)
- Stage 1 ⏳ = price > MA200, MA50 < MA200 (basing)
- Stage 3 ⚠️ = price < MA200, price > MA50 (topping)
- Stage 4 🔴 = price < MA50 < MA200 (downtrend, avoid)

### Base & Breakout Detection
Uses a rolling 8-week window (40 bars):
- Calculates base depth: (high − low) / high × 100
- Detects range contraction: recent 3-week range vs prior 5-week range
- Volume surge: today's volume > 40-day average × 1.4
- Pattern labels: Flat Base Breakout, Cup Breakout, Handle Forming, Tight Base, Near High (No Vol), Basing, Deep Correction

### Additional Metrics Per Stock
- `perf_1m_%`, `perf_3m_%`, `perf_6m_%` — price returns
- `rs_3m`, `rs_6m` — relative strength at 63 and 126 bar lookbacks
- `adr_%` — Average Daily Range % (volatility proxy, 20-day)
- `vs_50ma_%`, `vs_200ma_%` — % distance from moving averages
- `vol_surge_x` — recent 5-day avg volume / 50-day avg volume
- `near_52wh` — boolean, price ≥ 85% of 52-week high
- `pct_off_high_%` — how far below 52-week high
- `earn_momentum` — proxy: news count + 3m performance

---

## 6. Watchlist & Themes (config.yaml)

**8 US themes, 45 tickers total:**
- `ai_semis`: NVDA, MU, MRVL, ARM, AVGO, TSM, ASML
- `software_agentic`: SNOW, DDOG, CRWD, PANW, NET, ORCL, PLTR
- `cybersecurity`: CRWD, PANW, FTNT, ZS, CYBR
- `space_ev`: RKLB, TSLA, IONQ
- `fintech_payments`: V, MA, SQ, AFRM, SOFI, HOOD
- `biotech_health`: LLY, NVO, MRNA, ABBV, ISRG, HIMS
- `cloud_infra`: MSFT, AMZN, GOOGL, META, CRM
- `consumer_growth`: CELH, LULU, ONON, NKE, DUOL

Tickers are fully configurable in config.yaml. No code changes needed to add/remove stocks.

---

## 7. The 16 Dashboard Tabs

| Tab | Function | Data Source |
|---|---|---|
| 🏆 Leaderboard | Ranked scan results table + bar chart | yfinance scan |
| 📈 Chart Viewer | Candlestick + 50/200 MA + volume | yfinance |
| 🌍 Theme Heatmap | Sector momentum heatmap + bubble chart | yfinance scan |
| 💼 Portfolio Tracker | Live P&L, stop alerts, pie chart | yfinance live |
| 📅 Earnings Calendar | Next earnings dates + urgency flags | yfinance |
| 🔄 Sector Rotation | 11 sector ETF 1W/1M/3M performance | yfinance |
| 🔍 Stock Deep Dive | Full Apex Score for any ticker | yfinance |
| 🎯 Options Flow | Unusual options activity scanner | yfinance options chain |
| 🕵️ Insider Tracker | SEC Form 4 insider buys, cluster detection | OpenInsider |
| 📊 Dividend Calculator | DRIP compounding projection + chart | User input + yfinance yield |
| ⏱ Backtester | Historical strategy backtest + equity curve | yfinance historical |
| ⚖️ Risk Calculator | Position sizing + pyramiding plan | User input |
| 🤖 AI Briefing | Claude-written morning market briefing | Claude API + scan data |
| 📋 Watchlists | Named watchlist manager + per-list scanning | yfinance + local JSON |
| 🔔 Alert Settings | Telegram + Email config + auto-alerts | Telegram API + Gmail SMTP |
| 📖 Guide | In-app documentation | Static |

---

## 8. Alert System (modules/alerts.py)

**Channels:** Telegram Bot API + Gmail SMTP (both simultaneous)

**Alert types that fire automatically:**
1. **Breakout Alert** — fires after every Live Scan if score ≥ threshold
2. **Stop Loss Breach** — fires when portfolio holding drops below 50MA
3. **Earnings Warning** — fires when earnings < 7 days away
4. **Daily Briefing** — optional send of AI briefing to Telegram

**Message format:** Markdown-formatted for Telegram, plain text for email

**Configuration:** Stored in `data/alert_settings.json`

---

## 9. AI Briefing (modules/ai_briefing.py)

**Model:** claude-sonnet-4-6 via Anthropic API
**Endpoint:** `https://api.anthropic.com/v1/messages`
**Max tokens:** 1000
**Trigger:** Manual button in dashboard or post-scan

**What it produces:**
1. Market Pulse — overall tone from scan data
2. Top Setups to Watch — specific actionable commentary per stock
3. Active Breakouts — what to do right now
4. Theme Rotation Insight — which themes have momentum
5. Risk Reminder — specific to today's data

Briefings are saved to `data/briefings/briefing_YYYYMMDD_HHMM.txt`

---

## 10. Backtester (modules/backtester.py)

**Entry signal:** Apex Score ≥ threshold AND Stage 2 confirmed
**Exit signal:** Price closes below 50MA OR max hold days reached
**Metrics returned:** Win rate, avg return, best/worst trade, equity curve, full trade log

Works on single ticker or full watchlist (15 tickers max recommended for speed).

---

## 11. Risk Calculator (modules/risk_calc.py)

**Core formula:** `Shares = (Account × Risk%) ÷ (Entry − Stop)`

**Outputs:**
- Exact share count to buy
- Total position size in $
- Maximum dollar loss if stop is hit
- Reward:Risk ratio (warns if < 2:1)
- Breakeven price after commission
- Pyramiding plan (staged entries as stock rises)

---

## 12. Known Limitations

| Limitation | Impact |
|---|---|
| yfinance data is end-of-day only | No intraday scanning |
| yfinance options chain is limited vs paid sources | Options flow is approximate |
| OpenInsider scraping can break if site changes | Insider tracker may need maintenance |
| Finnhub free tier has no EPS data | Earnings momentum is a proxy only |
| No user authentication | Anyone with the URL can access |
| Flat file storage (JSON/CSV) | Not suitable for multi-user production |
| Claude API requires internet | AI briefing needs active connection |

---

## 13. Upgrade Paths for Next Platform

**Priority upgrades:**
1. **Real-time data** — Replace yfinance with Polygon.io ($29/mo) for live prices
2. **User authentication** — Add login system (Streamlit Auth or FastAPI + JWT)
3. **Database** — Replace JSON/CSV with PostgreSQL or SQLite for multi-user
4. **Real earnings data** — Alpha Vantage or Financial Modeling Prep for actual EPS
5. **Mobile app** — React Native frontend consuming a FastAPI backend
6. **Scheduled AI briefing** — Auto-generate and send every morning at 7AM via scheduler.py
7. **More tickers** — Current watchlist is 45; expand to full S&P 500 with pagination
8. **Paper trading** — Simulate trades from scan results without real money

**Architecture for SaaS version:**
- Backend: FastAPI (Python)
- Frontend: React or Next.js
- Database: PostgreSQL
- Queue: Celery + Redis for scheduled scans
- Hosting: Railway, Render, or AWS
- Auth: Auth0 or Supabase
- Pricing: $20–30/month per user

---

## 14. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
python -m streamlit run dashboard.py

# Run a manual scan
python scanner.py

# Schedule daily scans
python scheduler.py --loop --time 16:10
```

Browser opens at: `http://localhost:8501`

---

*ApexScan — Built with Python, Streamlit, yfinance, Plotly and Claude AI.*
*All data sources free tier. No paid subscriptions required.*
