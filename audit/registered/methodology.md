# ApexScan Wider Audit: Methodology (fixed before any results)

This document is written before any price data has been received or any outcome examined. Anything changed afterwards is listed as a dated amendment at the end, never edited in place.

## 1. What the export script review found

| Check | Result |
| --- | --- |
| Benchmark symbols | My first script downloaded ^RUT, IWM, ^RUA, IWV, SPY. ApexScan's relative strength does not use them. It uses ^GSPC, ^R25I (fallback SMMD) and ^RAG (fallback IWZ), from config.yaml. Fixed. A test now compares the script's list with config.yaml |
| yfinance column layout | Since yfinance 0.2.48 a single-ticker download also returns two-level columns. My first script handled only the flat form for a one-ticker batch, and would have logged that ticker as failed without saying why. The new script handles both level orders and the flat form, and refuses to guess when a multi-ticker batch comes back flat |
| Empty or failed tickers | Each is logged, retried one at a time (as the scanner does), and listed in download_log.csv with the reason |
| Missing values | Only rows where open, high, low, close and volume are all missing are dropped (the scanner's own rule). Partly missing rows stay as blanks. Nothing is filled or set to zero |
| Duplicates, gaps, bad values | validate_prices.py checks duplicate dates, missing sessions against the ^GSPC calendar, impossible prices, high below low, daily moves over 50%, stale closes, bars ending early, and writes DATA_QUALITY_REPORT.md |
| Tests | 19 tests on the exporter and validator. 10 deliberate breakages of the code were each caught by at least one test. These use synthetic data; the script has not been run against live Yahoo from this workspace |

## 2. Populations

- **A. The 895 recorded discoveries** (6 Aug to 31 Oct 2026). This is Test B evidence: what ApexScan actually produced. Most have fewer than 63 sessions since discovery, so returns are reported at the horizon available and marked immature.
- **B. Rally-selected stocks.** Chosen from price data alone, never from ApexScan output. A rally is a rise of at least +100% from a closing low to a later closing high within 126 sessions (sensitivity +60% and +150%). Drawn from the full frame only if the optional full-frame download is run; otherwise from the random sample.
- **C. Stratified random sample** from today's listed common stocks: 250 NASDAQ, 250 NYSE, 120 NYSE American. Source: the NASDAQ Trader symbol files. Seed 20261031. ETFs, test issues, warrants, rights, units and preferreds are excluded. CHDN is removed from the sample if drawn.
- **D. Failures** (failed breakouts, stocks that declined after a signal) are not sampled separately. They are the losing outcomes inside C and B, reported in full.

## 3. Periods

- Data from 1 Jan 2019, so the 200-day average is valid from late 2019.
- **Research period:** signals dated 1 Jul 2020 to 30 Jun 2023. Used to check that the implementation behaves, and for stability comparison. No threshold or rule is tuned on it.
- **Validation period:** signals dated 1 Jul 2023 to 31 Jul 2026, run once.
- From 1 Aug 2026 the 63-session horizon is incomplete. That period is reserved for Test B (population A).
- CHDN is excluded from every research and validation calculation. It is reported only as the case study.

## 4. The four entry approaches

Each is evaluated on day t using only bars up to t, with the scanner's own functions from an exact copy of commit ad28693 and a trailing 365-day window, as the scanner sees one year of history. The score gate (at least 20) cannot be replayed because order flow, earnings, weekly and fundamental inputs are not available, so the replay applies the other four gates. This is stated in every table.

| Code | Name | Signal |
| --- | --- | --- |
| E1 | 200-day confirmation (ApexScan as built) | First day of a new unbroken run in which close is above the 200-day average, 3-month return is at least 0%, relative strength is at least -30 against ^GSPC, and volume (larger of today and 20-day average) is at least 100,000 |
| E2 | Early recovery | Same gates, except the close is above the 50-day average and at or below the 200-day average. This uses the 50-day stage test ApexScan already applies to NGX and gems. First day of a new run |
| E3 | Breakout | E1 gates pass and the scanner's detect_base_breakout is true. First such day per episode |
| E4 | Breakout retest | After an E3 day b, level L = highest High of the 40 bars before b. Signal on the first day within 15 sessions of b when Low is at most 1.01 x L and Close is at least L, with E1 gates passing. Cancelled if any close falls below 0.97 x L first |

E4 is a research variant, not an ApexScan rule. For stocks where E2 fires and E1 follows within 60 sessions, a paired comparison measures the E2 entry against the E1 entry to the same exit date.

## 5. Execution, costs and outcomes

- **Entry:** the open of the session after the signal. **Exit:** the open 63 sessions after entry (sensitivity: 21 and 126). No stop in the primary test.
- **Costs per side**, by 20-day median dollar volume at signal: 0.10% at $20M or more, 0.30% from $2M to $20M, 0.75% below $2M. This covers spread and slippage. Sensitivity at half and double. Round trip counts both sides.
- **One signal per ticker per approach** until the horizon has passed, so overlapping trades are not counted as separate evidence.
- **Win:** net return above 0. **Severe failure:** net return of -15% or worse. **Substantial winner:** highest High within the horizon at least +30% above the entry price (sensitivity +20% and +50%). **Failed breakout** (E3): close back below L within 10 sessions of the breakout day.
- **Reported:** number of signals, win rate, mean and median net return, average win and loss, profit factor, maximum favourable and adverse excursion (from High and Low), results by exchange, by stage, by market regime (^GSPC above or below its 200-day average), and by research versus validation period.
- **Controls:** buy-and-hold ^GSPC over the same entry and exit dates, and a random-entry control (1,000 random entry dates per ticker in the same period). A signal must beat both to count as informative.
- **Uncertainty:** bootstrap confidence intervals resampling by ticker, and separately by calendar month.
- **Missed-winner metrics (B):** share of rallies where E1 qualified before the stock had risen 25% from the rally's low, after that point, or never; median rise before qualification; median delay in sessions.
- **Score bands:** the Apex Score cannot be replayed, so score bands are reported only from saved reports (population A), with the 11 reports that have zero relative strength excluded or shown separately.

## 6. Decision rule

A change to ApexScan is recommended only if the variant, compared with E1: (a) has a higher net mean return and profit factor in both periods, (b) has a validation-period 95% interval for the difference that excludes zero, (c) does not raise the share of losing trades by more than 5 percentage points, and (d) holds on at least two of the three exchanges. Otherwise the conclusion is that the evidence does not support a change.

## 7. Limitations known in advance

- **Survivorship:** the frame is today's listings. Delisted and bankrupt stocks are absent, which flatters every long strategy and probably flatters early-recovery entries most. No reliable free source of delisted prices has been found. I will not claim delisted coverage.
- **Universe mismatch:** ApexScan scans index members (S&P 500, NASDAQ-100, S&P 400, Russell 2000) and there is no historical membership. The sample includes smaller stocks outside those indexes. A secondary result restricts to price at least $5 and median dollar volume at least $2M.
- **No rejected-stock records:** the saved reports contain only stocks that passed. Which stocks the live app rejected, and why, is unknown.
- **No historical fundamentals, earnings or order-flow inputs**, so the score is not replayed.
- **Test A is a replay with daily scanning.** The live app was scanned by hand, so scan timing is measured separately in Test B.
- **Adjusted prices** are adjusted for later splits and dividends, which slightly changes historical closes against what was visible on the day.
- **NGX is outside this sample.** No verified price source for it has been found, so no NGX conclusion will be drawn.
- Yahoo data quality is unaudited until the validator has been run on the real download.

## 8. Status of deliverables

| Deliverable | Status |
| --- | --- |
| Validated export script and validator | Done and tested on synthetic data. Needs one run on your computer |
| Data-quality report | Generated by the validator after the download |
| CHDN reconstruction | Replay engine built and tested (8 tests, independent oracle for the 200-day date). Waiting for prices |
| Wider-sample methodology | This document |
| Entry-strategy comparison | After the download. Run once on validation |
| Confidence assessment and recommendations | After results |
