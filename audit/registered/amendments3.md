# Amendment 3 - Discovery-date provenance, Population A cohorts, and dependence reporting

Dated 3 November 2026. Written and approved before any price data from the live download has been examined. The registered
methodology (31 Oct 2026), Amendment 1 and Amendment 2 are unchanged and keep their recorded hashes. This amendment adds
definitions and reporting rules only. It changes no entry definition, threshold, sample size, research or validation period,
cost assumption, statistical test, or decision rule. It does not touch ApexScan's scanner, dashboard, strategies, scoring,
filters or configuration.

## 1. What was found (evidence from the repository's Git history, 3 Nov 2026)

- `data/discoveries.json` first appears in Git on 6 Aug 2026. At 10:51 it held 63 tickers and at 11:05 it held 605; all 63
  are among the 605. All 605 carry `discovered_at = 2026-08-06` and every one of their stored discovery prices is unchanged
  in the current file.
- ApexScan logs a ticker only if it is not already in the log, and stamps the date the log first saw it. On an empty log every
  stock that passes the scan is therefore "new". The 605 are the stocks that were already passing when the log began; they
  are not 605 qualifications that happened on 6 Aug.
- `gh_storage.py` (GitHub persistence) was created on 4 Aug and `dashboard.py` was edited at 10:45 on 6 Aug. The likeliest
  explanation is that the log first survived on that date. This is an inference; Git history alone does not prove it.
- The repository holds no scan data before 6 Aug. Saved scan reports start on 12 Aug (`last_report`), 23 Aug
  (`alpha_observations`) and 30 Aug (`previous_report`). No saved report or Git record can establish when any of the 605
  first qualified.
- There is no continuous record of scans, so a stock first recorded after 6 Aug may also have qualified earlier.

## 2. Cohort names (used in every table, chart and sentence)

- **A1 - Initial persistence/seeding cohort.** The records whose recorded discovery date is 2026-08-06 (605 at 30 Oct 2026).
- **A2 - Post-seeding first-recorded discovery cohort.** All other records (290 at 30 Oct 2026).

A2 is not described as confirmed genuine first-time qualifications. Some A2 records may have qualified before their recorded
date. The cohort rule is by date only, stated here so it cannot be adjusted after seeing outcomes.

## 3. Three separate date concepts (never interchanged)

| Concept | Meaning | Available for |
| --- | --- | --- |
| Recorded discovery date | `discovered_at` in the discovery log: the day the log first saw the ticker | All records |
| Reconstructed four-gate date | The first date of the unbroken run, containing the recorded date, in which the four price-based gates replayed from Yahoo history all passed (close above 200-day average; 3-month return at least 0%; relative strength at least -30; volume at least 100,000) | Records with price data |
| Actual full scanner qualification date | The date ApexScan itself first listed the stock, including the Apex Score of at least 20 | Not established for any record |

The reconstructed four-gate date is a **lower bound**: the four replayed gates are necessary but not sufficient, and the
historical Apex Score cannot be replayed, so full qualification cannot be earlier than this date and may be later. It is not
an ApexScan discovery date and is never described as one. If the run reaches back to the start of the available price history
it is flagged as left-censored (the true start may be earlier). If the four gates were not all met on the recorded date, that
is reported as such and no date is assigned.

## 4. How the dates are used

- Entry for any Population A calculation remains the open of the first session strictly after the recorded discovery date
  (Amendment 2, item 2). For A1 this is 7 Aug 2026 for every record. That is an executable entry date for a hypothetical
  purchase; it says nothing about how early ApexScan found the stock.
- The registered timing comparison between entry approaches (E1 to E4) uses replayed signal dates from price data and does
  not use `discovered_at`. It is unaffected by this issue.
- Claims about discovery timing use A2 only, and say they rest on first-recorded dates.

## 5. Reporting rules

1. A1 and A2 are reported separately. No pooled headline figure is produced. The 605 A1 records share one entry date and are
   one market episode, not 605 independent observations.
2. Stock-level outcomes are kept. Beside them, every table reports: unique tickers, discovery events (records), distinct
   recorded discovery dates, distinct entry dates, the largest share of records on one date, the largest number of holding
   windows open at the same session, the number of overlapping pairs of holding windows, and the number of non-overlapping
   trades per ticker.
3. Maturity rules are unchanged. Matured and immature observations are reported separately.
4. The limitation is stated wherever Population A appears: the original qualification dates of A1 cannot be established from
   saved reports or Git history.

## 6. How non-overlapping (independent) trades are counted

For one arm and one holding horizon H, taking each ticker separately:
1. Order that ticker's signals by entry session.
2. Accept the first signal. Its holding window runs from its entry session to the session H sessions later, inclusive.
3. Accept a later signal only if its entry session is strictly after the exit session of the last accepted trade. Otherwise
   it is skipped (and not counted as independent).
4. The count of accepted trades is the ticker's non-overlapping trade count. The sum over tickers is the arm's independent
   trade count, reported next to the raw signal count.
Two different tickers are never merged by this rule. Cross-ticker dependence (many stocks entering on the same dates) is shown
by the distinct entry dates and simultaneous-open-window figures in item 5.2.

The independent-trade count does not change the registered ticker-cluster bootstrap, the Holm correction, the evidence
classes or the decision rule. It is reported so that readers can see how many independent observations sit behind each figure.

## 7. Calendar-month analysis

Results are also shown by calendar month of entry, as a description only: count of trades, mean and median net return. It is
not used in any test, interval, evidence class or recommendation, and it cannot change the registered decision rule.

## 8. Limitations carried forward

- The original qualification dates of the 605 A1 records cannot be established. The reconstructed four-gate date is a lower
  bound, not a substitute.
- No Population A record has completed the registered 63-session horizon at 30 Oct 2026 (maximum 60). Any 21-session look is
  descriptive only and rests largely on one market episode.
