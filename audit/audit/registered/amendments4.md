# Amendment 4 - Treatment of data-quality findings and research-period run rules

Dated 4 November 2026. Written after the data-quality review of the real download (flags, counts and the coverage of tickers only) and
BEFORE any trade, signal, return or outcome statistic was computed from it. The registered methodology and Amendments 1 to 3 are unchanged and keep
their hashes. This amendment changes no entry definition, threshold, sample size, research or validation period, cost tier, statistical test or
decision rule. It records how the data-quality findings are handled and fixes implementation details that the registered text left open. It
changes nothing in ApexScan.

## 1. What the data-quality review found (counts only, no outcomes)
1,410 of 1,421 requested tickers returned data (2,461,137 rows). All 620 registered-sample stocks returned data. 11 tickers returned nothing after a
retry (APGE, AVB, DOMO, EQR, KVUE, LPRO, WBD, WBS and the benchmarks ^R25I, ^RAG, IWZ). The Russell 3000 Growth benchmark role is unavailable
(neither ^RAG nor IWZ); the four replayed gates use ^GSPC only, so they are unaffected. 185 one-day moves above 100% occur in 104 tickers (94 in
the registered sample), consistent with different securities' histories joined under one ticker or unadjusted splits. PSKY has 15 rows.

## 2. Decisions (approved by the owner on 4 Nov 2026 as the recommended handling)
1. **Large moves.** No price is removed, corrected or filled. The primary results use every trade that the registered rules allow. A trade is
   **affected** if, for its ticker, any one-session close-to-close move above 100% (in either direction) occurs in the 200 sessions up to and
   including the signal day, or from the entry session through the exit session. Every results table is also shown **without affected trades** as
   a disclosed sensitivity. The count and share of affected trades is reported for every approach, and a gap of more than 5 percentage points
   between approaches is flagged. The decision rule is evaluated on the primary (all-trades) results; a conclusion that holds only without
   affected trades, or only with them, is labelled as dependent on this treatment.
2. **The 11 tickers with no data** are accepted as unavailable. They are not retried and not replaced. They are recorded by ticker and reason.
3. **CHDN** is excluded from every research and validation calculation (Amendment 1). It is in the discovery log (recorded 31 Oct 2026, cohort A2),
   so any Population A table is shown with and without CHDN.
4. **PSKY** is accepted as downloaded. Its sessions are not evaluable under the registered missing-data rules and are recorded as such.

## 3. Scope of this run
Only the research period (signals dated 1 Jul 2020 to 30 Jun 2023, inclusive) on Population C (the 620-stock sample) is computed. No signal dated
on or after 1 Jul 2023 is generated and no validation-period statistic is computed. Exit prices of research-period trades may fall after
30 Jun 2023, because exits are measured in sessions after entry. The validation period is run once, later, after the owner has reviewed this run.
Population B is not used for any return statistic (Amendment 1, item 6). Nothing in the research-period run changes a rule or threshold.

## 4. Signal rules left open by the registered text
1. **Replay span.** Sessions are replayed from 1 Apr 2020 so that the start of a run can be seen before the research period begins. A signal
   counts only if its date lies in the research period.
2. **First day of a run (E1, E2, and E3 episodes).** A signal day is the first session on which the condition holds after at least one earlier
   session that was **evaluated and did not hold**. A session that could not be evaluated is not an evaluated failure (Amendment 3 continuity rule),
   so the first evaluable session of a ticker, or the first after an unevaluable stretch, is not a signal.
3. **E2 gates.** Gates 2 to 4 of E1 unchanged (3-month return at least 0%, relative strength at least -30 against ^GSPC, volume at least 100,000);
   gate 1 is replaced by close above the 50-day average and at or below the 200-day average.
4. **E3.** The condition is "E1 gates pass and the scanner's detect_base_breakout is true"; a signal is the first such session after a session on
   which this condition was evaluated and false. A session on which the breakout test cannot be evaluated is not an evaluated failure.
5. **E4.** Exactly as registered: after an E3 day b, L is the highest High of the 40 bars before b; the signal is the first day within 15 sessions
   of b with Low at most 1.01 x L and Close at least L and the E1 gates passing; it is cancelled if any close falls below 0.97 x L first.
6. **One signal per ticker per approach** until the horizon has passed: a later signal is kept only if its entry session is strictly after the exit
   session of the last kept trade (Amendment 3, item 6), judged separately for each horizon (21, 63, 126).
7. **Stage and regime labels** come from the replay row on the signal day (the scanner's stage label) and from ^GSPC close above or at-or-below its
   200-day average on that day.
8. **Secondary liquid subset (Amendment 2, item 1 proxy):** signal-day adjusted close at least 5 and median 20-day dollar volume at least 2,000,000.

## 5. What is reported (research period only)
For E1 to E4 at 63 sessions (21 and 126 as sensitivities): raw signals, independent trades, exclusions by reason, win rate, mean and median net
return, average win and loss, profit factor, severe-failure share (net return at most -15%), substantial-winner share (highest High at least +30%
above entry; +20% and +50% as sensitivities), maximum favourable and adverse excursion, results by exchange, stage and regime, ticker-cluster and
calendar-month bootstrap intervals for the mean (10,000 resamples, seed 1), the two registered controls (^GSPC open-to-open over the same dates;
1,000 random entry sessions per ticker, seed 1), cost sensitivity at 0.5x, 1x and 2x, the E2-versus-E1 paired comparison, the primary
and without-affected-trades views, and the liquid subset. The E1-versus-variant comparison uses the registered bootstrap and Holm correction but
is labelled research-period and descriptive: it cannot satisfy the decision rule, which needs the validation period.

## 6. Limitations carried forward
Survivorship (today's listings), the universe mismatch with ApexScan's index-member universe, no replay of the Apex Score, adjusted prices, and
the unresolved cause of the large moves all remain. Nothing in this run is evidence about ApexScan's live behaviour; that is Population A.
