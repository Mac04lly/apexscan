# ApexScan audit tools (additive)

Adds an **Audit** page to the sidebar. It holds the registered audit methodology (hash-checked), a read-only view of
live discoveries and how many have matured, an in-app Yahoo price download, a data-quality report and a replay of the
scanner's gates for one ticker (default CHDN).

What it does NOT do: change scanner.py, dashboard.py, strategies.py, config.yaml, any score, or anything in `data/`.

Rollback: delete `audit/`, `pages/Audit.py` and `tests/test_audit_*.py`. Nothing else depends on them.

## Amendment 3 (3 Nov 2026)
`audit/registered/amendments3.md` (hash-checked with the others) defines the Population A cohorts (A1 initial persistence/seeding,
A2 post-seeding first-recorded), keeps three date concepts apart (recorded discovery date, reconstructed four-gate date as a
lower bound, actual full scanner qualification date: not established), and sets how independent trades are counted.
`audit/cohorts.py` implements it; the Live discoveries, Data quality and CHDN replay tabs show it.

### Implementation clarification, 4 Nov 2026 (not part of the registered record)
Amendment 3 (registered 3 Nov 2026, text and hash unchanged) says a reconstructed four-gate run is flagged as left-censored when
it reaches "the start of the available price history". The implementation is stricter, and this note records the difference.

- **Available price history** is every daily bar downloaded for a ticker. **Available evaluable history** is the part of it on
  which the replay can actually judge the gates: at least 200 bars of history, no missing sessions or values in the lookback, and
  a usable benchmark. Evaluable history can begin well after the first downloaded bar.
- **Why evaluable history is used.** A session that cannot be judged is neither a pass nor a fail. Treating the first downloaded bar
  as the start would claim knowledge of sessions the replay could not assess, and skipping unjudgeable sessions would join runs
  that were never shown to be continuous. So a run is left-censored only when nothing earlier was evaluated: "Left-censored at the
  beginning of available evaluable history; the true qualification start may be earlier." The replay tab also starts on a date the
  user chooses, so a run that reaches that date is censored by the window; choose an earlier start to look further back.
- **Continuity.** A passing session starts or continues a verified run. A failed session breaks it. An unevaluable session breaks it,
  and so does a missing session row when the trading calendar is supplied. Nothing is bridged. If an evaluated session exists
  earlier but is separated from the run by an unevaluable session, the status is `start_uncertain_unevaluable_gap` and the date
  is the start of the verified run only.
- **One implementation.** `cohorts.four_gate_lower_bound` and `replay.first_pass_dates` / `first_pass_detail` (the CHDN replay
  tab) share `cohorts.session_states` and `cohorts.verified_run`, so they cannot disagree.
- **Status of this note.** It is an implementation clarification. The registered Amendment 3 is not rewritten and keeps its hash.
  No entry definition, threshold, period, cost, test, evidence class or decision rule changes.
