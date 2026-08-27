# ApexScan v27

## New — Apex Alpha Lab: Phases 4-7 (Setup, Feature, Conditional Research + Research Workspace)
Builds directly on the Phase 1-3 observation/outcome/score-validation layer (nothing about that layer changed) to answer the four questions the V9 spec set out for this stage:

- **Phase 4 — Setup Research.** "Does ApexScan know which setup types work?" New `compute_alpha_metrics_by_setup()` groups every discovery by its existing `setup_id` (pullback-to-50MA, fresh 200MA reclaim, breakout with/without volume confirmation, etc.) and reports win rate, expectancy, median return, profit factor, and a 95% CI per setup type.
- **Phase 5 — Feature Research.** "Which individual features actually carry predictive association?" New `FEATURE_REGISTRY` covers every technical/fundamental/valuation feature ApexScan already computes; `compute_alpha_metrics_by_feature()` buckets any of them (categorical or continuous, via rank-based quantile chunking) and measures outcomes per bucket; `rank_feature_alpha()` ranks features by the spread between their best- and worst-performing bucket. Explicitly described throughout as historical association within ApexScan's own observed universe — never framed as causation.
- **Phase 6 — Conditional Research.** "When do a feature's strongest signals actually work?" New `compute_conditional_alpha()` filters to a feature's top bucket and breaks the result down by market regime, sector, or setup. New `PRESET_COMBINATIONS` (6 fixed combinations — RS+Stage2, RS+Volume, Stage2+Volume, Stage2+RS+Volume, Fundamentals+RS, Institutional+RS) with `compute_combination_alpha()` / `get_combination_alpha_table()`. Deliberately a short, fixed list rather than an exhaustive search — testing many combinations and keeping only the best-looking one is a textbook way to manufacture a fake edge.
- **Phase 7 — Apex Alpha Lab workspace.** New `ui/alpha_lab.py` (extracted as its own module rather than growing dashboard.py further, per the spec's own refactoring guidance) with six tabs: Overview, Score Validation, Setup Alpha, Feature Alpha, Conditional Alpha, and Combinations & Findings. Findings are auto-generated from any setup/feature/combination with ≥10 resolved observations, each one traceable back to its exact source — no black-box claims. Replaces the old "Alpha Lab (Preview)" block and is no longer nested inside the legacy Discovery Tracker's `if not tracked.empty:` gate, so it renders on its own regardless of the legacy tracker's state.
- Model Comparison is scaffolded (every observation already carries a `model_version` tag) but not built out yet — there's only one model version to compare so far.

### Known state, honestly
- The 599 real observations logged so far have **zero resolved outcomes** at any horizon yet (too recent) — every new table will show `n=0` / "Too Small" until enough trading days pass. This is correct, not a bug.
- `market_regime` has only ever been observed as "Sideways" so far — regime-conditional breakdowns will show a single row until ApexScan has scanned across more varied market conditions.
- Fundamentals (`revenue_growth`, `institutional_ownership`, etc.) are still unpopulated on most logged observations — Fundamentals/Institutional-Ownership combination rows will read `n=0` until Alpha Vantage/yfinance enrichment fills them in for more discoveries.

### Testing
- New `tests/test_alpha_metrics_v9_phase4_7.py` — 19 regression tests against synthetic observations with known expected values (setup grouping, feature bucketing incl. a fixed bucket-collapse edge case, conditional filtering, combination matching, overview counts, findings min-n gating). All passing.
- Verified via AST that all 23 `with tabs[N]:` blocks in dashboard.py remain real executable code after the edit (per the project's own established regression discipline).
- Ran the existing `tests/test_smoke_app.py` smoke suite — passes identically to the unmodified file (the one failure in both is a sandboxed-environment Yahoo Finance network block, unrelated to this change).

# ApexScan v26

## Reliability
- Rebuilt the US universe cache layer with freshness, corruption, minimum-record, and atomic-write validation.
- Added per-index fetch/parser diagnostics, de-duplication, and explicit failures that do not overwrite a good cache.
- Added structured NGX Pulse and NGN Market session counters for API calls, cache hits, authentication, HTTP, and rate limits.
- Made missing or invalid provider credentials visible in logs without exposing the credential.

## Scanner
- Market intent now takes priority over ticker suffixes, with `.LG`/`.NG` detection retained as a fallback.
- US benchmarks are preloaded only for US-containing scans; NGX scans use the NGX All-Share source.
- Added scan-summary diagnostics and rejection/failure attribution.

## Product capabilities
- Added strategy variants: swing, position, long-term, dividend, and value. Existing `apex_score` remains unchanged.
- Added an optional, cached AI interpretation layer that is isolated from deterministic scanning and degrades safely.
