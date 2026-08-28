# ApexScan v28

## New — Model Governance, Walk-Forward Validation, Workspace Navigation, AI Explanation (V9 Phases 8-11)

- **Phase 8 — Model Governance.** New governance layer in `modules/model_registry.py`, persisted to `data/model_registry.json` via the same GitHub-backed pattern as every other store. Implements the spec's exact pipeline: `propose_research_model()` -> `record_walk_forward_result()` -> `approve_promotion()` (requires a named human approver AND a passing walk-forward result on file — refuses both otherwise) -> a human manually edits `MODEL_VERSION` and redeploys. Nothing in this layer can activate a version by itself, by design — that split is what makes "a model can improve without corrupting historical evidence" actually true rather than aspirational. Also adds `compare_models()`, comparing realized Alpha Metrics between any two `model_version` tags already present in observations.
- **Phase 9 — Walk-Forward Validation.** New `modules/walk_forward.py`. Chronological train/test splits only (never random k-fold, which would leak future observations into an earlier "training" window). `rolling_folds()` walks the observed date range forward in order; `evaluate_setup_stability()` checks whether the best-expectancy setup identified on each TRAIN window still held up (n and direction) on the TEST window immediately after it. `walk_forward_readiness()` refuses to produce a verdict below an estimated sample-size floor (min_n × (folds+1)) — reports exactly how much more history is needed instead of running a hollow analysis. This is the actual gate for Phase 8 approvals.
- **Phase 10 — Workspace Navigation.** New `ui/workspace_nav.py`, inserted as a single isolated call right above the existing tab bar in dashboard.py — the six workspaces (Discover / Market / Opportunities / Research / Portfolio / Alpha Lab) from the spec's §48, as a way-finding strip over the untouched 23 tabs. Deliberately NOT a hard router: Streamlit's stable public API has no supported way to programmatically switch an already-rendered tab, and a DOM-click JS hack would be exactly the kind of fragile dashboard.py change this project's own regression discipline warns against. Every one of the 23 `with tabs[N]:` blocks is unmodified and still verified present.
- **Phase 11 — AI Explanation.** New `build_decision_explanation_prompt()` in `ai/prompts.py` and `explain_decision()` in `ai/engine.py`, wired into a new "Explain a Decision" tab in the Alpha Lab. Explains an already-made deterministic decision (score, setup, and that setup's real historical win rate/expectancy/sample size from Alpha Lab) in plain English — explicitly instructed never to invent a price, statistic, or confidence level not given in the prompt, and never to issue a new buy/sell recommendation. Inactive (with a plain explanation why) unless `ai_enabled: true` and a real API key are configured — same gate used everywhere else the AI layer appears. Note: the existing "🤖 AI Briefing" tab uses a separate, template-based `generate_scan_briefing()` function that needs no API key; `ai/engine.py`'s LLM-calling path was present in the repo but not actually wired into any UI before this change.

### Testing
- New `tests/test_model_governance_v9_phase8.py` — 11 tests, GitHub storage mocked so the workflow's guard rails (missing approver, missing/failing walk-forward result, duplicate version IDs, the module constant never changing) are verified without live credentials.
- New `tests/test_walk_forward_v9_phase9.py` — 11 tests, including synthetic timelines that verify a genuinely-decaying setup edge is correctly flagged as NOT holding up out-of-sample.
- Full suite (58 tests across all `tests/` files) passes, including the pre-existing `test_every_tab_index_used_exactly_once` smoke test — confirms the Phase 10 workspace-nav insertion didn't disturb tab indexing.
- AST-verified all 23 `with tabs[N]:` blocks remain real executable code.

### Known state, honestly
- Every new capability here that depends on resolved outcomes (walk-forward, model comparison) will report "not ready" / n=0 against real current data, for the same reason noted in v27: zero resolved observations exist yet. This is correct.
- Model Governance currently has one version on record (the backfilled production entry) until someone actually proposes a research model.

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
