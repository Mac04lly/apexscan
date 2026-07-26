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
