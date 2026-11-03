# ApexScan audit tools (additive)

Adds an **Audit** page to the sidebar. It holds the registered audit methodology (hash-checked), a read-only view of
live discoveries and how many have matured, an in-app Yahoo price download, a data-quality report and a replay of the
scanner's gates for one ticker (default CHDN).

What it does NOT do: change scanner.py, dashboard.py, strategies.py, config.yaml, any score, or anything in `data/`.

Rollback: delete `audit/`, `pages/Audit.py` and `tests/test_audit_*.py`. Nothing else depends on them.
