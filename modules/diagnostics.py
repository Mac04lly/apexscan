"""
modules/diagnostics.py — ApexScan Scan Diagnostics
Tracks scan-level counts (scanned/passed/rejected) and rejection reasons
so a scan never "silently" drops from thousands of tickers to a handful
without an explanation in the logs.
"""

class ScanDiagnostics:
    FAIL_REASONS   = ["history", "api"]
    REJECT_REASONS = ["stage", "performance", "relative_strength", "volume", "score"]

    def __init__(self):
        self.scanned = 0
        self.passed  = 0
        self._counts = {r: 0 for r in self.FAIL_REASONS + self.REJECT_REASONS}

    def fail(self, reason: str):
        """Record a hard failure before any gate is even evaluated (e.g. no history)."""
        self._counts[reason] = self._counts.get(reason, 0) + 1

    def reject(self, reason: str):
        """Record a gate rejection (stage/performance/RS/volume/score)."""
        self._counts[reason] = self._counts.get(reason, 0) + 1

    @property
    def rejected_total(self) -> int:
        return sum(self._counts.get(r, 0) for r in self.REJECT_REASONS)

    @property
    def failed_total(self) -> int:
        return sum(self._counts.get(r, 0) for r in self.FAIL_REASONS)

    def summary(self) -> dict:
        return {
            "scanned":  self.scanned,
            "passed":   self.passed,
            "rejected": self.rejected_total,
            "failed":   self.failed_total,
            "breakdown": dict(self._counts),
        }

    def log_summary(self, log):
        s = self.summary()
        log.info("=" * 60)
        log.info("SCAN SUMMARY")
        log.info(f"Scanned              {s['scanned']}")
        log.info(f"Passed               {s['passed']}")
        log.info(f"Rejected             {s['rejected']}")
        log.info(f"  History            {self._counts.get('history', 0)}")
        log.info(f"  API                {self._counts.get('api', 0)}")
        log.info(f"  Stage              {self._counts.get('stage', 0)}")
        log.info(f"  Performance        {self._counts.get('performance', 0)}")
        log.info(f"  Relative Strength  {self._counts.get('relative_strength', 0)}")
        log.info(f"  Volume             {self._counts.get('volume', 0)}")
        log.info(f"  Score              {self._counts.get('score', 0)}")
        log.info("=" * 60)
