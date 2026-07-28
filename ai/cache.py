"""ai/cache.py — simple file-based cache for AI outputs to minimise API cost."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

CACHE_DIR = Path("data/ai_cache")
CACHE_TTL_SECONDS = 6 * 3600  # 6h — market conditions change slowly enough for this


class AICache:
    def __init__(self):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def key(self, *parts) -> str:
        raw = "|".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        try:
            p = self._path(key)
            if not p.exists():
                return None
            data = json.loads(p.read_text())
            if time.time() - data.get("_ts", 0) > CACHE_TTL_SECONDS:
                return None
            return data.get("text")
        except Exception:
            return None

    def set(self, key: str, text: str) -> None:
        try:
            self._path(key).write_text(json.dumps({"_ts": time.time(), "text": text}))
        except Exception:
            pass
