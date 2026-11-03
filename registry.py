"""audit/registry.py - integrity check of the registered methodology and amendments.

The text files in audit/registered/ are hashed (SHA-256, line endings normalised to LF so a Windows checkout cannot
change the result) and compared with the hashes recorded in PREREGISTRATION.txt on the day each document was registered.
A mismatch means the registered text was edited after registration."""
import hashlib, re
from pathlib import Path

REG_DIR = Path(__file__).resolve().parent / "registered"

def sha256_normalised(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()

def records(reg_dir: Path = REG_DIR):
    """[(registered_utc, sha256, filename, note)] parsed from PREREGISTRATION.txt."""
    out = []
    for line in (reg_dir / "PREREGISTRATION.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\S+)\s+([0-9a-f]{64})\s+(\S+)\s*(.*)$", line.strip())
        if m: out.append(m.groups())
    return out

def verify(reg_dir: Path = REG_DIR):
    """One row per registered document: recorded hash, current hash, status ok / CHANGED / missing."""
    rows = []
    for ts, h, fn, note in records(reg_dir):
        p = reg_dir / fn
        if not p.exists(): rows.append({"document": fn, "registered_utc": ts, "recorded": h, "current": "", "status": "missing"}); continue
        cur = sha256_normalised(p.read_bytes())
        rows.append({"document": fn, "registered_utc": ts, "recorded": h, "current": cur, "status": "ok" if cur == h else "CHANGED"})
    return rows

def all_intact(reg_dir: Path = REG_DIR) -> bool:
    r = verify(reg_dir); return bool(r) and all(x["status"] == "ok" for x in r)

def text(name: str, reg_dir: Path = REG_DIR) -> str:
    return (reg_dir / name).read_text(encoding="utf-8")
