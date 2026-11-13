"""Append-only, tamper-evident run log (JSON Lines + SHA-256 hash chain).

* The class exposes ONLY append() and verify(); there is no update/delete API.
* The file is opened in 'a' mode (OS-level O_APPEND), so writes can never overwrite old bytes.
* Each line stores the hash of the previous line, so editing any past line breaks the chain,
  which verify() detects.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

GENESIS = "0" * 64


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


class AppendOnlyRunLog:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        last = GENESIS
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)["hash"]
        return last

    def append(self, event: dict) -> str:
        entry = {"prev_hash": self._last_hash, **event}
        digest = hashlib.sha256(_canonical(entry).encode("utf-8")).hexdigest()
        line = _canonical({**entry, "hash": digest})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._last_hash = digest
        return digest

    def verify(self) -> tuple[bool, Optional[int]]:
        """Return (True, None) if the chain is intact, else (False, first_bad_line_number)."""
        prev = GENESIS
        with self.path.open("r", encoding="utf-8") as f:
            for n, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                claimed = entry.pop("hash", None)
                if entry.get("prev_hash") != prev:
                    return False, n
                if hashlib.sha256(_canonical(entry).encode("utf-8")).hexdigest() != claimed:
                    return False, n
                prev = claimed
        return True, None
