"""Content-addressed verdict cache.

The key is `(judge identity, trace content hash)`. Judge identity includes the model
name, the temperature and a hash of the rubric prompt, because changing any of those
changes the verdict and silently serving the old one would make a judge revision
invisible - which is the eval-tooling equivalent of caching on the model name and
ignoring the revision.

Storage is one JSON file per judge identity. Deliberately dumb: no database, no
server, and a file you can delete, diff or commit alongside the results it produced.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

__all__ = ["VerdictCache", "default_cache_dir"]


def default_cache_dir() -> Path:
    """`$LIVINGEVAL_CACHE`, else `~/.cache/livingeval`."""
    env = os.environ.get("LIVINGEVAL_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "livingeval"


def judge_fingerprint(**parts) -> str:
    """A stable short hash over everything that changes a judge's output."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class VerdictCache:
    """A flat, append-mostly JSON cache of verdicts."""

    def __init__(self, fingerprint: str, directory: Path | str | None = None, enabled: bool = True):
        self.fingerprint = fingerprint
        self.enabled = enabled
        self.dir = Path(directory) if directory is not None else default_cache_dir()
        self.path = self.dir / f"verdicts-{fingerprint}.json"
        self._data: dict[str, dict] = {}
        self._dirty = False
        if self.enabled and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # A corrupt cache is a performance problem, never a correctness one.
                self._data = {}

    def get(self, key: str) -> dict | None:
        return self._data.get(key) if self.enabled else None

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        self._data[key] = value
        self._dirty = True

    def flush(self) -> None:
        if not (self.enabled and self._dirty):
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False

    def __len__(self) -> int:
        return len(self._data)

    def __enter__(self) -> VerdictCache:
        return self

    def __exit__(self, *exc) -> None:
        self.flush()
