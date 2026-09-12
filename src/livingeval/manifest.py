"""What was generated, from what, and what a human has touched since.

`init` writes files into somebody's repository. `sync` writes into the same
files after the code has moved. The difference between those two being useful
and being a disaster is entirely this record.

Three things are fingerprinted, and each answers a different question on the
next run:

``sites``
    A digest per call site, over the facts that decide its metrics -- archetype,
    the roles present, the schema names. Not the line number: a call site that
    slid down twelve lines is the same call site, and reporting it as changed
    every time somebody adds an import would make `sync` output worthless.

``files``
    A digest of each generated file *as written*. If the digest still matches,
    nobody has edited it and it can be rewritten freely. If it does not, the user
    has changed something and `sync` must not clobber it without saying so.

``goldens``
    Per dataset, how many answers a human had filled in. This is the one that
    must never be lost. Regenerating a golden set that somebody spent an
    afternoon labelling, because a line number moved, would be the single worst
    thing this tool could do.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from livingeval._version import __version__

__all__ = ["FileRecord", "Manifest", "SiteRecord", "digest", "site_digest"]

MANIFEST_NAME = ".livingeval-manifest.json"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class SiteRecord:
    """A call site, fingerprinted over what decides its evals."""

    ident: str
    path: str
    function: str
    provider: str
    archetype: str
    fingerprint: str
    line: int = 0

    @property
    def stable_key(self) -> str:
        """Identity across edits. Line numbers move; the function does not."""
        return f"{self.path}::{self.function}"


@dataclass
class FileRecord:
    path: str
    written: str          # digest at the time livingeval wrote it
    kind: str             # eval | harness | registry | golden | doc


@dataclass
class Manifest:
    version: str = __version__
    generated_at: float = 0.0
    judge: str = ""
    scope: str = ""
    sites: list[SiteRecord] = field(default_factory=list)
    files: list[FileRecord] = field(default_factory=list)
    goldens: dict[str, dict] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)

    # -- lookup ------------------------------------------------------------

    def site_by_key(self, key: str) -> SiteRecord | None:
        return next((s for s in self.sites if s.stable_key == key), None)

    def file_record(self, relative: str) -> FileRecord | None:
        return next((f for f in self.files if f.path == relative), None)

    # -- persistence -------------------------------------------------------

    def save(self, package: Path) -> Path:
        path = package / MANIFEST_NAME
        payload = asdict(self)
        payload["generated_at"] = self.generated_at or time.time()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, package: Path) -> Manifest | None:
        path = package / MANIFEST_NAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        return cls(
            version=data.get("version", "?"),
            generated_at=data.get("generated_at", 0.0),
            judge=data.get("judge", ""),
            scope=data.get("scope", ""),
            sites=[SiteRecord(**s) for s in data.get("sites", [])],
            files=[FileRecord(**f) for f in data.get("files", [])],
            goldens=data.get("goldens", {}),
            metrics=data.get("metrics", []),
        )


def site_digest(site) -> str:
    """Fingerprint the facts that decide a site's metrics.

    Deliberately excludes the line number and the prompt text. Both change
    constantly without changing which evals apply, and a `sync` that reports
    every site as modified after a reformat is a `sync` nobody reads.
    """
    evidence = site.evidence
    material = "|".join([
        site.archetype,
        site.provider,
        ",".join(sorted(set(evidence.schemas))),
        ",".join(sorted(set(evidence.literal_enums))),
        "retrieval" if evidence.retrieval else "",
        "rerank" if evidence.rerank else "",
        "tools" if evidence.tools else "",
        "memory" if evidence.memory else "",
        "loop" if evidence.has_loop else "",
        "structured" if evidence.structured else "",
    ])
    return digest(material)
