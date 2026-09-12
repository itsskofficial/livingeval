"""`livingeval sync` -- the suite follows the code.

Code moves. A retriever gets added, a schema gains a field, a call site is
deleted, a new agent appears in a module nobody told the eval suite about. A
suite generated once and never revisited describes the application as it was on
the day somebody ran `init`, which is exactly the decay this library exists to
measure -- and it would be absurd to ship it in the tool itself.

`sync` rescans, diffs against the manifest, and reports before it writes. The
diff is over *what decides the metrics*, not over the source: a call site that
slid twelve lines down is unchanged, and a call site that gained a retriever is
a different thing needing three more evals.

**What it will never do is overwrite a human's work.** Three protections, in
order of how much it would hurt to lose:

1. **Golden answers are sacred.** A dataset a person has filled in is merged --
   new cases appended, existing answers untouched -- never regenerated. Losing
   an afternoon of labelling because a line number moved is the worst thing this
   tool could do, so it is the thing most carefully prevented.
2. **Edited files are reported, not replaced.** The manifest records each file's
   digest as written. If it no longer matches, the user has changed it, and
   `sync` says so and leaves it alone unless told otherwise.
3. **Deletions are proposed, never performed.** A pipeline whose call site
   disappeared is reported as stale. Whether that call site was deleted or
   merely moved somewhere the scanner cannot see is not a question static
   analysis can answer, and guessing wrong deletes working evals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from livingeval.discover.sites import CallSite

# `merge_golden` and `write_goldens_preserving` live in `generate.goldens`:
# merging is part of writing a golden set safely, and it has to happen on
# every write rather than only when sync happens to be the caller. Re-exported
# here because this is where a reader looks for them.
from livingeval.generate.goldens import (
    merge_golden,
    write_goldens_preserving,
)
from livingeval.manifest import Manifest, SiteRecord, digest, site_digest
from livingeval.plan.planner import Plan

__all__ = ["Change", "SyncPlan", "diff", "merge_golden",
           "protected_files", "write_goldens_preserving"]


@dataclass
class Change:
    kind: str          # added | removed | changed | unchanged
    subject: str
    detail: str = ""

    def line(self) -> str:
        marker = {"added": "+", "removed": "-", "changed": "~", "unchanged": " "}[self.kind]
        return f"  {marker} {self.subject}" + (f"  ({self.detail})" if self.detail else "")


@dataclass
class SyncPlan:
    """What changed, and what may safely be written."""

    sites: list[Change] = field(default_factory=list)
    metrics: list[Change] = field(default_factory=list)
    edited: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    preserved: dict[str, int] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return any(c.kind != "unchanged" for c in self.sites + self.metrics)

    def summary(self) -> str:
        added = sum(1 for c in self.sites if c.kind == "added")
        removed = sum(1 for c in self.sites if c.kind == "removed")
        changed = sum(1 for c in self.sites if c.kind == "changed")
        new_metrics = sum(1 for c in self.metrics if c.kind == "added")
        gone = sum(1 for c in self.metrics if c.kind == "removed")
        if not self.has_changes:
            return "nothing has changed since the suite was generated"
        return (f"{added} new call sites, {changed} changed, {removed} gone -> "
                f"{new_metrics} metrics to add, {gone} now unused")


def diff(manifest: Manifest, sites: list[CallSite], plan: Plan) -> SyncPlan:
    """Compare a fresh scan against what was generated last time."""
    out = SyncPlan()

    seen: set[str] = set()
    for site in sites:
        record = SiteRecord(
            ident=site.ident, path=str(site.path), function=site.function,
            provider=site.provider, archetype=site.archetype,
            fingerprint=site_digest(site), line=site.line)
        seen.add(record.stable_key)
        previous = manifest.site_by_key(record.stable_key)

        if previous is None:
            out.sites.append(Change("added", f"{site.path}:{site.line} {site.function}()",
                                    f"{site.archetype}"))
        elif previous.fingerprint != record.fingerprint:
            was = previous.archetype
            detail = (f"{was} -> {site.archetype}" if was != site.archetype
                      else "same archetype, different components")
            out.sites.append(Change("changed",
                                    f"{site.path}:{site.line} {site.function}()", detail))
        else:
            out.sites.append(Change("unchanged", f"{site.path} {site.function}()"))

    for record in manifest.sites:
        if record.stable_key not in seen:
            out.sites.append(Change(
                "removed", f"{record.path} {record.function}()",
                "no longer found; its evals are reported stale, not deleted"))

    current = {m.key for p in plan.pipelines for m in p.metrics}
    previous_metrics = set(manifest.metrics)
    for key in sorted(current - previous_metrics):
        out.metrics.append(Change("added", key))
    for key in sorted(previous_metrics - current):
        out.metrics.append(Change("removed", key, "no call site needs it any more"))

    return out


def protected_files(manifest: Manifest, package: Path) -> list[str]:
    """Generated files a human has since edited.

    Compared against the digest recorded at write time, so this is "changed
    since we wrote it", not "differs from what we would write now" -- the second
    would flag every file on every version bump.
    """
    edited = []
    for record in manifest.files:
        path = package / record.path
        if not path.exists():
            continue
        if digest(path.read_text(encoding="utf-8")) != record.written:
            edited.append(record.path)
    return edited
