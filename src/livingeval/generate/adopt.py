"""Confirmed review-queue cases, written back into the generated golden sets.

`drift` names the traffic the suite cannot see, `serve` queues cases mined from
it, and a person confirms them one by one. Without this module that loop stops
one step short: the confirmations land in the store, and the golden sets the
suite actually runs on stay exactly as this tool generated them -- synthetic
inputs and unanswered stubs, describing what the application *can* do rather
than what anyone does with it.

So this is the last hop. What it may and may not do is the whole design:

**A confirmed case brings a real input.** That is the part worth having. Every
generated input is a guess; a mined one came from somebody's traffic, and it
replaces a placeholder rather than joining it.

**A thumbs-up is not a reference answer.** A reviewer clicking "this was fine"
has said the output was acceptable, not that it is the answer a domain expert
would give. Those diverge on exactly the cases that matter -- an output that is
plausible, agreeable and subtly wrong is the one that survives review. So a
confirmed-good case fills in `expected` only when the reviewer typed one; a
thumbs-up alone brings the input and leaves the answer owed.

**A confirmed failure is worth more than a pass.** It arrives as a real input
with the answer still blank and a note saying the reviewed output was wrong,
which is the most useful row in any suite: a question known to break the system,
waiting on the answer it should have given.

Nothing here overwrites an answer a human already wrote: the only rows it may
take over are this tool's own unanswered placeholders. Everything else is
appended.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["AdoptionResult", "adopt_into_goldens"]

# A mined case carries no metric of its own. Correctness is where a real
# question with a real answer belongs, and it is the set that starts empty.
DEFAULT_TARGET = "application.correctness"


@dataclass
class AdoptionResult:
    added: int
    replaced_placeholders: int
    skipped_existing: int
    without_answers: int
    files: list[str]
    note: str = ""

    def summary(self) -> str:
        if not self.files and not self.added:
            return ("nothing to adopt: no confirmed case carried a question this "
                    "suite does not already have" + (f"\n  {self.note}" if self.note else ""))
        out = [f"adopted {self.added} confirmed case(s) into "
               f"{len(self.files)} golden set(s)"]
        if self.replaced_placeholders:
            out.append(f"  {self.replaced_placeholders} placeholder input(s) "
                       f"replaced with real questions")
        if self.without_answers:
            out.append(f"  {self.without_answers} arrived without a reference "
                       f"answer and are still awaiting one -- a reviewer "
                       f"approving an output is not the same as writing the "
                       f"answer it should have given, so they do not count "
                       f"towards the gate yet")
        if self.skipped_existing:
            out.append(f"  {self.skipped_existing} already in the set, left alone")
        for name in self.files:
            out.append(f"  wrote {name}")
        if self.note:
            out.append(f"  {self.note}")
        return "\n".join(out)


def _question(case) -> str:
    """The user's words. Falls back to the whole trace if there is no user turn."""
    for turn in case.trace.turns:
        if turn.role == "user":
            return turn.content.strip()
    return " ".join(t.content for t in case.trace.turns).strip()


def _answer(case) -> str | None:
    """What the reviewer says the answer should be -- not what the model said.

    A reviewer who corrects an answer types it into the review queue, and it
    lands in `meta`. Absent that, the confirmation is a verdict on the observed
    output and carries no reference.
    """
    for key in ("expected_output", "corrected", "answer"):
        value = case.meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _target_for(case) -> str:
    """Which golden set this case belongs in.

    Cases that came out of `drift` carry their metric as a tag, because
    `goldens_as_suite` put it there. Anything mined from raw traffic does not.
    """
    for tag in case.tags:
        if "." in tag and not tag.startswith("cluster"):
            return tag
    return DEFAULT_TARGET


def adopt_into_goldens(package: Path, cases: list, default_target: str = DEFAULT_TARGET
                       ) -> AdoptionResult:
    """Write confirmed cases into the generated golden sets under `package`.

    `cases` are `EvalCase`s with provenance "confirmed"; anything else is
    ignored rather than rejected, so this can be handed a whole suite.
    """
    directory = package / "goldens"
    if not directory.exists():
        return AdoptionResult(0, 0, 0, 0, [], note=f"no golden sets under {directory}")

    confirmed = [c for c in cases if c.provenance == "confirmed"]
    if not confirmed:
        return AdoptionResult(0, 0, 0, 0, [], note="no confirmed cases in the input")

    by_target: dict[str, list] = {}
    for case in confirmed:
        target = _target_for(case)
        if not (directory / f"{target.replace('.', '_')}.json").exists():
            target = default_target
        by_target.setdefault(target, []).append(case)

    added = replaced = skipped = unanswered = 0
    written: list[str] = []
    missing: list[str] = []

    for target, group in sorted(by_target.items()):
        path = directory / f"{target.replace('.', '_')}.json"
        if not path.exists():
            missing.append(target)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("cases", [])
        seen = {row.get("input", "").strip() for row in rows}

        # Placeholder rows are this tool's own boilerplate. A real question
        # takes one of their slots rather than sitting beside it, so the set
        # does not grow by five rows of prose nobody asked.
        slots = [i for i, row in enumerate(rows)
                 if row.get("expected") is None
                 and str(row.get("note", "")).startswith("TODO")
                 and row.get("provenance") != "confirmed"]

        fresh = []
        for case in group:
            question = _question(case)
            if not question or question in seen:
                skipped += 1
                continue
            seen.add(question)
            answer = _answer(case)
            row = {
                "id": f"confirmed-{case.case_id}"[:80],
                "input": question,
                "kind": "confirmed",
                "provenance": "confirmed",
                "reviewer": case.reviewer,
            }
            if answer:
                row["expected"] = answer
            else:
                unanswered += 1
                row["note"] = (
                    "TODO: write the answer this should have given. The "
                    "reviewer marked the output "
                    + ("acceptable, which is a verdict on what the system said, "
                       "not the reference answer this metric scores against"
                       if case.expected == 1 else
                       "wrong -- so this is a question known to break the "
                       "system, waiting on the answer it should have given"))
            fresh.append(row)

        if not fresh:
            continue

        for row in fresh:
            if slots:
                rows[slots.pop(0)] = row
                replaced += 1
            else:
                rows.append(row)
            added += 1

        data["cases"] = rows
        # `complete` is not this module's to grant, and it is recomputed
        # rather than left alone: a set is unfinished the moment a row is
        # added that still wants an answer, and a thumbs-up does not supply
        # one.
        data["complete"] = not any(
            str(row.get("note", "")).startswith("TODO")
            and row.get("expected") is None for row in rows)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        written.append(path.name)

    note = ""
    if missing:
        note = ("no golden set exists for " + ", ".join(sorted(missing)) +
                "; those cases went to " + default_target)
    return AdoptionResult(added, replaced, skipped, unanswered, written, note)
