"""Closing the loop: labels back into the suite, and out as training data.

Step 4 of the original design — "feed labels back" — is two different jobs, and running
them together is how an eval suite quietly starts grading itself.

**Back into the suite.** Confirmed proposals become gateable cases; rejected ones stay
out. `promote()` does that read-and-merge against a store, and it will only promote
what carries a reviewer's name. If nobody reviewed it, it does not go in.

**Out as training data.** Once you have human-confirmed labels, they are the correct
supervision for a distilled scorer — better than the judge's own labels, because they
are the thing the judge is *approximating*. `export_finetune_data` writes them in the
format your training path wants.

The ordering matters and is enforced: `export_finetune_data(..., confirmed_only=True)`
is the default, so the exported dataset is human-labelled by construction. Training a
scorer on judge labels and then measuring it against judge labels is a closed loop that
reports high agreement no matter how wrong the judge is.

Rolling retention is here too. A suite that only grows becomes a suite nobody runs, so
`promote(max_cases=...)` retires oldest-first and reports what it dropped — a silent
cap would make "coverage went up" indistinguishable from "coverage was recomputed on
fewer cases".
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from livingeval.suite.case import EvalCase
from livingeval.suite.suite import EvalSuite
from livingeval.trace.render import render_view

__all__ = ["FORMATS", "PromotionResult", "export_finetune_data", "promote"]


@dataclass
class PromotionResult:
    """What one turn of the loop changed about a suite."""

    suite: EvalSuite
    added: int
    retired: int
    skipped_unreviewed: int
    reviewers: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover - display only
        who = ", ".join(f"{k}:{v}" for k, v in sorted(self.reviewers.items())) or "nobody"
        lines = [
            f"promotion  {self.suite.name}  ->  {len(self.suite)} case(s)",
            f"  added {self.added} confirmed   retired {self.retired}   reviewers: {who}",
        ]
        if self.skipped_unreviewed:
            lines.append(
                f"  skipped {self.skipped_unreviewed} unreviewed proposal(s) - they stay in the "
                "queue and out of the gate"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "kind": "promotion",
            "suite": self.suite.name,
            "suite_hash": self.suite.content_hash(),
            "n_cases": len(self.suite),
            "added": self.added,
            "retired": self.retired,
            "skipped_unreviewed": self.skipped_unreviewed,
            "reviewers": self.reviewers,
        }


def promote(
    store,
    suite_name: str,
    max_cases: int | None = None,
    save: bool = True,
) -> PromotionResult:
    """Merge confirmed proposals into a suite.

    Reads the suite and its confirmed proposals from `store`, adds the confirmed cases,
    optionally retires the oldest to stay under `max_cases`, and writes the suite back.

    Only confirmed proposals are promoted. Pending ones are counted and left alone,
    which is the whole guard-rail: a mined case cannot reach a gate without a named
    human having looked at it.
    """
    suite = store.get_suite(suite_name)
    if suite is None:
        raise KeyError(f"no suite named {suite_name!r} in the store")

    confirmed = store.confirmed_cases(suite_name)
    pending = len(store.get_proposals(suite_name, status="pending"))

    existing = {c.case_id for c in suite}
    fresh = [c for c in confirmed if c.case_id not in existing]
    merged = EvalSuite([*suite.cases, *fresh], suite.name, suite.meta)

    retired = 0
    if max_cases is not None and len(merged) > max_cases:
        retired = len(merged) - max_cases
        merged = EvalSuite(merged.cases[retired:], suite.name, suite.meta)

    reviewers: dict[str, int] = {}
    for case in fresh:
        if case.reviewer:
            reviewers[case.reviewer] = reviewers.get(case.reviewer, 0) + 1

    if save:
        store.put_suite(merged)

    return PromotionResult(merged, len(fresh), retired, pending, reviewers)


# ---------------------------------------------------------------------------
# fine-tune data export
# ---------------------------------------------------------------------------

DEFAULT_INSTRUCTION = "Did the agent do the right thing? Answer PASS or FAIL."


def _chat(text: str, label: int, instruction: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
            {"role": "assistant", "content": "PASS" if label else "FAIL"},
        ]
    }


def _prompt_completion(text: str, label: int, instruction: str) -> dict:
    return {"prompt": f"{instruction}\n\n{text}\n\nAnswer:",
            "completion": " PASS" if label else " FAIL"}


def _classification(text: str, label: int, instruction: str) -> dict:
    return {"text": text, "label": int(label)}


#: `chat` suits OpenAI and Fireworks SFT. `prompt_completion` suits Unsloth and TRL's
#: plain text field. `classification` suits an encoder with a classification head,
#: which is what `scorer.finetune` with `backend="local"` consumes.
FORMATS = {
    "chat": _chat,
    "prompt_completion": _prompt_completion,
    "classification": _classification,
}


def export_finetune_data(
    cases: Iterable[EvalCase] | EvalSuite,
    path: str | Path,
    fmt: str = "chat",
    view: str = "full",
    instruction: str = DEFAULT_INSTRUCTION,
    confirmed_only: bool = True,
    split: float | None = None,
) -> dict:
    """Write human-confirmed labels as fine-tuning data.

    `confirmed_only=True` (the default) keeps curated and reviewer-confirmed cases and
    drops unreviewed mined ones. Turning it off trains your scorer on the judge's own
    labels, which makes the resulting agreement number circular — so it is possible,
    and it is not the default.

    `split=0.2` writes a matching `.eval.jsonl` holdout, split **by session** so a
    conversation cannot appear on both sides.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {sorted(FORMATS)}")
    render = FORMATS[fmt]

    all_cases = list(cases)
    kept = [c for c in all_cases if (not confirmed_only) or c.gateable]
    dropped = len(all_cases) - len(kept)

    rows = [(c.trace.group, render(render_view(c.trace, view), c.expected, instruction))
            for c in kept]

    train, holdout = rows, []
    if split:
        groups = sorted({g for g, _ in rows})
        n_holdout = max(1, round(len(groups) * split))
        # Group-aware split for the same reason the ladder's CV is: sessions are
        # near-duplicates, and a random row split would report a validation number
        # inflated by memorisation.
        held = set(groups[-n_holdout:])
        train = [(g, r) for g, r in rows if g not in held]
        holdout = [(g, r) for g, r in rows if g in held]

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for _, row in train:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    eval_path = None
    if holdout:
        eval_path = out.with_suffix(".eval.jsonl")
        with eval_path.open("w", encoding="utf-8") as fh:
            for _, row in holdout:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_pass = sum(1 for c in kept if c.expected == 1)
    return {
        "kind": "finetune_export",
        "path": str(out),
        "eval_path": str(eval_path) if eval_path else None,
        "format": fmt,
        "view": view,
        "n_train": len(train),
        "n_eval": len(holdout),
        "n_dropped_unconfirmed": dropped,
        "pass_rate": n_pass / len(kept) if kept else None,
        "confirmed_only": confirmed_only,
    }
