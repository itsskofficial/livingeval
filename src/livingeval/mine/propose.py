"""Close the loop: turn traffic the suite cannot see into candidate cases.

Selection is greedy k-centre restricted to blind clusters. Candidates are drawn from
the least-covered clusters first, and within a cluster each pick is the trace furthest
from everything already chosen, so twenty proposals are twenty different problems
rather than twenty rephrasings of the most common one.

**Nothing here writes a label you did not confirm.** A proposal carries a *suggested*
expected label from a judge, if one was supplied, and it enters a suite with
`provenance="mined"`, which is excluded from gate decisions until a named reviewer
confirms it. This is the guard-rail that stops the loop from converging on "the model
is correct because the cases came from the model" - an eval suite that grows itself
unsupervised eventually asserts current behaviour is right by definition. That failure
is silent, slow and total, and the only defence is procedural.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from livingeval.judge.base import Judge
from livingeval.mine.blindspots import BlindSpotReport
from livingeval.mine.blindspots import blindspots as _blindspots
from livingeval.mine.space import Space
from livingeval.suite.case import EvalCase
from livingeval.suite.suite import EvalSuite
from livingeval.trace.types import Trace, TraceSet

__all__ = ["Proposal", "ProposalSet", "propose"]


@dataclass
class Proposal:
    """A candidate eval case awaiting human confirmation."""

    trace: Trace
    cluster: int
    reason: str
    suggested_expected: int | None = None
    judge_rationale: str | None = None
    distance_to_suite: float = float("nan")

    def as_case(self, expected: int | None = None) -> EvalCase:
        """Materialise as an unconfirmed, non-gateable case."""
        value = expected if expected is not None else self.suggested_expected
        if value is None:
            raise ValueError(
                f"proposal {self.trace.trace_id} has no suggested label; pass `expected` "
                "explicitly - a case with a guessed label is worse than no case"
            )
        return EvalCase(
            case_id=f"mined-{self.trace.trace_id}",
            trace=self.trace,
            expected=int(value),
            provenance="mined",
            tags=("mined", f"cluster{self.cluster}"),
            meta={"reason": self.reason, "distance_to_suite": self.distance_to_suite},
        )


@dataclass
class ProposalSet:
    proposals: list[Proposal]
    report: BlindSpotReport | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.proposals)

    def __iter__(self):
        return iter(self.proposals)

    def __getitem__(self, i):
        return self.proposals[i]

    def cases(self) -> list[EvalCase]:
        """Unconfirmed cases. Legal to add to a suite; excluded from the gate."""
        return [p.as_case() for p in self.proposals]

    def confirm(self, reviewer: str, expected: dict[str, int] | None = None) -> list[EvalCase]:
        """Confirm every proposal under one reviewer's name.

        `expected` overrides individual labels by trace id, which is the common
        outcome of a review: most suggestions stand, a few were wrong, and the wrong
        ones are exactly the cases worth having.
        """
        if not reviewer:
            raise ValueError("confirm() requires a reviewer name; that is the entire point")
        overrides = expected or {}
        out = []
        for p in self.proposals:
            case = p.as_case(overrides.get(p.trace.trace_id))
            out.append(case.confirm(reviewer))
        return out

    def summary(self) -> str:  # pragma: no cover - display only
        if not self.proposals:
            return "no proposals: the suite already covers the traffic"
        lines = [f"{len(self.proposals)} proposed case(s), all UNCONFIRMED", ""]
        for p in self.proposals[:10]:
            label = "?" if p.suggested_expected is None else ("pass" if p.suggested_expected else "FAIL")
            first = next((t.content for t in p.trace.turns if t.role == "user"), "")
            lines.append(f"  [{p.trace.trace_id}] cluster {p.cluster}  suggested={label}  {p.reason}")
            lines.append(f"      {first[:96]}")
        if len(self.proposals) > 10:
            lines.append(f"  ... and {len(self.proposals) - 10} more")
        lines.append("")
        lines.append("  confirm with: proposals.confirm(reviewer='your-name')")
        return "\n".join(lines)


def propose(
    traces: TraceSet,
    suite: EvalSuite,
    n: int = 20,
    judge: Judge | None = None,
    report: BlindSpotReport | None = None,
    space: Space | None = None,
    prefer_failures: bool = True,
    seed: int = 0,
    view: str = "request",
) -> ProposalSet:
    """Propose `n` candidate cases from the traffic the suite covers least.

    With a `judge`, each proposal carries a suggested label and the judge's rationale,
    and failing traces are offered first when `prefer_failures` is set - a suite with
    no failures in a region cannot detect a regression there either.
    """
    space = space or Space.fit(traces, view=view, seed=seed)
    report = report or _blindspots(suite, traces, space=space, seed=seed, view=view)

    traffic_coords = space.transform(traces)
    suite_coords = space.transform(suite.traces) if len(suite) else np.zeros((0, traffic_coords.shape[1]))
    labels = report.clustering.labels

    # Budget across blind clusters in proportion to severity, so the worst spot gets
    # the most cases without the others getting none.
    ranked = [s for s in report.spots if s.coverage < report.threshold] or report.spots[:1]
    total = sum(s.severity for s in ranked) or 1.0
    budget = {s.cluster: max(1, round(n * s.severity / total)) for s in ranked}

    chosen: list[Proposal] = []
    for spot in ranked:
        want = budget.get(spot.cluster, 0)
        if want <= 0 or len(chosen) >= n:
            continue
        rows = np.flatnonzero(labels == spot.cluster)
        if rows.size == 0:
            continue

        if judge is not None and prefer_failures:
            verdicts = {int(i): judge(traces[int(i)]) for i in rows}
            failing = [i for i in rows if verdicts[int(i)].label == 0]
            rows = np.asarray(failing if failing else list(rows))
        else:
            verdicts = {}

        # Greedy k-centre seeded by the existing suite: the first pick is the trace
        # furthest from anything already covered.
        anchors = [suite_coords] if suite_coords.shape[0] else []
        picked: list[int] = []
        pool = rows.tolist()
        while pool and len(picked) < want and len(chosen) + len(picked) < n:
            coords = traffic_coords[pool]
            reference = np.vstack(anchors) if anchors else np.zeros((0, coords.shape[1]))
            if reference.shape[0]:
                d = Space.cosine_distance(coords, reference).min(axis=1)
            else:
                d = np.ones(len(pool))
            take = int(np.argmax(d))
            idx = pool.pop(take)
            picked.append(idx)
            anchors.append(traffic_coords[idx : idx + 1])

        for idx in picked:
            verdict = verdicts.get(idx) if verdicts else (judge(traces[idx]) if judge else None)
            d_suite = (
                float(Space.cosine_distance(traffic_coords[idx : idx + 1], suite_coords).min())
                if suite_coords.shape[0]
                else float("nan")
            )
            reason = (
                f"cluster {spot.cluster} carries {spot.traffic_share:.1%} of traffic at "
                f"{spot.coverage:.0%} coverage"
            )
            chosen.append(
                Proposal(
                    trace=traces[idx],
                    cluster=spot.cluster,
                    reason=reason,
                    suggested_expected=None if verdict is None else int(verdict.label),
                    judge_rationale=None if verdict is None else verdict.rationale,
                    distance_to_suite=d_suite,
                )
            )

    return ProposalSet(chosen[:n], report, {"judge": getattr(judge, "name", None), "seed": seed})
