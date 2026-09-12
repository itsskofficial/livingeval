"""Turning a scan into a plan.

The planner is the part that decides, and the part users will argue with. Two
consequences shape it.

**Every decision carries its reason.** A `Pipeline` records which metrics it
holds, which target they attach to, and why that target was classified the way
it was. `livingeval plan --explain` prints it. A tool that writes files into
somebody's repository on the strength of reasoning it will not show is a tool
that gets deleted the first time it is wrong.

**Nothing reference-based is ever silently filled in.** Metrics needing a known
answer are planned, emitted as runnable stubs, and reported as blocked with a
question attached. The tempting shortcut -- run the current code and record what
it says -- builds a suite that certifies today's behaviour as correct by
definition, which is exactly the failure `EvalCase.provenance` exists to stop.
It would also make the numbers meaningless in the only situation anyone cares
about, which is when the behaviour changes.

Pipelines are grouped the way people organise them by hand: one file per
(target, risk), so a run can be narrowed to "just safety" or "just the
retriever" without unpicking anything.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from livingeval.discover.sites import CallSite
from livingeval.plan.taxonomy import (
    Archetype,
    Level,
    Method,
    Metric,
    Reference,
    Risk,
    for_target,
)

__all__ = ["Pipeline", "Plan", "Question", "build_plan"]

# Component-level metrics name the role they attach to. A site only gets them
# when the evidence shows that role is actually present -- planning a retriever
# eval for a pipeline with no retriever wastes a run and reports a fake zero.
ROLE_EVIDENCE = {
    "retriever": lambda ev: bool(ev.retrieval),
    "generator": lambda ev: True,          # a model call is a generator
    "output_parser": lambda ev: bool(ev.schemas or ev.structured),
    "agent": lambda ev: bool(ev.tools),
}


@dataclass
class Question:
    """Something the planner cannot decide alone.

    Questions are not errors. Most plans have several, and a plan with none is
    usually a plan that guessed. Each carries a default so that `--yes` can
    proceed unattended and the user can see afterwards what was assumed.
    """

    kind: str
    subject: str
    ask: str
    options: tuple[str, ...] = ()
    default: str | None = None
    blocking: bool = False


@dataclass
class Pipeline:
    """One generated eval file: a target, a risk, and the metrics for both."""

    target: str            # "retriever", "workflow", "application", ...
    level: Level
    risk: Risk
    archetype: Archetype
    metrics: list[Metric] = field(default_factory=list)
    sites: list[CallSite] = field(default_factory=list)
    rationale: tuple[str, ...] = ()

    @property
    def slug(self) -> str:
        return f"eval_{self.target}_{self.risk.value}".replace("-", "_")

    @property
    def runnable(self) -> list[Metric]:
        """Metrics that work with no human input."""
        return [m for m in self.metrics if m.automatable]

    @property
    def blocked(self) -> list[Metric]:
        """Metrics waiting on somebody to write the answers."""
        return [m for m in self.metrics if m.reference is not Reference.FREE]

    @property
    def needs_wiring(self) -> list[Metric]:
        """Metrics waiting on a key `call_app` does not return yet.

        Separate from `blocked` because the remedy is somewhere else entirely:
        one wants a golden answer written, the other wants a line added to
        harness.py.
        """
        return [m for m in self.metrics if m.needs_instrumentation]

    @property
    def needs_judge(self) -> bool:
        return any(m.method is Method.MODEL_GRADED for m in self.metrics)


@dataclass
class Plan:
    """Everything the generator needs, and everything the user should see."""

    pipelines: list[Pipeline] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    sites: list[CallSite] = field(default_factory=list)

    @property
    def metric_count(self) -> int:
        return sum(len(p.metrics) for p in self.pipelines)

    @property
    def runnable_count(self) -> int:
        return sum(len(p.runnable) for p in self.pipelines)

    @property
    def blocked_count(self) -> int:
        return sum(len(p.blocked) for p in self.pipelines)

    @property
    def wiring_count(self) -> int:
        return sum(len(p.needs_wiring) for p in self.pipelines)

    def summary(self) -> str:
        sites = len(self.sites)
        tail = f"{self.blocked_count} awaiting golden answers"
        if self.wiring_count:
            tail += f", {self.wiring_count} awaiting instrumentation"
        return (f"{sites} call site{'s' if sites != 1 else ''} -> "
                f"{len(self.pipelines)} pipelines, {self.metric_count} metrics "
                f"({self.runnable_count} runnable now, {tail})")


def _dedupe(metrics: list[Metric]) -> list[Metric]:
    """One metric per key. The catalogue deliberately repeats names across
    levels -- workflow faithfulness is not component faithfulness -- and `key`
    already carries the level, so this only drops genuine duplicates arising
    from several sites sharing a pipeline."""
    seen: dict[str, Metric] = {}
    for metric in metrics:
        seen.setdefault(metric.key, metric)
    return list(seen.values())


def _component_pipelines(site: CallSite, archetype: Archetype) -> list[Pipeline]:
    """Component evals, one per role the evidence actually supports."""
    out: list[Pipeline] = []
    by_role: dict[str, list[Metric]] = defaultdict(list)
    for metric in for_target(archetype, level=Level.COMPONENT):
        role = metric.component or "generator"
        if ROLE_EVIDENCE.get(role, lambda _ev: True)(site.evidence):
            by_role[role].append(metric)

    for role, metrics in by_role.items():
        for risk in {m.risk for m in metrics}:
            selected = [m for m in metrics if m.risk is risk]
            out.append(Pipeline(
                target=role, level=Level.COMPONENT, risk=risk,
                archetype=archetype, metrics=_dedupe(selected), sites=[site],
                rationale=site.rationale))
    return out


def _merge(pipelines: list[Pipeline]) -> list[Pipeline]:
    """Collapse pipelines that share a target, level and risk."""
    merged: dict[tuple[str, Level, Risk], Pipeline] = {}
    for pipeline in pipelines:
        key = (pipeline.target, pipeline.level, pipeline.risk)
        if key in merged:
            existing = merged[key]
            existing.metrics = _dedupe(existing.metrics + pipeline.metrics)
            existing.sites += [s for s in pipeline.sites if s not in existing.sites]
            existing.rationale = tuple(dict.fromkeys(existing.rationale + pipeline.rationale))
        else:
            merged[key] = pipeline
    return list(merged.values())


def _questions(sites: list[CallSite], pipelines: list[Pipeline]) -> list[Question]:
    out: list[Question] = []

    # Low-confidence classification changes which metrics apply, so it is worth
    # one question rather than a silent guess.
    for site in sites:
        if site.confidence < 0.3:
            out.append(Question(
                kind="archetype", subject=site.ident,
                ask=(f"{site.path}:{site.line} in {site.function}() was classified "
                     f"as '{site.archetype}' with low confidence. What is it?"),
                options=tuple(a.value for a in Archetype),
                default=site.archetype))

    # Scope adherence and over-refusal cannot be judged without knowing what the
    # assistant is supposed to refuse. There is no sensible default.
    if any(m.name in {"scope_adherence", "over_refusal"}
           for p in pipelines for m in p.metrics):
        out.append(Question(
            kind="policy", subject="scope",
            ask=("In one sentence: what is this assistant for, and what should "
                 "it decline? Scope adherence and over-refusal are measured "
                 "against this."),
            blocking=True))

    # One question per reference-based metric family, not per metric: nobody
    # wants to answer the same thing six times.
    blocked = {m.name: m for p in pipelines for m in p.blocked}
    for name, metric in sorted(blocked.items()):
        out.append(Question(
            kind="goldens", subject=name,
            ask=(f"'{name}' needs {metric.needs[0] if metric.needs else 'a reference answer'}. "
                 "Generated as a stub with inputs filled in; the answers are yours "
                 "to write. Queue it for review?"),
            options=("review", "skip"), default="review"))

    return out


def build_plan(sites: list[CallSite], include_safety: bool = True,
               include_operational: bool = True) -> Plan:
    """Decide the whole suite from a scan.

    `include_safety` and `include_operational` exist because both are
    application-wide rather than per-site, and a user who already has a
    guardrail layer or an APM they trust should be able to say so rather than
    delete generated files.
    """
    if not sites:
        return Plan()

    # Declining a risk declines it at every level, not only application level.
    # A user who has an APM they trust does not want component latency evals
    # either, and half-honouring the flag is worse than not offering it.
    wanted_risks = {Risk.QUALITY}
    if include_safety:
        wanted_risks.add(Risk.SAFETY)
    if include_operational:
        wanted_risks.add(Risk.OPERATIONAL)

    pipelines: list[Pipeline] = []

    for site in sites:
        archetype = Archetype(site.archetype)
        pipelines += _component_pipelines(site, archetype)

        # Workflow evals exist because components that are individually correct
        # still compose into a broken pipeline -- the retriever that returns the
        # right document at rank five while the generator answers from rank one.
        workflow = _dedupe(list(for_target(archetype, level=Level.WORKFLOW)))
        for risk in {m.risk for m in workflow}:
            pipelines.append(Pipeline(
                target="workflow", level=Level.WORKFLOW, risk=risk,
                archetype=archetype,
                metrics=[m for m in workflow if m.risk is risk],
                sites=[site],
                rationale=(f"{site.ident} composes several components; "
                           "component-level passes do not imply the composition works",)))

    # Application-level metrics are per-app, not per-site. The archetype used is
    # the most common one, which only affects which archetype-specific quality
    # metrics come along; safety and operational metrics apply universally.
    counts: dict[str, int] = defaultdict(int)
    for site in sites:
        counts[site.archetype] += 1
    dominant = Archetype(max(counts.items(), key=lambda kv: kv[1])[0])

    for risk in wanted_risks:
        metrics = _dedupe(list(for_target(dominant, level=Level.APPLICATION, risk=risk)))
        if metrics:
            pipelines.append(Pipeline(
                target="application", level=Level.APPLICATION, risk=risk,
                archetype=dominant, metrics=metrics, sites=list(sites),
                rationale=("measured end to end: correct components can still "
                           "compose into an application that is unsafe or too "
                           "slow to ship",)))

    merged = _merge([p for p in pipelines if p.risk in wanted_risks])
    # Stable, readable order: component work first, then composition, then the
    # whole app; quality before safety before operations within each level.
    order = {Level.COMPONENT: 0, Level.WORKFLOW: 1, Level.APPLICATION: 2}
    risk_order = {Risk.QUALITY: 0, Risk.SAFETY: 1, Risk.OPERATIONAL: 2}
    merged.sort(key=lambda p: (order[p.level], risk_order[p.risk], p.target))

    return Plan(pipelines=merged, questions=_questions(sites, merged), sites=sites)
