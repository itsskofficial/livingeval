"""The planner's decisions, asserted as decisions rather than as output shape.

Each test names a judgement the planner is supposed to make. If one fails the
right question is "was the judgement wrong?", not "did the format change?".
"""

from __future__ import annotations

from pathlib import Path

from livingeval.discover.sites import CallSite, Evidence
from livingeval.plan import Level, Reference, Risk, build_plan
from livingeval.plan.taxonomy import CATALOGUE, Archetype


def site(archetype: Archetype, **evidence) -> CallSite:
    return CallSite(
        path=Path("app.py"), line=1, function="answer", provider="openai",
        evidence=Evidence(**evidence), archetype=archetype.value,
        confidence=0.8, rationale=("fixture",))


# ---------------------------------------------------------------------------
# the catalogue itself
# ---------------------------------------------------------------------------


def test_every_metric_states_what_it_catches():
    """A metric nobody can explain is a metric nobody will keep."""
    assert all(m.catches for m in CATALOGUE)


def test_reference_based_metrics_say_what_they_need():
    """The user has to be told exactly what to write."""
    for metric in CATALOGUE:
        if metric.reference is Reference.BASED:
            assert metric.needs, f"{metric.key} needs a human but does not say for what"


def test_judgement_metrics_carry_a_criterion():
    """Holistic scoring needs G-Eval, and G-Eval needs a criterion to expand."""
    from livingeval.plan.taxonomy import Mechanism, Method
    for metric in CATALOGUE:
        if metric.mechanism is Mechanism.JUDGEMENT and metric.method is Method.MODEL_GRADED:
            assert metric.geval_criterion, f"{metric.key} scores holistically with no criterion"


def test_lower_is_better_metrics_are_marked():
    """The regression comparison reads this. Getting it backwards turns a
    safety improvement into a reported regression."""
    lower = {m.name for m in CATALOGUE if not m.higher_is_better}
    assert {"toxicity", "pii_leakage", "latency_p95", "cost_per_query",
            "error_rate", "over_refusal"} <= lower


# ---------------------------------------------------------------------------
# what gets planned
# ---------------------------------------------------------------------------


def test_rag_gets_the_full_triad_at_the_workflow_level():
    plan = build_plan([site(Archetype.RAG, retrieval=["similarity_search"])])
    workflow = [p for p in plan.pipelines if p.level is Level.WORKFLOW]
    names = {m.name for p in workflow for m in p.metrics}
    assert {"faithfulness", "answer_relevancy", "contextual_relevance"} <= names


def test_workflow_evals_are_planned_even_when_components_pass():
    """The whole argument for workflow evals: components that are individually
    correct still compose into a broken pipeline."""
    plan = build_plan([site(Archetype.RAG, retrieval=["similarity_search"])])
    assert any(p.level is Level.WORKFLOW for p in plan.pipelines)


def test_retriever_metrics_are_skipped_when_there_is_no_retriever():
    """Planning a retriever eval for a pipeline with no retriever reports a
    fake zero and wastes a run."""
    plan = build_plan([site(Archetype.GENERIC)])
    targets = {p.target for p in plan.pipelines}
    assert "retriever" not in targets


def test_agent_gets_tool_and_trajectory_metrics():
    plan = build_plan([site(Archetype.AGENT, tools=["bind_tools"], has_loop=True)])
    names = {m.name for p in plan.pipelines for m in p.metrics}
    assert {"tool_selection", "parameter_correctness", "task_completion"} <= names


def test_extraction_gets_schema_assertions_that_need_no_human():
    """These are why a generated suite is worth anything on day one."""
    plan = build_plan([site(Archetype.EXTRACTION, schemas=["Invoice"],
                            structured=["response_format"])])
    runnable = {m.name for p in plan.pipelines for m in p.runnable}
    assert {"schema_validity", "required_fields"} <= runnable


def test_safety_and_operational_are_planned_for_every_app():
    plan = build_plan([site(Archetype.GENERIC)])
    risks = {p.risk for p in plan.pipelines}
    assert Risk.SAFETY in risks and Risk.OPERATIONAL in risks


def test_safety_and_operational_can_be_declined():
    plan = build_plan([site(Archetype.GENERIC)],
                      include_safety=False, include_operational=False)
    risks = {p.risk for p in plan.pipelines}
    assert Risk.SAFETY not in risks and Risk.OPERATIONAL not in risks


def test_over_refusal_ships_with_the_safety_metrics():
    """Without it, 'safe' and 'useless' are indistinguishable and every safety
    fix looks like a win."""
    plan = build_plan([site(Archetype.GENERIC)])
    names = {m.name for p in plan.pipelines for m in p.metrics}
    assert "over_refusal" in names


def test_empty_scan_produces_an_empty_plan_not_a_crash():
    plan = build_plan([])
    assert plan.pipelines == [] and plan.questions == []


# ---------------------------------------------------------------------------
# the human boundary
# ---------------------------------------------------------------------------


def test_reference_based_metrics_are_reported_blocked_never_auto_filled():
    """Filling these from the current system's own output builds a suite that
    certifies today's behaviour as correct by definition."""
    plan = build_plan([site(Archetype.RAG, retrieval=["similarity_search"])])
    blocked = {m.name for p in plan.pipelines for m in p.blocked}
    assert {"contextual_recall", "contextual_precision", "correctness"} <= blocked
    assert all(m.reference is Reference.BASED
               for p in plan.pipelines for m in p.blocked)


def test_most_of_a_rag_suite_runs_before_anybody_writes_an_answer():
    """The adoption claim, asserted."""
    plan = build_plan([site(Archetype.RAG, retrieval=["similarity_search"])])
    assert plan.runnable_count >= 3 * plan.blocked_count


def test_low_confidence_classification_raises_a_question():
    ambiguous = site(Archetype.RAG, retrieval=["similarity_search"])
    ambiguous.confidence = 0.1
    plan = build_plan([ambiguous])
    assert any(q.kind == "archetype" for q in plan.questions)


def test_scope_metrics_force_a_blocking_question():
    """Scope adherence cannot be judged without knowing the intended scope, and
    there is no sensible default for it."""
    plan = build_plan([site(Archetype.GENERIC)])
    scope = [q for q in plan.questions if q.kind == "policy"]
    assert scope and scope[0].blocking


def test_goldens_are_asked_once_per_metric_not_once_per_site():
    many = [site(Archetype.RAG, retrieval=["similarity_search"]) for _ in range(5)]
    for i, s in enumerate(many):
        s.line = i + 1
    plan = build_plan(many)
    asked = [q.subject for q in plan.questions if q.kind == "goldens"]
    assert len(asked) == len(set(asked))


# ---------------------------------------------------------------------------
# explainability
# ---------------------------------------------------------------------------


def test_every_pipeline_carries_a_reason():
    plan = build_plan([site(Archetype.RAG, retrieval=["similarity_search"])])
    assert all(p.rationale for p in plan.pipelines)


def test_summary_reports_the_human_boundary():
    plan = build_plan([site(Archetype.RAG, retrieval=["similarity_search"])])
    assert "runnable now" in plan.summary()
