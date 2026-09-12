# The decision framework

What an eval planner has to decide, and on what grounds.

This is the reference the planner in `livingeval.plan` implements. It is written
down separately from the code because the decisions are the product — the code
is just the part that executes them — and because a planner whose reasoning
cannot be inspected is a planner nobody will trust with their codebase.

The taxonomy follows the structure taught in CampusX's *LLM Evaluation* series
([playlist](https://www.youtube.com/playlist?list=PLEneLIDJFpcA)), which is the
most complete public treatment of the question "where do evals go and which ones
do I write". Where this document departs from it, it says so and says why.

---

## 1. The five questions

Every eval pipeline is the answer to five questions, asked in order. The planner
answers them for each thing it finds in a codebase.

| # | question | answer space |
|---|---|---|
| 1 | **Where** does this go? | component · workflow · application |
| 2 | **What** risk does it cover? | quality · safety · operational |
| 3 | **Which** metric measures it? | see §4 |
| 4 | **How** is it executed? | programmatic · model-graded · human |
| 5 | **What** data does it need? | reference-based · reference-free |

Answers cascade. The target and risk pick the metric; the metric determines
whether a golden answer is required; that determines what a human must supply
before the eval can run at all.

---

## 2. Where: the three levels

An LLM application fails at three levels, and passing at one does not imply
passing at the next. This is the argument for multiple pipelines per app rather
than one.

**Component.** Any single piece can fail on its own: retriever, reranker, query
rewriter, embedding model, vector store, output parser, tool, memory, guardrail,
or the system prompt itself.

**Workflow.** Components that are individually correct still compose into a
broken pipeline. The canonical case: a retriever with `k=5` returns the right
document at rank 5 — recall is satisfied — while the generator, instructed to
weight earlier context more heavily, answers from rank 1. Neither component
misbehaved. The workflow did. Only an eval at the interaction edge sees it, and
the fix (a reranker) is invisible from either component's own metrics.

**Application.** Even a correct workflow can be undeployable: ten seconds to
first answer, or a cost per query that makes the product uneconomic.

> **Planner rule.** Never emit component evals alone. Any discovered component
> that feeds another gets a workflow eval at the edge, and every entrypoint gets
> application-level operational evals.

## 3. What: the three risk categories

Orthogonal to level. The cross product of level × risk is the pipeline set.

**Quality** — does it do its job? Correctness, relevance, completeness,
instruction-following, plus archetype-specific concerns (§4).

**Safety** — can it cause harm? Toxicity, harmful content, bias, PII leakage,
prompt injection and jailbreak resistance, scope adherence.

**Operational** — can it run fast, cheap and reliably? Latency, cost per
request, token efficiency, error rate, behaviour under load.

---

## 4. Which metric: the catalogue

Metrics are selected by (level, risk, archetype). The archetype is what the
scanner infers from the code.

### 4.1 Retrieval-augmented

| level | metric | what it catches | reference |
|---|---|---|---|
| component (retriever) | contextual recall | relevant documents missed | based |
| component (retriever) | contextual precision | irrelevant documents fetched | based |
| component (generator) | faithfulness | claims not grounded in context | free |
| component (generator) | answer relevancy | grounded but does not answer the question | free |
| workflow | the three above, over *retrieved* rather than golden context | composition failures | mixed |
| workflow | contextual relevance | noise *inside* otherwise-relevant chunks | free |

The last row is the diagnostic that earns its place. Precision at 89% with
contextual relevance at 42% is not a contradiction: precision counts chunks that
contain something useful, contextual relevance counts useful *claims within*
them. High precision and low contextual relevance localises the fault precisely
— the chunks are right and too big — and the fix is chunk size, not the
retriever.

> **Planner rule.** Emit metrics that triangulate, not metrics that duplicate.
> A metric earns its place if some combination of it and its neighbours
> distinguishes a fault that neither isolates alone.

### 4.2 Generic generation

Correctness (against a reference), completeness (all parts of a multi-part
question addressed), instruction-following, style/tone, helpfulness, coherence.

**Correctness is not faithfulness.** Faithfulness asks whether the answer follows
from the retrieved context; correctness asks whether it is true of the world.
All four combinations occur, including faithful-and-wrong when the source itself
is wrong. An eval suite that measures only faithfulness certifies that the system
reproduces its corpus, including the corpus's errors.

### 4.3 Agents

Tool selection, parameter correctness, task completion, error recovery,
termination.

### 4.4 Multi-turn

Context retention across turns, clarification behaviour when the request is
underspecified.

### 4.5 Safety

Toxicity, PII leakage, system-prompt leakage, proprietary-content leakage, scope
adherence, jailbreak and injection resistance, bias.

### 4.6 Operational

Latency (mean, P50, P95, P99, and a per-component breakdown), time to first
token, cost per query split input/output, error rate, timeout rate, retry rate.

---

## 5. How: the three execution methods

The method is not a preference. It is forced by whether the criterion is
computable.

**Programmatic** when the criterion reduces to counting or comparison against a
key — recall against known-relevant document ids, classification accuracy,
latency, cost, error rate. Use it whenever it is available: it is free,
instant and exactly repeatable.

**Model-graded (LLM-as-judge)** when the criterion needs judgement and the
volume rules out people. Everything holistic: correctness, completeness, style,
helpfulness, most of safety.

**Human** for what neither can do: writing the golden set in the first place,
red-teaming, A/B testing on live users, and adjudicating cases the other two
methods flag as uncertain.

### 5.1 Count-based versus judgement-based, and why it matters

Model-graded metrics split into two mechanisms with very different reliability.

*Count-based* metrics decompose an output into claims, test each claim against
something, and report a ratio. Faithfulness, answer relevancy, contextual
relevance and toxicity all work this way. Each individual judgement is small and
local, so the aggregate is comparatively stable.

*Judgement-based* metrics score the whole output at once, because the property
does not survive decomposition. Style is the clearest case: an answer is written
in a house voice as a whole, not sentence by sentence. Correctness is another —
an analogy scored in isolation looks like an unsupported claim.

Naive holistic scoring is unreliable for two compounding reasons: a one-line
criterion is re-interpreted differently on each call, and asking for an integer
score makes the output hostage to which of two adjacent digits happens to win the
token sampling. Runs swing by two points with no change to the system.

**G-Eval** fixes both. It expands the criterion into explicit evaluation steps
via chain-of-thought (so the rubric stops drifting between calls), and it reads
the *log-probabilities* of the candidate score tokens and returns their weighted
average instead of the argmax (so a judge torn between 7 and 8 returns 7.4 rather
than flipping). Use it for every judgement-based metric.

> **Planner rule.** Programmatic where computable. G-Eval, never bare
> LLM-as-judge, where not. Human only where neither can go.

### 5.2 The judge-authoring ladder

A G-Eval metric is not written correctly the first time. The progression is:

1. **Criterion only.** Let the judge generate its own evaluation steps. Fastest
   to write, highest variance.
2. **Fixed evaluation steps.** After a few runs you have seen how it fails.
   Freeze the steps; the per-call step-generation variance disappears.
3. **Fixed rubric.** Pin the score bands explicitly. Now scoring variance goes
   too, and repeated runs land within a point of each other.

This ladder is mechanisable, and §8 is where it becomes the planner's job rather
than the user's.

---

## 6. What data: reference-based versus reference-free

One question decides it: **does the golden set contain the right answer?**

Reference-based needs a human to write the answer, or the document ids, or the
marks. Reference-free needs only inputs; the criterion is a rubric applied to
whatever comes out.

This is the single most important fact for a tool that generates evals, because
it is the boundary between what can be produced automatically and what cannot.
Reference-free metrics — faithfulness, answer relevancy, contextual relevance,
toxicity, PII leakage, scope adherence, and every operational metric — need
inputs and a rubric, both of which are derivable. Reference-based metrics need a
person.

> **Planner rule.** Everything reference-free ships working on day one.
> Everything reference-based ships as a stub with the inputs filled in, the
> answer column empty, and a review queue entry. It is never silently populated
> from the current system's own output — that produces a suite which certifies
> today's behaviour as correct by definition, which is the failure
> `EvalCase.provenance` exists to prevent.

Reference-free still means *inputs are needed*, which is easy to lose sight of:
a metric with a rubric and no cases does not fail, it vanishes from the report.
So livingeval generates an input set for every reference-free metric, shaped by
what the metric watches — an instruction-following metric scored on inputs
carrying no instruction measures nothing and scores well doing it. Those inputs
are the tool's guesses at your traffic and are marked as such, which is why
`drift` leaves them out of coverage.

There is a third category the source framework does not separate, because it
only appears once a tool is writing the harness rather than a person:
**reference-free but uninstrumented**. Time to first token needs a streamed
call; per-stage latency needs stage timings; the correctness of a tool call
needs the arguments. Nothing a person writes into a golden set will supply
these — they need one more key returned from the code under test. livingeval
counts them separately, names them every run, and the gate reports them BLIND
rather than passing them on no data.

### 6.1 Golden set composition

A safety golden set that contains only attacks measures the wrong thing. Three
case classes are needed:

- **Adversarial** — trying to induce the failure.
- **Benign** — normal requests that merely *look* like attacks. A question about
  how to evaluate toxicity is a legitimate question; a suite without these
  optimises the system into refusing it.
- **Mixed** — a legitimate request with an illegitimate rider, where the correct
  behaviour is to answer the first part and decline the second.

Benign and mixed cases exist to bound the false-positive rate. Without them,
"safe" and "useless" are indistinguishable.

---

## 7. The loop

The workflow is a loop, not a pipeline, and it does not stop at deploy.

```
define target -> define criteria -> build golden set -> choose method
      -> run -> score -> analyse failures -> improve -> re-run
      -> deploy -> monitor -> feed production failures back into the golden set
```

The closing edge is what makes a suite living. Failures found in production are
added to the offline golden set, so the suite grows toward the traffic it
actually sees.

### 7.1 Offline and online are different questions

**Offline** runs before deploy against a fixed golden set with an answer key. It
is cheap, repeatable, and gates releases. It asks: *is this correct?*

**Online** runs after deploy against live traffic with no answer key. It is
expensive, so it samples. It asks: *is this normal?*

Correctness is frequently unavailable in production — nobody has written the
answer for a question asked thirty seconds ago. What remains is distributional:
compare this week's score distribution against the established baseline and
alert on the shift. Plus reference-free metrics, which need no key and so work
identically in both settings, and captured behavioural signals — thumbs, hand-off
requests, repeated rephrasing, abandonment.

### 7.2 Sampling should be stratified

Uniform sampling of production traffic spends most of the evaluation budget on
conversations that went fine. Stratify and oversample the strata where failures
concentrate: thumbs-down, escalations, abrupt endings, repeated rephrasings of
the same question, and anything touching money.

### 7.3 Captured versus computed signals

**Captured** signals already exist and only need storing — latency, tokens, cost,
status codes, thumbs. They go straight to a dashboard and an alert.

**Computed** signals need an evaluator to produce them — faithfulness,
hallucination, toxicity. These are what sampling is for.

Logging requirements: non-blocking, durable and queryable, able to attach late
signals (an escalation that arrives the next day still belongs to its
conversation), and PII-masked at write time.

---

## 8. Where this framework stops, and what livingeval adds

The framework above is a complete account of *how to build an eval suite*. It is
deliberately silent on three questions, and those three are livingeval's reason
to exist.

**Is a change real, or is it noise?** The standard recipe — run the baseline ten
times, take twice the standard deviation as a noise threshold, call anything
smaller noise — is the right instinct implemented as a rule of thumb. It has no
false-positive rate, no power, and no notion of significance. livingeval replaces
it with a paired test and reports the p-value. *(`livingeval.stats`)*

**Could this suite have caught the regression at all?** Nobody asks. A suite that
reports no change might be reporting that nothing changed, or that it is
incapable of noticing. livingeval simulates a named regression and reports the
probability the gate turns red, alongside its false-alarm rate.
*(`livingeval.power`)*

**Does the suite still represent the traffic?** The framework names this problem
exactly — as the business changes its documents, a golden set written a year ago
quietly stops describing reality, and the suite keeps passing while the product
degrades. Its answer is to watch score distributions drift. livingeval measures
it directly: what share of today's traffic sits near something the suite
contains, at a radius calibrated from the suite's own spread. *(`livingeval.mine`)*

The gate is three-valued because of the second and third. `PASS`, `FAIL`, and
`BLIND` — "no regression detected, and this suite could not have detected one"
is a different claim from "no regression", and CI should be able to tell them
apart.

**And the framework is silent on authoring.** Every step above assumes a person
who already knows which of the fifteen metrics applies to the component in front
of them. That assumption is the barrier this tool removes: §§2–6 are a decision
procedure, decision procedures can be executed, and the inputs — components,
their wiring, their prompts, their schemas — are recoverable from source.
