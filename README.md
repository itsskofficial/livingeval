<br/>
<p align="center">

  <h3 align="center">livingeval</h3>

  <p align="center">
    Eval suites that tell you when they've gone blind — coverage, detection power and judge depth for LLM agents
    <br/>
    <br/>
    <a href="./docs/index.md">Read The Docs</a>
    .
    <a href="https://github.com/itsskofficial/livingeval/issues">Report Bug</a>
    .
    <a href="https://github.com/itsskofficial/livingeval/issues">Request Feature</a>
  </p>
</p>

![License](https://img.shields.io/github/license/itsskofficial/livingeval)
![Python](https://img.shields.io/pypi/pyversions/livingeval)
![PyPI](https://img.shields.io/pypi/v/livingeval)

## Table Of Contents

* [About the Project](#about-the-project)

* [Built With](#built-with)

* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)

  * [Installation](#installation)

* [Usage](#usage)

* [How It Works](#how-it-works)

* [Design Decisions](#design-decisions)

* [Roadmap](#roadmap)

* [Contributing](#contributing)

* [License](#license)

* [Authors](#authors)

* [Acknowledgements](#acknowledgements)

## About The Project

**Point it at your repository. It writes the evals.**

```sh
pip install livingeval
livingeval init
```

It reads your code, finds every LLM call, works out what each one *is* — a RAG
pipeline, a structured extractor, a classifier, an agent — and writes the eval
suite that shape of thing needs. It explains every choice as it goes, asks you
about the handful of things it genuinely cannot know, and never invents an
answer it should have asked for.

On a typical RAG application, one call site produces:

```
1 call site -> 7 pipelines, 22 metrics (18 runnable now, 4 awaiting golden answers)

  eval_retriever_quality        contextual_recall, contextual_precision
  eval_generator_quality        faithfulness, answer_relevancy
  eval_workflow_quality         faithfulness, answer_relevancy, contextual_relevance
  eval_application_quality      correctness, completeness, instruction_following, style
  eval_application_safety       toxicity, pii_leakage, prompt_leakage, scope_adherence,
                                injection_resistance, over_refusal
  eval_application_operational  latency_p95, time_to_first_token, cost_per_query, error_rate
```

**18 of those run before you write a single line.** Everything reference-free —
grounding, relevance, safety, latency, cost — needs only inputs and a rubric,
and both are derivable. The other four need a human, and are written as stubs
with the inputs filled in and the answer column deliberately empty.

That last point is the one that matters. The easy way to generate a "complete"
suite is to run your code and record what it says. That builds a suite which
passes because the cases were harvested from what the model already does — a
change detector wearing a correctness test's clothes. livingeval will not do it,
and says so in the file.

### Why these evals and not others

Every generated file explains itself, and `livingeval scan --explain` shows the
reasoning before anything is written:

```
myapp/rag.py:14  answer()
    openai -> rag (50% confident)
      - retrieval before generation (store.similarity_search)

eval_workflow_quality  [workflow/quality]
     faithfulness             grounding failures that only appear on real retrieved context
     contextual_relevance     noise inside chunks that are themselves relevant -- high
                              precision beside low relevance means the chunks are too big
```

The decision procedure is written down in
[docs/decision-framework.md](docs/decision-framework.md): where an eval goes
(component, workflow, application), which risk it covers (quality, safety,
operational), which metric measures it, how it is executed (programmatic,
model-graded, human), and whether it needs a reference answer. The catalogue is
[data, not code](src/livingeval/plan/taxonomy.py), so you can read the whole
thing and disagree with it.

### And then the part nobody else does

A suite tells you your scores changed. It cannot tell you whether the change was
real, whether it could have caught the regression at all, or whether it still
describes your traffic. livingeval measures all three.

**Coverage** is the share of today's traffic sitting near something the suite
contains, at a radius calibrated from the suite's own spread rather than a
constant somebody picked. **Detection power** is the probability the gate turns
red under a regression you *name*, established by simulation and always reported
beside its false-alarm rate. **Judge depth** is the cheapest scorer that
reproduces your LLM judge — if a keyword rule matches it, the judge was
decoration, and the tool says so.

So the gate is three-valued rather than two. `PASS (0)`, `FAIL (1)`, and
`BLIND (2)`, because "no regression detected, and this suite could not have
detected one" is a different claim from "no regression", and CI should be able
to treat them differently.

```
-> BLIND  (exit 2)
   score fell by 0.0000 but the paired test does not separate it from noise (p=1)
   coverage 30.0% is below 70%: 70% of traffic is unrepresented by this suite
   simulated detection power 1.2% is below 80% for the regression tested
```

Measuring the gap is only half of it. The platform mines real traces from the
clusters your suite cannot see, queues them for a human to confirm, and merges
the confirmed ones back into the suite — which is what makes it *living* rather
than a linter.

## Built With

This section lists the major frameworks and tools used in the project:

* [NumPy](https://numpy.org/)

* [SciPy](https://scipy.org/)

* [scikit-learn](https://scikit-learn.org/)

* [FastAPI](https://fastapi.tiangolo.com/)

* [PostgreSQL](https://www.postgresql.org/)

* [Supabase](https://supabase.com/)

* [Ollama](https://ollama.com/)

* [PyTorch](https://pytorch.org/)

* [Transformers](https://huggingface.co/docs/transformers)

* [Langfuse](https://langfuse.com/)

* [Docker](https://www.docker.com/)

* [Terraform](https://www.terraform.io/)

## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

* **Python:**

  Ensure you have Python 3.10 or above installed.

  ```sh
  python --version
  ```

* **Docker** *(optional)*:

  Only needed for the full local Supabase stack.

  ```sh
  docker --version
  ```

* **Ollama** *(optional)*:

  Only needed if you want a local judge instead of a hosted one.

  ```sh
  ollama --version
  ```

### Installation

1. **Install from PyPI**

   ```sh
   pip install livingeval
   ```

   The default install is NumPy, SciPy and scikit-learn — no torch, no API client, no web framework and no network, so the whole gating test suite runs on a laptop CPU in seconds.

2. **Or install the extras you need**

   ```sh
   pip install 'livingeval[all]'
   ```

   This adds judges, embeddings, fine-tuning and the local platform.

3. **Or clone and install for development**

   ```sh
   git clone https://github.com/itsskofficial/livingeval.git
   cd livingeval
   pip install -e '.[dev,all]'
   ```

4. **Configure what you need** *(optional)*

   ```sh
   cp .env.example .env
   ```

   Nothing is required to run locally. The keys are for things with no local equivalent — a hosted LLM judge, a hosted trace store — and every feature that does not need them works without them.

## Usage

### Write a suite for your codebase

```sh
livingeval scan --explain        # what is there, and what it would write
livingeval init                  # interactive: judge, scope, then generate
livingeval init --yes            # take every default, report what was assumed
```

`init` asks about four things and decides the rest: which model grades the
judgement-based metrics, its API key (written to `.env`, never committed),
what the assistant is *for* — scope adherence and over-refusal are unmeasurable
without it — and any call site it could not classify confidently.

It writes `livingeval_evals/`: one file per pipeline, the datasets, a metric
registry with direction and noise tolerance, a suite runner, and `WHY.md`
explaining every choice.

```sh
python -m livingeval_evals.run_suite     # run it, write baseline.json
livingeval baseline --runs 10            # measure real noise thresholds
# ... change something ...
python -m livingeval_evals.run_suite     # writes candidate.json
livingeval gate                          # PASS (0) / FAIL (1) / BLIND (2)
```

### Keep it in step with the code

```sh
livingeval sync --dry-run    # what changed, and what would be written
livingeval sync              # update the suite to match the code
```

A suite generated once describes the application as it was that day, which is
the exact decay this library exists to measure — shipping it in the tool itself
would be absurd. `sync` rescans, diffs against a manifest, and reports before it
writes:

```
1 new call sites, 0 changed, 0 gone -> 5 metrics to add, 0 now unused

  + svc/agent.py:8 act()  (agent)

  + component.tool_selection
  + workflow.task_completion
  ...
```

The diff is over *what decides the metrics*, not the source, so a call site that
slid twelve lines down is unchanged and one that gained a retriever is not.

**It will not destroy your work.** Golden answers are merged, never
regenerated — new cases appended, existing answers untouched. Generated files
you have edited are reported and skipped. Call sites that vanished are reported
stale, never deleted, because static analysis cannot tell "deleted" from "moved
somewhere I can't see".

```
  wrote 14 files
  preserved 3 human-written answers across 1 datasets
  skipped 1 edited files
```

### Watch it drift away from your traffic

```sh
livingeval drift --traces 'traces/*.jsonl'        # or a Langfuse/OTel export
```

The suite is written from source code, and source code says nothing about what
users actually send. Every metric can be green while most of your traffic sits
somewhere none of the cases go:

```
-> BLIND  coverage 8.3% of 120 traces
   below 70%: most of your traffic is unrepresented, so a green suite is not
   evidence the application is working

  traffic this suite cannot see:

     24.2% of traffic,  0.0% covered  billing charged, this month, twice
     24.2% of traffic,  0.0% covered  refund for, refund, the annual plan
     15.8% of traffic,  0.0% covered  subscription, transfer, to colleague
```

A number tells you to act; the cluster table tells you what to write. Those
clusters feed the review queue in `livingeval serve`, and `livingeval promote`
merges the cases a human confirms back into the suite — which is the loop that
makes it *living*.

`gate` compares every metric, direction-aware, and where the suite recorded
per-case outcomes it runs exact McNemar on them rather than comparing against a
threshold. Twenty-two metrics is twenty-two tests, so it corrects across the
family with Benjamini-Hochberg — uncorrected, a suite that size manufactures a
false alarm most runs, which is how teams learn to ignore their own gate.

```
-> FAIL  (exit 1)
   workflow.faithfulness fell 0.3500 (p=0.000, q=0.011): grounding failures
   that only appear on real retrieved context

  - workflow.faithfulness       0.900 ->  0.550  REGRESSED     p=0.000 q=0.011
    application.toxicity        0.150 ->  0.150  within noise  p=1.000 q=1.000
```

Until `livingeval baseline` has measured the real spread, metrics without
per-case data are compared against shipped estimates, and the gate returns
**BLIND rather than PASS** — comparing against a guess is not evidence of
stability:

```
-> BLIND  (exit 2)
   4 of 22 metrics were compared against estimated noise thresholds with no
   per-case data: run `livingeval baseline --runs 10` to measure them
```

### See it work on bundled data

```sh
livingeval serve --demo
```

Open `http://localhost:8000` for the dashboard and the review queue.

### Point it at your own traces

```sh
livingeval ingest     --traces 'traces/*.jsonl'          # or --follow to poll Langfuse
livingeval coverage   --suite suite.json --traces 'traces/*.jsonl'
livingeval blindspots --suite suite.json --traces 'traces/*.jsonl'
livingeval ladder     --suite suite.json --judge openai:gpt-4o-mini
livingeval gate       --suite suite.json --traces 'traces/*.jsonl'
```

### As a library

```python
import livingeval as le

traces = le.trace.ingest.read_jsonl("traces/*.jsonl")
suite = le.EvalSuite.load("suite.json")

cov  = le.mine.coverage(suite, traces)
gaps = le.mine.blindspots(suite, traces)               # ranked by traffic share

run = le.gate.run_suite(suite, judge)
pw  = le.power.estimate(suite, run, le.power.on_meta("intent", "refund", effect=0.4))

print(le.gate.evaluate(run, baseline, coverage=cov.coverage, power=pw.power.point))
```

### In CI

```yaml
- run: livingeval gate --suite suite.json --traces traces.jsonl
  # exit 0 PASS · 1 FAIL · 2 BLIND
```

### Keeping the suite alive

```sh
livingeval serve                       # dashboard + review queue on :8000
livingeval promote --suite my-suite    # confirmed cases join the suite
```

Nothing enters the suite until a person clicks. A suite labelled entirely by the judge it is used to check is circular.

### Running the whole stack locally

Real Postgres, real Storage, real row-level security — the only thing that changes in production is credentials.

```sh
python scripts/dev.py up       # local Supabase, migrations, bucket, seed
python scripts/dev.py serve    # the platform against it
```

## How It Works

```
  your traces (Langfuse / OTel / JSONL)
         │
         ▼
   ┌───────────┐  coverage · detection power · judge depth
   │  measure  │
   └───────────┘
         │
         ├──► the gaps, ranked by traffic share
         │         │
         │         ▼
         │   ┌───────────┐  real traces from those gaps,
         │   │  propose  │  queued for a human to confirm
         │   └───────────┘
         │         │
         │         ▼
         │   ┌───────────┐  confirmed cases join the suite;
         │   │  promote  │  oldest retire past a cap
         │   └───────────┘
         │
         └──► CI gate:  PASS (0) · FAIL (1) · BLIND (2)
```

1. **Ingest.** Traces arrive from Langfuse, OpenTelemetry or JSONL into a store — SQLite locally, Postgres or Supabase in production.

2. **Measure.** Coverage compares this week's traffic against the suite in an embedding space. Power simulates a named regression a few hundred times through the real gate. The ladder cross-validates cheap scorers against the judge, grouped by session so no conversation straddles a train/test split.

3. **Mine.** Traffic is clustered, and the clusters the suite covers least are ranked by how much traffic each carries.

4. **Review.** Real traces from those clusters are queued in a web UI for a human to confirm or reject.

5. **Promote.** Confirmed cases merge into the suite, and coverage and power are measured again on the same traffic.

6. **Gate.** CI exits non-zero on `FAIL`, and differently on `BLIND`.

## Design Decisions

* **Every number ships with its control.** Detection power means nothing without the false-alarm rate from the same code path at zero effect, so both are always reported.

* **Power is defined against a named alternative.** "Your power is 0.4" says nothing without stating *against what*, so a regression is an object you construct and it goes into the record.

* **The judge-depth ladder can embarrass the method.** It was built to be able to report that a keyword rule reproduces your LLM judge. A metric that can only flatter the tool is not a measurement.

* **`BLIND` is a separate exit code, not a warning.** Folding it into `PASS` would hide exactly the failure this project exists to name.

* **Confidence intervals everywhere.** Small suites produce unstable estimates — a ranking computed on 60 traces inverted when recomputed on 480. Intervals make that visible rather than hiding it behind three significant figures.

* **Humans confirm every mined case.** Auto-labelling proposals with the judge and then using them to check that judge is circular.

* **The engine knows nothing about your domain.** An import-linter contract forbids scenario code in `src/livingeval/`, which is what lets it point at anyone's agent.

* **The default install stays offline.** Torch, API clients and the web stack are extras resolved lazily, and a subprocess test asserts they never leak into the default import path.

Every choice, and the alternative that was rejected, is recorded in [DECISIONS.md](./DECISIONS.md).

## Roadmap

* More trace adapters — LangSmith, Braintrust, Arize, Phoenix

* Sequential testing, so CI can stop early without inflating the false-alarm rate

* Multi-turn coverage, since conversations that fail on turn six are a different distribution

* Cost-aware suite selection: maximise detection power per judge dollar

* Drift alerting via Prometheus metrics

* Calibrated judge ensembles with abstention

See [ROADMAP.md](./ROADMAP.md) for what v0.1 deliberately does not do, and why.

## Contributing

Contributions make the open source community a fantastic place to learn, inspire, and create. All contributions are highly appreciated.

If you have ideas for improvement, feel free to fork the repo and create a pull request or open an issue with the tag "enhancement". Don't forget to give the repository a star!

### Creating A Pull Request

1. Fork the Project

2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)

3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)

4. Push to the Branch (`git push origin feature/AmazingFeature`)

5. Open a Pull Request

Before opening a pull request, please run the checks:

```sh
pytest
ruff check src/ tests/
mypy src/
lint-imports
```

## License

Distributed under the Apache License 2.0. See [LICENSE](https://github.com/itsskofficial/livingeval/blob/main/LICENSE) for more information.

## Authors

* **Sarthak Karandikar** - [itsskofficial](https://github.com/itsskofficial)

## Acknowledgements

* [Hewitt & Liang, *Designing and Interpreting Probes with Control Tasks*](https://aclanthology.org/D19-1275/) — the selectivity argument this project borrows

* [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

* [Langfuse Documentation](https://langfuse.com/docs)

* [Supabase Documentation](https://supabase.com/docs)

* [FastAPI Documentation](https://fastapi.tiangolo.com/)
