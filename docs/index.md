# livingeval documentation

- **[guide.md](guide.md)** — the long-form walkthrough: the problem, the three
  measurements, the ladder, the platform, and how to run any of it on your own traces.
- **[decision-framework.md](decision-framework.md)** — the procedure `init` follows to
  decide where an eval goes, which risk it covers and which metric measures it, with
  its sources and the three places livingeval departs from them.
- **[concepts.md](concepts.md)** — what coverage, power and judge-depth actually mean,
  and how each is computed. Start here if any of the words below are unfamiliar; every
  term is defined before it is used.
- **[recipes.md](recipes.md)** — wiring it into CI, pointing it at your own traces,
  running the ladder against a real judge.
- **[platform.md](platform.md)** — the local dashboard, the review queue, per-turn
  online scoring, storage and live ingestion.
- **[../DECISIONS.md](../DECISIONS.md)** — why each methodological choice was made, and
  what was rejected.
- **[../ROADMAP.md](../ROADMAP.md)** — what is deliberately out of scope.

## The one-paragraph version

An eval score is a score on the distribution the suite was drawn from. `livingeval`
measures three things that decide whether that score means anything today:
**coverage** (does the suite still represent your traffic?), **power** (would it fire
if the regression you fear were present?) and **judge depth** (could something far
cheaper have produced the judge's labels?). It returns all three next to the score
rather than requiring you to ask, and its CI gate has three outcomes — `PASS`, `FAIL`
and `BLIND` — because "the suite passed" and "the suite is no longer capable of
failing" are different states.

## Install

```bash
pip install livingeval                    # numpy, scipy, scikit-learn. Nothing else.
pip install 'livingeval[openai]'          # + OpenAI judges and embeddings
pip install 'livingeval[anthropic]'       # + Anthropic judges
pip install 'livingeval[embed]'           # + local Hugging Face embeddings
pip install 'livingeval[finetune]'        # + the fine-tuned rung
pip install 'livingeval[serve]'           # + the local platform
pip install 'livingeval[postgres]'        # + Supabase / Postgres storage
pip install 'livingeval[figures]'         # + matplotlib
pip install 'livingeval[all]'             # everything that installs without a GPU
```

Every extra is resolved lazily. A CI job fails if any of it leaks into the default
import path, and a subprocess test checks `sys.modules` to be sure.

## Thirty seconds

```bash
python examples/01_quickstart.py     # the three measurements, in a terminal
python scripts/demo.py --fresh       # the whole loop end to end, ~15s
livingeval serve --demo              # the platform, on http://127.0.0.1:8000
```

## Or point it at your own codebase

```bash
livingeval scan --explain        # the call sites it found, and what it would write
livingeval init                  # asks about the judge and the scope, then generates
python -m livingeval_evals.run_suite
livingeval baseline --runs 10    # replace the guessed noise thresholds with measured ones
livingeval sync                  # after the code moves; your answers are preserved
livingeval drift --traces langfuse:limit=1000    # is the suite still about your traffic
livingeval gate                  # PASS (0) / FAIL (1) / BLIND (2)
```

Every generated file explains why it exists, and `livingeval_evals/WHY.md` collects the
reasoning in one place — including which metrics it cannot measure yet and the single
line of instrumentation each one needs.
