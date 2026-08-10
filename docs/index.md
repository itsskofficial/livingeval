# livingeval documentation

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
