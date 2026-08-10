# The platform

```bash
pip install 'livingeval[serve]'
livingeval serve --demo          # http://127.0.0.1:8000
```

`--demo` seeds a drifting corpus and a suite curated from its oldest window, then fits
and deploys a scorer. Nothing about the demo is staged: the suite really is six windows
old, so coverage really does fall and the blind spot really is the intent that appeared
later.

Three pages and a REST API over one SQLite file. **Local by design** — it binds
`127.0.0.1`, has no authentication, and is a control surface for your own machine. See
[DECISIONS.md #25](../DECISIONS.md) for why it deliberately stops there.

---

## The dashboard — `/`

The three numbers that decide whether a score means anything, live, with a button for
each measurement.

| card | what it shows |
|---|---|
| **Coverage** | fraction of traffic the suite represents, and what fraction is not |
| **Detection power** | simulated, averaged over clusters — with the **worst cluster** called out beside it, because the average is what hides a blind spot |
| **CI gate** | `PASS` / `FAIL` / `BLIND`, colour-coded, with the reason |
| **Blind spots** | ranked clusters with a coverage bar; zero-case rows in red |
| **Ladder** | κ per rung against the judge, the recommended rung marked, the reading spelled out |
| **Online scoring** | score a turn or burst 200, with a latency sparkline |

Buttons: run each measurement, deploy a scorer, mine 20 cases. Everything is persisted
as a record, so the dashboard and the JSON you commit cannot disagree.

## The review queue — `/review`

One proposal at a time: the conversation rendered readably, the judge's suggestion, the
cluster it came from and why it was proposed.

- **`p`** — correct, PASS
- **`f`** — correct, FAIL
- **`x`** — not a useful case

`Promote confirmed → suite` merges them at the bottom.

**A reviewer name is required.** The API returns 422 without one, the store raises, and
the buttons refuse. That is the guard-rail that makes the loop safe to run unattended:
an eval suite growing by absorbing production traces converges, unwatched, on asserting
that current behaviour is correct by definition.

Correcting the judge is the most valuable thing that happens here. Those are the cases
where the judge was wrong, and they are what makes the suite better rather than bigger.

## Per-turn online scoring — `POST /v1/score`

```bash
curl -X POST localhost:8000/v1/score -H 'Content-Type: application/json' \
     -d '{"text": "I am unable to verify that right now."}'
```

```json
{"label": 0, "pass": false, "scorer": "keyword", "latency_us": 40.9, "kappa_vs_judge": 1.0}
```

Answers from the fitted rung held in memory. Measured across 200 turns the model time is
~18 µs; the original spec asked for under 50 ms.

Two things it does that a `pipeline.predict` would not: it **ships its own κ against the
judge with every score** (a label with no agreement number is a random number generator
with good latency), and it **refuses to load a rung the ladder did not clear** unless you
pass `force=True` — measuring depth is only useful if it constrains the deployment.

Accepts `{"text": ...}` or `{"turns": [...]}`. The turns form stores the trace, so live
traffic accumulates for the next coverage run.

## The API

```
POST /v1/score                    one turn, microseconds
POST /v1/scorer/refit             fit the rung the ladder cleared
GET  /v1/scorer                   deployed rung, κ, call count, mean and p95 latency

POST /v1/traces                   ingest (native or loose shape)
GET  /v1/traces?limit=

GET  /v1/coverage?refresh=        calibrated-radius coverage, per cluster
GET  /v1/blindspots?top=          ranked, with distinctive terms
GET  /v1/ladder?finetune=         κ per rung, judge_depth, the reading
GET  /v1/power?effect=&n_sim=     expected power + the false-alarm rate
GET  /v1/judge/validate           κ against human labels, or UNVALIDATED
GET  /v1/gate?mode=               PASS / FAIL / BLIND

POST /v1/mine?n=                  queue proposals from blind clusters
GET  /v1/proposals?status=
POST /v1/proposals/{id}/confirm   {"reviewer": "...", "expected": 0|1}
POST /v1/proposals/{id}/reject    {"reviewer": "..."}
POST /v1/promote                  {"max_cases": 400}
POST /v1/export/finetune          {"format": "chat", "split": 0.2}

GET  /v1/status                   store counts, trace span, scorer, online latency
GET  /v1/records?kind=            every measurement ever run
```

Interactive docs at `/docs`.

## Storage

```python
from livingeval.store import open_store

store = open_store("sqlite:///livingeval.db")     # default
store = open_store("postgresql://user:pw@host/db")  # Supabase works unchanged
```

Six tables: `traces`, `verdicts`, `suites`, `proposals`, `records`, `online_scores`.
No ORM, no migrations.

`put_traces` is idempotent on `trace_id`, so replaying a file or re-polling an
overlapping window is harmless.

```bash
livingeval serve --db postgresql://...     # same command, different URL
```

## Live ingestion

```bash
# once, from files, and curate a realistically stale suite while you are there
livingeval ingest --traces 'traces/*.jsonl' --db sqlite:///live.db \
                  --suite-from-first-window --suite-size 200

# continuously, from a Langfuse project
export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...
livingeval ingest --follow --interval 30 --db sqlite:///live.db
```

The poller is a foreground loop, not a daemon: the process that stops when you close the
terminal is the one you can reason about. Because `put_traces` deduplicates, it needs no
cursor bookkeeping and a restart costs nothing.

## Configuration

```bash
livingeval serve \
  --db sqlite:///livingeval.db \
  --suite support \
  --judge openai:gpt-4o-mini \
  --embed hf:sentence-transformers/all-MiniLM-L6-v2 \
  --port 8000
```

| flag | default | |
|---|---|---|
| `--db` | `sqlite:///livingeval.db` | any store URL |
| `--suite` | `default` | which suite the measurements are about |
| `--judge` | `oracle` | `rule:mod:fn`, `openai:model`, `anthropic:model` |
| `--embed` | tf-idf + SVD | `hashing`, `hf:<model>`, `openai:<model>` |
| `--demo` | off | seed a corpus if the store is empty |
| `--no-refit` | off | skip deploying a scorer at startup |

## Recording a walkthrough

```bash
python scripts/demo.py --fresh --embed hashing   # ~15s, prints the whole story
livingeval serve --db sqlite:///demo.db          # same database, now on screen
```

The script narrates the loop in the order the story needs — score, coverage, blind
spots, power, gate, ladder, deploy, mine, review, promote, re-measure, export — so the
dashboard afterwards is showing numbers the viewer has already been told the meaning of.

On the bundled corpus one review round moves coverage 0.432 → 0.573 and power
0.013 → 0.427. Both numbers are real and both are printed.
