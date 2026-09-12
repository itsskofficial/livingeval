"""`livingeval init` -- the interactive setup.

The tool arrives knowing nothing and has to leave knowing enough to write a
suite somebody will keep. Four things it cannot work out alone: which model
grades the judgement-based metrics, the key for that model, what the assistant
is *for* (scope adherence and over-refusal are unmeasurable without it), and
whether the archetypes it inferred are right.

Everything else it decides, and says why. The running commentary is the product
as much as the files are: a tool that writes evals into your repository without
explaining its reasoning gets deleted the first time it is wrong, and it will be
wrong sometimes.

Design rules, learned from tools that get this badly wrong:

- **Never block on something with a sensible default.** Every question carries
  one, and `--yes` takes them all.
- **Show before asking.** The scan is printed before the first question, so the
  answers are informed.
- **Secrets go to `.env`, never to the committed config.** `config.py` reads
  credentials only from the environment; the wizard writes them where that
  will find them, and checks the file is ignored by git.
- **Never overwrite silently.** A second run reports what exists and touches
  only what the user confirms.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from livingeval.discover import scan
from livingeval.discover.sites import CallSite
from livingeval.generate import build_goldens, emit
from livingeval.plan import Plan, build_plan
from livingeval.plan.taxonomy import Archetype

__all__ = ["Answers", "run_init"]

JUDGES = [
    ("openai:gpt-4o-mini", "OPENAI_API_KEY", "cheap, and the model G-Eval was tuned against"),
    ("openai:gpt-4o", "OPENAI_API_KEY", "stronger judge, roughly 15x the cost"),
    ("anthropic:claude-sonnet-4-5", "ANTHROPIC_API_KEY", "strong judge, different failure modes to GPT"),
    ("ollama:llama3.1", None, "local, free, and a noticeably weaker judge"),
]


def _tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    """Colour, when the terminal will take it."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(text: str) -> str:
    return _c(text, "1")


def dim(text: str) -> str:
    return _c(text, "2")


def green(text: str) -> str:
    return _c(text, "32")


def yellow(text: str) -> str:
    return _c(text, "33")


def cyan(text: str) -> str:
    return _c(text, "36")


def rule(title: str = "") -> None:
    width = min(shutil.get_terminal_size((80, 24)).columns, 88)
    if title:
        print(f"\n{bold(title)}\n{dim('-' * width)}")
    else:
        print(dim("-" * width))


def ask(prompt: str, default: str | None = None, options: list[str] | None = None,
        assume_yes: bool = False) -> str:
    """One question. Returns the default unattended or when input is not a tty."""
    if assume_yes or not _tty():
        return default or ""
    suffix = f" {dim('[' + default + ']')}" if default else ""
    if options:
        for i, option in enumerate(options, 1):
            marker = green("*") if option == default else " "
            print(f"   {marker} {i}. {option}")
        raw = input(f"   {cyan('>')} choose 1-{len(options)}{suffix}: ").strip()
        if not raw:
            return default or options[0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        return raw
    raw = input(f"   {cyan('>')} {prompt}{suffix}: ").strip()
    return raw or (default or "")


def confirm(prompt: str, default: bool = True, assume_yes: bool = False) -> bool:
    if assume_yes or not _tty():
        return default
    hint = "Y/n" if default else "y/N"
    raw = input(f"   {cyan('>')} {prompt} {dim('[' + hint + ']')}: ").strip().lower()
    return default if not raw else raw.startswith("y")


@dataclass
class Answers:
    """What the wizard learned, and what it assumed."""

    judge: str = "openai:gpt-4o-mini"
    key_var: str | None = "OPENAI_API_KEY"
    api_key: str | None = None
    scope: str = ""
    include_safety: bool = True
    include_operational: bool = True
    overrides: dict[str, str] = field(default_factory=dict)
    assumed: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------


def show_scan(sites: list[CallSite]) -> None:
    rule("1. What I found")
    if not sites:
        print(yellow("   No LLM calls found."))
        print(dim("   Supported: openai, anthropic, langchain, litellm, ollama, google."))
        print(dim("   If your calls go through a wrapper, point --path at the module "
                  "that makes the outbound call."))
        return
    print(f"   {len(sites)} call site{'s' if len(sites) != 1 else ''}:\n")
    for site in sites:
        conf = (green if site.confidence >= 0.5 else
                yellow if site.confidence > 0 else dim)(f"{site.confidence:.0%}")
        print(f"   {bold(site.function + '()')}  {dim(f'{site.path}:{site.line}')}")
        print(f"      {site.provider} -> {cyan(site.archetype)} {dim('(' + conf + ' confident)')}")
        for reason in site.rationale[:2]:
            print(dim(f"      - {reason}"))
        print()


def choose_judge(answers: Answers, assume_yes: bool) -> None:
    rule("2. The judge")
    print(dim("   Model-graded metrics need one model to score them. Using a single\n"
              "   judge across the suite keeps scores comparable and makes an\n"
              "   upgrade a one-line change.\n"))
    labels = [f"{name}  {dim('-- ' + note)}" for name, _, note in JUDGES]
    picked = ask("judge", default=labels[0], options=labels, assume_yes=assume_yes)
    index = labels.index(picked) if picked in labels else 0
    answers.judge, answers.key_var, _ = JUDGES[index]
    if assume_yes:
        answers.assumed.append(f"judge = {answers.judge}")

    if not answers.key_var:
        print(green(f"\n   {answers.judge} runs locally. No key needed."))
        return

    if os.environ.get(answers.key_var):
        print(green(f"\n   {answers.key_var} found in your environment."))
        return

    print(yellow(f"\n   {answers.key_var} is not set."))
    print(dim("   It will be written to .env, which is read at run time and never\n"
              "   committed. livingeval reads credentials from the environment only."))
    if not assume_yes and _tty():
        import getpass
        entered = getpass.getpass(f"   {cyan('>')} {answers.key_var} (blank to skip): ").strip()
        answers.api_key = entered or None
    if not answers.api_key:
        print(dim(f"   Skipped. Set {answers.key_var} before running the suite."))


def ask_scope(answers: Answers, plan: Plan, assume_yes: bool) -> None:
    needs = any(m.name in {"scope_adherence", "over_refusal"}
                for p in plan.pipelines for m in p.metrics)
    if not needs:
        return
    rule("3. What is this assistant for?")
    print(dim("   Two safety metrics are measured against this and cannot be\n"
              "   evaluated without it:\n"
              "     scope_adherence  -- does it decline what it should?\n"
              "     over_refusal     -- does it still answer what it should?\n\n"
              "   The second is the one people forget. Without it a system that\n"
              "   refuses everything scores perfectly on safety.\n"))
    default = "answering questions about this product"
    answers.scope = ask("one sentence", default=default, assume_yes=assume_yes)
    if answers.scope == default:
        answers.assumed.append(f"scope = {default!r} (generic; edit the probe sets)")


def confirm_archetypes(answers: Answers, sites: list[CallSite], assume_yes: bool) -> None:
    unsure = [s for s in sites if s.confidence < 0.3]
    if not unsure:
        return
    rule("4. Calls I am unsure about")
    print(dim("   The archetype decides which metrics apply, so a wrong guess here\n"
              "   produces a suite that measures the wrong things.\n"))
    options = [a.value for a in Archetype]
    for site in unsure:
        print(f"   {bold(site.function + '()')} {dim(f'{site.path}:{site.line}')} "
              f"-- guessed {cyan(site.archetype)}")
        for reason in site.rationale:
            print(dim(f"      {reason}"))
        picked = ask("what is it", default=site.archetype, options=options,
                     assume_yes=assume_yes)
        if picked != site.archetype and picked in options:
            answers.overrides[site.ident] = picked
            site.archetype = picked
        elif assume_yes:
            answers.assumed.append(f"{site.ident} = {site.archetype} (low confidence)")


def explain_plan(plan: Plan) -> None:
    rule("5. The suite I would write")
    print(f"   {plan.summary()}\n")
    for pipeline in plan.pipelines:
        head = f"{pipeline.slug}"
        tag = dim(f"{pipeline.level.value}/{pipeline.risk.value}")
        print(f"   {bold(head)}  {tag}")
        for metric in pipeline.metrics:
            mark = green("ok") if metric.automatable else yellow("needs you")
            print(f"      {mark:<22} {metric.name}")
            print(dim(f"{'':<25}{metric.catches}"))
        print()
    if plan.blocked_count:
        print(yellow(f"   {plan.blocked_count} metrics need reference answers.") +
              dim(" They are written as stubs with\n   inputs filled in and the answer "
                  "column empty, and excluded from the gate\n   until you confirm them. "
                  "Filling them from your current output would\n   build a suite that "
                  "passes because it was harvested from what the\n   model already does."))
        print()


def write_env(root: Path, answers: Answers) -> Path | None:
    if not (answers.api_key and answers.key_var):
        return None
    env = root / ".env"
    line = f"{answers.key_var}={answers.api_key}\n"
    existing = env.read_text(encoding="utf-8") if env.exists() else ""
    if answers.key_var in existing:
        return env
    env.write_text(existing + ("" if existing.endswith("\n") or not existing else "\n") + line,
                   encoding="utf-8")
    gitignore = root / ".gitignore"
    ignored = gitignore.exists() and ".env" in gitignore.read_text(encoding="utf-8")
    if not ignored:
        with gitignore.open("a", encoding="utf-8") as handle:
            handle.write("\n# livingeval: credentials, never committed\n.env\n")
    return env


def write_project_config(root: Path, answers: Answers, plan: Plan) -> Path:
    """Non-secret settings, committed. Regeneration reads this back."""
    path = root / "livingeval.toml"
    overrides = "\n".join(f'"{k}" = "{v}"' for k, v in answers.overrides.items())
    path.write_text(f'''# livingeval project settings. Safe to commit -- no credentials here.
# Credentials are read from the environment; see .env.

[judge]
# One judge for the whole suite, so scores stay comparable.
model = "{answers.judge}"

[application]
# What this assistant is for. scope_adherence and over_refusal are measured
# against this sentence, so keep it accurate as the product changes.
scope = "{answers.scope}"

[suite]
safety = {str(answers.include_safety).lower()}
operational = {str(answers.include_operational).lower()}

[archetypes]
# Corrections to what the scanner inferred. Anything here wins over detection.
{overrides}
''', encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_init(root: Path, assume_yes: bool = False, dry_run: bool = False,
             include_tests: bool = False) -> int:
    print()
    print(bold("  livingeval init"))
    print(dim(f"  scanning {root.resolve()}"))

    sites = scan(root, include_tests=include_tests)
    show_scan(sites)
    if not sites:
        return 1

    answers = Answers()
    choose_judge(answers, assume_yes)

    plan = build_plan(sites)
    ask_scope(answers, plan, assume_yes)
    confirm_archetypes(answers, sites, assume_yes)

    # Rebuild after any override: the archetype decides the metric set.
    if answers.overrides:
        plan = build_plan(sites, include_safety=answers.include_safety,
                          include_operational=answers.include_operational)

    explain_plan(plan)

    if dry_run:
        print(dim("   --dry-run: nothing written."))
        return 0

    if not confirm("Write this suite?", default=True, assume_yes=assume_yes):
        print(dim("   Nothing written."))
        return 0

    rule("6. Writing")
    metrics = [m for p in plan.pipelines for m in p.metrics]
    goldens = build_goldens(metrics, scope=answers.scope or "this product")
    emission = emit(plan, goldens, root, judge=answers.judge, scope=answers.scope)
    config = write_project_config(root, answers, plan)
    env = write_env(root, answers)

    print(f"   {green('+')} livingeval_evals/           "
          f"{len(emission.files)} files")
    print(f"   {green('+')} livingeval_evals/goldens/   "
          f"{len(emission.goldens)} datasets")
    print(f"   {green('+')} {config.name}")
    if env:
        print(f"   {green('+')} {env.name} {dim('(gitignored)')}")

    complete = sum(1 for g in goldens.values() if g.complete)
    print()
    print(f"   {green(str(emission.runnable) + ' metrics run now')} "
          f"across {len(plan.pipelines)} pipelines.")
    print(f"   {complete} datasets are complete; "
          f"{len(goldens) - complete} await your answers.")

    if answers.assumed:
        print()
        print(yellow("   Assumed, unattended:"))
        for item in answers.assumed:
            print(dim(f"      - {item}"))

    rule("Next")
    print(f"   {bold('livingeval_evals/WHY.md')}  {dim('every choice, explained')}")
    print(f"   {bold('python -m livingeval_evals.run_suite')}  {dim('run it, write a baseline')}")
    print(f"   {bold('livingeval baseline')}  {dim('measure real noise thresholds')}")
    print(f"   {bold('livingeval gate')}  {dim('PASS / FAIL / BLIND against the baseline')}")
    print()
    return 0
