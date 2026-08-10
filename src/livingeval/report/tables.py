"""Markdown tables from records.

These render the audit output that goes in the README. They read records, never live
objects, so a table in a document and the JSON committed beside it cannot disagree.
"""

from __future__ import annotations

from livingeval.report.record import Record

__all__ = ["coverage_table", "decay_table", "false_alarm_table", "ladder_table"]


def _fmt(x, spec=".4f", dash="-"):
    return dash if x is None else format(x, spec)


def ladder_table(record: Record | list[dict], title: str = "") -> str:
    """One row per (scenario, rung): kappa against the judge, and what it cost."""
    results = record.of_kind("ladder") if isinstance(record, Record) else record
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += [
        "| scenario | designed depth | measured depth | majority | length | keyword | bow | charngram |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        by_name = {x["rung"]: x["kappa"]["point"] for x in r["rungs"]}
        scenario = r.get("meta", {}).get("scenario", r.get("subgroup") or r["judge"])
        designed = r.get("meta", {}).get("designed_depth") or "-"
        measured = r.get("judge_depth") or "none"
        cells = " | ".join(_fmt(by_name.get(k), ".3f") for k in
                           ("majority", "length", "keyword", "bow", "charngram"))
        lines.append(f"| `{scenario}` | {designed} | **{measured}** | {cells} |")
    return "\n".join(lines)


def coverage_table(record: Record | list[dict], title: str = "") -> str:
    results = record.of_kind("coverage") if isinstance(record, Record) else record
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += [
        "| suite | cases | traffic | coverage | 95% CI | radius | q25 | q75 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ci = r.get("ci") or {}
        sens = r.get("sensitivity", {})
        lines.append(
            f"| `{r['suite']}` | {r['n_cases']} | {r['n_traces']} | "
            f"{_fmt(r.get('coverage'), '.3f')} | "
            f"[{_fmt(ci.get('lo'), '.3f')}, {_fmt(ci.get('hi'), '.3f')}] | "
            f"{_fmt(r.get('radius'), '.3f')} | "
            f"{_fmt(sens.get('25.0'), '.3f')} | {_fmt(sens.get('75.0'), '.3f')} |"
        )
    return "\n".join(lines)


def decay_table(record: Record | list[dict], title: str = "") -> str:
    """Frozen against mined, window by window.

    Three rows per suite, not one, because they say different things. Coverage falls
    first. The worst cluster's power falls next. The *average* power over clusters
    barely moves at all - which is the point, since the average is the number a
    dashboard would show you.
    """
    results = record.of_kind("decay") if isinstance(record, Record) else record
    if not results:
        return ""
    windows = sorted({p["window"] for r in results for p in r["points"]})
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += [
        "| suite | metric | " + " | ".join(f"w{w}" for w in windows) + " |",
        "|---" * (len(windows) + 2) + "|",
    ]
    metrics = [
        ("coverage", "coverage", ".2f"),
        ("power", "power (averaged over clusters)", ".2f"),
        ("worst_power", "power in its **worst** cluster", ".3f"),
    ]
    for r in results:
        by_window = {p["window"]: p for p in r["points"]}
        for key, label, spec in metrics:
            if not any(by_window[w].get(key) is not None for w in by_window):
                continue
            cells = " | ".join(
                _fmt(by_window[w].get(key), spec) if w in by_window else "-" for w in windows
            )
            lines.append(f"| `{r['label']}` | {label} | {cells} |")
    return "\n".join(lines)


def false_alarm_table(record: Record | list[dict], title: str = "") -> str:
    """The Type-I error rate of each gate design, by suite size."""
    results = record.of_kind("false_alarm") if isinstance(record, Record) else record
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += [
        "| suite size | threshold | threshold gate fires | paired gate fires |",
        "|---|---|---|---|",
    ]
    for r in results:
        t, p = r.get("threshold_rate"), r.get("paired_rate")
        tci, pci = r.get("threshold_ci") or [None, None], r.get("paired_ci") or [None, None]
        lines.append(
            f"| {r['n_cases']} | {_fmt(r.get('threshold'), '.3f')} | "
            f"**{_fmt(t, '.3f')}** [{_fmt(tci[0], '.3f')}, {_fmt(tci[1], '.3f')}] | "
            f"{_fmt(p, '.3f')} [{_fmt(pci[0], '.3f')}, {_fmt(pci[1], '.3f')}] |"
        )
    return "\n".join(lines)
