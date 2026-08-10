"""Figures, drawn from records.

matplotlib is an optional extra and imported inside each function, so the default
install has no plotting dependency and the test suite never opens a display.

Every figure here reads a `Record`. None of them accept a live suite, judge or trace
set - which is the point: regenerating the whole figure set is a second on a laptop
and cannot change a number.
"""

from __future__ import annotations

from pathlib import Path

from livingeval.report.record import Record

__all__ = ["all_figures", "coverage_figure", "decay_figure", "ladder_figure", "staleness_figure"]

_COLORS = {"frozen": "#c0392b", "mined": "#27ae60", "accent": "#2c3e50", "muted": "#95a5a6"}


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError("pip install 'livingeval[figures]'") from e
    return plt


def decay_figure(record: Record, out: str | Path, title: str = "Detection power over time") -> Path:
    """The staleness curve: power per window, one line per suite strategy."""
    plt = _plt()
    curves = record.of_kind("decay")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for curve in curves:
        pts = curve["points"]
        xs = [p["window"] for p in pts]
        color = _COLORS.get(curve["label"], _COLORS["accent"])
        ax.plot(xs, [p["power"] for p in pts], marker="o", color=color, linewidth=2,
                label=f"{curve['label']} - averaged over clusters")
        if any(p.get("worst_power") is not None for p in pts):
            ax.plot(xs, [p.get("worst_power") for p in pts], marker="s", color=color,
                    linewidth=2, linestyle="--", alpha=0.85,
                    label=f"{curve['label']} - worst cluster")
    ax.axhline(0.5, linestyle=":", color=_COLORS["muted"], linewidth=1)
    ax.text(0.02, 0.52, "coin flip", transform=ax.get_yaxis_transform(),
            color=_COLORS["muted"], fontsize=8)
    ax.set_xlabel("time window")
    ax.set_ylabel("detection power (simulated)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def staleness_figure(record: Record, out: str | Path,
                     title: str = "Coverage falls before anything else does") -> Path:
    """Coverage per window, one line per suite strategy.

    The cheapest of the three signals and the earliest to move, which is why it is
    the one to put in a dashboard.
    """
    plt = _plt()
    curves = record.of_kind("decay")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for curve in curves:
        pts = [p for p in curve["points"] if p.get("coverage") is not None]
        if not pts:
            continue
        color = _COLORS.get(curve["label"], _COLORS["accent"])
        ax.plot([p["window"] for p in pts], [p["coverage"] for p in pts],
                marker="o", label=curve["label"], color=color, linewidth=2)
    ax.set_xlabel("time window")
    ax.set_ylabel("fraction of traffic covered")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def ladder_figure(record: Record, out: str | Path, title: str = "Judge-complexity ladder") -> Path:
    """Kappa against the judge at each rung, one group of bars per scenario."""
    plt = _plt()
    import numpy as np

    ladders = record.of_kind("ladder")
    rungs = ["majority", "length", "keyword", "bow", "charngram"]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    width = 0.8 / max(1, len(ladders))
    x = np.arange(len(rungs))
    cmap = plt.get_cmap("viridis")
    for i, lad in enumerate(ladders):
        by_name = {r["rung"]: r["kappa"]["point"] for r in lad["rungs"]}
        label = lad.get("meta", {}).get("scenario", lad["judge"])
        ax.bar(x + i * width, [by_name.get(r, 0.0) for r in rungs], width,
               label=label, color=cmap(i / max(1, len(ladders))))
    ax.axhline(ladders[0]["threshold"] if ladders else 0.8, linestyle="--",
               color=_COLORS["muted"], linewidth=1)
    ax.set_xticks(x + width * (len(ladders) - 1) / 2)
    ax.set_xticklabels(rungs)
    ax.set_ylabel("Cohen's kappa vs the judge")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def coverage_figure(record: Record, out: str | Path, title: str = "Coverage by traffic cluster") -> Path:
    """Traffic share against coverage per cluster - blind spots sit bottom-right."""
    plt = _plt()
    results = record.of_kind("coverage")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for res in results:
        shares = res.get("cluster_shares", {})
        cov = res.get("per_cluster", {})
        for cluster, share in shares.items():
            c = cov.get(cluster, 0.0)
            ax.scatter(share, c, s=60, color=_COLORS["frozen"] if c < 0.3 else _COLORS["accent"],
                       alpha=0.8)
            ax.annotate(cluster, (share, c), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.axhline(0.30, linestyle="--", color=_COLORS["muted"], linewidth=1)
    ax.set_xlabel("share of production traffic")
    ax.set_ylabel("fraction covered by the suite")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def all_figures(record: Record, out_dir: str | Path) -> list[Path]:
    """Draw whatever the record supports and skip the rest."""
    out_dir = Path(out_dir)
    made: list[Path] = []
    if record.of_kind("decay"):
        made.append(decay_figure(record, out_dir / "decay.png"))
        made.append(staleness_figure(record, out_dir / "staleness.png"))
    if record.of_kind("ladder"):
        made.append(ladder_figure(record, out_dir / "ladder.png"))
    if record.of_kind("coverage"):
        made.append(coverage_figure(record, out_dir / "coverage.png"))
    return made
