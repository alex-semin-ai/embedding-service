"""Recall vs latency plots (needs the optional 'report' extra: matplotlib)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .runner import Result


def plot_recall_latency(results: list[Result], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_filter: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_filter[r.filter].append(r)

    paths: list[Path] = []
    for fname, rows in by_filter.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        by_index: dict[str, list[Result]] = defaultdict(list)
        for r in rows:
            by_index[r.index].append(r)
        for name, points in by_index.items():
            points.sort(key=lambda r: r.p50_ms)
            ax.plot([p.p50_ms for p in points], [p.recall_at_k for p in points], "o-", label=name)
        ax.set_xscale("log")
        ax.set_xlabel("p50 latency, ms (log)")
        ax.set_ylabel("recall@10")
        ax.set_title(f"filter {fname}, selectivity {rows[0].selectivity:.1%}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"recall_latency_{fname}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths
