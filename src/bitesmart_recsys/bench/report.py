"""Write benchmark results as CSV, JSON and a Markdown report."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .runner import Result


def write_csv(results: list[Result], path: Path) -> None:
    if not results:
        return
    rows = [r.as_dict() for r in results]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def to_markdown(results: list[Result], k: int = 10) -> str:
    by_filter: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_filter[r.filter].append(r)

    lines = [f"# Index benchmark (recall@{k} vs latency and memory)", ""]
    if results:
        lines.append(f"Dataset size n={results[0].n}. Latency is per single query, one thread.")
        lines.append("")
    for fname, rows in by_filter.items():
        lines += [
            f"## Filter `{fname}` (selectivity {rows[0].selectivity:.1%})",
            "",
            f"| index | search params | recall@{k} | p50 ms | p95 ms | QPS | index MB | build RSS MB | build s |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(rows, key=lambda x: (x.index, x.p50_ms)):
            mem = "n/a" if r.index_mb is None else f"{r.index_mb:.1f}"
            rss = "n/a" if r.build_rss_mb is None else f"{r.build_rss_mb:.0f}"
            lines.append(
                f"| {r.index} | {r.search_params} | {r.recall_at_k:.3f} | {r.p50_ms:.2f} | "
                f"{r.p95_ms:.2f} | {r.qps:.0f} | {mem} | {rss} | {r.build_s:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_report(results: list[Result], out_dir: Path, k: int = 10) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out_dir / "results.csv",
        "json": out_dir / "results.json",
        "markdown": out_dir / "report.md",
    }
    write_csv(results, paths["csv"])
    paths["json"].write_text(json.dumps([r.as_dict() for r in results], indent=2), encoding="utf-8")
    paths["markdown"].write_text(to_markdown(results, k), encoding="utf-8")
    return paths
