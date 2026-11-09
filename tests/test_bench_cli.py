from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from bitesmart_recsys.bench import (
    IndexSpec,
    benchmark_filters,
    make_clustered,
    recall_at_k,
    run_benchmark,
    to_markdown,
    write_report,
)
from bitesmart_recsys.bench.embed_load import compare_batching
from bitesmart_recsys.cli import app
from bitesmart_recsys.embedding import HashEmbedder
from bitesmart_recsys.logging_setup import JsonFormatter, configure_logging


def test_recall_at_k() -> None:
    truth = np.array([[1, 2, 3], [4, 5, -1], [-1, -1, -1]])
    found = np.array([[3, 9, 1], [4, 5, 6], [7, 8, 9]])
    # query 1 finds 2 of 3, query 2 finds 2 of 2, query 3 has no truth and is skipped
    assert recall_at_k(found, truth) == pytest.approx((2 / 3 + 1.0) / 2)
    assert np.isnan(recall_at_k(found[2:], truth[2:]))


@pytest.fixture(scope="module")
def tiny_data():  # type: ignore[no-untyped-def]
    return make_clustered(600, dim=32, n_queries=10, seed=1)


def test_benchmark_flat_is_exact_and_hnsw_improves_with_ef(tiny_data) -> None:  # type: ignore[no-untyped-def]
    specs = [
        IndexSpec("faiss_flat"),
        IndexSpec(
            "faiss_hnsw", {"m": 16, "ef_construction": 100}, [{"ef_search": 8}, {"ef_search": 128}]
        ),
        IndexSpec("hnswlib"),  # skipped quietly when the optional extra is missing
    ]
    results = run_benchmark(tiny_data, specs, benchmark_filters(), measure_memory=False)
    flat = [r for r in results if r.index == "faiss_flat"]
    assert flat
    assert all(r.recall_at_k == 1.0 for r in flat if r.selectivity > 0)

    low = [r for r in results if r.index == "faiss_hnsw" and r.search_params == "{'ef_search': 8}"]
    high = [
        r for r in results if r.index == "faiss_hnsw" and r.search_params == "{'ef_search': 128}"
    ]
    assert np.nanmean([r.recall_at_k for r in high]) >= np.nanmean([r.recall_at_k for r in low])
    assert all(r.index_mb and r.p50_ms >= 0 and r.qps > 0 for r in results)
    assert {r.filter for r in results} == set(benchmark_filters())


def test_isolated_memory_measurement(tiny_data) -> None:  # type: ignore[no-untyped-def]
    results = run_benchmark(
        tiny_data, [IndexSpec("faiss_flat")], {"none": None}, measure_memory=True
    )
    assert results[0].build_rss_mb is not None


def test_reports_are_written(tiny_data, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    results = run_benchmark(
        tiny_data, [IndexSpec("faiss_flat")], {"none": None}, measure_memory=False
    )
    markdown = to_markdown(results)
    assert "recall@10" in markdown
    assert "faiss_flat" in markdown
    paths = write_report(results, tmp_path / "out")
    assert paths["csv"].read_text().startswith("index,")
    assert json.loads(paths["json"].read_text())[0]["index"] == "faiss_flat"
    assert to_markdown([]).startswith("# Index benchmark")


def test_plots_are_rendered(tiny_data, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("matplotlib")
    from bitesmart_recsys.bench.plots import plot_recall_latency

    results = run_benchmark(
        tiny_data, [IndexSpec("faiss_flat")], {"none": None}, measure_memory=False
    )
    paths = plot_recall_latency(results, tmp_path)
    assert paths[0].exists()


def test_dynamic_batching_beats_no_batching_under_load() -> None:
    class SlowEmbedder(HashEmbedder):
        """Fixed cost per model call, like a real model with small batches."""

        def embed(self, texts):  # type: ignore[no-untyped-def]
            import time

            time.sleep(0.004)
            return super().embed(texts)

    rows = asyncio.run(
        compare_batching(SlowEmbedder(dim=16), ["a b", "c d"], concurrency=16, requests=64)
    )
    no_batch, _, big = rows
    assert no_batch.mean_batch == 1.0
    assert big.mean_batch > 2
    assert big.requests_per_s > no_batch.requests_per_s


def test_json_logging_format() -> None:
    record = logging.LogRecord("x.y", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.batch_size = 5
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["batch_size"] == 5
    assert payload["ts"].endswith("+00:00")

    try:
        raise ValueError("bad")
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    assert "ValueError" in json.loads(JsonFormatter().format(record))["exc"]


def test_configure_logging_installs_json_handler() -> None:
    configure_logging("DEBUG")
    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0].formatter, JsonFormatter)


runner = CliRunner()


def test_cli_generate_embed_recommend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITESMART_EMBEDDER_BACKEND", "hash")
    out = tmp_path / "cat.jsonl"
    result = runner.invoke(app, ["generate-catalog", "--n", "150", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()

    result = runner.invoke(app, ["embed", "greek yogurt", "yoghurt", "--embedder", "hash"])
    assert result.exit_code == 0, result.output
    assert "cosine matrix" in result.output

    first_id = json.loads(out.read_text().splitlines()[0])["id"]
    result = runner.invoke(
        app,
        [
            "recommend",
            first_id,
            "--catalog",
            str(out),
            "--backend",
            "faiss_flat",
            "--forbid-allergen",
            "en:milk",
            "--embedder",
            "hash",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"source_id"' in result.output
