"""Command line interface. Human facing output goes through typer.echo, logs through logging."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Annotated

import httpx
import numpy as np
import typer

from bitesmart_recsys.logging_setup import configure_logging
from bitesmart_recsys.settings import Settings

app = typer.Typer(help="BiteSmart RecSys, embeddings server, index benchmark and recommender.")
logger = logging.getLogger(__name__)

CatalogOpt = Annotated[Path, typer.Option(help="Catalog JSONL file")]


def _settings(**overrides: object) -> Settings:
    base = Settings()
    configure_logging(base.log_level)
    return base.model_copy(update={k: v for k, v in overrides.items() if v is not None})


@app.command()
def serve(
    host: str | None = None,
    port: int | None = None,
    catalog: Annotated[str | None, typer.Option(help="Catalog JSONL to index")] = None,
    index: Annotated[str | None, typer.Option(help="Index backend for /v1/search")] = None,
    embedder: Annotated[str | None, typer.Option(help="onnx or hash")] = None,
) -> None:
    """Run the OpenAI-compatible embeddings server (plus /v1/search and /v1/recommend)."""
    import uvicorn

    from bitesmart_recsys.api import create_app

    settings = _settings(
        host=host, port=port, catalog_path=catalog, index_backend=index, embedder_backend=embedder
    )
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)


@app.command("generate-catalog")
def generate_catalog(n: int = 2000, seed: int = 42, out: Path = Path("data/catalog.jsonl")) -> None:
    """Write a seeded synthetic Open Food Facts style catalog."""
    from bitesmart_recsys.recsys import generate_products, write_products

    _settings()
    out.parent.mkdir(parents=True, exist_ok=True)
    count = write_products(generate_products(n, seed), out)
    typer.echo(f"wrote {count} products to {out}")


@app.command()
def embed(
    text: list[str],
    embedder: Annotated[str | None, typer.Option(help="onnx or hash")] = None,
) -> None:
    """Embed texts locally and print the first values and pairwise cosine similarities."""
    from bitesmart_recsys.embedding import create_embedder, with_prefix

    settings = _settings(embedder_backend=embedder)
    model = create_embedder(settings)
    vectors = model.embed([with_prefix(t, settings.default_prefix) for t in text])
    typer.echo(f"model={model.model_name} dim={model.dim}")
    for t, v in zip(text, vectors, strict=True):
        typer.echo(f"{t!r}: {np.round(v[:4], 4).tolist()} ...")
    if len(text) > 1:
        typer.echo("cosine matrix:")
        typer.echo(str(np.round(vectors @ vectors.T, 3)))


@app.command()
def bench(
    n: int = 20000,
    queries: int = 200,
    dim: int = 384,
    k: int = 10,
    out: Path = Path("bench_out"),
    qdrant_url: Annotated[str | None, typer.Option(envvar="BITESMART_QDRANT_URL")] = None,
    qdrant_local: Annotated[bool, typer.Option(help="Add embedded (exact) Qdrant")] = False,
    plots: bool = False,
) -> None:
    """Compare indexes, recall@k vs latency and memory, with and without metadata filters."""
    from bitesmart_recsys.bench import default_specs, make_clustered, run_benchmark, write_report

    _settings()
    typer.echo(f"generating {n} vectors, dim {dim}")
    data = make_clustered(n, dim, queries)
    results = run_benchmark(
        data, default_specs(qdrant_url, qdrant_local), k=k, progress=lambda m: logger.info(m)
    )
    paths = write_report(results, out, k)
    if plots:
        from bitesmart_recsys.bench.plots import plot_recall_latency

        plot_recall_latency(results, out)
    typer.echo(paths["markdown"].read_text(encoding="utf-8"))
    typer.echo(f"files in {out}")


@app.command("bench-embed")
def bench_embed(
    embedder: Annotated[str | None, typer.Option(help="onnx or hash")] = None,
    concurrency: int = 32,
    requests: int = 256,
) -> None:
    """Throughput with and without dynamic batching under concurrent single-text requests."""
    from bitesmart_recsys.bench.embed_load import compare_batching
    from bitesmart_recsys.embedding import create_embedder
    from bitesmart_recsys.recsys import generate_products

    settings = _settings(embedder_backend=embedder)
    model = create_embedder(settings)
    texts = ["query: " + p.product_name for p in generate_products(64)]
    rows = asyncio.run(compare_batching(model, texts, concurrency, requests))
    typer.echo("| max_batch | max_wait_ms | req/s | p50 ms | p95 ms | mean batch |")
    typer.echo("|---|---|---|---|---|---|")
    for r in rows:
        typer.echo(
            f"| {r.max_batch_size} | {r.max_wait_ms} | {r.requests_per_s} | {r.p50_ms} | "
            f"{r.p95_ms} | {r.mean_batch} |"
        )


@app.command()
def recommend(
    source_id: str,
    catalog: CatalogOpt = Path("data/catalog.jsonl"),
    backend: str = "faiss_hnsw",
    top_n: int = 7,
    vegan: bool = False,
    forbid_allergen: Annotated[list[str] | None, typer.Option()] = None,
    same_category: bool = False,
    embedder: Annotated[str | None, typer.Option(help="onnx or hash")] = None,
) -> None:
    """Recommend substitutes for one product, fully in process."""
    from bitesmart_recsys.embedding import create_embedder
    from bitesmart_recsys.indexes import create_index
    from bitesmart_recsys.recsys import (
        Catalog,
        RecommendationConfig,
        Recommender,
        read_products,
    )

    settings = _settings(embedder_backend=embedder)
    cat = Catalog.from_products(read_products(catalog), create_embedder(settings))
    rec = Recommender.build(cat, create_index(backend))
    cfg = RecommendationConfig()
    hard = cfg.candidate_search.hard_constraints
    hard.dietary.vegan = vegan
    hard.forbidden_allergens = list(forbid_allergen or [])
    hard.same_category_as_source = same_category
    typer.echo(rec.recommend(source_id, cfg, top_n).model_dump_json(indent=2))


@app.command("build-subset")
def build_subset_cmd(
    core: Annotated[Path, typer.Option(help="dataset_grade_core .jsonl.zst, read as a stream")],
    index: Annotated[Path, typer.Option(help="index_grade .jsonl.zst, read as a stream")],
    out: Path = Path("data/real/off_100k.jsonl"),
    size: int = 100_000,
    seed: int = 42,
    pairs_csv: Annotated[
        Path | None, typer.Option(help="Pair CSV whose src_id and alt_id are always kept")
    ] = None,
) -> None:
    """Build a reproducible Open Food Facts subset. Source files are only read, never copied."""
    from bitesmart_recsys.data import build_subset, read_labelled_ids
    from bitesmart_recsys.recsys import write_products

    _settings()
    labelled = read_labelled_ids(pairs_csv) if pairs_csv else set()
    result = build_subset(
        core, index, size, seed, labelled, progress=lambda m: typer.echo(f"  {m}", err=True)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    count = write_products(result.products, out)
    manifest = out.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(result.stats, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"wrote {count} products to {out} and the manifest to {manifest}")


def _wait_for(client: httpx.Client, attempts: int = 90) -> dict[str, object]:
    for _ in range(attempts):
        try:
            response = client.get("/healthz")
            if response.status_code == 200:
                return dict(response.json())
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise typer.Exit(code=1)


@app.command()
def demo(
    api_url: str = "http://localhost:8000",
    catalog: CatalogOpt = Path("data/catalog.jsonl"),
    bench_n: int = 5000,
) -> None:
    """Exercise the running server end to end. Meant for `docker compose up`."""
    from bitesmart_recsys.bench import default_specs, make_clustered, run_benchmark, to_markdown
    from bitesmart_recsys.recsys import read_products

    settings = _settings()
    api_key = settings.api_key
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(base_url=api_url, headers=headers, timeout=120) as client:
        health = _wait_for(client)
        model = str(health["model"])
        typer.echo(f"\n== 1. server is up: {health}")

        typer.echo("\n== 2. OpenAI-compatible /v1/embeddings")
        texts = ["greek yogurt with honey", "natural greek yoghurt", "chocolate bar with hazelnuts"]
        body = client.post("/v1/embeddings", json={"model": model, "input": texts}).json()
        vecs = np.array([d["embedding"] for d in body["data"]])
        typer.echo(f"dim={vecs.shape[1]} usage={body['usage']}")
        similar, different = vecs[0] @ vecs[1], vecs[0] @ vecs[2]
        typer.echo(
            f"cosine(yogurt, yoghurt)={similar:.3f}  cosine(yogurt, chocolate)={different:.3f}"
        )

        typer.echo("\n== 3. dynamic batching, 48 concurrent single-text requests")

        async def burst() -> None:
            async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=120) as ac:
                await asyncio.gather(
                    *(
                        ac.post("/v1/embeddings", json={"model": model, "input": f"product {i}"})
                        for i in range(48)
                    )
                )

        asyncio.run(burst())
        typer.echo(json.dumps(client.get("/metrics").json()["batcher"]))

        typer.echo("\n== 4. filtered semantic search, vegan and without milk allergen")
        search = client.post(
            "/v1/search",
            json={
                "query": "fruit yogurt",
                "k": 5,
                "constraints": {"forbidden_allergens": ["en:milk"], "dietary": {"vegan": True}},
            },
        ).json()
        for hit in search["hits"]:
            typer.echo(f"  {hit['similarity']:.3f}  {hit['product_name']}")

        typer.echo("\n== 5. substitution recommendation with hard constraints")
        products = read_products(catalog)
        source = next(p for p in products if p.categories_hierarchy[-1] == "en:fruit-yogurts")
        cfg = {
            "candidate_search": {
                "hard_constraints": {
                    "forbidden_allergens": ["en:milk"],
                    "dietary": {"vegan": True},
                    "source_nutrients": {"enforce": True, "sugars_lte_source": True},
                }
            },
            "category_transition": {"enabled": True},
        }
        rec = client.post(
            "/v1/recommend", json={"source_id": source.id, "top_n": 5, "config": cfg}
        ).json()
        typer.echo(
            f"source: {rec['source_name']} (sugars {source.nutriments['sugars_100g']} g/100g)"
        )
        for item in rec["items"]:
            typer.echo(
                f"  {item['score']:.3f}  {item['product_name']}  "
                f"sugars={item['nutriments'].get('sugars_100g')}  {item['reasons'][:1]}"
            )
        typer.echo(f"stats: {rec['stats']}")

    typer.echo(f"\n== 6. index benchmark on {bench_n} vectors")
    data = make_clustered(bench_n, 384, 100)
    qdrant_url = os.environ.get("BITESMART_QDRANT_URL")
    results = run_benchmark(data, default_specs(qdrant_url, qdrant_local=qdrant_url is None))
    typer.echo(to_markdown(results))
    typer.echo("demo finished")


if __name__ == "__main__":  # pragma: no cover
    app()
