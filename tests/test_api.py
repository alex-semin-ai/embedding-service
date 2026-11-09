from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from bitesmart_recsys.api import create_app
from bitesmart_recsys.embedding import HashEmbedder
from bitesmart_recsys.settings import Settings

MODEL = "hash-embedder"


def post_embed(client: TestClient, **body: object):  # type: ignore[no-untyped-def]
    return client.post("/v1/embeddings", json={"model": MODEL, **body})


def test_embeddings_single_and_batch(client: TestClient) -> None:
    one = post_embed(client, input="greek yogurt").json()
    assert one["object"] == "list"
    assert one["data"][0]["index"] == 0
    assert len(one["data"][0]["embedding"]) == 64
    assert one["usage"]["prompt_tokens"] == one["usage"]["total_tokens"] > 0

    many = post_embed(client, input=["a", "b", "c"]).json()
    assert [d["index"] for d in many["data"]] == [0, 1, 2]


def test_default_prefix_matches_explicit_query_prefix(client: TestClient) -> None:
    plain = post_embed(client, input="yogurt").json()["data"][0]["embedding"]
    explicit = post_embed(client, input="query: yogurt").json()["data"][0]["embedding"]
    assert plain == explicit


def test_base64_encoding_roundtrip(client: TestClient) -> None:
    floats = post_embed(client, input="yogurt").json()["data"][0]["embedding"]
    b64 = post_embed(client, input="yogurt", encoding_format="base64").json()["data"][0][
        "embedding"
    ]
    decoded = np.frombuffer(base64.b64decode(b64), dtype="<f4")
    assert np.allclose(decoded, floats, atol=1e-6)


@pytest.mark.parametrize(
    ("body", "status", "param"),
    [
        ({"model": "other", "input": "x"}, 404, "model"),
        ({"model": MODEL, "input": ""}, 400, "input"),
        ({"model": MODEL, "input": []}, 400, "input"),
        ({"model": MODEL, "input": ["ok", "  "]}, 400, "input"),
        ({"model": MODEL, "input": "x", "dimensions": 8}, 400, "dimensions"),
        ({"model": MODEL, "input": [1, 2, 3]}, 400, None),  # token ids are not supported
        ({"input": "x"}, 400, None),
    ],
)
def test_embeddings_errors_use_openai_format(
    client: TestClient, body: dict[str, object], status: int, param: str | None
) -> None:
    response = client.post("/v1/embeddings", json=body)
    assert response.status_code == status
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["message"]
    if param:
        assert error["param"] == param


def test_models_health_and_metrics(client: TestClient) -> None:
    assert client.get("/v1/models").json()["data"][0]["id"] == MODEL
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["catalog_size"] == 400
    assert health["index"] == "faiss_flat"
    post_embed(client, input="x")
    metrics = client.get("/metrics").json()
    assert metrics["batcher"]["items"] >= 1
    assert metrics["requests"] >= 3


def test_api_key_is_enforced(settings: Settings, embedder: HashEmbedder) -> None:
    secured = settings.model_copy(update={"api_key": "secret"})
    with TestClient(create_app(secured, embedder)) as c:
        denied = c.post("/v1/embeddings", json={"model": MODEL, "input": "x"})
        assert denied.status_code == 401
        assert denied.json()["error"]["type"] == "authentication_error"
        ok = c.post(
            "/v1/embeddings",
            json={"model": MODEL, "input": "x"},
            headers={"Authorization": "Bearer secret"},
        )
        assert ok.status_code == 200
        assert c.get("/healthz").status_code == 200  # health stays open for orchestrators


def test_official_openai_client_works_against_the_server(client: TestClient) -> None:
    """The point of OpenAI compatibility, the stock SDK talks to us unchanged."""
    sdk = OpenAI(api_key="unused", base_url="http://testserver/v1", http_client=client)
    result = sdk.embeddings.create(model=MODEL, input=["one", "two"])
    assert len(result.data) == 2
    assert len(result.data[0].embedding) == 64
    assert result.usage.prompt_tokens > 0
    # Without encoding_format the SDK asks for base64 and decodes it, so the call above
    # already covers the base64 path. An explicit value is passed through untouched.
    raw = sdk.embeddings.create(model=MODEL, input="one", encoding_format="base64")
    decoded = np.frombuffer(base64.b64decode(str(raw.data[0].embedding)), dtype="<f4")
    assert np.allclose(decoded, result.data[0].embedding, atol=1e-6)


def test_search_respects_constraints(client: TestClient) -> None:
    body = {
        "query": "yogurt",
        "k": 8,
        "constraints": {"forbidden_allergens": ["en:milk"], "dietary": {"vegan": True}},
    }
    hits = client.post("/v1/search", json=body).json()["hits"]
    assert hits
    products = {h["id"] for h in hits}
    payloads = {p["id"]: p for p in client.app.state.s.recommender.catalog.payloads}  # type: ignore[attr-defined]
    for pid in products:
        assert "en:milk" not in payloads[pid]["allergens_hierarchy"]


def test_recommend_endpoint_and_errors(client: TestClient) -> None:
    recommender = client.app.state.s.recommender  # type: ignore[attr-defined]
    source = recommender.catalog.products[0].id
    ok = client.post("/v1/recommend", json={"source_id": source, "top_n": 3})
    assert ok.status_code == 200
    data = ok.json()
    assert data["source_id"] == source
    assert len(data["items"]) <= 3
    assert source not in {i["id"] for i in data["items"]}

    missing = client.post("/v1/recommend", json={"source_id": "nope"})
    assert missing.status_code == 404
    assert missing.json()["error"]["param"] == "source_id"


def test_search_without_catalog_returns_503(settings: Settings, embedder: HashEmbedder) -> None:
    bare = settings.model_copy(update={"catalog_path": None})
    with TestClient(create_app(bare, embedder)) as c:
        response = c.post("/v1/search", json={"query": "x"})
        assert response.status_code == 503
        assert c.get("/healthz").json()["index"] is None
