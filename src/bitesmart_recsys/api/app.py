"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from bitesmart_recsys.embedding import (
    DynamicBatcher,
    Embedder,
    QueueFullError,
    create_embedder,
    with_prefix,
)
from bitesmart_recsys.indexes import create_index
from bitesmart_recsys.recsys import (
    PAYLOAD_SCHEMA,
    Catalog,
    RecommendationConfig,
    Recommender,
    read_products,
)
from bitesmart_recsys.settings import Settings, get_settings

from .schemas import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorBody,
    ErrorResponse,
    Health,
    ModelCard,
    ModelList,
    RecommendRequest,
    SearchHit,
    SearchRequest,
    SearchResponse,
    Usage,
)

logger = logging.getLogger(__name__)

MAX_INPUTS = 2048


@dataclass
class AppState:
    settings: Settings
    embedder: Embedder
    batcher: DynamicBatcher
    recommender: Recommender | None = None
    counters: dict[str, int] = field(default_factory=lambda: {"requests": 0, "errors": 0})


class ApiError(Exception):
    def __init__(
        self, status: int, message: str, type_: str = "invalid_request_error", param: str | None = None
    ) -> None:
        self.status, self.message, self.type_, self.param = status, message, type_, param


def _error(status: int, message: str, type_: str, param: str | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(message=message, type=type_, param=param))
    return JSONResponse(status_code=status, content=body.model_dump())


def _build_recommender(settings: Settings, embedder: Embedder) -> Recommender | None:
    if not settings.catalog_path:
        return None
    products = read_products(settings.catalog_path)
    catalog = Catalog.from_products(products, embedder)
    if settings.index_backend == "qdrant":
        index = create_index(
            "qdrant", url=settings.qdrant_url, payload_schema=PAYLOAD_SCHEMA, collection="catalog"
        )
    else:
        index = create_index(settings.index_backend)
    logger.info("building index", extra={"backend": index.name, "products": len(catalog)})
    return Recommender.build(catalog, index)


def create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = embedder or await asyncio.to_thread(create_embedder, settings)
        batcher = DynamicBatcher(
            model, settings.max_batch_size, settings.max_wait_ms, settings.max_queue_size
        )
        await batcher.start()
        recommender = await asyncio.to_thread(_build_recommender, settings, model)
        app.state.s = AppState(settings, model, batcher, recommender)
        logger.info("server ready", extra={"model": model.model_name, "dim": model.dim})
        try:
            yield
        finally:
            await batcher.stop()

    app = FastAPI(title="BiteSmart RecSys", version="0.1.0", lifespan=lifespan)

    def state() -> AppState:
        return app.state.s  # type: ignore[no-any-return]

    async def require_key(authorization: Annotated[str | None, Header()] = None) -> None:
        key = settings.api_key
        if key and authorization != f"Bearer {key}":
            raise ApiError(401, "Incorrect API key provided.", "authentication_error")

    Auth = Depends(require_key)  # noqa: N806

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.message, exc.type_, exc.param)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        param = ".".join(str(p) for p in first["loc"] if p != "body") or None
        return _error(400, str(first["msg"]), "invalid_request_error", param)

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail), "invalid_request_error")

    @app.middleware("http")
    async def _log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        st = getattr(app.state, "s", None)
        if st is not None:
            st.counters["requests"] += 1
            st.counters["errors"] += int(response.status_code >= 400)
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round(1000 * (time.perf_counter() - started), 2),
            },
        )
        return response

    @app.get("/healthz", response_model=Health)
    async def healthz() -> Health:
        st = state()
        rec = st.recommender
        return Health(
            model=st.embedder.model_name,
            dim=st.embedder.dim,
            catalog_size=len(rec.catalog) if rec else 0,
            index=rec.index.name if rec else None,
        )

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        st = state()
        return {"batcher": st.batcher.stats.as_dict(), **st.counters}

    @app.get("/v1/models", response_model=ModelList, dependencies=[Auth])
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=state().embedder.model_name)])

    @app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=[Auth])
    async def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
        st = state()
        if req.model != st.embedder.model_name:
            raise ApiError(404, f"The model `{req.model}` does not exist.", "invalid_request_error", "model")
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        if not texts or len(texts) > MAX_INPUTS:
            raise ApiError(400, f"input must contain 1 to {MAX_INPUTS} items.", param="input")
        if any(not t.strip() for t in texts):
            raise ApiError(400, "input must not contain empty strings.", param="input")
        if req.dimensions is not None and req.dimensions != st.embedder.dim:
            raise ApiError(
                400, f"This model only supports dimensions={st.embedder.dim}.", param="dimensions"
            )
        prepared = [with_prefix(t, st.settings.default_prefix) for t in texts]
        try:
            vectors = await st.batcher.submit(prepared)
        except QueueFullError as exc:
            raise ApiError(429, str(exc), "rate_limit_error") from exc
        tokens = st.embedder.count_tokens(prepared)
        data = [
            EmbeddingData(
                index=i,
                embedding=(
                    base64.b64encode(np.asarray(v, "<f4").tobytes()).decode()
                    if req.encoding_format == "base64"
                    else v.tolist()
                ),
            )
            for i, v in enumerate(vectors)
        ]
        return EmbeddingResponse(
            data=data, model=st.embedder.model_name, usage=Usage(prompt_tokens=tokens, total_tokens=tokens)
        )

    def need_recommender() -> Recommender:
        rec = state().recommender
        if rec is None:
            raise ApiError(503, "No catalog loaded, set BITESMART_CATALOG_PATH.", "service_unavailable")
        return rec

    @app.post("/v1/search", response_model=SearchResponse, dependencies=[Auth])
    async def search(req: SearchRequest) -> SearchResponse:
        st = state()
        rec = need_recommender()
        vector = (await st.batcher.submit([with_prefix(req.query, "query: ")]))[0]
        config = RecommendationConfig()
        config.candidate_search.hard_constraints = req.constraints
        config.candidate_search.limit = max(req.k, config.candidate_search.limit)
        config.candidate_search.top_k = min(config.candidate_search.top_k, config.candidate_search.limit)
        hits = await asyncio.to_thread(rec.search_vector, vector, req.k, config)
        return SearchResponse(
            index=rec.index.name,
            hits=[
                SearchHit(
                    id=h.id,
                    product_name=h.payload["product_name"],
                    brand=h.payload.get("brand"),
                    similarity=round(h.similarity, 6),
                )
                for h in hits
            ],
        )

    @app.post("/v1/recommend", dependencies=[Auth])
    async def recommend(req: RecommendRequest) -> dict[str, Any]:
        rec = need_recommender()
        try:
            result = await asyncio.to_thread(rec.recommend, req.source_id, req.config, req.top_n)
        except KeyError as exc:
            raise ApiError(404, str(exc.args[0]), "invalid_request_error", "source_id") from exc
        return result.model_dump()

    return app
