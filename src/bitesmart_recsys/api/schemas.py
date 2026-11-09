"""Request and response models. The /v1/embeddings shapes follow the OpenAI API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bitesmart_recsys.recsys.config import HardConstraints, RecommendationConfig


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: str | list[str]
    model: str
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "bitesmart"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorBody(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(10, ge=1, le=200)
    constraints: HardConstraints = Field(default_factory=HardConstraints)


class SearchHit(BaseModel):
    id: str
    product_name: str
    brand: str | None
    similarity: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    index: str


class RecommendRequest(BaseModel):
    source_id: str
    top_n: int = Field(7, ge=1, le=50)
    config: RecommendationConfig | None = None


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    model: str
    dim: int
    catalog_size: int
    index: str | None
    extra: dict[str, Any] = Field(default_factory=dict)
