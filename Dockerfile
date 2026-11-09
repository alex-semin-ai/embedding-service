# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.2 /uv /bin/uv
# hnswlib has no wheels on PyPI, it is compiled here and only the venv is copied on
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# Dependencies first so this layer is cached until the lock file changes
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra ann

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ann


FROM python:3.12-slim AS runtime

# Set to true to bake the e5 model into the image, so the container works offline
ARG PRELOAD_MODEL=false

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /models && chown app /models
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY data/catalog.jsonl ./data/catalog.jsonl
COPY configs ./configs

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1 \
    BITESMART_CATALOG_PATH=/app/data/catalog.jsonl

RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
      python -c "from huggingface_hub import hf_hub_download as d; [d('intfloat/multilingual-e5-small', f) for f in ('onnx/model.onnx', 'onnx/tokenizer.json')]" \
      && chown -R app /models; \
    fi

USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=240s --retries=5 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

CMD ["bitesmart", "serve"]
