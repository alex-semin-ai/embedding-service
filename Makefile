.PHONY: install lint format type test check serve bench bench-embed demo docker-build clean

install:
	uv sync --extra report
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

type:
	uv run mypy

test:
	uv run pytest

check: lint type test

serve:
	uv run bitesmart serve --catalog data/catalog.jsonl --index faiss_hnsw

bench:
	uv run bitesmart bench --n 20000 --plots --out bench_out

bench-embed:
	uv run bitesmart bench-embed

demo:
	docker compose --profile demo up --build --abort-on-container-exit --exit-code-from demo

docker-build:
	docker build -t bitesmart-recsys .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage bench_out
