from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bitesmart_recsys.embedding import (
    DynamicBatcher,
    HashEmbedder,
    QueueFullError,
    create_embedder,
    with_prefix,
)
from bitesmart_recsys.embedding.onnx_embedder import OnnxEmbedder
from bitesmart_recsys.settings import Settings


class RecordingEmbedder(HashEmbedder):
    """Hash embedder that remembers the size of every model call."""

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__(dim=16)
        self.calls: list[int] = []
        self.delay = delay

    def embed(self, texts: list[str]):  # type: ignore[no-untyped-def]
        import time

        self.calls.append(len(texts))
        time.sleep(self.delay)
        return super().embed(texts)


def test_hash_embedder_is_normalised_deterministic_and_lexical() -> None:
    e = HashEmbedder(dim=128)
    a, b, c = e.embed(["greek yogurt honey", "greek yogurt honey", "dark chocolate bar"])
    assert np.allclose(np.linalg.norm(a), 1.0, atol=1e-5)
    assert np.array_equal(a, b)
    assert a @ b > a @ c
    assert e.count_tokens(["a b c"]) == 3


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello", "query: hello"),
        ("query: hello", "query: hello"),
        ("passage: hello", "passage: hello"),
    ],
)
def test_with_prefix(text: str, expected: str) -> None:
    assert with_prefix(text, "query: ") == expected


async def test_concurrent_requests_are_merged_into_batches() -> None:
    model = RecordingEmbedder(delay=0.01)
    batcher = DynamicBatcher(model, max_batch_size=8, max_wait_ms=50)
    await batcher.start()
    results = await asyncio.gather(*(batcher.submit([f"text {i}"]) for i in range(20)))
    await batcher.stop()

    assert len(results) == 20
    assert batcher.stats.items == 20
    assert batcher.stats.max_batch <= 8
    assert batcher.stats.batches < 20  # fewer model calls than requests
    assert batcher.stats.mean_batch > 1
    assert sum(model.calls) == 20


async def test_results_keep_input_order() -> None:
    model = RecordingEmbedder()
    batcher = DynamicBatcher(model, max_batch_size=3, max_wait_ms=5)
    await batcher.start()
    texts = [f"item {i} unique{i}" for i in range(10)]
    got = await batcher.submit(texts)
    await batcher.stop()
    assert np.allclose(got, model.embed(texts))


async def test_max_wait_flushes_a_partial_batch() -> None:
    model = RecordingEmbedder()
    batcher = DynamicBatcher(model, max_batch_size=64, max_wait_ms=5)
    await batcher.start()
    out = await asyncio.wait_for(batcher.submit(["only one"]), timeout=2)
    await batcher.stop()
    assert out.shape == (1, 16)
    assert model.calls == [1]


async def test_model_error_reaches_all_waiters() -> None:
    model = RecordingEmbedder()
    batcher = DynamicBatcher(model, max_batch_size=4, max_wait_ms=5)
    await batcher.start()
    with patch.object(model, "embed", side_effect=RuntimeError("boom")):
        results = await asyncio.gather(
            batcher.submit(["a"]), batcher.submit(["b"]), return_exceptions=True
        )
    await batcher.stop()
    assert all(isinstance(r, RuntimeError) for r in results)


async def test_queue_full_and_not_started() -> None:
    model = RecordingEmbedder()
    batcher = DynamicBatcher(model, max_queue_size=2)
    with pytest.raises(RuntimeError, match="not started"):
        await batcher.submit(["x"])
    await batcher.start()
    with pytest.raises(QueueFullError):
        await batcher.submit(["a", "b", "c"])
    assert (await batcher.submit([])).shape == (0, 16)
    await batcher.stop()


def test_batch_size_validation() -> None:
    with pytest.raises(ValueError, match="max_batch_size"):
        DynamicBatcher(RecordingEmbedder(), max_batch_size=0)


def test_factory_selects_backend() -> None:
    assert isinstance(create_embedder(Settings(embedder_backend="hash")), HashEmbedder)


def test_onnx_embedder_pools_and_normalises(tmp_path: Path) -> None:
    """onnxruntime and the tokenizer are replaced by fakes, we test pooling and wiring."""
    hidden = np.array(
        [[[1.0, 0.0], [3.0, 0.0], [99.0, 99.0]], [[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]]],
        dtype=np.float32,
    )
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(), MagicMock()]
    session.get_inputs.return_value[0].name = "input_ids"
    session.get_inputs.return_value[1].name = "attention_mask"
    session.get_outputs.return_value = [MagicMock(shape=[None, None, 2])]
    session.run.return_value = [hidden]

    enc = [
        MagicMock(ids=[5, 6, 1], attention_mask=[1, 1, 0]),
        MagicMock(ids=[7, 8, 9], attention_mask=[1, 1, 1]),
    ]
    tok = MagicMock()
    tok.encode_batch.return_value = enc
    tok.token_to_id.return_value = 1

    with (
        patch("onnxruntime.InferenceSession", return_value=session),
        patch("tokenizers.Tokenizer.from_file", return_value=tok),
    ):
        model = OnnxEmbedder(model_dir=tmp_path, max_tokens=8)
        out = model.embed(["a", "b"])
        assert model.embed([]).shape == (0, 2)
        assert model.count_tokens(["a", "b"]) == 5

    assert model.dim == 2
    # padding positions are ignored by the mean, then L2 normalised
    assert np.allclose(out[0], [1.0, 0.0])
    assert np.allclose(out[1], [0.0, 1.0])
    feeds = session.run.call_args.args[1]
    assert set(feeds) == {"input_ids", "attention_mask"}
    tok.enable_truncation.assert_called_once_with(max_length=8)
