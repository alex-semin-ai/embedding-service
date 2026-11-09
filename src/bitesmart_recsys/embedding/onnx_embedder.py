"""ONNX Runtime embedder for intfloat/multilingual-e5-small (mean pooling + L2 norm)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import Matrix, l2_normalize

logger = logging.getLogger(__name__)


class OnnxEmbedder:
    """Runs an E5 style encoder exported to ONNX. No PyTorch dependency, small image."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        onnx_file: str = "onnx/model.onnx",
        max_tokens: int = 256,
        model_dir: str | Path | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_name = model_name
        model_path, tokenizer_path = self._resolve_files(model_name, onnx_file, model_dir)

        options = ort.SessionOptions()
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=max_tokens)
        pad_id = self._tokenizer.token_to_id("<pad>")
        self._tokenizer.enable_padding(pad_id=1 if pad_id is None else pad_id, pad_token="<pad>")
        self.dim = int(self._session.get_outputs()[0].shape[-1])
        logger.info("onnx embedder loaded", extra={"model": model_name, "dim": self.dim})

    @staticmethod
    def _resolve_files(
        model_name: str, onnx_file: str, model_dir: str | Path | None
    ) -> tuple[Path, Path]:
        if model_dir is not None:
            base = Path(model_dir)
            return base / onnx_file, base / "tokenizer.json"
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(model_name, onnx_file)
        tokenizer_path = hf_hub_download(model_name, "onnx/tokenizer.json")
        return Path(model_path), Path(tokenizer_path)

    def _encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encodings = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        return ids, mask

    def embed(self, texts: list[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        ids, mask = self._encode(texts)
        feeds: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feeds)[0]
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        return l2_normalize(pooled.astype(np.float32))

    def count_tokens(self, texts: list[str]) -> int:
        return int(sum(sum(e.attention_mask) for e in self._tokenizer.encode_batch(texts)))
