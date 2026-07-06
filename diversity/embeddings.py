"""Embedders that turn code snippets into numeric vectors.

The core metrics only need numpy arrays, so embedding is kept separate and
pluggable. The default :class:`CodeT5pEmbedder` matches the model used in the
original HSEvo analysis but is imported lazily so that the rest of the library
works without ``torch``/``transformers`` installed.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable

import numpy as np

DEFAULT_CHECKPOINT = "Salesforce/codet5p-110m-embedding"


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps a batch of code strings to a 2D array of embeddings."""

    def embed(self, code_snippets: Sequence[str]) -> np.ndarray:
        """Return a ``(n, d)`` array of embeddings for ``code_snippets``."""
        ...


class CodeT5pEmbedder:
    """Default embedder backed by a CodeT5+ embedding model.

    ``torch`` and ``transformers`` are imported lazily on first use, so simply
    importing this class (e.g. for type hints) has no heavy dependencies.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cpu",
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "CodeT5pEmbedder requires the 'transformers' and 'torch' packages. "
                "Install them with `pip install transformers torch` (or "
                "`pip install .[embeddings]`), or pass precomputed embeddings / a "
                "custom embedder instead."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.checkpoint, trust_remote_code=True
        ).to(self.device)

    def embed(self, code_snippets: Sequence[str]) -> np.ndarray:
        import torch

        self._ensure_loaded()

        embeddings: List[np.ndarray] = []
        for code in code_snippets:
            inputs = self._tokenizer.encode(code, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embedding = self._model(inputs)[0].cpu().numpy()
            embeddings.append(embedding.reshape(1, -1))

        if not embeddings:
            return np.empty((0, 0), dtype=np.float64)
        return np.vstack(embeddings)
