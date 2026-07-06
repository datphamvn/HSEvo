"""High-level API for computing diversity metrics.

Typical usage::

    from diversity import compute_diversity
    result = compute_diversity(list_of_code_strings)
    print(result.swdi, result.cdi)

Framework-agnostic usage with your own embeddings::

    from diversity import compute_diversity_from_embeddings
    result = compute_diversity_from_embeddings(my_numpy_embeddings)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .embeddings import CodeT5pEmbedder, Embedder
from .metrics import (
    ArrayLike,
    cumulative_diversity_index,
    shannon_wiener_index,
)

DEFAULT_THRESHOLD = 0.95


@dataclass
class DiversityResult:
    """Container for the two diversity metrics."""

    swdi: float
    cdi: float

    def as_dict(self) -> dict[str, float]:
        return {"swdi": self.swdi, "cdi": self.cdi}


def compute_diversity_from_embeddings(
    embeddings: ArrayLike,
    threshold: float = DEFAULT_THRESHOLD,
) -> DiversityResult:
    """Compute SWDI and CDI from precomputed embeddings.

    Args:
        embeddings: A ``(n, d)`` array or a sequence of ``n`` embedding vectors.
        threshold: Cosine-similarity threshold used by SWDI clustering.
    """
    return DiversityResult(
        swdi=shannon_wiener_index(embeddings, threshold=threshold),
        cdi=cumulative_diversity_index(embeddings),
    )


def compute_diversity(
    code_snippets: Sequence[str],
    embedder: Optional[Embedder] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> DiversityResult:
    """Embed ``code_snippets`` and compute SWDI and CDI.

    Args:
        code_snippets: The code strings (e.g. heuristics) to measure.
        embedder: An object implementing :class:`~diversity.embeddings.Embedder`.
            Defaults to :class:`~diversity.embeddings.CodeT5pEmbedder`.
        threshold: Cosine-similarity threshold used by SWDI clustering.
    """
    if embedder is None:
        embedder = CodeT5pEmbedder()
    embeddings = embedder.embed(code_snippets)
    return compute_diversity_from_embeddings(embeddings, threshold=threshold)
