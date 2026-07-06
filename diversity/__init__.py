"""Diversity metrics for LLM-EPS populations.

A small, dependency-light, framework-agnostic library for the two diversity
metrics introduced in HSEvo:

- **SWDI** (Shannon-Wiener Diversity Index): point-in-time diversity.
- **CDI** (Cumulative Diversity Index): overall spread via a minimum spanning tree.

Quick start::

    from diversity import compute_diversity
    result = compute_diversity(list_of_code_strings)
    print(result.swdi, result.cdi)
"""

from .core import (
    DEFAULT_THRESHOLD,
    DiversityResult,
    compute_diversity,
    compute_diversity_from_embeddings,
)
from .embeddings import DEFAULT_CHECKPOINT, CodeT5pEmbedder, Embedder
from .metrics import (
    cosine_similarity_matrix,
    cumulative_diversity_index,
    shannon_wiener_index,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_CHECKPOINT",
    "DiversityResult",
    "compute_diversity",
    "compute_diversity_from_embeddings",
    "shannon_wiener_index",
    "cumulative_diversity_index",
    "cosine_similarity_matrix",
    "CodeT5pEmbedder",
    "Embedder",
]

__version__ = "0.1.0"
