"""Shared types for retrieval modules."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata.

    Used as the common return type across dense, sparse, and fused retrieval.
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
