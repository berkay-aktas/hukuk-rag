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


def infer_source(result: RetrievalResult) -> str:
    """Best-effort source label across our corpora's heterogeneous metadata.

    Different index bundles store source info under different keys:

    * Shared (``phase0_shared_ft``): ``metadata['source']`` is ``'ORICON'`` or
      ``'YARGITAY'`` (the latter being HGK Kamulastırma case-law).
    * Mevzuat (``mevzuat_ft``): no source field; ``source_file`` is the PDF name.
    * Yargıtay legacy base index: ``metadata['_source']`` is the parquet name
      (typically ``'yargitay_700k'``).

    Returns a lowercase label so callers can use simple set-membership tests
    (e.g., ``source.startswith('yarg')`` or ``source == 'yargitay'``). Falls
    back to ``'unknown'`` when nothing matches — callers should treat that as
    "do not filter on source for this result."
    """
    md = result.metadata or {}
    explicit = md.get('source') or md.get('_source')
    if explicit:
        s = str(explicit).strip().lower()
        # Normalize the common case-law label variants to 'yargitay'.
        if 'yarg' in s:
            return 'yargitay'
        return s
    sf = (md.get('source_file') or '').lower()
    if 'mevzuat' in sf:
        return 'mevzuat'
    if 'yarg' in sf:
        return 'yargitay'
    return 'unknown'
