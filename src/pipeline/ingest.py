"""Corpus ingestion: turn any input into FAISS + BM25 indexes the rest of the pipeline can use.

Accepted input formats (auto-detected):

1. **JSONL** — one record per line, supports two schemas:
   - Pre-chunked: ``{"id"|"chunk_id": str, "text": str, ...metadata}``
     (shared classmate dataset uses ``id``; our own format uses ``chunk_id``)
   - Raw documents: ``{"id"|"doc_id": str, "text": str, ...metadata}`` where
     ``text`` is a full document — we chunk it via :func:`legal_aware_chunk`.
     Distinguished from pre-chunked by ``text`` length: >2000 chars triggers chunking.

2. **Parquet** — must contain ``chunk_id`` + ``text`` columns (legacy format from
   prior sessions' ``chunks.parquet``).

3. **JSON array** — top-level array of records, same record schema as JSONL.

4. **Directory** — recursively gathers ``.txt``, ``.md``, ``.pdf`` files. PDFs are
   parsed with ``pypdf``. Each file becomes one document → multiple chunks via
   :func:`legal_aware_chunk`. This is the path the course evaluator's custom
   corpus likely takes.

Output layout (written to ``output_dir``):

::

    output_dir/
    ├── chunks.parquet           # canonical chunked corpus (one row per chunk)
    ├── faiss.index              # FAISS IVF-PQ over the chosen embedding
    ├── faiss.mapping.pkl        # chunk_id → {text, metadata} mapping
    ├── bm25.pkl                 # pickled {"index": BM25Okapi, "mapping": [...]}
    └── manifest.json            # ingestion metadata: source paths, # chunks, model id, timestamp

The output is a self-contained "index bundle" that ``benchmark.py``, ``query.py``,
and the demo can load without knowing how it was built.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import pandas as pd

logger = logging.getLogger(__name__)

InputFormat = Literal["jsonl", "json_array", "parquet", "directory"]

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Chunking heuristic: text longer than this counts as a raw document needing
# segmentation. Shorter rows are treated as already-chunked passages.
RAW_DOC_THRESHOLD = 2_000


def detect_input_format(path: Path | str) -> InputFormat:
    """Detect the input format of a corpus source.

    Args:
        path: Path to a file or directory.

    Returns:
        One of ``"jsonl"``, ``"json_array"``, ``"parquet"``, ``"directory"``.

    Raises:
        ValueError: If the path doesn't exist or its format can't be determined.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Input does not exist: {p}")
    if p.is_dir():
        return "directory"
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        with open(p, encoding="utf-8") as f:
            head = f.read(64).lstrip()
        return "json_array" if head.startswith("[") else "jsonl"
    raise ValueError(f"Cannot determine format for {p}")


def build_indexes(
    inputs: Path | str | Iterable[Path | str],
    output_dir: Path | str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    *,
    build_bm25: bool = True,
    build_faiss: bool = True,
    chunker_max_tokens: int = 480,
    chunker_overlap_tokens: int = 64,
    faiss_nlist: int = 256,
    faiss_m: int = 32,
    faiss_nbits: int = 8,
    embedding_batch_size: int = 64,
) -> dict[str, Any]:
    """Ingest one or more corpus sources and build FAISS + BM25 indexes.

    Args:
        inputs: A single path or list of paths. Each may be a JSONL/JSON/Parquet
            file or a directory of raw documents. All inputs are merged into a
            single corpus.
        output_dir: Directory to write the index bundle into.
        embedding_model: HuggingFace model id or local path. Defaults to
            multilingual-e5-large. Pass a Drive path to use a fine-tuned checkpoint.
        build_bm25: Whether to build the BM25 sparse index.
        build_faiss: Whether to build the FAISS dense index.
        chunker_max_tokens: Max tokens per chunk when chunking raw documents.
        chunker_overlap_tokens: Overlap between consecutive chunks.
        faiss_nlist: FAISS IVF cells. Tune down for small corpora (<10k chunks).
        faiss_m: PQ sub-quantizers. Must divide the embedding dimension.
        faiss_nbits: Bits per sub-quantizer.
        embedding_batch_size: Batch size for embedding encoder.

    Returns:
        A manifest dict (also written to ``output_dir/manifest.json``).
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    input_paths = [Path(p) for p in inputs]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    logger.info("Ingesting %d source(s) into %s", len(input_paths), output_dir)

    chunks: list[dict[str, Any]] = []
    per_source_counts: dict[str, int] = {}
    for src in input_paths:
        fmt = detect_input_format(src)
        logger.info("  source %s [%s]", src, fmt)
        before = len(chunks)
        chunks.extend(_load_chunks(src, fmt, chunker_max_tokens, chunker_overlap_tokens))
        per_source_counts[str(src)] = len(chunks) - before

    if not chunks:
        raise RuntimeError("No chunks produced from any input source.")

    logger.info("Total chunks: %d", len(chunks))

    # Auto-tune FAISS nlist for small corpora. Faiss requires nlist * 39 training
    # vectors; if we have fewer chunks than that we need a smaller nlist.
    if build_faiss and len(chunks) < faiss_nlist * 40:
        adjusted = max(8, len(chunks) // 40)
        if adjusted < faiss_nlist:
            logger.warning(
                "Corpus too small for nlist=%d; auto-tuning to %d",
                faiss_nlist, adjusted,
            )
            faiss_nlist = adjusted

    # 1. Save canonical chunks parquet (allows downstream tools to re-read without rebuilding)
    chunks_df = pd.DataFrame(chunks)
    chunks_parquet_path = output_dir / "chunks.parquet"
    chunks_df.to_parquet(chunks_parquet_path, index=False)
    logger.info("Wrote chunks parquet: %s (%d rows)", chunks_parquet_path, len(chunks_df))

    # 2. Build BM25
    bm25_path: Path | None = None
    if build_bm25:
        from src.retrieval.bm25 import build_bm25_index

        bm25_path = output_dir / "bm25.pkl"
        build_bm25_index(chunks, save_path=bm25_path)
        logger.info("Built BM25 index: %s", bm25_path)

    # 3. Build FAISS
    faiss_path: Path | None = None
    if build_faiss:
        from src.retrieval.dense import build_faiss_index

        faiss_path = output_dir / "faiss.index"
        build_faiss_index(
            chunks,
            model_name=embedding_model,
            save_path=faiss_path,
            batch_size=embedding_batch_size,
            nlist=faiss_nlist,
            m=faiss_m,
            nbits=faiss_nbits,
        )
        logger.info("Built FAISS index: %s", faiss_path)

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(time.time() - started_at, 1),
        "sources": [str(p) for p in input_paths],
        "per_source_chunk_counts": per_source_counts,
        "total_chunks": len(chunks),
        "embedding_model": embedding_model if build_faiss else None,
        "faiss": {
            "path": str(faiss_path.relative_to(output_dir)) if faiss_path else None,
            "nlist": faiss_nlist,
            "m": faiss_m,
            "nbits": faiss_nbits,
        } if build_faiss else None,
        "bm25": {
            "path": str(bm25_path.relative_to(output_dir)) if bm25_path else None,
        } if build_bm25 else None,
        "chunker": {
            "max_tokens": chunker_max_tokens,
            "overlap_tokens": chunker_overlap_tokens,
            "type": "legal_aware",
        },
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Wrote manifest: %s", manifest_path)

    return manifest


def _load_chunks(
    source: Path,
    fmt: InputFormat,
    chunker_max_tokens: int,
    chunker_overlap_tokens: int,
) -> Iterator[dict[str, Any]]:
    """Load chunks from a single source, chunking raw documents on the fly."""
    if fmt == "directory":
        yield from _load_chunks_from_directory(source, chunker_max_tokens, chunker_overlap_tokens)
    elif fmt == "parquet":
        yield from _load_chunks_from_parquet(source)
    elif fmt == "jsonl":
        yield from _load_chunks_from_jsonl(source, chunker_max_tokens, chunker_overlap_tokens)
    elif fmt == "json_array":
        yield from _load_chunks_from_json_array(source, chunker_max_tokens, chunker_overlap_tokens)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _load_chunks_from_parquet(path: Path) -> Iterator[dict[str, Any]]:
    df = pd.read_parquet(path)
    if "text" not in df.columns:
        raise ValueError(f"Parquet {path} missing 'text' column. Columns: {list(df.columns)}")
    if "chunk_id" not in df.columns:
        df = df.copy()
        df["chunk_id"] = [f"{path.stem}__row_{i:09d}" for i in range(len(df))]
    for rec in df.to_dict(orient="records"):
        yield _normalize_record(rec)


def _load_chunks_from_jsonl(
    path: Path,
    chunker_max_tokens: int,
    chunker_overlap_tokens: int,
) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSONL line %d in %s: %s", line_no, path, e)
                continue
            yield from _record_to_chunks(rec, path.stem, line_no, chunker_max_tokens, chunker_overlap_tokens)


def _load_chunks_from_json_array(
    path: Path,
    chunker_max_tokens: int,
    chunker_overlap_tokens: int,
) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{path} is not a JSON array")
    for i, rec in enumerate(records):
        yield from _record_to_chunks(rec, path.stem, i, chunker_max_tokens, chunker_overlap_tokens)


def _load_chunks_from_directory(
    dir_path: Path,
    chunker_max_tokens: int,
    chunker_overlap_tokens: int,
) -> Iterator[dict[str, Any]]:
    """Walk a directory of raw documents and chunk each one."""
    from src.data.chunker import legal_aware_chunk

    supported_suffixes = {".txt", ".md", ".pdf"}
    files = sorted(p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in supported_suffixes)
    if not files:
        logger.warning("No supported documents found under %s", dir_path)
        return
    logger.info("  found %d document(s) under %s", len(files), dir_path)
    for file_path in files:
        try:
            text = _read_document(file_path)
        except Exception as e:
            logger.warning("Skipping %s: %s", file_path, e)
            continue
        if not text.strip():
            continue
        doc_id = str(file_path.relative_to(dir_path))
        metadata = {"source_file": str(file_path), "doc_id": doc_id}
        chunks = legal_aware_chunk(
            text=text,
            metadata=metadata,
            parent_doc_id=doc_id,
            max_tokens=chunker_max_tokens,
            overlap_tokens=chunker_overlap_tokens,
        )
        for c in chunks:
            yield _normalize_record({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "parent_doc_id": c.parent_doc_id,
                **c.metadata,
            })


def _read_document(path: Path) -> str:
    """Read a single document file to text."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError(
                "PDF ingestion requires pypdf. Install with: pip install pypdf"
            ) from e
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported document suffix: {suffix}")


def _record_to_chunks(
    rec: dict[str, Any],
    source_stem: str,
    record_idx: int,
    chunker_max_tokens: int,
    chunker_overlap_tokens: int,
) -> Iterator[dict[str, Any]]:
    """Convert a single input record to one or more chunks.

    If the record's text is short, treat it as already-chunked.
    If the text is long, chunk it with the legal-aware chunker.
    """
    text = rec.get("text") or rec.get("content") or rec.get("passage")
    if not text:
        return
    text = str(text)

    if len(text) <= RAW_DOC_THRESHOLD:
        # Already-chunked: pass through with id normalization
        yield _normalize_record(rec, fallback_id=f"{source_stem}__row_{record_idx:09d}")
        return

    # Raw document: needs chunking
    from src.data.chunker import legal_aware_chunk

    doc_id = rec.get("doc_id") or rec.get("id") or rec.get("chunk_id") or f"{source_stem}__doc_{record_idx:09d}"
    metadata = {k: v for k, v in rec.items() if k not in ("text", "content", "passage")}
    chunks = legal_aware_chunk(
        text=text,
        metadata=metadata,
        parent_doc_id=str(doc_id),
        max_tokens=chunker_max_tokens,
        overlap_tokens=chunker_overlap_tokens,
    )
    for c in chunks:
        yield _normalize_record({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "parent_doc_id": c.parent_doc_id,
            **c.metadata,
        })


def _normalize_record(rec: dict[str, Any], fallback_id: str | None = None) -> dict[str, Any]:
    """Normalize a record to ``{chunk_id, text, ...metadata}`` shape.

    Handles three id-field variants: ``chunk_id`` (our format), ``id`` (shared
    format), ``_id`` (some MongoDB exports).
    """
    rec = dict(rec)  # don't mutate caller's dict

    if "chunk_id" not in rec:
        for k in ("id", "_id", "chunk_id"):
            if k in rec:
                rec["chunk_id"] = str(rec.pop(k))
                break
        else:
            if fallback_id is None:
                raise ValueError(f"Record has no id field and no fallback: {list(rec.keys())}")
            rec["chunk_id"] = fallback_id

    if "text" not in rec:
        for k in ("content", "passage"):
            if k in rec:
                rec["text"] = rec.pop(k)
                break
        else:
            raise ValueError(f"Record has no text field: {list(rec.keys())}")

    # Flatten nested 'metadata' dict if present (shared dataset uses this)
    if "metadata" in rec and isinstance(rec["metadata"], dict):
        meta = rec.pop("metadata")
        for k, v in meta.items():
            if k not in rec:  # don't overwrite top-level keys
                rec[k] = v

    rec["chunk_id"] = str(rec["chunk_id"])
    rec["text"] = str(rec["text"])
    return rec
