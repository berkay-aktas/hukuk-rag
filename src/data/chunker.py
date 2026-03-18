"""Legal-aware document chunking for Turkish legal texts."""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

MADDE_PATTERN = re.compile(r'(?:^|\n)\s*(Madde\s+\d+)', re.MULTILINE)
SECTION_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:BÖLÜM|KISIM|FASIL|BAŞLIK)\s+',
    re.MULTILINE | re.IGNORECASE,
)
PARAGRAPH_BREAK = re.compile(r'\n\s*\n')


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    text: str
    chunk_id: str
    parent_doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def legal_aware_chunk(
    text: str,
    metadata: dict[str, Any],
    parent_doc_id: str,
    max_tokens: int = 480,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Split text using legal-aware boundaries.

    Tries to split on 'Madde X' boundaries first, then section headers,
    then paragraph breaks, falling back to token-based splitting.

    Args:
        text: Document text to chunk.
        metadata: Metadata to attach to each chunk.
        parent_doc_id: ID of the parent document.
        max_tokens: Maximum tokens per chunk (whitespace-tokenized).
        overlap_tokens: Token overlap between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    if not text or not text.strip():
        return []

    segments = _split_by_legal_markers(text)

    chunks = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        token_count = len(segment.split())
        if token_count <= max_tokens:
            chunks.append(segment)
        else:
            sub_chunks = _recursive_split(segment, max_tokens, overlap_tokens)
            chunks.extend(sub_chunks)

    return [
        Chunk(
            text=chunk_text,
            chunk_id=str(uuid.uuid4()),
            parent_doc_id=parent_doc_id,
            metadata=metadata.copy(),
        )
        for chunk_text in chunks
        if chunk_text.strip()
    ]


def _split_by_legal_markers(text: str) -> list[str]:
    """Split text on legal markers (Madde X, section headers).

    Args:
        text: Input text.

    Returns:
        List of text segments.
    """
    madde_positions = [m.start() for m in MADDE_PATTERN.finditer(text)]

    if len(madde_positions) >= 2:
        segments = []
        for i, pos in enumerate(madde_positions):
            end = madde_positions[i + 1] if i + 1 < len(madde_positions) else len(text)
            segments.append(text[pos:end])

        if madde_positions[0] > 0:
            segments.insert(0, text[:madde_positions[0]])
        return segments

    section_positions = [m.start() for m in SECTION_PATTERN.finditer(text)]
    if len(section_positions) >= 2:
        segments = []
        for i, pos in enumerate(section_positions):
            end = section_positions[i + 1] if i + 1 < len(section_positions) else len(text)
            segments.append(text[pos:end])
        if section_positions[0] > 0:
            segments.insert(0, text[:section_positions[0]])
        return segments

    paragraphs = PARAGRAPH_BREAK.split(text)
    if len(paragraphs) >= 2:
        return paragraphs

    return [text]


def _recursive_split(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into chunks by token count with overlap.

    Args:
        text: Input text.
        max_tokens: Max tokens per chunk.
        overlap_tokens: Overlap between chunks.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap_tokens

    return chunks


def chunk_corpus(
    corpus_df: pd.DataFrame,
    text_column: str = "text",
    id_column: str | None = None,
    max_tokens: int = 480,
    overlap_tokens: int = 64,
) -> pd.DataFrame:
    """Apply legal-aware chunking to entire corpus.

    Args:
        corpus_df: Corpus DataFrame with text column.
        text_column: Name of the text column.
        id_column: Name of the document ID column. Auto-generated if None.
        max_tokens: Max tokens per chunk.
        overlap_tokens: Overlap between chunks.

    Returns:
        DataFrame with columns: chunk_id, parent_doc_id, text, + original metadata.
    """
    all_chunks = []

    for idx, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Chunking"):
        text = row.get(text_column, "")
        if not isinstance(text, str) or not text.strip():
            continue

        doc_id = str(row[id_column]) if id_column and id_column in row.index else str(idx)
        metadata = {
            col: row[col]
            for col in row.index
            if col not in (text_column, id_column)
        }

        chunks = legal_aware_chunk(
            text=text,
            metadata=metadata,
            parent_doc_id=doc_id,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        for chunk in chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "parent_doc_id": chunk.parent_doc_id,
                "text": chunk.text,
                **chunk.metadata,
            }
            all_chunks.append(chunk_dict)

    result = pd.DataFrame(all_chunks)
    logger.info(
        f"Chunked {len(corpus_df)} documents into {len(result)} chunks "
        f"(avg {len(result)/max(len(corpus_df),1):.1f} chunks/doc)"
    )
    return result


def chunk_parquet_streaming(
    input_paths: list[tuple[Path, str]],
    output_path: Path,
    text_column: str = "text",
    max_tokens: int = 480,
    overlap_tokens: int = 64,
    batch_size: int = 10_000,
) -> int:
    """Chunk multiple parquet files in batches without loading all into RAM.

    Reads each parquet file in batches, chunks each batch, and appends
    results to the output parquet file incrementally.

    Args:
        input_paths: List of (parquet_path, source_name) tuples.
        output_path: Output parquet file for all chunks.
        text_column: Name of the text column.
        max_tokens: Max tokens per chunk.
        overlap_tokens: Overlap between chunks.
        batch_size: Rows to process at a time.

    Returns:
        Total number of chunks created.
    """
    import pyarrow.parquet as pq

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    writer = None

    for parquet_path, source_name in input_paths:
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            logger.warning(f"Skipping {parquet_path} — file not found")
            continue

        pf = pq.ParquetFile(parquet_path)
        n_rows = pf.metadata.num_rows
        logger.info(f"Processing {source_name}: {n_rows:,} rows in batches of {batch_size}")

        doc_offset = total_chunks
        batch_num = 0

        for batch in tqdm(
            pf.iter_batches(batch_size=batch_size, columns=[text_column] if text_column in pf.schema.names else None),
            total=(n_rows + batch_size - 1) // batch_size,
            desc=f"Chunking {source_name}",
        ):
            batch_df = batch.to_pandas()
            if text_column not in batch_df.columns:
                # Try to find a text-like column
                str_cols = batch_df.select_dtypes(include="object").columns
                if len(str_cols) > 0:
                    text_column_actual = str_cols[0]
                else:
                    continue
            else:
                text_column_actual = text_column

            chunk_rows = []
            for i, row in batch_df.iterrows():
                text = row.get(text_column_actual, "")
                if not isinstance(text, str) or not text.strip():
                    continue

                doc_id = f"{source_name}_{doc_offset + batch_num * batch_size + i}"
                chunks = legal_aware_chunk(
                    text=text,
                    metadata={"_source": source_name},
                    parent_doc_id=doc_id,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
                for c in chunks:
                    chunk_rows.append({
                        "chunk_id": c.chunk_id,
                        "parent_doc_id": c.parent_doc_id,
                        "text": c.text,
                        "_source": source_name,
                    })

            if chunk_rows:
                import pyarrow as pa
                chunk_table = pa.Table.from_pylist(chunk_rows)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_path), chunk_table.schema)
                writer.write_table(chunk_table)
                total_chunks += len(chunk_rows)

            batch_num += 1
            del batch_df, chunk_rows
            import gc; gc.collect()

    if writer:
        writer.close()

    logger.info(f"Total chunks created: {total_chunks:,}")
    return total_chunks
