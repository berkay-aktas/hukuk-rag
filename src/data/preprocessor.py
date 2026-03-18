"""Dataset downloading, merging, and preprocessing."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_datasets(config: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Download all configured datasets.

    Args:
        config: Project config dict.
        output_dir: Directory to save raw downloads.

    Returns:
        Dict mapping dataset name to download path.
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for ds_info in config["datasets"]["huggingface"]:
        name = ds_info["name"]
        ds_path = output_dir / f"{name}.parquet"
        if ds_path.exists():
            logger.info(f"Skipping {name}, already downloaded")
            paths[name] = ds_path
            continue

        logger.info(f"Downloading {ds_info['id']}...")
        ds = load_dataset(ds_info["id"], split="train")
        ds.to_parquet(str(ds_path))
        paths[name] = ds_path
        logger.info(f"Saved {name} ({len(ds)} rows) to {ds_path}")

    for ds_info in config["datasets"]["kaggle"]:
        name = ds_info["name"]
        kaggle_dir = output_dir / name
        if kaggle_dir.exists():
            logger.info(f"Skipping {name}, already downloaded")
            paths[name] = kaggle_dir
            continue

        logger.info(f"Downloading Kaggle dataset {ds_info['slug']}...")
        _download_kaggle(ds_info["slug"], kaggle_dir)
        paths[name] = kaggle_dir

    return paths


def _download_kaggle(slug: str, output_dir: Path) -> None:
    """Download a Kaggle dataset.

    Args:
        slug: Kaggle dataset slug (owner/dataset-name).
        output_dir: Directory to extract files into.
    """
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(output_dir), "--unzip"],
        check=True,
    )


def merge_and_dedup(
    raw_paths: dict[str, Path],
    text_column: str = "text",
) -> pd.DataFrame:
    """Merge all datasets and deduplicate by text hash.

    Args:
        raw_paths: Dict mapping dataset name to file/directory path.
        text_column: Column containing the main text content.

    Returns:
        Deduplicated DataFrame with source metadata.
    """
    frames = []

    for name, path in tqdm(raw_paths.items(), desc="Loading datasets"):
        path = Path(path)
        if path.is_file() and path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.is_dir():
            parquet_files = list(path.glob("*.parquet"))
            csv_files = list(path.glob("*.csv"))
            json_files = list(path.glob("*.json"))

            if parquet_files:
                df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
            elif csv_files:
                df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            elif json_files:
                df = pd.concat([pd.read_json(f) for f in json_files], ignore_index=True)
            else:
                logger.warning(f"No supported files found in {path}")
                continue
        else:
            logger.warning(f"Unsupported path: {path}")
            continue

        df["_source"] = name
        frames.append(df)
        logger.info(f"Loaded {name}: {len(df)} rows, columns: {list(df.columns)}")

    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"Total rows before dedup: {len(merged)}")

    text_col = _find_text_column(merged, text_column)
    if text_col:
        merged["_text_hash"] = merged[text_col].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest() if pd.notna(x) else None
        )
        merged = merged.drop_duplicates(subset="_text_hash").drop(columns="_text_hash")

    logger.info(f"Total rows after dedup: {len(merged)}")
    return merged


def _find_text_column(df: pd.DataFrame, preferred: str) -> str | None:
    """Find the main text column in a DataFrame.

    Args:
        df: Input DataFrame.
        preferred: Preferred column name.

    Returns:
        Column name or None.
    """
    if preferred in df.columns:
        return preferred

    candidates = ["text", "content", "document", "passage", "context", "input"]
    for col in candidates:
        if col in df.columns:
            return col

    str_cols = df.select_dtypes(include="object").columns
    if len(str_cols) > 0:
        longest = max(str_cols, key=lambda c: df[c].str.len().mean())
        return longest

    return None


def save_corpus(df: pd.DataFrame, path: Path) -> None:
    """Save corpus to Parquet.

    Args:
        df: Corpus DataFrame.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved corpus ({len(df)} rows) to {path}")


def load_corpus(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load corpus from Parquet with optional column selection.

    Args:
        path: Path to Parquet file.
        columns: Optional list of columns to load (reduces memory).

    Returns:
        Corpus DataFrame.
    """
    return pd.read_parquet(path, columns=columns)
