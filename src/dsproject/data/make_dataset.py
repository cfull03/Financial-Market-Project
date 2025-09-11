"""Data ingestion and splitting utilities (EDA-first).

Functions:
- ingest_csv: copy a source CSV into data/raw with a canonical name
- split_csv: deterministic train/test split saved under data/processed

Notes:
- Uses robust CSV reading with encoding fallbacks.
- Stratifies on the target when it looks like classification (low cardinality).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

__all__ = ["ingest_csv", "split_csv"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_csv_robust(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with common encoding fallbacks.
    Why: avoid breaking the pipeline on BOM/encoding differences.
    """
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1", engine="python", **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_csv(src_csv: Path, raw_dir: Path, name: str) -> Path:
    """Copy data into versioned raw path. Preserves provenance.

    Parameters
    ----------
    src_csv : Path
        Path to the source CSV (any location).
    raw_dir : Path
        Project's raw data directory (e.g., data/raw).
    name : str
        Base filename to save as (no extension), e.g., "finance".

    Returns
    -------
    Path
        The path of the canonical raw CSV written under raw_dir.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = _read_csv_robust(Path(src_csv))
    out = raw_dir / f"{name}.csv"
    df.to_csv(out, index=False)
    logger.info(
        "Ingested %s → %s (rows=%d, cols=%d)", src_csv, out, df.shape[0], df.shape[1]
    )
    return out


def split_csv(
    raw_csv: Path,
    processed_dir: Path,
    *,
    target_col: Optional[str],
    test_size: float,
    random_state: int,
) -> Tuple[Path, Path]:
    """Create train/test CSVs from a raw dataset.

    Stratifies on `target_col` if present and low-cardinality (< 50 unique values).

    Parameters
    ----------
    raw_csv : Path
        Path to canonical raw CSV (e.g., data/raw/finance.csv).
    processed_dir : Path
        Output directory for splits (e.g., data/processed).
    target_col : Optional[str]
        Column name used for stratified split when classification-like.
    test_size : float
        Fraction for test set (0 < test_size < 1).
    random_state : int
        Random seed for deterministic splits.

    Returns
    -------
    (Path, Path)
        Paths to train and test CSVs under processed_dir.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = _read_csv_robust(Path(raw_csv))

    stratify = None
    if target_col and target_col in df.columns:
        n_unique = df[target_col].nunique(dropna=True)
        if n_unique > 1 and n_unique < 50:
            stratify = df[target_col]
            logger.info(
                "Stratifying split on target '%s' (unique=%d)", target_col, n_unique
            )
        else:
            logger.info(
                "Not stratifying: target '%s' unique values = %d", target_col, n_unique
            )
    else:
        if target_col:
            logger.warning("Target column '%s' not found; proceeding without stratify.", target_col)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )

    stem = Path(raw_csv).stem
    train_path = processed_dir / f"{stem}_train.csv"
    test_path = processed_dir / f"{stem}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Split %s → train=%s test=%s", raw_csv, train_path, test_path)
    return train_path, test_path
