# ============================
# File: src/dsproject/data/make_dataset.py
# ============================
"""Data ingestion and timestamped splitting (EDA-first, no ML/DL).

Why
----
- Keep **raw** data canonical and stable (no timestamp in filename).
- Write **processed** outputs with a timestamp suffix for traceability.

Public API
----------
- `ingest_csv(src_csv, raw_dir, name) -> Path`
- `split_csv(raw_csv, processed_dir, *, target_col, test_size, random_state) -> (Path, Path)`
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from dsproject.utils.io import to_process, write_csv

logger = logging.getLogger(__name__)

__all__ = ["ingest_csv", "split_csv"]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _read_csv_robust(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with encoding fallbacks. Why: BOM/encoding can vary."""
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
    """Ingest a CSV into **raw** as `<name>.csv` (no timestamp).

    Rationale: a stable, canonical filename in raw simplifies diffs & provenance.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = _read_csv_robust(Path(src_csv))
    out = raw_dir / f"{name}.csv"
    write_csv(df, out, index=False)
    logger.info("Ingested %s → %s (rows=%d, cols=%d)", src_csv, out, df.shape[0], df.shape[1])
    return out


def split_csv(
    raw_csv: Path,
    processed_dir: Path,
    *,
    target_col: Optional[str],
    test_size: float,
    random_state: int,
) -> Tuple[Path, Path]:
    """Split a raw CSV into train/test and save to **processed** with timestamps.

    Stratifies on `target_col` when present and low-cardinality (< 50 uniques).
    Filenames are `<stem>_train_YYYYMMDD_HHMMSS.csv` and `<stem>_test_...` via `to_process`.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = _read_csv_robust(Path(raw_csv))

    stratify = None
    if target_col and target_col in df.columns:
        n_unique = df[target_col].nunique(dropna=True)
        if 1 < n_unique < 50:
            stratify = df[target_col]
            logger.info("Stratifying on '%s' (unique=%d)", target_col, n_unique)
        else:
            logger.info("Not stratifying: '%s' unique=%d", target_col, n_unique)
    elif target_col:
        logger.warning("Target column '%s' not found; proceeding without stratify.", target_col)

    train_df, test_df = train_test_split(
        df, test_size=float(test_size), random_state=int(random_state), stratify=stratify
    )

    stem = Path(raw_csv).stem
    train_path = to_process(train_df, f"{stem}_train", processed_dir)
    test_path = to_process(test_df, f"{stem}_test", processed_dir)

    logger.info("Split %s → train=%s test=%s", raw_csv, train_path.name, test_path.name)
    return train_path, test_path
