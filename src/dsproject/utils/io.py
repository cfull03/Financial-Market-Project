"""IO utilities for the EDA-first pipeline.

Focus on safe filesystem ops and small conveniences used by CLI steps.
Only the essentials; no ML/DL dependencies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional
import hashlib
import json
import os
import tempfile

import pandas as pd
import yaml

__all__ = [
    "ensure_dirs",
    "dataset_split_paths",
    "read_csv",
    "write_csv",
    "read_json",
    "write_json",
    "read_yaml",
    "write_yaml",
    "file_hash",
    "infer_id_columns",
    "infer_datetime_columns",
]


# ------------------------------ dirs & paths ------------------------------

def ensure_dirs(*dirs: Path | str) -> None:
    """Create directories if missing.
    Why: avoid first-run errors when paths don't exist yet.
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def dataset_split_paths(name: str, processed_dir: Path) -> tuple[Path, Path]:
    """Canonical locations for split CSVs for a dataset name."""
    train_path = Path(processed_dir) / f"{name}_train.csv"
    test_path = Path(processed_dir) / f"{name}_test.csv"
    return train_path, test_path


# ------------------------------ atomic writes -----------------------------

def _atomic_write_bytes(data: bytes, path: Path) -> None:
    """Write bytes atomically.
    Why: prevents partial files if the process is interrupted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def write_json(obj: Any, path: Path, *, indent: int = 2) -> None:
    _atomic_write_bytes(json.dumps(obj, indent=indent).encode("utf-8"), Path(path))


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_yaml(obj: Any, path: Path) -> None:
    _atomic_write_bytes(yaml.safe_dump(obj, sort_keys=False).encode("utf-8"), Path(path))


def read_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------ CSV helpers -------------------------------

def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Robust CSV reader with sane fallbacks.
    Why: handle BOM/encoding quirks without breaking the pipeline.
    """
    path = Path(path)
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1", engine="python", **kwargs)


def write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


# ------------------------------ misc utils --------------------------------

def file_hash(path: Path, *, algo: str = "sha256", chunk_size: int = 1 << 16) -> str:
    """Return hex digest of a file. Useful for lightweight data versioning."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_id_columns(df: pd.DataFrame, *, max_unique_frac: float = 0.98, exclude: Optional[Iterable[str]] = None) -> List[str]:
    """Heuristically find ID-like columns with near-unique values.
    Why: these are often dropped for EDA/featurization.
    """
    exclude = set(exclude or [])
    ids: List[str] = []
    n = max(len(df), 1)
    for c in df.columns:
        if c in exclude:
            continue
        frac = df[c].nunique(dropna=True) / n
        if frac >= max_unique_frac:
            ids.append(c)
    return ids


def infer_datetime_columns(df: pd.DataFrame, *, min_parse_rate: float = 0.9) -> List[str]:
    """Detect columns that parse cleanly as datetimes.
    Why: helps configure `datetime_columns` in config.
    """
    dt_cols: List[str] = []
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        rate = 1.0 - float(s.isna().mean())
        if rate >= min_parse_rate:
            dt_cols.append(c)
    return dt_cols
