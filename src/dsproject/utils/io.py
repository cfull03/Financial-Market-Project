# ============================
# File: src/dsproject/utils/io.py
# ============================
"""Lightweight I/O helpers for data projects.

Why
----
- Keep file formats consistent and safe across scripts.
- Provide **timestamped** saves to `data/interim` and `data/processed`.
- Make it easy to locate the **latest** train/test splits even when timestamped.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime as _dt
from pathlib import Path
from typing import Any, List, Optional, Union

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
    # timestamped stage saves (preferred)
    "to_interm",
    "to_process",
    # generic stage saver
    "to_stage",
]


# ---------------------------------------------------------------------------
# Dirs
# ---------------------------------------------------------------------------


def ensure_dirs(*dirs: Union[str, Path]) -> None:
    """Create directories if missing (idempotent)."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CSV / JSON / YAML
# ---------------------------------------------------------------------------


def read_csv(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """Robust CSV reader with common encoding fallbacks.
    Why: BOM/encoding differences otherwise cause brittle pipelines.
    """
    p = Path(path)
    try:
        return pd.read_csv(p, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(p, encoding="utf-8-sig", **kwargs)
        except UnicodeDecodeError:
            return pd.read_csv(p, encoding="latin-1", engine="python", **kwargs)


def write_csv(df: pd.DataFrame, path: Union[str, Path], *, index: bool = False) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index, encoding="utf-8")
    return p


def read_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return p


def read_yaml(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(obj: Any, path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    return p


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------


def file_hash(path: Union[str, Path], *, algo: str = "sha256", chunk: int = 1 << 16) -> str:
    """Hash a file for integrity tracking (default sha256)."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Inference helpers (EDA convenience)
# ---------------------------------------------------------------------------


def infer_id_columns(
    df: pd.DataFrame, *, uniqueness_ratio: float = 0.9, max_cols: int = 8
) -> List[str]:
    """Heuristic: columns with high uniqueness look like IDs; suggest dropping for EDA."""
    out: List[str] = []
    n = max(len(df), 1)
    for c in df.columns:
        ratio = df[c].nunique(dropna=True) / n
        if ratio >= uniqueness_ratio:
            out.append(c)
            if len(out) >= max_cols:
                break
    return out


def infer_datetime_columns(
    df: pd.DataFrame, *, min_parse_ratio: float = 0.7, max_cols: int = 8
) -> List[str]:
    """Heuristic: columns that parse cleanly to datetime are likely timestamps."""
    out: List[str] = []
    for c in df.columns:
        try:
            s = pd.to_datetime(df[c], errors="coerce")
            ratio = 1.0 - float(s.isna().mean())
            if ratio >= min_parse_ratio:
                out.append(c)
                if len(out) >= max_cols:
                    break
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Locate latest split files (handles timestamped names)
# ---------------------------------------------------------------------------


def _latest_by_glob(dir_: Path, pattern: str) -> Path:
    matches = [p for p in dir_.glob(pattern) if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"No files match {pattern} in {dir_}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def dataset_split_paths(name: str, processed_dir: Union[str, Path]) -> tuple[Path, Path]:
    """Return latest `{name}_train*.csv` and `{name}_test*.csv` from `processed_dir`.
    Why: splits are timestamped; callers should still get the newest pair easily.
    """
    d = Path(processed_dir)
    train = _latest_by_glob(d, f"{name}_train*.csv")
    test = _latest_by_glob(d, f"{name}_test*.csv")
    return train, test


# ---------------------------------------------------------------------------
# Timestamped stage saves (preferred API)
# ---------------------------------------------------------------------------


def _timestamp_for_name() -> str:
    return _dt.now().strftime("%Y%m%d_%H%M%S")


def _as_df(df_or_path: Union[pd.DataFrame, str, Path], **read_kwargs: Any) -> pd.DataFrame:
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path
    return read_csv(Path(df_or_path), **read_kwargs)


def to_stage(
    df_or_path: Union[pd.DataFrame, str, Path],
    name: str,
    stage_dir: Union[str, Path],
    *,
    index: bool = False,
    read_kwargs: Optional[dict] = None,
) -> Path:
    """Save to a stage with a `_YYYYMMDD_HHMMSS` suffix (uniform convention)."""
    read_kwargs = read_kwargs or {}
    df = _as_df(df_or_path, **read_kwargs)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    out = stage_dir / f"{name}_{_timestamp_for_name()}.csv"
    return write_csv(df, out, index=index)


def to_interm(
    df_or_path: Union[pd.DataFrame, str, Path],
    name: str,
    interim_dir: Union[str, Path],
    *,
    index: bool = False,
    read_kwargs: Optional[dict] = None,
) -> Path:
    """Save into `data/interim` as `<name>_YYYYMMDD_HHMMSS.csv`."""
    return to_stage(df_or_path, name, interim_dir, index=index, read_kwargs=read_kwargs)


def to_process(
    df_or_path: Union[pd.DataFrame, str, Path],
    name: str,
    processed_dir: Union[str, Path],
    *,
    index: bool = False,
    read_kwargs: Optional[dict] = None,
) -> Path:
    """Save into `data/processed` as `<name>_YYYYMMDD_HHMMSS.csv`."""
    return to_stage(df_or_path, name, processed_dir, index=index, read_kwargs=read_kwargs)
