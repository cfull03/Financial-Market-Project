"""Feature-building helpers for an EDA-first workflow (no ML components).

Primary entry point:
- `simple_clean(csv_in, csv_out, id_columns, datetime_columns, ...)`

Design goals:
- Coerce datetimes so downstream EDA is reliable.
- Drop ID-like columns to avoid skewing stats/plots.
- Optional: expand datetime parts, drop constant columns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from dsproject.utils.io import read_csv, write_csv

logger = logging.getLogger(__name__)

__all__ = [
    "simple_clean",
    "expand_datetime_parts",
]


# ------------------------------ helpers ------------------------------------


def _to_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Coerce provided columns to datetime where possible.
    Why: avoids mixed/ambiguous types that break grouping/plotting.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
    return df


def _drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with a single unique non-null value.
    Why: such columns add no information and clutter EDA.
    """
    dropped: list[str] = []
    for c in list(df.columns):
        nun = df[c].nunique(dropna=True)
        if nun <= 1:
            df = df.drop(columns=[c])
            dropped.append(c)
    return df, dropped


def expand_datetime_parts(
    df: pd.DataFrame, dt_cols: Iterable[str], *, prefix: Optional[str] = None
) -> pd.DataFrame:
    """Expand datetime columns into common parts (year, month, day, dow, quarter).
    Why: quick, interpretable temporal features for EDA and simple models later.
    """
    for col in dt_cols:
        if col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        base = f"{prefix}_{col}" if prefix else col
        s = df[col]
        df[f"{base}_year"] = s.dt.year
        df[f"{base}_month"] = s.dt.month
        df[f"{base}_day"] = s.dt.day
        df[f"{base}_dow"] = s.dt.dayofweek
        df[f"{base}_quarter"] = s.dt.quarter
    return df


# ------------------------------ public API ---------------------------------


def simple_clean(
    csv_in: Path,
    csv_out: Path,
    *,
    id_columns: List[str],
    datetime_columns: List[str],
    add_timeparts: bool = False,
    keep_original_datetime: bool = True,
    drop_constants: bool = False,
) -> Path:
    """Apply minimal, opinionated cleaning and save to `csv_out`.

    Parameters
    ----------
    csv_in : Path
        Input CSV path.
    csv_out : Path
        Output CSV path (written, directories created if needed).
    id_columns : list[str]
        Columns to drop (IDs/high-cardinality identifiers).
    datetime_columns : list[str]
        Columns to parse as datetimes.
    add_timeparts : bool
        If True, expand datetime parts (year/month/day/dow/quarter).
    keep_original_datetime : bool
        If False and `add_timeparts=True`, drop original datetime columns after expansion.
    drop_constants : bool
        If True, drop columns with a single unique value.

    Returns
    -------
    Path
        The path written to (`csv_out`).
    """
    df = read_csv(csv_in)

    # Datetimes first to ensure type-aware operations
    df = _to_datetime(df, datetime_columns)

    # Drop explicit ID columns
    drop_cols = [c for c in id_columns if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info("Dropped id columns: %s", drop_cols)

    # Optional: expand datetime parts
    if add_timeparts and datetime_columns:
        before_cols = set(df.columns)
        df = expand_datetime_parts(df, datetime_columns)
        added = sorted(set(df.columns) - before_cols)
        logger.info("Added datetime parts: %s", added[:10] + (["..."] if len(added) > 10 else []))
        if not keep_original_datetime:
            keep = [c for c in df.columns if c not in datetime_columns]
            df = df[keep]

    # Optional: drop constant columns
    if drop_constants:
        df, dropped = _drop_constant_columns(df)
        if dropped:
            logger.info("Dropped constant columns: %s", dropped)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(df, csv_out, index=False)
    logger.info("Saved cleaned CSV → %s (rows=%d, cols=%d)", csv_out, df.shape[0], df.shape[1])
    return csv_out
