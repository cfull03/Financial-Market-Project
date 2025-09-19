# ============================
# File: src/dsproject/data/make_sample.py
# ============================
"""Simple helpers to pick the latest CSV from a stage and sample N rows.

Two tiny entry points:
- `get_latest_csv(stage: str = "processed", name: str | None = None) -> Path`
- `sample_latest(n: int, stage: str = "processed", name: str | None = None) -> Path`

Stages: "raw", "interim", or "processed".
Outputs go to `data/sample/` and are named `<stem>_sample_n<N>.csv`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from dsproject.config import Config, load_config
from dsproject.utils.io import ensure_dirs, read_csv, write_csv

__all__ = ["get_latest_csv", "sample_latest"]


def _stage_dir(cfg: Config, stage: str) -> Path:
    stage = stage.lower()
    if stage == "raw":
        return cfg.paths.raw_dir
    if stage == "interim":
        return cfg.paths.interim_dir
    if stage == "processed":
        return cfg.paths.processed_dir
    raise ValueError("stage must be one of: 'raw', 'interim', 'processed'")


def _candidates(dir_: Path, stage: str, name: Optional[str]) -> list[Path]:
    """Prefer cleaner, more canonical files. Why: avoids accidental temp files."""
    pats: list[str]
    if stage == "processed":
        if name:
            pats = [f"{name}_train_clean.csv", f"{name}_train.csv", f"{name}.csv"]
        else:
            pats = ["*_train_clean.csv", "*_train.csv", "*.csv"]
    else:
        pats = [f"{name}.csv" if name else "*.csv"]
    out: list[Path] = []
    for p in pats:
        out.extend(dir_.glob(p))
    return [p for p in dict.fromkeys(out) if p.is_file()]  # uniq, keep order


def get_latest_csv(
    stage: str = "processed", name: Optional[str] = None, *, config: Optional[Path] = None
) -> Path:
    """Return the newest CSV by modification time from a stage directory.
    `stage` in {raw, interim, processed}. Optional `name` narrows the search.
    """
    cfg = load_config(config or Path("configs/default.yaml"))
    base = _stage_dir(cfg, stage)
    cands = _candidates(base, stage.lower(), name)
    if not cands:
        raise FileNotFoundError(f"No CSVs found in {base} (stage={stage!r}, name={name!r})")
    return max(cands, key=lambda p: p.stat().st_mtime)


def sample_latest(
    n: int, stage: str = "processed", name: Optional[str] = None, *, config: Optional[Path] = None
) -> Path:
    """Sample `n` rows from the latest CSV in the given stage and write to `data/sample/`.
    Only `n` is required for the common case to keep UX minimal.
    """
    cfg = load_config(config or Path("configs/default.yaml"))

    src = get_latest_csv(stage=stage, name=name, config=config)
    df = read_csv(src)

    k = min(int(n), len(df))
    if k <= 0:
        raise ValueError("n must be > 0")
    sampled = df.sample(n=k, random_state=cfg.random_state)

    out_dir = cfg.paths.processed_dir.parent / "sample"
    ensure_dirs(out_dir)
    out = out_dir / f"{src.stem}_sample_n{int(n)}.csv"
    write_csv(sampled, out, index=False)
    return out
