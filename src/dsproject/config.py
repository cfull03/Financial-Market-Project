"""Minimal, working config loader (EDA-only).
Keeps your original structure; fixes the misplaced `@staticmethod`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PathsCfg:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    figures_dir: Path
    metrics_dir: Path


@dataclass
class Config:
    project_name: str
    random_state: int
    input: Dict[str, Any]
    split: Dict[str, Any]
    paths: PathsCfg

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        p = cfg["paths"]
        paths = PathsCfg(
            raw_dir=Path(p["raw_dir"]),
            interim_dir=Path(p["interim_dir"]),
            processed_dir=Path(p["processed_dir"]),
            figures_dir=Path(p["figures_dir"]),
            metrics_dir=Path(p["metrics_dir"]),
        )
        return cls(
            project_name=str(cfg["project_name"]),
            random_state=int(cfg["random_state"]),
            input=dict(cfg["input"]),
            split=dict(cfg["split"]),
            paths=paths,
        )


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Convenience wrapper for scripts/notebooks."""
    return Config.load(Path(path))
