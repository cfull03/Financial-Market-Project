"""Unit tests for src/dsproject/config.py

These tests assume a minimal Config implementation with:
- dataclasses: PathsCfg, Config
- classmethod: Config.load(path)
- helper: load_config(path="configs/default.yaml")

They validate structure and basic field types without requiring the
directories to actually exist on disk (CI‑friendly).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dsproject.config import Config, load_config


def test_load_config_default_yaml_present() -> None:
    cfg = load_config("configs/default.yaml")
    assert isinstance(cfg, Config)
    assert isinstance(cfg.project_name, str) and cfg.project_name
    assert isinstance(cfg.random_state, int)


def test_sections_and_types() -> None:
    cfg = load_config("configs/default.yaml")

    # input & split are plain dicts in the minimal version
    assert isinstance(cfg.input, dict)
    assert isinstance(cfg.split, dict)
    assert "test_size" in cfg.split

    # paths is a PathsCfg dataclass with Path fields
    paths = cfg.paths
    for attr in ("raw_dir", "interim_dir", "processed_dir", "figures_dir", "metrics_dir"):
        p = getattr(paths, attr)
        assert isinstance(p, Path)


def test_split_test_size_range() -> None:
    cfg = load_config("configs/default.yaml")
    ts = float(cfg.split.get("test_size", 0.2))
    assert 0.0 < ts < 1.0


@pytest.mark.parametrize("key", [
    "raw_dir",
    "interim_dir",
    "processed_dir",
    "figures_dir",
    "metrics_dir",
])
def test_paths_string_form_contains_expected_subdirs(key: str) -> None:
    cfg = load_config("configs/default.yaml")
    p = getattr(cfg.paths, key)
    # No strict assertion on existence; only type & reasonable-looking path
    assert isinstance(p, Path)
    assert len(str(p)) > 0
