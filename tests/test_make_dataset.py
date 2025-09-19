from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from dsproject.config import Config, PathsCfg
from dsproject.data.make_dataset import ingest_csv, split_csv

TS_RE = re.compile(r"_\d{8}_\d{6}\.csv$")


def _cfg(tmp: Path) -> Config:
    paths = PathsCfg(
        raw_dir=tmp / "raw",
        interim_dir=tmp / "interim",
        processed_dir=tmp / "processed",
        figures_dir=tmp / "reports" / "figures",
        metrics_dir=tmp / "reports" / "metrics",
    )
    for d in (
        paths.raw_dir,
        paths.interim_dir,
        paths.processed_dir,
        paths.figures_dir,
        paths.metrics_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    return Config(
        project_name="toyproject",
        random_state=7,
        input={},
        split={"test_size": 0.25},
        paths=paths,
    )


def test_ingest_and_split_timestamped(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    # make a tiny raw csv to ingest
    src = tmp_path / "src.csv"
    pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}).to_csv(src, index=False)

    raw_csv = ingest_csv(src, cfg.paths.raw_dir, name="dataset")
    assert raw_csv.exists()
    assert raw_csv.name == "dataset.csv"

    train, test = split_csv(
        raw_csv,
        cfg.paths.processed_dir,
        target_col=None,
        test_size=cfg.split["test_size"],
        random_state=cfg.random_state,
    )
    assert train.exists() and test.exists()
    assert train.parent == cfg.paths.processed_dir and test.parent == cfg.paths.processed_dir
    assert train.name.startswith("dataset_train_") and TS_RE.search(train.name)
    assert test.name.startswith("dataset_test_") and TS_RE.search(test.name)
