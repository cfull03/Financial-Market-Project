from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd
import yaml

from dsproject.data.make_sample import sample_latest

TS_RE = re.compile(r"_\d{8}_\d{6}\.csv$")


def _write_config(tmp: Path) -> Path:
    cfg = {
        "project_name": "toy",
        "random_state": 0,
        "input": {},
        "split": {"test_size": 0.2},
        "paths": {
            "raw_dir": str(tmp / "raw"),
            "interim_dir": str(tmp / "interim"),
            "processed_dir": str(tmp / "processed"),
            "figures_dir": str(tmp / "reports" / "figures"),
            "metrics_dir": str(tmp / "reports" / "metrics"),
        },
    }
    for d in (
        tmp / "raw",
        tmp / "interim",
        tmp / "processed",
        tmp / "reports" / "figures",
        tmp / "reports" / "metrics",
        tmp / "data" / "sample",
    ):
        d.mkdir(parents=True, exist_ok=True)
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_sample_latest_picks_newest_and_writes_sample(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)

    processed = tmp_path / "processed"
    sample_dir = tmp_path / "data" / "sample"

    df1 = pd.DataFrame({"a": range(50)})
    df2 = pd.DataFrame({"a": range(100)})

    f1 = processed / "toy_train_20240101_000000.csv"
    f2 = processed / "toy_train_20241231_235959.csv"
    df1.to_csv(f1, index=False)
    df2.to_csv(f2, index=False)

    # ensure mtime reflects order (newest = f2)
    os.utime(f1, (time.time() - 1000, time.time() - 1000))
    os.utime(f2, None)

    out = sample_latest(n=10, stage="processed", name="toy", config=cfg_path)
    assert out.exists()
    assert out.parent == sample_dir
    assert out.name.startswith("toy_sample_") and TS_RE.search(out.name)

    df_sample = pd.read_csv(out)
    assert len(df_sample) == 10
