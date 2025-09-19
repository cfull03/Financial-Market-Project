from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from dsproject.utils.io import to_interm, to_process

TS_RE = re.compile(r"_\d{8}_\d{6}\.csv$")


def test_to_interm_and_to_process_append_timestamp(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    interim_dir.mkdir()
    processed_dir.mkdir()

    p1 = to_interm(df, "toy", interim_dir)
    p2 = to_process(df, "toy", processed_dir)

    assert p1.exists() and p2.exists()
    assert p1.parent == interim_dir and p2.parent == processed_dir

    assert TS_RE.search(p1.name), p1.name
    assert TS_RE.search(p2.name), p2.name
