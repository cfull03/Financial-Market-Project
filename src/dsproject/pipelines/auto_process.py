# ============================
# File: src/dsproject/pipelines/auto_process.py
# ============================
"""Auto-process pipeline: consume CSVs from **interim/**, clean, write to **processed/**,
then archive the source so it's handled **once**.

Usage
-----
Python module:
    python -m dsproject.pipelines.auto_process --config configs/default.yaml

CLI via Make (example target):
    auto:
        python -m dsproject.pipelines.auto_process --config $(CONFIG)

Behavior
--------
- Scans `cfg.paths.interim_dir` for `*.csv`.
- Skips files that are already archived under `interim/archived/`.
- For each file:
    * Clean with `simple_clean_df` using config (`id_columns`, `datetime_columns`).
    * Save a timestamped CSV to `processed/` via `to_process` (suffix `_clean`).
    * Move the original CSV into `interim/archived/<name>_archived_YYYYMMDD_HHMMSS.csv`.
    * Record an entry in `reports/metrics/auto_process_manifest.json` (idempotency log).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dsproject.config import Config
from dsproject.features.build_features import simple_clean_df
from dsproject.utils.io import ensure_dirs, to_process

logger = logging.getLogger(__name__)


# ------------------------------ utilities -----------------------------------


@dataclass(slots=True)
class ProcessResult:
    src: Path
    out: Path
    archived: Path


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _manifest_path(cfg: Config) -> Path:
    return cfg.paths.metrics_dir / "auto_process_manifest.json"


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"items": []}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {"items": []}


def _save_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ------------------------------- core logic ---------------------------------


def _clean_and_write(cfg: Config, src_csv: Path) -> Path:
    """Return path to processed output (timestamped)."""
    df_clean = simple_clean_df(
        src_csv,
        id_columns=list(cfg.input.get("id_columns", [])),
        datetime_columns=list(cfg.input.get("datetime_columns", [])),
        add_timeparts=False,
        keep_original_datetime=True,
        drop_constants=False,
    )
    base = src_csv.stem
    out = to_process(df_clean, f"{base}_clean", cfg.paths.processed_dir)
    return out


def _archive_src(cfg: Config, src_csv: Path) -> Path:
    arch_dir = cfg.paths.interim_dir / "archived"
    arch_dir.mkdir(parents=True, exist_ok=True)
    archived = arch_dir / f"{src_csv.stem}_archived_{_now_stamp()}{src_csv.suffix}"
    src_csv.replace(archived)  # atomic move within same filesystem
    return archived


def _already_archived(cfg: Config, path: Path) -> bool:
    return path.parent.name == "archived"


def _should_skip(manifest: dict, src_csv: Path) -> bool:
    # Skip if an entry exists with same absolute path (before archive)
    src_str = str(src_csv.resolve())
    for item in manifest.get("items", []):
        if item.get("src_resolved") == src_str:
            return True
    return False


def process_inbox(cfg: Config) -> List[ProcessResult]:
    """Process every CSV in `interim/` once.

    Returns a list of results for files that were processed in this run.
    """
    interim = cfg.paths.interim_dir
    ensure_dirs(interim, cfg.paths.processed_dir, cfg.paths.metrics_dir)

    manifest_path = _manifest_path(cfg)
    manifest = _load_manifest(manifest_path)

    processed_now: List[ProcessResult] = []

    for src_csv in sorted(interim.glob("*.csv")):
        if _already_archived(cfg, src_csv):
            continue
        if _should_skip(manifest, src_csv):
            logger.info("Skipping already processed: %s", src_csv.name)
            continue

        logger.info("Processing: %s", src_csv.name)
        out = _clean_and_write(cfg, src_csv)
        archived = _archive_src(cfg, src_csv)

        manifest.setdefault("items", []).append(
            {
                "src": str(src_csv),
                "src_resolved": str(src_csv.resolve()),
                "output": str(out),
                "archived": str(archived),
                "processed_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        processed_now.append(ProcessResult(src=src_csv, out=out, archived=archived))

    _save_manifest(manifest_path, manifest)
    return processed_now


# ------------------------------- entrypoint ---------------------------------


def main(config: Optional[Path] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Auto-process interim CSVs → processed.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config YAML.",
    )
    args = parser.parse_args([] if config is not None else None)
    cfg_path = config or args.config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = Config.load(cfg_path)

    results = process_inbox(cfg)
    if not results:
        logger.info("No new CSVs to process.")
    else:
        for r in results:
            logger.info("OK: %s → %s (archived: %s)", r.src.name, r.out.name, r.archived.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
