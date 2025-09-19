# ============================
# File: src/dsproject/pipelines/cli.py
# ============================
"""EDA-first CLI for this project (no ML/DL).

Commands
--------
- ingest       : copy a source CSV into `data/raw/` (stable filename)
- split        : write train/test into `data/processed/` (timestamped)
- clean        : minimal cleaning to *_train/_test (timestamped *_clean)
- validate     : validate a CSV against `configs/schema.yaml`
- eda-report   : quick plots (hists/bars/corr) to `reports/figures/`
- make-sample  : sample N rows from the latest dataset (by stage)
- auto-process : consume interim/*.csv → processed (once), archive sources

Notes
-----
- Uses `configs/default.yaml` unless `--config` is passed.
- Processed & interim saves are timestamped by helpers in `utils.io`.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import typer

from dsproject.config import Config
from dsproject.data.make_dataset import ingest_csv, split_csv
from dsproject.data.make_sample import sample_latest
from dsproject.features.build_features import simple_clean
from dsproject.pipelines.auto_process import process_inbox
from dsproject.utils.io import dataset_split_paths, ensure_dirs

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
logger = logging.getLogger(__name__)


# ------------------------------ shared utils -------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_cfg(path: Path) -> Config:
    return Config.load(path)


def _default_name(cfg: Config) -> str:
    """Use project_name from config (slugified) as default dataset name."""
    base = cfg.project_name or "dataset"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()
    return slug or "dataset"


# ------------------------------ typer root ---------------------------------


@app.callback()
def app_init(
    ctx: typer.Context,
    config: Path = typer.Option(Path("configs/default.yaml"), help="Path to config YAML."),
):
    """Load config once and stash it in context."""
    _setup_logging()
    cfg = _load_cfg(config)
    ctx.obj = {"cfg": cfg, "config_path": config}


# -------------------------------- ingest -----------------------------------


@app.command()
def ingest(
    ctx: typer.Context,
    csv: Path = typer.Option(..., exists=True, help="Source CSV path."),
    name: Optional[str] = typer.Option(None, help="Base name (defaults to config.project_name)."),
):
    cfg: Config = ctx.obj["cfg"]
    ensure_dirs(cfg.paths.raw_dir)
    ds_name = name or _default_name(cfg)
    out = ingest_csv(csv, cfg.paths.raw_dir, ds_name)
    typer.echo(str(out))


# -------------------------------- split ------------------------------------


@app.command()
def split(
    ctx: typer.Context,
    name: Optional[str] = typer.Option(
        None, help="Dataset base name (defaults to config.project_name)."
    ),
):
    cfg: Config = ctx.obj["cfg"]
    ds_name = name or _default_name(cfg)
    raw_csv = cfg.paths.raw_dir / f"{ds_name}.csv"
    train_path, test_path = split_csv(
        raw_csv,
        cfg.paths.processed_dir,
        target_col=cfg.input.get("target"),
        test_size=float(cfg.split.get("test_size", 0.2)),
        random_state=int(cfg.random_state),
    )
    typer.echo(json.dumps({"train": str(train_path), "test": str(test_path)}))


# -------------------------------- clean ------------------------------------


@app.command()
def clean(
    ctx: typer.Context,
    name: Optional[str] = typer.Option(
        None, help="Dataset base name (defaults to config.project_name)."
    ),
    add_timeparts: bool = typer.Option(False, help="Expand Date_* parts (year, month, ...)."),
    drop_constants: bool = typer.Option(False, help="Drop constant columns."),
):
    """Clean the latest train/test splits and save timestamped *_clean CSVs."""
    cfg: Config = ctx.obj["cfg"]

    ds_name = name or _default_name(cfg)
    train_csv, test_csv = dataset_split_paths(ds_name, cfg.paths.processed_dir)

    out_train = simple_clean(
        train_csv,
        cfg.paths.processed_dir / f"{ds_name}_train_clean.csv",
        id_columns=list(cfg.input.get("id_columns", [])),
        datetime_columns=list(cfg.input.get("datetime_columns", [])),
        add_timeparts=add_timeparts,
        drop_constants=drop_constants,
        append_timestamp=True,
    )
    out_test = simple_clean(
        test_csv,
        cfg.paths.processed_dir / f"{ds_name}_test_clean.csv",
        id_columns=list(cfg.input.get("id_columns", [])),
        datetime_columns=list(cfg.input.get("datetime_columns", [])),
        add_timeparts=add_timeparts,
        drop_constants=drop_constants,
        append_timestamp=True,
    )
    typer.echo(json.dumps({"train_clean": str(out_train), "test_clean": str(out_test)}))


# ------------------------------- validate ----------------------------------


def _compile_feature_rules(schema: dict):
    rules = []
    for r in schema.get("feature_rules", []) or []:
        try:
            rx = re.compile(r.get("pattern", ".*"))
        except re.error as e:
            logger.warning("Invalid feature_rules regex %r: %s", r.get("pattern"), e)
            continue
        rules.append(
            {
                "regex": rx,
                "type": r.get("type"),
                "max_missing": float(r.get("max_missing", 1.0)),
                "drop_if_constant": bool(r.get("drop_if_constant", False)),
            }
        )
    return rules


def _load_schema(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@app.command()
def validate(
    ctx: typer.Context,
    csv: Optional[Path] = typer.Option(
        None, help="CSV to validate; defaults to latest processed train."
    ),
    name: Optional[str] = typer.Option(
        None, help="Base name to locate split if --csv omitted (defaults to config.project_name)."
    ),
    schema: Optional[Path] = typer.Option(
        None, help="Schema YAML path (defaults next to the loaded config)."
    ),
):
    import pandas as pd

    cfg: Config = ctx.obj["cfg"]
    ds_name = name or _default_name(cfg)
    df_path = csv or dataset_split_paths(ds_name, cfg.paths.processed_dir)[0]
    schema_path = schema or (ctx.obj["config_path"].parent / "schema.yaml")
    df = pd.read_csv(df_path)
    sch = _load_schema(schema_path)

    errors: list[str] = []

    # presence & types
    required = sch.get("columns", {}).keys()
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    type_map = {
        "int": pd.api.types.is_integer_dtype,
        "float": pd.api.types.is_float_dtype,
        "number": pd.api.types.is_numeric_dtype,
        "string": pd.api.types.is_string_dtype,
        "bool": pd.api.types.is_bool_dtype,
        "datetime": pd.api.types.is_datetime64_any_dtype,
    }

    for col, spec in sch.get("columns", {}).items():
        if col not in df.columns:
            continue
        expected = spec.get("type")
        if expected:
            ok = type_map.get(expected, lambda s: True)(df[col])
            if not ok:
                errors.append(f"Type mismatch for {col}: expected {expected}")
        if spec.get("allowed") is not None:
            bad = set(df[col].dropna().unique()) - set(spec["allowed"])
            if bad:
                errors.append(f"Unexpected values in {col}: {sorted(bad)[:10]} ...")
        if spec.get("max_missing") is not None:
            miss = float(df[col].isna().mean())
            if miss > float(spec["max_missing"]):
                errors.append(f"Missing ratio for {col}={miss:.3f} exceeds {spec['max_missing']}")

    # extras & engineered
    extras = [c for c in df.columns if c not in sch.get("columns", {})]
    rules = _compile_feature_rules(sch)
    if extras:
        if sch.get("no_extra_columns", False) and not rules:
            errors.append(f"Unexpected extra columns (no feature_rules): {extras}")
        else:
            for col in extras:
                matched = None
                for r in rules:
                    if r["regex"].search(col):
                        matched = r
                        break
                if not matched:
                    errors.append(f"Extra column not allowed by feature_rules: {col}")
                    continue
                miss = float(df[col].isna().mean())
                if miss > float(matched.get("max_missing", 1.0)):
                    errors.append(
                        f"Feature {col}: missing {miss:.3f} exceeds {matched.get('max_missing')}"
                    )

    # report
    cfg.paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.metrics_dir / f"{Path(df_path).stem}_schema_validation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ok": len(errors) == 0, "errors": errors}, f, indent=2)

    typer.echo(json.dumps({"ok": len(errors) == 0, "report": str(out)}))
    if errors:
        raise typer.Exit(code=1)


# ------------------------------- eda-report --------------------------------


@app.command("eda-report")
def eda_report(
    ctx: typer.Context,
    name: Optional[str] = typer.Option(
        None, help="Dataset base name (defaults to config.project_name)."
    ),
    clean: bool = typer.Option(True, help="Use *_train_clean*.csv if available."),
    top_k: int = typer.Option(12, help="Top categories to plot for categoricals."),
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    cfg: Config = ctx.obj["cfg"]
    ds_name = name or _default_name(cfg)
    figures = cfg.paths.figures_dir
    figures.mkdir(parents=True, exist_ok=True)

    base_train, _ = dataset_split_paths(ds_name, cfg.paths.processed_dir)
    if clean and "_train_clean" not in base_train.stem:
        try:
            candidates = list(Path(cfg.paths.processed_dir).glob(f"{ds_name}_train_clean*.csv"))
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                base_train = candidates[0]
        except Exception:
            pass

    df = pd.read_csv(base_train)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure()
            df[col].hist(bins=30)
            plt.title(f"{col} distribution")
            plt.xlabel(col)
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(figures / f"{ds_name}_{col}_hist.png")
            plt.close()

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            vc = df[col].astype(str).value_counts().head(top_k)
            if len(vc) == 0:
                continue
            plt.figure()
            vc.plot(kind="bar")
            plt.title(f"{col} top-{len(vc)}")
            plt.xlabel(col)
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(figures / f"{ds_name}_{col}_bar.png")
            plt.close()

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)
        plt.figure()
        plt.imshow(corr, interpolation="nearest")
        plt.title("Correlation (numeric)")
        plt.xticks(range(corr.shape[1]), corr.columns, rotation=90)
        plt.yticks(range(corr.shape[1]), corr.columns)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(figures / f"{ds_name}_corr_heatmap.png")
        plt.close()

    typer.echo(json.dumps({"figures_dir": str(figures)}))


# ------------------------------ make-sample --------------------------------


@app.command("make-sample")
def make_sample_cmd(
    ctx: typer.Context,
    n: int = typer.Option(..., help="Number of rows to sample."),
    stage: str = typer.Option("processed", help="Stage to sample from: raw|interim|processed"),
    name: Optional[str] = typer.Option(
        None, help="Optional base name to prefer (defaults to config.project_name)."
    ),
):
    cfg_path: Path = ctx.obj["config_path"]
    cfg: Config = ctx.obj["cfg"]
    ds_name = name or _default_name(cfg)
    out = sample_latest(n=n, stage=stage, name=ds_name, config=cfg_path)
    typer.echo(str(out))


# ------------------------------ auto-process --------------------------------


@app.command("auto-process")
def auto_process_cmd(ctx: typer.Context):
    """Process each CSV in `interim/` once: clean → processed (timestamped) → archive source."""
    cfg: Config = ctx.obj["cfg"]
    results = process_inbox(cfg)
    payload = [{"src": str(r.src), "out": str(r.out), "archived": str(r.archived)} for r in results]
    typer.echo(json.dumps({"processed": payload}))


# --------------------------- console entry point ---------------------------


def main() -> None:  # pragma: no cover
    app()
