# ============================
# File: src/dsproject/pipelines/__init__.py
# ============================
"""Pipelines package: CLI entry points and batch utilities.


- `cli_main` exposes the Typer app entrypoint (see `cli.py`).
"""
from __future__ import annotations

from .cli import main as cli_main  # re-export for convenience

__all__ = ["cli_main"]
