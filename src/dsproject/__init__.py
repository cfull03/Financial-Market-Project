"""
Data Science Project (EDA-first)

Lightweight package initializer. Keeps imports minimal to avoid side effects during installation
and fast interpreter startup. Exposes only version metadata.
"""

from __future__ import annotations

from importlib import metadata


def _safe_version(dist_name: str) -> str:
    """Return installed distribution version or a fallback.
    Why: allows running the package from a source tree without an installed dist.
    """
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "0.0.0"


__version__: str = _safe_version("dsproject")

__all__ = ["__version__"]
