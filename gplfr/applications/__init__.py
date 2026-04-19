"""Bridge package for source-tree GPLFR applications."""

from __future__ import annotations

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[2] / "applications"),
]
