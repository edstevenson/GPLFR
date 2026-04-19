"""Local matplotlib style helper for the copied toy app."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ._paths import find_project_root


def get_paper_style_path() -> str:
    style_path = find_project_root(Path(__file__).resolve().parent) / "analysis" / "paper.mplstyle"
    if not style_path.exists():
        raise FileNotFoundError(f"Style file not found at {style_path}")
    return str(style_path)


def use_paper_style() -> None:
    plt.style.use(get_paper_style_path())
