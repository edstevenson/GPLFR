"""Local path helpers for the copied PyXOpto app."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    start = start or Path(__file__).resolve().parent
    for parent in [start] + list(start.parents):
        if (parent / 'pyproject.toml').exists() or (parent / '.git').exists():
            return parent
    return start


def resolve_path(p: str | Path, base: Path | None = None) -> Path:
    p_str = str(p)
    if p_str.startswith('ROOT/'):
        return (find_project_root() / p_str[5:]).resolve()
    path = Path(p).expanduser()
    return path.resolve() if path.is_absolute() else ((base or Path.cwd()) / path).resolve()
