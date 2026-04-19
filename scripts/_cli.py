from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

HELP_FLAGS = {"-h", "--help"}


def wants_help(argv: list[str]) -> bool:
    return any(arg in HELP_FLAGS for arg in argv)


def run(command: str, argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if wants_help(argv):
        print(
            f"Usage: python gplfr/scripts/{command}.py [config=path] [key=value ...]\n"
            f"Thin wrapper for python -m gplfr.applications.exoclimate.{command}."
        )
        return
    module = import_module(f"gplfr.applications.exoclimate.{command}")
    module.main(argv)
