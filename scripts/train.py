#!/usr/bin/env python
"""Thin wrapper: delegates to gplfr.applications.exoclimate.train.main."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _cli import run


def main(argv: list[str] | None = None) -> None:
    run("train", argv)


if __name__ == "__main__":
    main()
