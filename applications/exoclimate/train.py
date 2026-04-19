"""Benchmark-facing exoclimate train-mean wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ._common import fit_train_mean


def train(config: str | Path | dict[str, Any] | None = None, *, argv: list[str] | None = None) -> dict[str, Any]:
    return fit_train_mean(config, argv=argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(train(argv=sys.argv[1:] if argv is None else argv), indent=2, default=str))


if __name__ == "__main__":
    main()
