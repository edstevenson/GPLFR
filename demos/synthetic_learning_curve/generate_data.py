from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gplfr import create_synthetic_data


HERE = Path(__file__).resolve().parent


def main() -> None:
    cfg = json.loads((HERE / parse_args().config).read_text())
    data_path, split_path = HERE / cfg["data_path"], HERE / cfg["split_path"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, **create_synthetic_data(**cfg["dataset"]))
    write_split(data_path, split_path, cfg["split"])
    print(f"wrote {data_path}")
    print(f"wrote {split_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    return p.parse_args()


def write_split(data_path: Path, out_path: Path, cfg: dict[str, Any]) -> None:
    n = int(np.load(data_path)["X"].shape[0])
    perm = np.random.default_rng(int(cfg["seed"])).permutation(n)
    n_pool, n_test = int(cfg["n_train_pool"]), int(cfg["n_test"])
    np.savez(out_path, train_idx=np.sort(perm[:n_pool]), test_idx=np.sort(perm[n_pool : n_pool + n_test]))


if __name__ == "__main__":
    main()
