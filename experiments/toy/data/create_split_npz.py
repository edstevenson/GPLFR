"""Create a train/val/test split NPZ from a mechanism-toy data.npz file.

Output keys:
  - train_idx: (n_train,)
  - val_idx: (n_val,)
  - test_idx: (n_test,)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def create_split_npz(
    data_path: str | Path,
    out_path: str | Path,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
) -> None:
    data_path = Path(data_path)
    out_path = Path(out_path)
    if not data_path.is_absolute():
        data_path = (Path(__file__).parent / data_path).resolve()
    if not out_path.is_absolute():
        out_path = (Path(__file__).parent / out_path).resolve()

    z = np.load(data_path)
    n = int(z["X"].shape[0])
    if n_train + n_val + n_test > n:
        raise ValueError(f"Requested n_train+n_val+n_test={n_train+n_val+n_test} > n_available={n}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train : n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val : n_train + n_val + n_test])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"Saved {out_path} (n={n}, n_train={n_train}, n_val={n_val}, n_test={n_test})")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Create split indices NPZ from mechanism-toy data.npz")
    p.add_argument("data_path", help="Input data.npz (relative to this folder if not absolute)")
    p.add_argument("out_path", help="Output split NPZ path (relative to this folder if not absolute)")
    p.add_argument("--n-train", type=int, required=True)
    p.add_argument("--n-val", type=int, default=0)
    p.add_argument("--n-test", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    create_split_npz(
        args.data_path,
        args.out_path,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
    )

