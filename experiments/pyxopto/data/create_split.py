"""Create a train/val/test split NPZ from refl.npz, stratified by HG g.

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

    data = np.load(data_path)
    X = np.asarray(data["X"], dtype=np.float64)
    n = int(X.shape[0])
    g = X[:, 0]
    g_values = np.unique(g)
    S = int(g_values.shape[0])

    if n_train < 0 or n_val < 0 or n_test < 0:
        raise ValueError("n_train, n_val, and n_test must be >= 0")
    if n_train + n_val + n_test > n:
        raise ValueError(f"Requested n_train+n_val+n_test={n_train+n_val+n_test} > n_available={n}")
    if any(v % S != 0 for v in (n_train, n_val, n_test)):
        raise ValueError(f"n_train/n_val/n_test must be divisible by n_tasks={S} (stratified by g)")

    n_train_g, n_val_g, n_test_g = n_train // S, n_val // S, n_test // S
    rng = np.random.default_rng(seed)

    train_idx_list: list[np.ndarray] = []
    val_idx_list: list[np.ndarray] = []
    test_idx_list: list[np.ndarray] = []
    for gv in g_values:
        idx = np.where(g == gv)[0]
        if n_train_g + n_val_g + n_test_g > int(idx.shape[0]):
            raise ValueError(f"Not enough points for g={gv}: need {n_train_g+n_val_g+n_test_g}, have {idx.shape[0]}")
        perm = rng.permutation(idx.shape[0])
        idx = idx[perm]
        train_idx_list.append(idx[:n_train_g])
        val_idx_list.append(idx[n_train_g : n_train_g + n_val_g])
        test_idx_list.append(idx[n_train_g + n_val_g : n_train_g + n_val_g + n_test_g])

    train_idx = np.sort(np.concatenate(train_idx_list))
    val_idx = np.sort(np.concatenate(val_idx_list))
    test_idx = np.sort(np.concatenate(test_idx_list))

    np.savez(out_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"Saved {out_path} (n={n}, tasks={S}, n_train={n_train}, n_val={n_val}, n_test={n_test})")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Create stratified split indices NPZ from refl.npz")
    p.add_argument("data_path", help="Input refl.npz (X,Y)")
    p.add_argument("out_path", help="Output split NPZ path (e.g. split_idx.npz)")
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

