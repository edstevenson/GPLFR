from __future__ import annotations

import numpy as np


def median_1nn_distance(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, d = (int(X.shape[0]), int(X.shape[1]))
    if n < 2:
        raise ValueError("Need at least 2 points to compute 1-NN distances")
    out = np.empty((d,), dtype=float)
    for k in range(d):
        u, inv = np.unique(X[:, k], return_inverse=True)
        if u.size < 2:
            raise ValueError(f"Need at least 2 unique values in dim={k} to compute 1-NN distances")
        diffs = np.diff(u)
        prev = np.concatenate(([np.inf], diffs))
        nxt = np.concatenate((diffs, [np.inf]))
        d_unique = np.minimum(prev, nxt)
        d_point = d_unique[inv]
        d_point = d_point[np.isfinite(d_point)]
        if d_point.size == 0:
            raise ValueError(f"No finite neighbor distances found in dim={k}")
        out[k] = float(np.median(d_point))
    return out


def min_lengthscale_1nn(X: np.ndarray, *, factor: float = 1.0) -> np.ndarray:
    return float(factor) * median_1nn_distance(X)
