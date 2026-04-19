"""Simplex (Delaunay) piecewise-linear interpolation baseline for PyXOpto.

Interpolates in log10(Y + eps) space for better behavior on log-normal outputs,
then exponentiates back to raw space.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, KDTree


class SimplexInterpolator:
    def __init__(self, X: np.ndarray, Y: np.ndarray, *, eps: float = 1e-8) -> None:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self._eps = eps

        # Standardize X for well-conditioned triangulation
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0).clip(min=1e-12)
        self._Xs = (X - self._x_mean) / self._x_std

        self._tri = Delaunay(self._Xs)
        self._kdtree = KDTree(self._Xs)
        self._Y_log = np.log10(Y + eps)

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        X_query = np.asarray(X_query, dtype=np.float64)
        Xq_s = (X_query - self._x_mean) / self._x_std

        simplices = self._tri.find_simplex(Xq_s)
        out = np.empty((X_query.shape[0], self._Y_log.shape[1]), dtype=np.float64)

        for i in range(X_query.shape[0]):
            si = int(simplices[i])
            if si < 0:
                # Outside hull: nearest-neighbor fallback
                _, idx = self._kdtree.query(Xq_s[i], k=1)
                out[i] = self._Y_log[int(idx)]
            else:
                # Barycentric interpolation in log space
                T = self._tri.transform[si]
                d = self._tri.ndim
                b = T[:d] @ (Xq_s[i] - T[d])
                w = np.append(b, 1.0 - b.sum())
                verts = self._tri.simplices[si]
                out[i] = w @ self._Y_log[verts]

        return 10.0**out - self._eps
