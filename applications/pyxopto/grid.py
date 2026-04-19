from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class PyXOptoParams:
    g: float
    mua: float
    musr: float

    def as_array(self, *, dtype=np.float64) -> np.ndarray:
        return np.array([self.g, self.mua, self.musr], dtype=dtype)


class PyXOptoGrid:
    """PyXOpto MCML reflectance dataset loaded from NPZ."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        r_mm: np.ndarray | None = None,
        g_values: np.ndarray | None = None,
    ) -> None:
        self._X_full = np.asarray(X, dtype=np.float64)
        self._Y = np.asarray(Y, dtype=np.float32)
        if self._X_full.ndim != 2 or self._X_full.shape[1] != 3:
            raise ValueError("Expected X shape (N,3) with columns (g,mua,musr)")
        if self._Y.ndim != 2 or self._Y.shape[0] != self._X_full.shape[0]:
            raise ValueError("Expected Y shape (N,P) matching X")

        self.g = self._X_full[:, 0]
        self.X = self._X_full[:, 1:]  # (mua, musr)
        self.g_values = np.unique(self.g) if g_values is None else np.asarray(g_values, dtype=np.float64)
        if not set(np.unique(self.g)).issubset(set(self.g_values)):
            raise ValueError("Subset contains g values not present in provided g_values")
        self.g_to_idx = {float(g): i for i, g in enumerate(self.g_values)}
        self.s = np.array([self.g_to_idx[float(g)] for g in self.g], dtype=int)

        if r_mm is None:
            edges_m = np.linspace(0.0, 0.005, int(self._Y.shape[1]) + 1)
            r_mm = 1e3 * 0.5 * (edges_m[1:] + edges_m[:-1])
        self.r_mm = np.asarray(r_mm, dtype=np.float64)

        self.params = [PyXOptoParams(g=x[0], mua=x[1], musr=x[2]) for x in self._X_full]
        self._index = {(float(p.g), float(p.mua), float(p.musr)): i for i, p in enumerate(self.params)}

    @classmethod
    def load(cls, path: str | Path) -> "PyXOptoGrid":
        data = np.load(path)
        return cls(X=data["X"], Y=data["Y"])

    def _subset(self, idx: np.ndarray) -> "PyXOptoGrid":
        return PyXOptoGrid(self._X_full[idx], self._Y[idx], r_mm=self.r_mm, g_values=self.g_values)

    def curve_at(self, g: float, mua: float, musr: float) -> np.ndarray:
        idx = self._index[(float(g), float(mua), float(musr))]
        return self._Y[idx]

    def load_matrix(self, *, dtype: np.dtype = np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_cont, s, r_mm, Y)."""
        return self.X.copy(), self.s.copy(), self.r_mm.copy(), self._Y.astype(dtype, copy=False)

    def iter_curves(self, *, dtype: np.dtype = np.float32) -> Iterator[tuple[PyXOptoParams, np.ndarray]]:
        for i, p in enumerate(self.params):
            yield p, self._Y[i].astype(dtype, copy=False)

    @property
    def n_models(self) -> int:
        return int(self._X_full.shape[0])

    @property
    def n_pix(self) -> int:
        return int(self._Y.shape[1])
