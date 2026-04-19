"""NPZ-backed structured-output toy grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MechanismToyGrid:
    X: np.ndarray
    axis: np.ndarray
    _Y: np.ndarray
    _Y_sig: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path) -> "MechanismToyGrid":
        z = np.load(Path(path).expanduser().resolve())
        return cls(
            X=np.asarray(z["X"], dtype=np.float64),
            axis=np.asarray(z["axis"], dtype=np.float64),
            _Y=np.asarray(z["Y"], dtype=np.float32),
            _Y_sig=(None if "Y_sig" not in z.files else np.asarray(z["Y_sig"], dtype=np.float32)),
        )

    @property
    def n_models(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_pix(self) -> int:
        return int(self.axis.shape[0])

    @property
    def Y_sig(self) -> np.ndarray | None:
        return self._Y_sig

    def _subset(self, idx: np.ndarray) -> "MechanismToyGrid":
        idx = np.asarray(idx, dtype=int)
        return MechanismToyGrid(self.X[idx], self.axis, self._Y[idx], None if self._Y_sig is None else self._Y_sig[idx])

    def load_matrix(self, *, dtype: np.dtype = np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.X.astype(np.float64, copy=False), self.axis.copy(), self._Y.astype(dtype, copy=False)

