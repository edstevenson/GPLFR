"""Metrics for the mechanism toy benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {a.shape}")
    return a


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    out = []
    for d in range(int(yt.shape[1])):
        y = yt[:, d]
        yh = yp[:, d]
        denom = float(np.sum((y - y.mean()) ** 2))
        out.append(float("nan") if denom == 0.0 else float(1.0 - np.sum((yh - y) ** 2) / denom))
    return out


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, y_sig_true: np.ndarray | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"rmse_obs": rmse(y_true, y_pred)}
    if y_sig_true is not None:
        out["rmse_sig"] = rmse(y_sig_true, y_pred)
    return out

