"""Metrics for comparing reflectance-curve emulators (computed in log10 space)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import log10_transform


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {a.shape}")
    return a


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return np.sqrt(np.mean((yp - yt) ** 2, axis=1))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return np.mean(np.abs(yp - yt), axis=1)


def maqe(y_true: np.ndarray, y_pred: np.ndarray, *, q: float = 0.95) -> np.ndarray:
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    err = np.abs(yp - yt)
    thr = np.quantile(err, q, axis=1, keepdims=True)
    mask = err >= thr
    return (err * mask).sum(axis=1) / mask.sum(axis=1)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-30) -> dict[str, Any]:
    yt = log10_transform(y_true, eps=eps)
    yp = log10_transform(y_pred, eps=eps)
    return {
        "rmse": float(rmse(yt, yp).mean()),
        "mae": float(mae(yt, yp).mean()),
        "maqe_0.95": float(maqe(yt, yp, q=0.95).mean()),
    }

