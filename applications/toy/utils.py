"""Small helpers for the toy benchmark runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ._paths import resolve_path as _resolve_path


def parse_config_and_overrides(argv: list[str]) -> tuple[Path, dict[str, str]]:
    cfg_path: Path | None = None
    overrides: dict[str, str] = {}
    for arg in argv:
        if "=" not in arg and cfg_path is None and arg.lower().endswith((".yaml", ".yml")):
            cfg_path = Path(arg).expanduser()
            continue
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        if k == "config":
            cfg_path = Path(v).expanduser()
        else:
            overrides[k] = v
    if cfg_path is None:
        cfg_path = Path(__file__).resolve().with_name("config.yaml")
    return cfg_path, overrides


def load_run_config(cfg_path: str | Path, *, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    cfg_path = Path(cfg_path).expanduser().resolve()
    cfg_dir = cfg_path.parent
    root = Path(__file__).resolve().parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected YAML mapping at top-level, got {type(cfg).__name__}")
    if overrides:
        cfg.update(overrides)

    for key in ("data_path", "split_path"):
        if key in cfg:
            cfg[key] = str(_resolve_path(cfg[key], root))
    if "out_dir" in cfg:
        cfg["out_dir"] = str(_resolve_path(cfg["out_dir"], cfg_dir))

    cfg["emulators"] = [str(e).lower() for e in cfg.get("emulators", [])]
    return cfg


def fit_x_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    std = X.std(axis=0)
    if not np.all(std > 0):
        raise ValueError(f"Non-varying input dims: {np.where(std <= 0)[0].tolist()}")
    return X.mean(axis=0), std


def apply_x_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=np.float64) - mean[None, :]) / std[None, :]


def fit_y_standardizer(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Y = np.asarray(Y, dtype=np.float64)
    std = Y.std(axis=0)
    if not np.all(std > 0):
        raise ValueError(f"Non-varying output dims: {np.where(~(std > 0))[0].tolist()}")
    return Y.mean(axis=0), std


def apply_y_standardizer(Y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(Y, dtype=np.float64) - mean[None, :]) / std[None, :]


def invert_y_standardizer(Y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        return Y * std + mean
    if Y.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {Y.shape}")
    return Y * std[None, :] + mean[None, :]


def fit_preprocessors(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean, x_std = fit_x_standardizer(X)
    y_mean, y_std = fit_y_standardizer(Y)
    return x_mean, x_std, y_mean, y_std
