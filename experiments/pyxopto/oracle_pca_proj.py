"""Oracle PCA projection baseline for PyXOpto.

This baseline answers: "What RMSE would we get if we knew the test curve's
coordinates in the rank-k PCA subspace learned from the training subset?"

Concretely:
  - Fit PCA on standardized log-curves from the training subset.
  - For test curves, compute PCA scores using the *true* standardized log-curve
    (oracle access), reconstruct, invert standardization + log transform, then
    compute metrics in the same log10 space used by the benchmark.

Outputs `runs.json` (per-seed) and `metrics.json` (aggregated) in a format that
`learning_curve.py` and `compression_curve.py` can load.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.decomposition import PCA

from . import metrics as metrics_mod
from .grid import PyXOptoGrid
from .utils import (
    apply_y_standardizer,
    fit_y_standardizer,
    invert_y_standardizer,
    inv_log10_transform,
    load_run_config,
    log10_transform,
    parse_config_and_overrides,
)


def _metric_stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float).reshape(-1)
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {
        "mean": float(a.mean()),
        "std": std,
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for r in runs:
        vid = str(r.get("variant_id", ""))
        key = (str(r["emulator"]), int(r["n_train"]), vid)
        g = grouped.setdefault(key, {"seeds": set(), "metrics": {}})
        g["seeds"].add(int(r.get("seed", r.get("data_seed", 0))))
        for m, v in (r.get("metrics") or {}).items():
            g["metrics"].setdefault(str(m), []).append(float(v))

    out: list[dict[str, Any]] = []
    for (emu, n_train, vid), g in grouped.items():
        entry: dict[str, Any] = {
            "emulator": emu,
            "n_train": n_train,
            "n_seeds": len(g["seeds"]),
            "metrics": {m: _metric_stats(vs) for m, vs in g["metrics"].items()},
        }
        if vid:
            entry["variant_id"] = vid
        out.append(entry)

    out.sort(key=lambda d: (d["emulator"], d["n_train"], d.get("variant_id", "")))
    return out


def _as_int_list(x: Any, *, name: str, min_value: int = 0) -> list[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out = [int(v) for v in x]
    else:
        out = [int(x)]
    if any(v < min_value for v in out):
        raise ValueError(f"{name} must be >= {min_value}, got {out}")
    return out


def _oracle_pca_predict(
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    *,
    eps: float,
    n_components: int,
) -> tuple[np.ndarray, dict[str, Any], float, float]:
    t_train0 = time.perf_counter()
    Y_train_log = log10_transform(Y_train, eps=eps)
    y_mean, y_std = fit_y_standardizer(Y_train_log)
    Y_train_log_z = apply_y_standardizer(Y_train_log, y_mean, y_std)
    n_components = int(min(n_components, max(int(Y_train_log_z.shape[0]) - 1, 0), int(Y_train_log_z.shape[1])))
    if n_components < 1:
        raise ValueError(f"Oracle PCA requires n_train>=2 and n_pix>=1, got {Y_train_log_z.shape}")

    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(Y_train_log_z)
    t_train = time.perf_counter() - t_train0

    t_pred0 = time.perf_counter()
    Y_test_log = log10_transform(Y_test, eps=eps)
    Y_test_log_z = apply_y_standardizer(Y_test_log, y_mean, y_std)
    scores = pca.transform(Y_test_log_z)
    Y_hat_test_log_z = pca.inverse_transform(scores)
    Y_hat_test_log = invert_y_standardizer(Y_hat_test_log_z, y_mean, y_std)
    Y_hat_test = np.maximum(inv_log10_transform(Y_hat_test_log, eps=eps), 0.0)
    t_pred = time.perf_counter() - t_pred0

    extra = {
        "n_components": int(n_components),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_.tolist()],
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    return Y_hat_test, extra, float(t_train), float(t_pred)


def main(argv: list[str] | None = None) -> None:
    argv_in = sys.argv[1:] if argv is None else argv
    cfg_path, overrides = parse_config_and_overrides(argv_in)
    cfg = load_run_config(cfg_path, overrides=overrides)

    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg["data_path"]).expanduser().resolve()
    split_path = Path(cfg["split_path"]).expanduser().resolve()
    eps = float(cfg.get("log_eps", 1e-30))

    eval_set = str(cfg.get("eval_set", "test")).lower()
    if eval_set not in ("test", "val"):
        raise ValueError(f"Unknown eval_set={eval_set!r} (expected 'test' or 'val').")

    grid = PyXOptoGrid.load(data_path)
    split = np.load(split_path)
    pool_idx = np.asarray(split["train_idx"], dtype=int)
    eval_key = "val_idx" if eval_set == "val" else "test_idx"
    if eval_key not in split.files:
        raise ValueError(f"eval_set={eval_set!r} requires split NPZ to contain {eval_key!r} (got keys={split.files}).")
    test_idx = np.asarray(split[eval_key], dtype=int)
    test_grid = grid._subset(test_idx)
    _, _, _, y_true = test_grid.load_matrix(dtype=np.float64)

    S = int(grid.g_values.shape[0])
    pool_s = grid.s[pool_idx]
    pools = [pool_idx[pool_s == t] for t in range(S)]
    pool_n_per = [int(p.shape[0]) for p in pools]
    if len(set(pool_n_per)) != 1:
        raise ValueError(f"Expected equal pool sizes per task, got {pool_n_per}")

    sweep_cfg = cfg.get("sweep") or {}
    seeds = _as_int_list(sweep_cfg.get("seeds", None), name="sweep.seeds")
    n_train_list = _as_int_list((sweep_cfg.get("grid") or {}).get("n_train", None), name="sweep.grid.n_train", min_value=1)
    n_components_list = _as_int_list((sweep_cfg.get("grid") or {}).get("n_components", None), name="sweep.grid.n_components", min_value=1)

    if not seeds or not n_train_list or not n_components_list:
        raise ValueError("oracle_pca_proj requires sweep.seeds, sweep.grid.n_train, and sweep.grid.n_components")
    if any(n % S != 0 for n in n_train_list):
        raise ValueError(f"All n_train must be divisible by n_tasks={S} for balanced subsets (got {n_train_list})")

    runs: list[dict[str, Any]] = []
    for data_seed in seeds:
        rng = np.random.default_rng(int(data_seed))
        perms = [p[rng.permutation(p.shape[0])] for p in pools]

        for n_train in n_train_list:
            n_per = int(n_train) // S
            train_idx = np.sort(np.concatenate([p[:n_per] for p in perms]))
            train_grid = grid._subset(train_idx)
            _, _, _, Y_train = train_grid.load_matrix(dtype=np.float64)

            for k in n_components_list:
                t0 = time.perf_counter()
                y_pred, extra, t_train, t_pred = _oracle_pca_predict(Y_train, y_true, eps=eps, n_components=int(k))
                met = metrics_mod.evaluate_metrics(y_true, y_pred, eps=eps)
                t_total = time.perf_counter() - t0
                runs.append(
                    {
                        "emulator": "oracle_pca_proj",
                        "seed": int(data_seed),
                        "data_seed": int(data_seed),
                        "n_train": int(n_train),
                        "variant_id": f"n_components={int(k)}",
                        "variant": {"n_components": int(k)},
                        "metrics": met,
                        "timing_s": {"train": float(t_train), "predict": float(t_pred), "total": float(t_total)},
                        "extra": extra,
                    }
                )

    variants = _summarize_runs(runs)
    summary = {
        f"{v.get('emulator')}:{v.get('variant_id','')}:n{int(v['n_train'])}": float(v["metrics"]["rmse"]["median"])
        for v in variants
        if "rmse" in (v.get("metrics") or {})
    }

    (out_dir / "cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out_dir / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({"summary": summary, "variants": variants}, indent=2), encoding="utf-8")
    print(f"[oracle_pca_proj] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
