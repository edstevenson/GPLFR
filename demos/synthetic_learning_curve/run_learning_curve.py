from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gplfr import GPLFR


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"


def main() -> None:
    args = parse_args()
    cfg = json.loads((HERE / args.config).read_text())
    tasks = build_tasks(cfg)
    if args.task_id is not None:
        run_task(cfg, tasks[int(args.task_id)], force=args.force)
        return
    for task in tasks:
        run_task(cfg, task, force=args.force)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--task-id", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.task_id is None and not args.all:
        p.error("pass --task-id N or --all")
    return args


def build_tasks(cfg: dict[str, Any]) -> list[dict[str, int]]:
    return [{"task_id": i, "n_train": int(n), "latent_dim": int(cfg["latent_dim"]), "seed": int(seed), "num_steps": int(cfg["num_steps"])} for i, (n, seed) in enumerate((n, seed) for n in cfg["n_train"] for seed in cfg["seeds"])]


def run_task(cfg: dict[str, Any], task: dict[str, int], *, force: bool = False) -> None:
    run_dir = OUT_DIR / f"n_train_{task['n_train']}" / f"seed_{task['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        print(f"[skip] {result_path}")
        return
    print(f"[run] {task} start={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    torch.set_num_threads(int(os.environ.get("GPLFR_NUM_THREADS", "1")))
    data = prepare_data(cfg, task["n_train"], task["seed"])
    model = GPLFR(
        latent_dim=task["latent_dim"],
        kernel=cfg["kernel"],
        ell_mode=cfg["lengthscale_grouping"],
        sigma_f_mode="fixed" if cfg["amplitude_grouping"] == "fixed" else cfg["amplitude_grouping"],
        sigma_f_fixed=float(cfg["amplitude"]) if cfg["amplitude_grouping"] == "fixed" else None,
        data_fit_scale=float(cfg["beta"]),
        sigma_xi2=float(cfg["latent_noise"]),
        jitter=float(cfg["jitter"]),
        dtype=getattr(torch, cfg["dtype"]),
        device=pick_device(cfg["device"]),
    )
    t0 = time.perf_counter()
    fit = fit_with_checkpoints(model, data, cfg, task, run_dir)
    fit.update(
        status="completed",
        fit_seconds=time.perf_counter() - t0,
        device=str(model.device),
        config={k: task[k] for k in ("task_id", "n_train", "latent_dim", "seed", "num_steps")} | {"beta": float(cfg["beta"]), "latent_noise": float(cfg["latent_noise"])},
    )
    result_path.write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n")
    print(f"[done] task_id={task['task_id']} best_step={fit['best_step']} rmse_sig={fit['test_metrics']['rmse_sig']:.6g}", flush=True)


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def prepare_data(cfg: dict[str, Any], n_train: int, seed: int) -> dict[str, np.ndarray]:
    data = np.load(HERE / cfg["data_path"])
    split = np.load(HERE / cfg["split_path"])
    pool_idx = np.asarray(split["train_idx"], dtype=int)
    train_idx = np.sort(pool_idx[np.random.default_rng(seed).permutation(pool_idx.shape[0])[:n_train]])
    val_idx, test_idx = np.asarray(split["val_idx"], dtype=int), np.asarray(split["test_idx"], dtype=int)
    x_mean, x_std, y_mean, y_std = fit_preprocessors(data["X"][train_idx], data["Y"][train_idx])
    return {
        "X_train": standardize_x(data["X"][train_idx], x_mean, x_std),
        "Y_train": standardize_y(data["Y"][train_idx], y_mean, y_std),
        "X_val": standardize_x(data["X"][val_idx], x_mean, x_std),
        "Y_val": standardize_y(data["Y"][val_idx], y_mean, y_std),
        "Y_val_sig": standardize_y(data["Y_sig"][val_idx], y_mean, y_std),
        "X_test": standardize_x(data["X"][test_idx], x_mean, x_std),
        "Y_test": np.asarray(data["Y"][test_idx], dtype=np.float64),
        "Y_test_sig": np.asarray(data["Y_sig"][test_idx], dtype=np.float64),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def fit_with_checkpoints(model: GPLFR, data: dict[str, np.ndarray], cfg: dict[str, Any], task: dict[str, int], run_dir: Path) -> dict[str, Any]:
    fit = model.fit(
        data["X_train"],
        data["Y_train"],
        num_steps=int(task["num_steps"]),
        lr_Z=float(cfg["lr_Z"]),
        lr_global=float(cfg["lr_global"]),
        log_every=int(cfg["log_every"]),
        record_loss_curve=True,
        val_X=data["X_val"],
        val_Y=data["Y_val"],
        val_Y_sig=data["Y_val_sig"],
        val_y_std=data["y_std"],
        early_stop_metric=cfg["selection_metric"],
        early_stop_patience_evals=None,
        seed=int(task["seed"]),
        verbose=True,
    )
    rows = fit.loss_curve or []
    best_row = min(rows, key=lambda row: row[cfg["selection_metric"]])
    write_curve(run_dir / "training_curve.csv", rows)
    y_test = invert_y(model.predict(data["X_test"]), data["y_mean"], data["y_std"])
    np.savez_compressed(run_dir / "test_predictions.npz", y_pred=y_test, y_true=data["Y_test"], y_sig=data["Y_test_sig"])
    return {"best_step": int(best_row["step"]), "best_validation": best_row, "final_validation": rows[-1], "selection_metric": cfg["selection_metric"], "test_metrics": evaluate_metrics(data["Y_test"], y_test, data["Y_test_sig"])}


def current_samples(model: GPLFR, guide: Any) -> dict[str, torch.Tensor]:
    params = guide(model.Y_train_, model.X_train_)
    samples = {k: v.detach().unsqueeze(0) for k, v in params.items() if isinstance(v, torch.Tensor)}
    samples["amplitude"] = model.X_train_.new_tensor(float(model.amplitude)).view(1)
    return samples


def set_samples(model: GPLFR, samples: dict[str, torch.Tensor]) -> None:
    model.posterior_samples_ = {k: v.to(model.device) for k, v in samples.items()}
    model._build_cached_state()


def fit_preprocessors(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return x.mean(0), x.std(0), y.mean(0), y.std(0)


def standardize_x(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mean[None, :]) / std[None, :]


def standardize_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(y, dtype=np.float64) - mean[None, :]) / std[None, :]


def invert_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) * std[None, :] + mean[None, :]


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_sig_true: np.ndarray) -> dict[str, float]:
    return {"rmse_obs": rmse(y_true, y_pred), "rmse_sig": rmse(y_sig_true, y_pred)}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def write_curve(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
