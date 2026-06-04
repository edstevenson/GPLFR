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
    for task in tasks if args.task_id is None else [tasks[int(args.task_id)]]:
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
    return [{"task_id": i, "n_train": int(n), "seed": int(seed)} for i, (n, seed) in enumerate((n, seed) for n in cfg["n_train"] for seed in cfg["seeds"])]


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
        latent_dim=int(cfg["latent_dim"]),
        kernel=cfg["kernel"],
        lengthscale_grouping=cfg["lengthscale_grouping"],
        amplitude_grouping=cfg["amplitude_grouping"],
        amplitude=float(cfg["amplitude"]),
        inverse_temperature=float(cfg["inverse_temperature"]),
        latent_noise=float(cfg["latent_noise"]),
        jitter=float(cfg["jitter"]),
        dtype=getattr(torch, cfg["dtype"]),
        device=cfg["device"],
    )
    t0 = time.perf_counter()
    fit = model.fit(
        data["X_train"],
        data["Y_train"],
        num_steps=int(cfg["num_steps"]),
        lr_Z=float(cfg["lr_Z"]),
        lr_global=float(cfg["lr_global"]),
        log_every=int(cfg["log_every"]),
        record_loss_curve=True,
        seed=int(task["seed"]),
        verbose=False,
    )
    y_pred = invert_y(model.predict(data["X_test"]), data["y_mean"], data["y_std"])
    write_curve(run_dir / "training_curve.csv", fit.loss_curve or [])
    np.savez_compressed(run_dir / "test_predictions.npz", y_pred=y_pred, y_true=data["Y_test"], y_sig=data["Y_test_sig"])
    result = {
        "config": {**task, "latent_dim": int(cfg["latent_dim"]), "num_steps": int(cfg["num_steps"]), "inverse_temperature": float(cfg["inverse_temperature"]), "latent_noise": float(cfg["latent_noise"])},
        "status": "completed",
        "device": str(model.device),
        "fit_seconds": time.perf_counter() - t0,
        "final_loss": float(fit.final_loss),
        "test_metrics": {"rmse_sig": rmse(data["Y_test_sig"], y_pred), "rmse_obs": rmse(data["Y_test"], y_pred)},
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[done] task_id={task['task_id']} rmse_sig={result['test_metrics']['rmse_sig']:.6g}", flush=True)


def prepare_data(cfg: dict[str, Any], n_train: int, seed: int) -> dict[str, np.ndarray]:
    data = np.load(HERE / cfg["data_path"])
    split = np.load(HERE / cfg["split_path"])
    pool_idx = np.asarray(split["train_idx"], dtype=int)
    train_idx = np.sort(pool_idx[np.random.default_rng(seed).permutation(pool_idx.shape[0])[:n_train]])
    test_idx = np.asarray(split["test_idx"], dtype=int)
    x_mean, x_std, y_mean, y_std = fit_preprocessors(data["X"][train_idx], data["Y"][train_idx])
    return {
        "X_train": standardize(data["X"][train_idx], x_mean, x_std),
        "Y_train": standardize(data["Y"][train_idx], y_mean, y_std),
        "X_test": standardize(data["X"][test_idx], x_mean, x_std),
        "Y_test": np.asarray(data["Y"][test_idx], dtype=np.float64),
        "Y_test_sig": np.asarray(data["Y_sig"][test_idx], dtype=np.float64),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def fit_preprocessors(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return x.mean(0), x.std(0), y.mean(0), y.std(0)


def standardize(a: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) - mean[None, :]) / std[None, :]


def invert_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) * std[None, :] + mean[None, :]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def write_curve(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
