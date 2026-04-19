"""Unified sweep runner for the PyXOpto benchmark.

Supports optional SLURM submission via:
  sweep:
    slurm:
      enabled: true
      partition: ampere|icelake|sapphire|pvc9
      account_cpu: ...
      account_gpu: ...
      account_dawn: ...
      cpus_per_task: 12
      time: "02:00:00"
      gres: "gpu:1"   # required on GPU partitions
      exclude: "pvc-s-19"  # optional (or list[str])
      nodelist: "pvc-s-8,pvc-s-16"  # optional (or list[str])
      exclusive: false  # optional
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ._paths import resolve_path

from . import metrics as metrics_mod
from .grid import PyXOptoGrid
from .utils import (
    apply_x_standardizer,
    fit_preprocessors,
    load_run_config,
    log10_transform,
    parse_config_and_overrides,
)
from . import train as train_mod


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "slurm" / "run.slurm").exists():
            return p
    raise FileNotFoundError("Could not locate repo root containing slurm/run.slurm")


def _maybe_submit_slurm(*, cfg_path: Path, overrides: dict[str, str], cfg: dict[str, Any]) -> bool:
    sweep_cfg = cfg.get("sweep") or {}
    slurm_cfg = (sweep_cfg.get("slurm") or {}) if isinstance(sweep_cfg, dict) else {}
    if not bool(slurm_cfg.get("enabled", False)):
        return False
    if os.environ.get("XCE_PYXOPTO_SLURM_BATCH"):
        return False
    if bool(sweep_cfg.get("parallel_seeds", False)):
        seeds = _as_int_list(sweep_cfg.get("seeds"), name="sweep.seeds")
        if not seeds:
            raise ValueError("sweep.parallel_seeds requires sweep.seeds")
        out_dir = _effective_out_dir(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        for seed in seeds:
            seed_cfg = deepcopy(cfg)
            seed_cfg["out_dir"] = str(out_dir / f"seed_{seed}")
            seed_cfg["sweep"]["seeds"] = [seed]
            seed_cfg["sweep"]["parallel_seeds"] = False
            seed_cfg["sweep"]["name"] = None
            seed_cfg_path = Path(seed_cfg["out_dir"]) / "cfg.yaml"
            seed_cfg_path.parent.mkdir(parents=True, exist_ok=True)
            seed_cfg_path.write_text(yaml.safe_dump(seed_cfg, sort_keys=False))
            _maybe_submit_slurm(cfg_path=seed_cfg_path, overrides=overrides, cfg=seed_cfg)
        return True

    out_dir = _effective_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot an immutable cfg *before* queueing so editing the source YAML later can't
    # change what the compute node runs (prevents out_dir/log mismatches across runs).
    slurm_cfg_path = (out_dir / "cfg.yaml").resolve()
    slurm_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[pyxopto.sweep] cfg snapshot: {slurm_cfg_path}")

    partition = str(slurm_cfg.get("partition", "")).strip()
    if not partition:
        raise ValueError("sweep.slurm.partition is required when sweep.slurm.enabled is true")

    account_cpu = str(slurm_cfg.get("account_cpu", "")).strip()
    account_gpu = str(slurm_cfg.get("account_gpu", "")).strip()
    account_dawn = str(slurm_cfg.get("account_dawn", "")).strip()
    if partition == "pvc9":
        if not account_dawn:
            raise ValueError("sweep.slurm.account_dawn is required when partition is pvc9")
        account = account_dawn
    elif partition == "ampere":
        if not account_gpu:
            raise ValueError("sweep.slurm.account_gpu is required when partition is ampere")
        account = account_gpu
    else:
        if not account_cpu:
            raise ValueError("sweep.slurm.account_cpu is required for CPU partitions")
        account = account_cpu

    cpus_per_task = int(slurm_cfg.get("cpus_per_task", 1))
    if cpus_per_task <= 0:
        raise ValueError("sweep.slurm.cpus_per_task must be > 0")

    time_limit = str(slurm_cfg.get("time", "")).strip()
    if not time_limit:
        raise ValueError("sweep.slurm.time is required (e.g. \"5:00:00\")")

    gpu_partitions = {"ampere", "pvc9"}
    gres_args: list[str] = []
    if partition in gpu_partitions:
        gres = str(slurm_cfg.get("gres", "")).strip()
        if not gres:
            raise ValueError(f"sweep.slurm.gres is required when partition is one of {sorted(gpu_partitions)}")
        gres_args = [f"--gres={gres}"]

    exclude = slurm_cfg.get("exclude", None)
    if isinstance(exclude, (list, tuple)):
        exclude = ",".join(str(x).strip() for x in exclude if str(x).strip())
    exclude_arg = [f"--exclude={str(exclude).strip()}"] if str(exclude or "").strip() else []

    nodelist = slurm_cfg.get("nodelist", None)
    if isinstance(nodelist, (list, tuple)):
        nodelist = ",".join(str(x).strip() for x in nodelist if str(x).strip())
    nodelist_arg = [f"--nodelist={str(nodelist).strip()}"] if str(nodelist or "").strip() else []

    exclusive_arg = ["--exclusive"] if bool(slurm_cfg.get("exclusive", False)) else []

    repo_root = _find_repo_root()
    run_slurm = repo_root / "slurm" / "run.slurm"

    job_name = f"{out_dir.name}"
    cmd: list[str] = [
        "sbatch",
        "-A", account,
        "-p", partition,
        *exclude_arg,
        *nodelist_arg,
        *exclusive_arg,
        f"--cpus-per-task={cpus_per_task}",
        f"--time={time_limit}",
        *gres_args,
        f"--job-name={job_name}",
        "-o", str(out_dir / "slurm.out"),
        "-e", str(out_dir / "slurm.err"),
        f"--export=XCE_CORES={cpus_per_task},XCE_PYXOPTO_SLURM_BATCH=1",
        str(run_slurm),
        "-m", "gplfr.experiments.pyxopto.sweep",
        f"config={slurm_cfg_path}",
    ]

    print("[pyxopto.sweep] SLURM enabled; submitting:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("sbatch stdout:", result.stdout.strip())
        print("sbatch stderr:", result.stderr.strip())
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    print(result.stdout.strip())
    return True


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


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


def _variant_id(variant: dict[str, Any]) -> str:
    return ",".join(f"{k}={variant[k]}" for k in sorted(variant.keys())) if variant else ""


def _iter_emulator_variants(grid: dict[str, Any], emulator: str) -> list[dict[str, Any]]:
    emu_grid = grid.get(emulator, {})
    if not emu_grid or not isinstance(emu_grid, dict):
        return [{}]
    keys = list(emu_grid.keys())
    values = [list(emu_grid[k]) for k in keys]
    return [{k: v for k, v in zip(keys, combo)} for combo in itertools.product(*values)]


def _summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for r in runs:
        vid = str(r.get("variant_id", ""))
        key = (str(r["emulator"]), int(r["n_train"]), vid)
        g = grouped.setdefault(key, {"seeds": set(), "metrics": {}})
        g["seeds"].add(int(r["seed"]))
        for m, v in (r.get("metrics") or {}).items():
            g["metrics"].setdefault(str(m), []).append(float(v))

    summary: list[dict[str, Any]] = []
    for (emu, n_train, vid), g in grouped.items():
        entry: dict[str, Any] = {
            "emulator": emu,
            "n_train": n_train,
            "n_seeds": len(g["seeds"]),
            "metrics": {m: _metric_stats(vs) for m, vs in g["metrics"].items()},
        }
        if vid:
            entry["variant_id"] = vid
        summary.append(entry)

    summary.sort(key=lambda d: (d["emulator"], d["n_train"], d.get("variant_id", "")))
    return summary


def _effective_out_dir(cfg: dict[str, Any]) -> Path:
    base = Path(cfg["out_dir"]).expanduser().resolve()
    sweep_cfg = cfg.get("sweep") or {}
    name = sweep_cfg.get("name") if isinstance(sweep_cfg, dict) else None
    return base.parent / name if name else base


def _fit_and_predict(
    emulator: str,
    train_grid: PyXOptoGrid,
    X_test_z: np.ndarray,
    s_test: np.ndarray,
    emu_cfg: dict[str, Any],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    eps: float,
    *,
    X_test_full: np.ndarray | None = None,
    val_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None, float, float]:
    if emulator == "simplex":
        from .simplex import SimplexInterpolator

        _, _, _, Y_train_raw = train_grid.load_matrix(dtype=np.float64)
        t_train0 = time.perf_counter()
        emu_s = SimplexInterpolator(train_grid._X_full, Y_train_raw, eps=eps)
        t_train = time.perf_counter() - t_train0
        t_pred0 = time.perf_counter()
        y_pred = emu_s.predict(X_test_full)
        t_pred = time.perf_counter() - t_pred0
        return np.asarray(y_pred, dtype=np.float64), None, t_train, t_pred

    if emulator == "gplfr":
        t_train0 = time.perf_counter()
        emu, extra = train_mod._fit_gplfr(
            train_grid,
            cfg=emu_cfg,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            eps=eps,
            val_data=val_data,
        )
        t_train = time.perf_counter() - t_train0

        t_pred0 = time.perf_counter()
        y_pred_log_z = emu.predict(X_test_z, s_test)
        y_pred = train_mod.inv_log10_transform(train_mod.invert_y_standardizer(y_pred_log_z, y_mean, y_std), eps=eps)
        t_pred = time.perf_counter() - t_pred0
        return np.asarray(y_pred, dtype=np.float64), extra, t_train, t_pred

    if emulator == "pca_mlp":
        t_train0 = time.perf_counter()
        emu, extra = train_mod._fit_pca_mlp(
            train_grid,
            cfg=emu_cfg,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            eps=eps,
            val_data=val_data,
        )
        t_train = time.perf_counter() - t_train0

        t_pred0 = time.perf_counter()
        y_pred_log_z = emu.predict_curve(X_test_z, s_test)
        y_pred = train_mod.inv_log10_transform(train_mod.invert_y_standardizer(y_pred_log_z, y_mean, y_std), eps=eps)
        t_pred = time.perf_counter() - t_pred0
        return np.asarray(y_pred, dtype=np.float64), extra, t_train, t_pred

    if emulator != "pca_icm":
        raise ValueError(f"Unknown emulator {emulator!r}")
    t_train0 = time.perf_counter()
    emu, extra = train_mod._fit_pca_icm(
        train_grid,
        cfg=emu_cfg,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        eps=eps,
        val_data=val_data,
    )
    t_train = time.perf_counter() - t_train0

    t_pred0 = time.perf_counter()
    y_pred_log_z = emu.predict_curve(X_test_z, s_test)
    y_pred = train_mod.inv_log10_transform(train_mod.invert_y_standardizer(y_pred_log_z, y_mean, y_std), eps=eps)
    t_pred = time.perf_counter() - t_pred0
    return np.asarray(y_pred, dtype=np.float64), extra, t_train, t_pred


def main(argv: list[str] | None = None) -> None:
    argv_in = sys.argv[1:] if argv is None else argv
    cfg_path, overrides = parse_config_and_overrides(argv_in)
    cfg_path = cfg_path.expanduser().resolve()
    cfg = load_run_config(cfg_path, overrides=overrides)

    sweep_cfg = cfg.get("sweep") or {}
    if not (isinstance(sweep_cfg, dict) and bool(sweep_cfg.get("enabled", False))):
        train_mod.main(argv=argv_in)
        return

    data_path = resolve_path(cfg["data_path"])
    split_path = resolve_path(cfg["split_path"])
    out_dir = _effective_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _maybe_submit_slurm(cfg_path=cfg_path, overrides=overrides, cfg=cfg):
        return

    if cfg.get("metrics") is not None:
        raise ValueError("metrics selection is not supported; metrics are fixed to rmse/mae/maqe_0.95 in log10 space.")

    eps = float(cfg.get("log_eps", 1e-30))
    (out_dir / "cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    grid = PyXOptoGrid.load(data_path)
    split = np.load(split_path)
    pool_idx = np.asarray(split["train_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    val_data = None
    if "val_idx" in split.files:
        val_grid = grid._subset(np.asarray(split["val_idx"], dtype=int))
        Xv, sv, _, Yv = val_grid.load_matrix(dtype=np.float64)
        val_data = (Xv, sv, Yv)

    test_grid = grid._subset(test_idx)
    X_test, s_test, _, y_true = test_grid.load_matrix(dtype=np.float64)

    S = int(grid.g_values.shape[0])
    pool_s = grid.s[pool_idx]
    pools = [pool_idx[pool_s == t] for t in range(S)]
    pool_n_per = [int(p.shape[0]) for p in pools]
    if len(set(pool_n_per)) != 1:
        raise ValueError(f"Expected equal pool sizes per task, got {pool_n_per}")

    sweep_grid = sweep_cfg.get("grid", {})
    n_train_list = _as_int_list(sweep_grid.get("n_train", sweep_cfg.get("n_train")), name="sweep.grid.n_train", min_value=1)
    seeds = _as_int_list(sweep_cfg.get("seeds"), name="sweep.seeds")
    emulators = [str(e).lower() for e in sweep_cfg.get("emulators", cfg.get("emulators", []))]

    if not n_train_list or not seeds or not emulators:
        raise ValueError("sweep requires sweep.grid.n_train, sweep.seeds, and sweep.emulators (or top-level emulators).")
    if any(n % S != 0 for n in n_train_list):
        raise ValueError(f"All n_train must be divisible by n_tasks={S} for balanced subsets (got {n_train_list})")

    max_n = max(n_train_list) // S
    if max_n > pool_n_per[0]:
        raise ValueError(f"Max per-task n_train={max_n} exceeds per-task pool size={pool_n_per[0]}")

    total_variants = sum(len(_iter_emulator_variants(sweep_grid, e)) for e in emulators)
    total_runs = len(seeds) * len(n_train_list) * total_variants
    print(f"[pyxopto.sweep] pool_n={int(pool_idx.shape[0])} n_test={int(test_idx.shape[0])} n_pix={int(test_grid.n_pix)} n_tasks={S}")
    print(f"[pyxopto.sweep] emulators={emulators} n_train={n_train_list} seeds={seeds} total_runs={total_runs}")

    runs: list[dict[str, Any]] = []
    for data_seed in seeds:
        rng = np.random.default_rng(int(data_seed))
        perms = [p[rng.permutation(p.shape[0])] for p in pools]

        for n_train in n_train_list:
            n_per = int(n_train) // S
            train_idx = np.sort(np.concatenate([p[:n_per] for p in perms]))
            train_grid = grid._subset(train_idx)

            X_train, _, _, Y_train = train_grid.load_matrix(dtype=np.float64)
            Y_train_log = log10_transform(Y_train, eps=eps)
            x_mean, x_std, y_mean, y_std = fit_preprocessors(X_train, Y_train_log)
            X_test_z = apply_x_standardizer(X_test, x_mean, x_std)

            for emulator in emulators:
                variants = _iter_emulator_variants(sweep_grid, emulator)
                for variant in variants:
                    emu_cfg: dict[str, Any] = deepcopy(cfg[emulator])
                    _deep_update(emu_cfg, variant)
                    if "random_seed" in emu_cfg and "random_seed" not in variant:
                        emu_cfg["random_seed"] = int(data_seed)

                    vid = _variant_id(variant)
                    t0 = time.perf_counter()
                    print(
                        f"[pyxopto.sweep] data_seed={data_seed} n_train={int(train_grid.n_models)} emulator={emulator}"
                        + (f" variant={vid}" if vid else "")
                        + " train..."
                    )

                    y_pred, extra, t_train, t_pred = _fit_and_predict(
                        emulator, train_grid, X_test_z, s_test, emu_cfg, x_mean, x_std, y_mean, y_std, eps,
                        X_test_full=test_grid._X_full,
                        val_data=(val_data if emu_cfg.get("early_stop_patience_evals", None) is not None and int(emu_cfg["early_stop_patience_evals"]) > 0 else None),
                    )
                    met = metrics_mod.evaluate_metrics(y_true, y_pred, eps=eps)
                    t_total = time.perf_counter() - t0
                    print(
                        f"[pyxopto.sweep] data_seed={data_seed} n_train={int(train_grid.n_models)} emulator={emulator}"
                        + (f" variant={vid}" if vid else "")
                        + f" done mae={met.get('mae', float('nan')):.6g} train={t_train:.3f}s pred={t_pred:.3f}s total={t_total:.3f}s"
                    )

                    run_entry: dict[str, Any] = {
                        "emulator": emulator,
                        "seed": int(data_seed),
                        "data_seed": int(data_seed),
                        "n_train": int(train_grid.n_models),
                        "metrics": met,
                        "timing_s": {"train": float(t_train), "predict": float(t_pred), "total": float(t_total)},
                    }
                    if vid:
                        run_entry["variant_id"] = vid
                        run_entry["variant"] = {k: v for k, v in variant.items()}
                    if extra is not None:
                        run_entry["extra"] = extra
                    runs.append(run_entry)

    variants = _summarize(runs)
    summary = {
        (v.get("variant_id") or v["emulator"]): v["metrics"]["mae"]["median"]
        for v in variants if "mae" in v.get("metrics", {})
    }
    (out_dir / "metrics.json").write_text(json.dumps({"summary": summary, "variants": variants}, indent=2), encoding="utf-8")
    (out_dir / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"[pyxopto.sweep] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
