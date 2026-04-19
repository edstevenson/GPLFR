"""Unified sweep runner for the toy benchmark.

Supports:
- learning curves (varying n_train)
- per-emulator hyperparameter sweeps

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

from . import metrics as metrics_mod
from .grid import MechanismToyGrid
from .utils import (
    apply_x_standardizer,
    apply_y_standardizer,
    fit_preprocessors,
    invert_y_standardizer,
    load_run_config,
    parse_config_and_overrides,
)
from . import train as train_mod


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
    out = [int(v) for v in x] if isinstance(x, (list, tuple)) else [int(x)]
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


def _encode_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return str(v)
    if isinstance(v, dict):
        keys = sorted(str(k) for k in v.keys())
        items = [f"{k}={_encode_value(v[k])}" for k in keys]
        return "{" + ";".join(items) + "}"
    if isinstance(v, (list, tuple, np.ndarray)):
        items = [_encode_value(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
        return "[" + ";".join(items) + "]"
    return str(v).replace(",", ";")


def _variant_id(*, emulator: str, n_train: int, variant: dict[str, Any]) -> str:
    parts = {"emulator": emulator, "n_train": int(n_train), **(variant or {})}
    return ",".join(f"{k}={_encode_value(parts[k])}" for k in sorted(parts.keys()))


def _iter_emulator_variants(grid: dict[str, Any], emulator: str) -> list[dict[str, Any]]:
    emu_grid = grid.get(emulator, {})
    if not emu_grid or not isinstance(emu_grid, dict):
        return [{}]
    keys = list(emu_grid.keys())
    values = []
    for k in keys:
        v = emu_grid[k]
        if not isinstance(v, (list, tuple)):
            raise TypeError(f"sweep.grid.{emulator}.{k} must be a list, got {type(v).__name__}")
        values.append(list(v))
    return [{k: v for k, v in zip(keys, combo)} for combo in itertools.product(*values)]


def _summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for r in runs:
        key = (str(r["emulator"]), int(r["n_train"]), str(r.get("variant_id", "")))
        g = grouped.setdefault(key, {"seeds": set(), "metrics": {}})
        g["seeds"].add(int(r["seed"]))
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


def _effective_out_dir(cfg: dict[str, Any]) -> Path:
    base = Path(cfg["out_dir"]).expanduser().resolve()
    sweep_cfg = cfg.get("sweep") or {}
    name = sweep_cfg.get("name") if isinstance(sweep_cfg, dict) else None
    return base.parent / name if name else base


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
    if os.environ.get("XCE_TOY_SLURM_BATCH"):
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
            seed_cfg_path.write_text(yaml.safe_dump(seed_cfg, sort_keys=False), encoding="utf-8")
            _maybe_submit_slurm(cfg_path=seed_cfg_path, overrides=overrides, cfg=seed_cfg)
        return True

    out_dir = _effective_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    slurm_cfg_path = (out_dir / "cfg.yaml").resolve()
    slurm_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[toy.sweep] cfg snapshot: {slurm_cfg_path}")

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

    exclude = str(slurm_cfg.get("exclude", "")).strip()
    exclude_args = [] if not exclude else [f"--exclude={exclude}"]

    repo_root = _find_repo_root()
    run_slurm = repo_root / "slurm" / "run.slurm"

    cmd: list[str] = [
        "sbatch",
        "-A", account,
        "-p", partition,
        *exclude_args,
        f"--cpus-per-task={cpus_per_task}",
        f"--time={time_limit}",
        *gres_args,
        *([] if not bool(slurm_cfg.get("exclusive", False)) else ["--exclusive"]),
        f"--job-name={out_dir.name}",
        "-o", str(out_dir / "slurm.out"),
        "-e", str(out_dir / "slurm.err"),
        f"--export=XCE_CORES={cpus_per_task},XCE_TOY_SLURM_BATCH=1",
        str(run_slurm),
        "-m", "gplfr.experiments.toy.sweep",
        f"config={slurm_cfg_path}",
    ]

    print("[toy.sweep] SLURM enabled; submitting:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("sbatch stdout:", result.stdout.strip())
        print("sbatch stderr:", result.stderr.strip())
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    print(result.stdout.strip())
    return True


def _compact_extra(emulator: str, extra: dict[str, Any] | None) -> dict[str, Any] | None:
    if not extra:
        return None
    return {"fit_result": extra.get("fit_result")} if emulator == "gplfr" else extra


def main(argv: list[str] | None = None) -> None:
    argv_in = sys.argv[1:] if argv is None else argv
    cfg_path, overrides = parse_config_and_overrides(argv_in)
    cfg_path = cfg_path.expanduser().resolve()
    cfg = load_run_config(cfg_path, overrides=overrides)

    sweep_cfg = cfg.get("sweep") or {}
    if not (isinstance(sweep_cfg, dict) and bool(sweep_cfg.get("enabled", False))):
        train_mod.main(argv=argv_in)
        return

    out_dir = _effective_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    if _maybe_submit_slurm(cfg_path=cfg_path, overrides=overrides, cfg=cfg):
        return
    (out_dir / "cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    grid = MechanismToyGrid.load(cfg["data_path"])
    split = np.load(Path(cfg["split_path"]).expanduser().resolve())
    pool_idx = np.asarray(split["train_idx"], dtype=int)
    val_idx = np.asarray(split["val_idx"], dtype=int) if "val_idx" in split.files else None
    test_idx = np.asarray(split["test_idx"], dtype=int)

    test_grid = grid._subset(test_idx)
    X_test = np.asarray(test_grid.X, dtype=np.float64)
    y_true = np.asarray(test_grid._Y, dtype=np.float64)
    y_sig_true = None if test_grid.Y_sig is None else np.asarray(test_grid.Y_sig, dtype=np.float64)

    val_data = None
    if val_idx is not None:
        val_grid = grid._subset(val_idx)
        val_data = (np.asarray(val_grid.X, dtype=np.float64), np.asarray(val_grid._Y, dtype=np.float64))
        if val_grid.Y_sig is not None:
            val_data = (*val_data, np.asarray(val_grid.Y_sig, dtype=np.float64))

    sweep_grid = sweep_cfg.get("grid", {})
    n_train_list = _as_int_list(sweep_grid.get("n_train", sweep_cfg.get("n_train")), name="sweep.grid.n_train", min_value=1)
    seeds = _as_int_list(sweep_cfg.get("seeds"), name="sweep.seeds")
    emulators = sweep_cfg.get("emulators", cfg.get("emulators", []))
    emulators = [str(e).lower() for e in emulators]
    if not n_train_list or not seeds or not emulators:
        raise ValueError("sweep requires non-empty n_train, seeds, and emulators")

    pool_n = int(pool_idx.shape[0])
    if max(n_train_list) > pool_n:
        raise ValueError(f"max(n_train)={max(n_train_list)} > pool_size={pool_n} (split.train_idx)")

    runs: list[dict[str, Any]] = []
    fitted_params_runs: list[dict[str, Any]] = []
    for data_seed in seeds:
        perm = np.random.default_rng(int(data_seed)).permutation(pool_n)
        for n_eff in n_train_list:
            train_idx = np.sort(pool_idx[perm[: int(n_eff)]])
            train_grid = grid._subset(train_idx)

            X_tr = np.asarray(train_grid.X, dtype=np.float64)
            Y_tr = np.asarray(train_grid._Y, dtype=np.float64)
            x_mean, x_std, y_mean, y_std = fit_preprocessors(X_tr, Y_tr)
            prep = cfg.get("preprocess", {}) or {}
            if str(prep.get("x", "zscore")).lower() == "none":
                x_mean = np.zeros_like(x_mean)
                x_std = np.ones_like(x_std)
            if str(prep.get("y", "zscore")).lower() == "none":
                y_mean = np.zeros_like(y_mean)
                y_std = np.ones_like(y_std)
            X_test_z = apply_x_standardizer(X_test, x_mean, x_std)

            for emulator in emulators:
                variants = _iter_emulator_variants(sweep_grid, emulator)
                for variant in variants:
                    emu_cfg = deepcopy(cfg[emulator])
                    _deep_update(emu_cfg, variant)
                    emu_cfg.setdefault("random_seed", int(data_seed))

                    vid = _variant_id(emulator=emulator, n_train=int(n_eff), variant=variant)
                    t0 = time.perf_counter()
                    print(f"[toy.sweep] seed={data_seed} n_train={n_eff} emulator={emulator}" + (f" variant={vid}" if vid else "") + " train...")

                    if emulator == "pca_gp":
                        t_train0 = time.perf_counter()
                        emu, extra = train_mod._fit_pca_gp(
                            train_grid,
                            cfg=emu_cfg,
                            x_mean=x_mean,
                            x_std=x_std,
                            y_mean=y_mean,
                            y_std=y_std,
                            val_data=val_data,
                        )
                        t_train = time.perf_counter() - t_train0
                        t_pred0 = time.perf_counter()
                        y_pred_z = emu.predict_flux(X_test_z)
                        y_pred = invert_y_standardizer(y_pred_z, y_mean, y_std)
                        t_pred = time.perf_counter() - t_pred0

                        y_true_z = apply_y_standardizer(y_true, y_mean, y_std)
                        w_true = (y_true_z - emu.pca_mean[None, :]) @ emu.eigenspectra.T
                        w_pred = (np.asarray(y_pred_z, dtype=np.float64) - emu.pca_mean[None, :]) @ emu.eigenspectra.T
                        r2 = metrics_mod.r2_per_dim(w_true, w_pred)
                        extra["diagnostics"] = {"pca_score_r2": r2, "pca_score_r2_mean": float(np.nanmean(r2))}
                        s2 = emu.sigma_xi2
                        fitted_params = {
                            "kernel": str(emu.kernel),
                            "n_components": int(emu.n_components),
                            "sigma_xi2_mode": str(emu.sigma_xi2_mode),
                            "sigma_xi2": (float(s2) if not isinstance(s2, np.ndarray) else [float(x) for x in s2]),
                            "variance": [float(x) for x in emu.variances],
                            "lengthscales": [[float(x) for x in row] for row in emu.lengthscales],
                        }

                    elif emulator == "gplfr":
                        if val_data is None:
                            raise ValueError("gplfr sweep requires val_idx in split NPZ (for optional early stopping).")
                        t_train0 = time.perf_counter()
                        emu, extra = train_mod._fit_gplfr(
                            train_grid,
                            cfg=emu_cfg,
                            x_mean=x_mean,
                            x_std=x_std,
                            y_mean=y_mean,
                            y_std=y_std,
                            val_data=val_data,
                        )
                        t_train = time.perf_counter() - t_train0
                        t_pred0 = time.perf_counter()
                        y_pred = invert_y_standardizer(emu.predict(X_test_z), y_mean, y_std)
                        t_pred = time.perf_counter() - t_pred0
                        post = getattr(emu, "posterior_samples_", None) or {}
                        sigma = None if "sigma" not in post else float(post["sigma"][0].detach().cpu().item())
                        ell = (
                            post["ell"][0].detach().cpu().numpy().tolist()
                            if "ell" in post
                            else emu_cfg.get("ell_fixed", emu_cfg.get("ell", None))
                        )
                        sigma_f = (
                            post["sigma_f"][0].detach().cpu().numpy().tolist()
                            if "sigma_f" in post
                            else (
                                emu_cfg.get("sigma_f_fixed", emu_cfg.get("sigma_f", None))
                                if str(emu_cfg.get("sigma_f_mode", "null")).lower() == "fixed"
                                else None
                            )
                        )
                        sigma_xi2 = None
                        if "sigma_xi2" in post:
                            s2 = post["sigma_xi2"][0].detach().cpu().numpy()
                            sigma_xi2 = float(s2) if s2.shape == () else s2.tolist()
                        Z_T_rms = None
                        if "Z_T" in post:
                            Z_T = post["Z_T"][0].detach().cpu().numpy()
                            Z_T_rms = float(np.sqrt(np.mean(Z_T**2)))
                        fitted_params = {
                            "tau": float(emu_cfg.get("tau", 1.0)),
                            "data_fit_scale": float(getattr(emu, "data_fit_scale", emu_cfg.get("data_fit_scale", 1.0))),
                            "sigma_xi2_mode": str(emu_cfg.get("sigma_xi2_mode", "float")),
                            "sigma_xi2": float(emu_cfg.get("sigma_xi2", 1e-6)) if sigma_xi2 is None else sigma_xi2,
                            "sigma": sigma,
                            "ell": ell,
                            "sigma_f": sigma_f,
                            "Z_T_rms": Z_T_rms,
                        }

                    else:
                        raise ValueError(f"Unknown emulator {emulator!r}")

                    met = metrics_mod.evaluate_metrics(y_true, y_pred, y_sig_true=y_sig_true)
                    if emulator == "pca_gp":
                        met["pca_score_r2_mean"] = float(extra["diagnostics"]["pca_score_r2_mean"])
                    t_total = time.perf_counter() - t0
                    print(f"[toy.sweep] seed={data_seed} n_train={n_eff} emulator={emulator}" + (f" variant={vid}" if vid else "") + f" done rmse_sig={met.get('rmse_sig', float('nan')):.6g} total={t_total:.3f}s")

                    run_entry: dict[str, Any] = {
                        "emulator": emulator,
                        "seed": int(data_seed),
                        "n_train": int(n_eff),
                        "metrics": met,
                        "timing_s": {"train": float(t_train), "predict": float(t_pred), "total": float(t_total)},
                    }
                    run_entry["variant_id"] = vid
                    extra_out = _compact_extra(emulator, extra)
                    if extra_out is not None:
                        run_entry["extra"] = extra_out
                    run_entry["fitted_params"] = fitted_params
                    runs.append(run_entry)
                    fitted_params_runs.append(
                        {
                            "emulator": emulator,
                            "seed": int(data_seed),
                            "n_train": int(n_eff),
                            "variant_id": vid,
                            "fitted_params": fitted_params,
                        }
                    )

    variants = _summarize(runs)
    key_metric = "rmse_sig" if any("rmse_sig" in (r.get("metrics") or {}) for r in runs) else "rmse_obs"
    summary = {
        (v.get("variant_id") or v["emulator"]): v["metrics"][key_metric]["median"]
        for v in variants
        if key_metric in v.get("metrics", {})
    }
    (out_dir / "metrics.json").write_text(json.dumps({"summary": summary, "variants": variants}, indent=2), encoding="utf-8")
    (out_dir / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    (out_dir / "fitted_params.json").write_text(json.dumps({"runs": fitted_params_runs}, indent=2), encoding="utf-8")
    print(f"[toy.sweep] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
