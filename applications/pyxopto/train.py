"""PyXOpto benchmark runner (PCAICM baseline + others)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import metrics as metrics_mod
from .grid import PyXOptoGrid
from .pca_icm import PCAICM
from .pca_mlp import PCAMLP
from .utils import (
    apply_x_standardizer,
    apply_y_standardizer,
    fit_preprocessors,
    inv_log10_transform,
    invert_y_standardizer,
    load_run_config,
    log10_transform,
    parse_config_and_overrides,
)


def _balanced_subset(grid: PyXOptoGrid, n_total: int, *, seed: int = 0) -> PyXOptoGrid:
    S = int(grid.g_values.shape[0])
    if n_total % S != 0:
        raise ValueError(f"n_train must be divisible by n_tasks={S} (got {n_total})")
    n_per = n_total // S
    rng = np.random.default_rng(seed)
    idx_list = [rng.choice(np.where(grid.s == t)[0], size=n_per, replace=False) for t in range(S)]
    return grid._subset(np.sort(np.concatenate(idx_list)))


def _fit_pca_icm(
    train: PyXOptoGrid,
    *,
    cfg: dict[str, Any],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    eps: float,
    val_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[PCAICM, dict[str, Any]]:
    n_components = int(cfg.get("n_components", 25))
    kernel = str(cfg.get("kernel", "rbf"))
    per_latent_kernel = bool(cfg.get("per_latent_kernel", False))
    per_latent_noise = bool(cfg.get("per_latent_noise", True))
    train_hyperparams = bool(cfg.get("train_hyperparams", True))
    device = cfg.get("device", "auto")
    dtype = str(cfg.get("dtype", "float64")).lower()

    X, s, r_mm, Y = train.load_matrix(dtype=np.float32)
    Xz = apply_x_standardizer(X, x_mean, x_std)
    Y_log_z = apply_y_standardizer(log10_transform(Y, eps=eps), y_mean, y_std)

    max_allowed = int(min(max(int(Y_log_z.shape[0]) - 1, 0), int(Y_log_z.shape[1])))
    if max_allowed < 1:
        raise ValueError(f"[pca_icm] PCA requires n_train>=2 and n_features>=1, got {Y_log_z.shape}")
    n_components = min(n_components, max_allowed)

    emu = PCAICM.from_curves(
        grid_points=Xz,
        tasks=s,
        axis=r_mm,
        curves=Y_log_z,
        n_components=n_components,
        kernel=kernel,  # type: ignore[arg-type]
        per_latent_kernel=per_latent_kernel,
        per_latent_noise=per_latent_noise,
        dtype=dtype,
        device=device,
    )
    print(f"[pyxopto] emulator=pca_icm device={emu.torch_device} dtype={emu.torch_dtype}")
    if train_hyperparams:
        Xv = sv = Yv = None
        if val_data is not None:
            Xv = apply_x_standardizer(np.asarray(val_data[0], dtype=np.float64), x_mean, x_std)
            sv = np.asarray(val_data[1], dtype=int).reshape(-1)
            Yv = apply_y_standardizer(log10_transform(np.asarray(val_data[2], dtype=np.float64), eps=eps), y_mean, y_std)
        emu.train(
            num_steps=int(cfg.get("num_steps", 200)),
            lr=float(cfg.get("lr", 5e-2)),
            seed=int(cfg.get("random_seed", 0)),
            val_X=Xv,
            val_tasks=sv,
            val_Y=Yv,
            val_y_std=y_std,
            early_stop_metric=str(cfg.get("early_stop_metric", "mae_val")),
            early_stop_patience_evals=cfg.get("early_stop_patience_evals", None),
            early_stop_min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
            log_every=int(cfg.get("log_every", 100)),
            verbose=bool(cfg.get("verbose", False)),
        )

    B = (np.tril(emu.coreg_L) @ np.tril(emu.coreg_L).T).astype(np.float64)
    B = B / np.sqrt(np.outer(np.diag(B), np.diag(B)))

    hyperparams: dict[str, Any] = {
        "kernel": kernel,
        "per_latent_kernel": per_latent_kernel,
        "per_latent_noise": per_latent_noise,
        "ell_min": [float(x) for x in emu.ell_min],
        "sigma_z2": (
            [float(np.exp(emu.hyperparams[f"log_sigma_z2:{m}"])) for m in range(emu.n_components)]
            if per_latent_noise
            else float(np.exp(emu.hyperparams["log_sigma_z2"]))
        ),
        "B": [[float(x) for x in row] for row in B],
    }
    if per_latent_kernel:
        hyperparams["variance"] = [float(np.exp(emu.hyperparams[f"log_variance:{m}"])) for m in range(emu.n_components)]
        hyperparams["lengthscales"] = [
            [float(emu.ell_min[d] + np.exp(emu.hyperparams[f"log_lengthscale:{m}:{d}"])) for d in range(2)] for m in range(emu.n_components)
        ]
    else:
        hyperparams["variance"] = float(np.exp(emu.hyperparams["log_variance"]))
        hyperparams["lengthscales"] = [float(emu.ell_min[d] + np.exp(emu.hyperparams[f"log_lengthscale:{d}"])) for d in range(2)]

    return emu, {
        "n_components": n_components,
        "kernel": kernel,
        "per_latent_kernel": per_latent_kernel,
        "per_latent_noise": per_latent_noise,
        "train_hyperparams": train_hyperparams,
        "early_stop_patience_evals": cfg.get("early_stop_patience_evals", None),
        "hyperparams": hyperparams,
    }


def _fit_pca_mlp(
    train: PyXOptoGrid,
    *,
    cfg: dict[str, Any],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    eps: float,
    val_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[PCAMLP, dict[str, Any]]:
    n_components = int(cfg.get("n_components", 25))
    hidden_width = int(cfg.get("hidden_width", 128))
    activation = str(cfg.get("activation", "silu")).lower()
    train_hyperparams = bool(cfg.get("train_hyperparams", True))
    device = cfg.get("device", "auto")
    dtype = str(cfg.get("dtype", "float64")).lower()

    X, s, r_mm, Y = train.load_matrix(dtype=np.float32)
    Xz = apply_x_standardizer(X, x_mean, x_std)
    Y_log_z = apply_y_standardizer(log10_transform(Y, eps=eps), y_mean, y_std)

    max_allowed = int(min(max(int(Y_log_z.shape[0]) - 1, 0), int(Y_log_z.shape[1])))
    if max_allowed < 1:
        raise ValueError(f"[pca_mlp] PCA requires n_train>=2 and n_features>=1, got {Y_log_z.shape}")
    n_components = min(n_components, max_allowed)

    emu = PCAMLP.from_curves(
        grid_points=Xz,
        tasks=s,
        axis=r_mm,
        curves=Y_log_z,
        n_components=n_components,
        hidden_width=hidden_width,
        activation=activation,  # type: ignore[arg-type]
        dtype=dtype,
        device=device,
    )
    print(f"[pyxopto] emulator=pca_mlp device={emu.torch_device} dtype={emu.torch_dtype}")

    if train_hyperparams:
        Xv = sv = Yv = None
        if val_data is not None:
            Xv = apply_x_standardizer(np.asarray(val_data[0], dtype=np.float64), x_mean, x_std)
            sv = np.asarray(val_data[1], dtype=int).reshape(-1)
            Yv = apply_y_standardizer(log10_transform(np.asarray(val_data[2], dtype=np.float64), eps=eps), y_mean, y_std)
        emu.train(
            num_steps=int(cfg.get("num_steps", 3000)),
            lr=float(cfg.get("lr", 1e-3)),
            weight_decay=float(cfg.get("weight_decay", 1e-3)),
            seed=int(cfg.get("random_seed", 0)),
            val_X=Xv,
            val_tasks=sv,
            val_Y=Yv,
            val_y_std=y_std,
            early_stop_metric=str(cfg.get("early_stop_metric", "mae_val")),
            early_stop_patience_evals=cfg.get("early_stop_patience_evals", None),
            early_stop_min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
            log_every=int(cfg.get("log_every", 100)),
            verbose=bool(cfg.get("verbose", False)),
        )

    return emu, {
        "n_components": n_components,
        "hidden_width": hidden_width,
        "activation": activation,
        "train_hyperparams": train_hyperparams,
        "optimizer": {"lr": float(cfg.get("lr", 1e-3)), "weight_decay": float(cfg.get("weight_decay", 1e-3))},
        "early_stop_patience_evals": cfg.get("early_stop_patience_evals", None),
        "axis_mm": r_mm.tolist(),
    }


def _fit_gplfr(
    train: PyXOptoGrid,
    *,
    cfg: dict[str, Any],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    eps: float,
    val_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from .gplfr import GPLFR
    from gplfr.gplfr.kernels import compute_sim_type_kernel
    import torch

    dtype = str(cfg.get("dtype", "float64")).lower()
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype)
    if torch_dtype is None:  # pragma: no cover
        raise ValueError(f"Unknown gplfr.dtype {dtype!r} (expected 'float32' or 'float64')")

    model_kwargs = {
        "latent_dim": int(cfg.get("latent_dim", 8)),
        "kernel": str(cfg.get("kernel", "matern52")),
        "ell_mode": str(cfg.get("ell_mode", "shared")),
        "coreg_mode": str(cfg.get("coreg_mode", "lkj")),
        "eta_lkj": float(cfg.get("eta_lkj", 1.0)),
        "sigma_f_mode": (None if cfg.get("sigma_f_mode", None) is None else str(cfg["sigma_f_mode"])),
        "sigma_f_fixed": (None if cfg.get("sigma_f_fixed", None) is None else float(cfg["sigma_f_fixed"])),
        "tau": float(cfg.get("tau", 1.0)),
        "data_fit_scale": float(cfg.get("data_fit_scale", 1.0)),
        "sigma_xi2_mode": str(cfg.get("sigma_xi2_mode", "float")),
        "sigma_xi2": float(cfg.get("sigma_xi2", 1e-6)),
        "jitter": float(cfg.get("jitter", 0.0)),
        "dtype": torch_dtype,
        "device": cfg.get("device", "auto"),
    }

    X, s, r_mm, Y = train.load_matrix(dtype=np.float64)
    Xz = apply_x_standardizer(X, x_mean, x_std)
    Y_log_z = apply_y_standardizer(log10_transform(Y, eps=eps), y_mean, y_std)

    Dy = int(Y_log_z.shape[1])
    d_eff = cfg.get("D_eff", cfg.get("d_eff", None))
    if d_eff is not None:
        model_kwargs["data_fit_scale"] = float(d_eff) / float(Dy)
    model_kwargs["data_fit_scale"] = float(model_kwargs["data_fit_scale"]) * float(cfg.get("data_fit_scale_mult", 1.0))

    Xv = sv = Yv = None
    if val_data is not None:
        Xv = apply_x_standardizer(np.asarray(val_data[0], dtype=np.float64), x_mean, x_std)
        sv = np.asarray(val_data[1], dtype=int).reshape(-1)
        Yv = apply_y_standardizer(log10_transform(np.asarray(val_data[2], dtype=np.float64), eps=eps), y_mean, y_std)

    max_allowed = int(min(max(int(Y_log_z.shape[0]) - 1, 0), int(Y_log_z.shape[1])))
    if max_allowed < 1:
        raise ValueError(f"[gplfr] PCA init requires n_train>=2 and n_features>=1, got {Y_log_z.shape}")
    if int(model_kwargs["latent_dim"]) > max_allowed:
        model_kwargs["latent_dim"] = max_allowed

    emu = GPLFR(**model_kwargs)
    print(f"[pyxopto] emulator=gplfr device={emu.device} dtype={emu.dtype}")
    pca_init_cfg = cfg.get("pca_init", None) or {}
    pca_init = {k: v for k, v in pca_init_cfg.items() if k != "enabled"} if bool(pca_init_cfg.get("enabled", False)) else None

    fit_result = emu.fit(
        Xz,
        s,
        Y_log_z,
        num_steps=int(cfg.get("num_steps", 3000)),
        lr_Z=float(cfg.get("lr_Z", 1.0e-2)),
        lr_global=float(cfg.get("lr_global", 1.0e-4)),
        lr_gamma=(None if cfg.get("lr_gamma", None) is None else float(cfg["lr_gamma"])),
        frozen_Z_start=float(cfg.get("frozen_Z_start", 0.0)),
        log_every=int(cfg.get("log_every", 100)),
        record_loss_curve=bool(cfg.get("record_loss_curve", False)),
        pca_init=pca_init,
        tau_schedule=cfg.get("tau_schedule", None),
        homotopy=cfg.get("homotopy", None),
        annealing=cfg.get("annealing", None),
        val_X=Xv,
        val_s=sv,
        val_Y=Yv,
        val_y_std=(None if Xv is None else y_std),
        early_stop_metric=str(cfg.get("early_stop_metric", "mae_val")),
        early_stop_patience_evals=cfg.get("early_stop_patience_evals", None),
        early_stop_min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
        seed=int(cfg.get("random_seed", 0)),
        verbose=bool(cfg.get("verbose", True)),
    )

    samples = emu.posterior_samples_ or {}
    hyper: dict[str, Any] = {
        "kernel": model_kwargs["kernel"],
        "ell_mode": model_kwargs["ell_mode"],
        "coreg_mode": model_kwargs["coreg_mode"],
        "tau": float(getattr(emu, "tau", model_kwargs["tau"])),
        "data_fit_scale": float(getattr(emu, "data_fit_scale", model_kwargs["data_fit_scale"])),
        "sigma_xi2_mode": model_kwargs["sigma_xi2_mode"],
        "sigma_xi2_extra": float(getattr(emu, "sigma_xi2_extra", 0.0)),
    }
    if samples:
        ell = samples["ell"][0].detach().cpu().numpy()
        hyper["ell"] = ell.tolist()
        hyper["sigma"] = float(samples["sigma"][0].detach().cpu())
        if str(model_kwargs.get("sigma_xi2_mode", "float")).lower() == "float":
            hyper["sigma_xi2"] = float(model_kwargs["sigma_xi2"])
        else:
            hyper["sigma_xi2"] = float(samples["sigma_xi2"][0].detach().cpu())
        if "sigma_f" in samples:
            hyper["sigma_f"] = samples["sigma_f"][0].detach().cpu().numpy().tolist()
        hyper["r_sim"] = samples["r_sim"][0].detach().cpu().numpy().tolist()
        Ks = compute_sim_type_kernel(samples["L_corr"][0], samples["r_sim"][0]).detach().cpu().numpy()
        corr = Ks / np.sqrt(np.outer(np.diag(Ks), np.diag(Ks)))
        hyper["B_corr"] = corr.tolist()

    return emu, {
        "latent_dim": int(model_kwargs["latent_dim"]),
        "kernel": str(model_kwargs["kernel"]),
        "fit_result": {"final_loss": float(fit_result.final_loss), "num_train_steps": int(fit_result.num_train_steps)},
        "hyperparams": hyper,
        "axis_mm": r_mm.tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    cfg_path, overrides = parse_config_and_overrides(sys.argv[1:] if argv is None else argv)
    cfg = load_run_config(cfg_path, overrides=overrides)

    data_path = Path(cfg["data_path"]).expanduser().resolve()
    split_path = Path(cfg["split_path"]).expanduser().resolve()
    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_set = str(cfg.get("eval_set", "test")).lower()
    if eval_set not in ("test", "val"):
        raise ValueError(f"Unknown eval_set={eval_set!r} (expected 'test' or 'val').")

    eps = float(cfg.get("log_eps", 1e-30))

    full_grid = PyXOptoGrid.load(data_path)
    split = np.load(split_path)
    train_grid = full_grid._subset(np.asarray(split["train_idx"], dtype=int))
    eval_key = "val_idx" if eval_set == "val" else "test_idx"
    if eval_key not in split.files:
        raise ValueError(f"eval_set={eval_set!r} requires split NPZ to contain {eval_key!r} (got keys={split.files}).")
    test_grid = full_grid._subset(np.asarray(split[eval_key], dtype=int))

    X_test, s_test, _, y_true = test_grid.load_matrix(dtype=np.float64)
    print(f"[pyxopto.train] n_train={train_grid.n_models} n_eval={test_grid.n_models} eval_set={eval_set}")

    X_train, _, _, Y_train = train_grid.load_matrix(dtype=np.float64)
    Y_train_log = log10_transform(Y_train, eps=eps)
    x_mean, x_std, y_mean, y_std = fit_preprocessors(X_train, Y_train_log)
    X_test_z = apply_x_standardizer(X_test, x_mean, x_std)

    if cfg.get("metrics") is not None:
        raise ValueError("metrics selection is not supported; metrics are fixed to rmse/mae/maqe_0.95 in log10 space.")

    results: dict[str, Any] = {}
    for name in cfg.get("emulators", []):
        t0 = time.perf_counter()
        print(f"[pyxopto.train] emulator={name} train...")

        if name == "pca_icm":
            t_train0 = time.perf_counter()
            val_data = None
            pat = cfg["pca_icm"].get("early_stop_patience_evals", None)
            if pat is not None and int(pat) > 0:
                if "val_idx" not in split.files:
                    raise ValueError(f"pca_icm early stopping requires split NPZ to contain 'val_idx' (got keys={split.files}).")
                val_grid = full_grid._subset(np.asarray(split["val_idx"], dtype=int))
                Xv, sv, _, Yv = val_grid.load_matrix(dtype=np.float64)
                val_data = (Xv, sv, Yv)
            emu, extra = _fit_pca_icm(
                train_grid,
                cfg=cfg["pca_icm"],
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
            y_pred = inv_log10_transform(invert_y_standardizer(y_pred_log_z, y_mean, y_std), eps=eps)
            t_pred = time.perf_counter() - t_pred0
        elif name == "pca_mlp":
            t_train0 = time.perf_counter()
            val_data = None
            pat = cfg["pca_mlp"].get("early_stop_patience_evals", None)
            if pat is not None and int(pat) > 0:
                if "val_idx" not in split.files:
                    raise ValueError(f"pca_mlp early stopping requires split NPZ to contain 'val_idx' (got keys={split.files}).")
                val_grid = full_grid._subset(np.asarray(split["val_idx"], dtype=int))
                Xv, sv, _, Yv = val_grid.load_matrix(dtype=np.float64)
                val_data = (Xv, sv, Yv)
            emu, extra = _fit_pca_mlp(
                train_grid,
                cfg=cfg["pca_mlp"],
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
            y_pred = inv_log10_transform(invert_y_standardizer(y_pred_log_z, y_mean, y_std), eps=eps)
            t_pred = time.perf_counter() - t_pred0
        elif name == "gplfr":
            gplfr_cfg = cfg["gplfr"]
            ## Optional per-emulator subsampling (matches sweep n_train behavior)
            gplfr_train = train_grid
            n_train_gplfr = gplfr_cfg.get("n_train", None)
            if n_train_gplfr is not None:
                n_train_gplfr = int(n_train_gplfr)
                if n_train_gplfr < train_grid.n_models:
                    gplfr_train = _balanced_subset(train_grid, n_train_gplfr, seed=int(gplfr_cfg.get("random_seed", 0)))
                    print(f"[pyxopto.train] gplfr n_train={gplfr_train.n_models} (subsampled from {train_grid.n_models})")
            ## Recompute preprocessors on the (possibly subsampled) training data
            X_g, _, _, Y_g = gplfr_train.load_matrix(dtype=np.float64)
            Y_g_log = log10_transform(Y_g, eps=eps)
            xm, xs, ym, ys = fit_preprocessors(X_g, Y_g_log)
            X_test_z_g = apply_x_standardizer(X_test, xm, xs)

            t_train0 = time.perf_counter()
            val_data = None
            pat = gplfr_cfg.get("early_stop_patience_evals", None)
            if pat is not None and int(pat) > 0:
                if "val_idx" not in split.files:
                    raise ValueError(f"gplfr early stopping requires split NPZ to contain 'val_idx' (got keys={split.files}).")
                val_grid = full_grid._subset(np.asarray(split["val_idx"], dtype=int))
                Xv, sv, _, Yv = val_grid.load_matrix(dtype=np.float64)
                val_data = (Xv, sv, Yv)
            emu, extra = _fit_gplfr(
                gplfr_train,
                cfg=gplfr_cfg,
                x_mean=xm,
                x_std=xs,
                y_mean=ym,
                y_std=ys,
                eps=eps,
                val_data=val_data,
            )
            t_train = time.perf_counter() - t_train0

            t_pred0 = time.perf_counter()
            y_pred_log_z = emu.predict(X_test_z_g, s_test)
            y_pred = inv_log10_transform(invert_y_standardizer(y_pred_log_z, ym, ys), eps=eps)
            t_pred = time.perf_counter() - t_pred0
        elif name == "simplex":
            from .simplex import SimplexInterpolator

            t_train0 = time.perf_counter()
            emu_s = SimplexInterpolator(train_grid._X_full, Y_train, eps=eps)
            t_train = time.perf_counter() - t_train0
            t_pred0 = time.perf_counter()
            y_pred = emu_s.predict(test_grid._X_full)
            t_pred = time.perf_counter() - t_pred0
            extra = {}
        else:
            raise ValueError(f"Unknown emulator {name!r}")

        met = metrics_mod.evaluate_metrics(y_true, y_pred, eps=eps)
        t_total = time.perf_counter() - t0
        print(f"[pyxopto.train] emulator={name} done train={t_train:.3f}s pred={t_pred:.3f}s total={t_total:.3f}s")
        results[name] = {"metrics": met, **extra}

        if cfg.get("save_model", False):
            import torch
            pre = (xm, xs, ym, ys) if name == "gplfr" else (x_mean, x_std, y_mean, y_std)
            bundle = {
                "emulator_name": name,
                "emulator": emu if name != "simplex" else emu_s,
                "preprocessors": {"x_mean": pre[0], "x_std": pre[1], "y_mean": pre[2], "y_std": pre[3]},
                "eps": eps,
            }
            model_path = out_dir / f"{name}_model.pkl"
            torch.save(bundle, model_path)
            print(f"[pyxopto.train] saved model to {model_path}")

    (out_dir / "cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[pyxopto.train] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
