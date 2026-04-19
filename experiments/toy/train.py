"""Toy benchmark runner (PCA+GP vs GPLFR).

Run with:
  micromamba run -n xapm python -m gplfr.experiments.toy.train config=/abs/path/to/config.yaml
Or via wrapper:
  micromamba run -n xapm python scripts/toy_run.py config=/abs/path/to/config.yaml

Optional CLI overrides (top-level only):
  data_path=/path/to/data.npz  split_path=/path/to/split.npz  out_dir=/path/to/outputs
"""

from __future__ import annotations

import json
import sys
import time
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
from .pca_gp import PCAGP


def _warn_yellow(msg: str) -> None:
    print(f"\033[33m{msg}\033[0m")


def _fit_pca_gp(
    train: MechanismToyGrid,
    *,
    cfg: dict[str, Any],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    val_data: tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[PCAGP, dict[str, Any]]:
    n_components = int(cfg.get("n_components", 25))
    train_hyperparams = bool(cfg.get("train_hyperparams", False))
    kernel = str(cfg.get("kernel", "rbf"))
    sigma_xi2_mode = str(cfg.get("sigma_xi2_mode", "shared")).lower()
    sigma_xi2 = cfg.get("sigma_xi2", 1.0)
    device = cfg.get("device", "auto")
    dtype = str(cfg.get("dtype", "float64")).lower()
    random_seed = int(cfg.get("random_seed", 0))

    X, axis, Y = train.load_matrix(dtype=np.float32)
    Xz = apply_x_standardizer(X, x_mean, x_std)
    Yz = apply_y_standardizer(Y, y_mean, y_std)
    Xv = Yv = Yv_sig = None
    if val_data is not None:
        Xv = apply_x_standardizer(np.asarray(val_data[0], dtype=np.float64), x_mean, x_std)
        Yv = apply_y_standardizer(np.asarray(val_data[1], dtype=np.float64), y_mean, y_std)
        if len(val_data) == 3:
            Yv_sig = apply_y_standardizer(np.asarray(val_data[2], dtype=np.float64), y_mean, y_std)
    patience = cfg.get("early_stop_patience_evals", None)
    if patience is not None and int(patience) > 0 and (Xv is None or Yv is None):
        raise ValueError("pca_gp early stopping requires val_data (provide a split with val_idx).")

    max_allowed = int(min(max(int(Yz.shape[0]) - 1, 0), int(Yz.shape[1])))
    if max_allowed < 1:
        raise ValueError(f"[pca_gp] PCA requires n_train>=2 and n_features>=1, got {Yz.shape}")
    if n_components > max_allowed:
        _warn_yellow(f"[WARN][pca_gp] n_components={n_components} > max_allowed={max_allowed}; using {max_allowed}")
        n_components = max_allowed

    emu = PCAGP.from_spectra(
        grid_points=Xz,
        wavelength_nm=axis,
        fluxes=Yz,
        n_components=n_components,
        kernel=kernel,  # type: ignore[arg-type]
        sigma_xi2_mode=sigma_xi2_mode,  # type: ignore[arg-type]
        sigma_xi2=sigma_xi2,
        device=device,
        dtype=dtype,
    )
    print(f"[toy.train] emulator=pca_gp device={emu.torch_device} dtype={emu.torch_dtype}")
    if train_hyperparams:
        opt_kwargs = dict(cfg.get("opt_kwargs", {}) or {})
        opt_kwargs.setdefault("restart_seed", random_seed)
        opt_kwargs.setdefault(
            "early_stop_patience_evals",
            None if cfg.get("early_stop_patience_evals", None) is None else int(cfg["early_stop_patience_evals"]),
        )
        opt_kwargs.setdefault("early_stop_min_delta", float(cfg.get("early_stop_min_delta", 0.0)))
        opt_kwargs.setdefault("early_stop_metric", str(cfg.get("early_stop_metric", "rmse_val_obs")))
        opt_kwargs.setdefault("log_every", int(cfg.get("log_every", 100)))
        opt_kwargs.setdefault("verbose", bool(cfg.get("verbose", False)))
        opt_kwargs.setdefault("val_X", Xv)
        opt_kwargs.setdefault("val_Y", Yv)
        opt_kwargs.setdefault("val_Y_sig", Yv_sig)
        opt_kwargs.setdefault("val_y_std", y_std.astype(np.float64))
        emu.train(**opt_kwargs)

    fit_metrics = None
    if bool(cfg.get("log_fit_metrics", False)):
        train_pred = invert_y_standardizer(emu.predict_flux(Xz), y_mean, y_std)
        fit_metrics = {
            "train": metrics_mod.evaluate_metrics(
                np.asarray(Y, dtype=np.float64),
                train_pred,
                y_sig_true=(None if train.Y_sig is None else np.asarray(train.Y_sig, dtype=np.float64)),
            )
        }
        if val_data is not None and Xv is not None:
            val_pred = invert_y_standardizer(emu.predict_flux(Xv), y_mean, y_std)
            fit_metrics["val"] = metrics_mod.evaluate_metrics(
                np.asarray(val_data[1], dtype=np.float64),
                val_pred,
                y_sig_true=(None if len(val_data) != 3 else np.asarray(val_data[2], dtype=np.float64)),
            )

    sigma_xi2_fit = emu.sigma_xi2
    hyperparams = {
        "kernel": kernel,
        "sigma_xi2_mode": str(sigma_xi2_mode),
        "sigma_xi2": (float(sigma_xi2_fit) if not isinstance(sigma_xi2_fit, np.ndarray) else [float(x) for x in sigma_xi2_fit]),
        "sigma_m2": [float(x) for x in emu.variances],
        "lengthscales": [[float(x) for x in row] for row in emu.lengthscales],
    }
    return emu, {
        "n_components": n_components,
        "kernel": kernel,
        "sigma_xi2_mode": str(sigma_xi2_mode),
        "train_hyperparams": train_hyperparams,
        "hyperparams": hyperparams,
        **({} if fit_metrics is None else {"fit_metrics": fit_metrics}),
    }


def _fit_gplfr(
    train: MechanismToyGrid,
    *,
    cfg: dict[str, Any],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    val_data: tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from .gplfr import GPLFR, PRIOR_CFG as GPLFR_PRIOR_CFG
    import torch

    dtype = str(cfg.get("dtype", "float64")).lower()
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype)
    if torch_dtype is None:  # pragma: no cover
        raise ValueError(f"Unknown gplfr.dtype {dtype!r} (expected 'float32' or 'float64')")

    b_keys = {
        "B_mode",
        "B_ell_mode",
        "B_ell_fixed",
        "B_ell",
        "B_rho_mode",
        "B_rho_fixed",
        "B_rho",
        "B_jitter",
        "B_stop_grad_U",
        "output_hw",
        "output_coords",
    }
    if any(k in cfg for k in b_keys):
        raise ValueError("toy GPLFR no longer supports B-mode output correlations (remove B_* keys from config).")
    model_kwargs = {
        "latent_dim": int(cfg.get("latent_dim", 8)),
        "kernel": str(cfg.get("kernel", "matern52")),
        "ell_mode": str(cfg.get("ell_mode", "shared")),
        "ell_fixed": cfg.get("ell_fixed", cfg.get("ell", None)),
        "sigma_f_mode": (None if cfg.get("sigma_f_mode", None) is None else str(cfg["sigma_f_mode"])),
        "sigma_f_fixed": (None if cfg.get("sigma_f_fixed", cfg.get("sigma_f", None)) is None else float(cfg.get("sigma_f_fixed", cfg.get("sigma_f")))),
        "tau": float(cfg.get("tau", 1.0)),
        "data_fit_scale": float(cfg.get("data_fit_scale", 1.0)),
        "sigma_xi2_mode": str(cfg.get("sigma_xi2_mode", "float")),
        "sigma_xi2": float(cfg.get("sigma_xi2", 1e-6)),
        "jitter": float(cfg.get("jitter", 0.0)),
        "dtype": torch_dtype,
        "device": cfg.get("device", "auto"),
    }
    emu = GPLFR(**model_kwargs)
    prior_cfg = cfg.get("prior_cfg", None) or {}
    if not isinstance(prior_cfg, dict):
        raise TypeError("gplfr.prior_cfg must be a mapping/dict")
    GPLFR_PRIOR_CFG.update(prior_cfg)
    n_restarts = int(cfg.get("n_restarts", 0) or 0)

    X = np.asarray(train.X, dtype=np.float64)
    Y = np.asarray(train._Y, dtype=np.float64)
    Xz = apply_x_standardizer(X, x_mean, x_std)
    Yz = apply_y_standardizer(Y, y_mean, y_std)

    Dy = int(Yz.shape[1])
    d_eff = cfg.get("D_eff", cfg.get("d_eff", None))
    if d_eff is not None:
        model_kwargs["data_fit_scale"] = float(d_eff) / float(Dy)
    model_kwargs["data_fit_scale"] = float(model_kwargs["data_fit_scale"]) * float(cfg.get("data_fit_scale_mult", 1.0))
    emu.data_fit_scale = float(model_kwargs["data_fit_scale"])
    print(f"[toy.train] emulator=gplfr device={emu.device} dtype={emu.dtype} data_fit_scale={emu.data_fit_scale:.6g}")

    max_allowed = int(min(max(int(Yz.shape[0]) - 1, 0), int(Yz.shape[1])))
    if max_allowed < 1:
        raise ValueError(f"[gplfr] PCA requires n_train>=2 and n_features>=1, got {Yz.shape}")
    if int(model_kwargs["latent_dim"]) > max_allowed:
        _warn_yellow(f"[WARN][gplfr] latent_dim={model_kwargs['latent_dim']} > max_allowed={max_allowed}; using {max_allowed}")
        model_kwargs["latent_dim"] = max_allowed
        emu.latent_dim = max_allowed

    Xv = Yv = Yv_sig = None
    if val_data is not None:
        Xv = apply_x_standardizer(np.asarray(val_data[0], dtype=np.float64), x_mean, x_std)
        Yv = apply_y_standardizer(np.asarray(val_data[1], dtype=np.float64), y_mean, y_std)
        if len(val_data) == 3:
            Yv_sig = apply_y_standardizer(np.asarray(val_data[2], dtype=np.float64), y_mean, y_std)
    patience = cfg.get("early_stop_patience_evals", None)
    if patience is not None and int(patience) > 0 and (Xv is None or Yv is None):
        raise ValueError("gplfr early stopping requires val_data (provide a split with val_idx).")

    pca_init_cfg = cfg.get("pca_init", None) or {}
    pca_init = {k: v for k, v in pca_init_cfg.items() if k != "enabled"} if bool(pca_init_cfg.get("enabled", False)) else None

    seed0 = int(cfg.get("random_seed", 0))
    homotopy = cfg.get("homotopy", None)
    if homotopy is None and cfg.get("annealing", None) is not None:
        homotopy = {"likelihood_tempering": cfg.get("annealing", None)}
    fit_kwargs = {
        "num_steps": int(cfg.get("num_steps", 5000)),
        "lr_Z": float(cfg.get("lr_Z", 1.0e-2)),
        "lr_global": float(cfg.get("lr_global", 1.0e-4)),
        "lr_gamma": (float(cfg["lr_gamma"]) if cfg.get("lr_gamma") is not None else None),
        "frozen_Z_start": float(cfg.get("frozen_Z_start", 0.0)),
        "early_stop_patience_evals": (None if cfg.get("early_stop_patience_evals", 10) is None else int(cfg.get("early_stop_patience_evals", 10))),
        "early_stop_min_delta": float(cfg.get("early_stop_min_delta", 0.0)),
        "early_stop_metric": str(cfg.get("early_stop_metric", "rmse_val_obs")),
        "log_every": int(cfg.get("log_every", 50)),
        "record_loss_curve": bool(cfg.get("record_loss_curve", cfg.get("log_every", None) is not None)),
        "seed": seed0,
        "verbose": bool(cfg.get("verbose", True)),
        "linear_trend_cfg": cfg.get("linear_trend", None),
        "pca_init": pca_init,
        "tau_schedule": cfg.get("tau_schedule", None),
        "homotopy": homotopy,
        "val_X": Xv,
        "val_Y": Yv,
        "val_Y_sig": Yv_sig,
        "val_y_std": y_std.astype(np.float64),
    }

    best = best_fit = best_samples = best_seed = None
    for r in range(n_restarts + 1):
        fit_kwargs["seed"] = seed0 + r
        fit_r = emu.fit(Xz, Yz, **fit_kwargs)
        if best is None or float(fit_r.final_loss) < float(best):
            best = float(fit_r.final_loss)
            best_fit = fit_r
            best_seed = int(fit_kwargs["seed"])
            best_samples = {k: v.detach().clone() for k, v in emu.posterior_samples_.items()}  # type: ignore[union-attr]
    assert best_fit is not None and best_samples is not None and best_seed is not None
    emu.posterior_samples_ = best_samples
    emu.fit_result_ = best_fit
    emu._build_cached_state()

    sigma = float(emu.posterior_samples_["sigma"][0].detach().cpu().item()) if emu.posterior_samples_ is not None else None
    fit_result_out = {k: v for k, v in best_fit.__dict__.items() if v is not None}
    fit_result_out.update({"n_restarts": int(n_restarts), "best_seed": int(best_seed), "sigma": sigma})

    fit_kwargs_logged = {k: v for k, v in fit_kwargs.items() if k not in ("val_X", "val_Y")}
    extra = {
        "fit_result": fit_result_out,
        "cfg": {**{k: v for k, v in model_kwargs.items() if k != "dtype"}, "dtype": dtype, "n_restarts": int(n_restarts), **fit_kwargs_logged},
    }
    return emu, extra


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

    full_grid = MechanismToyGrid.load(data_path)
    split = np.load(split_path)
    train_grid = full_grid._subset(np.asarray(split["train_idx"], dtype=int))
    val_grid = full_grid._subset(np.asarray(split["val_idx"], dtype=int)) if "val_idx" in split.files else None
    eval_idx = split["val_idx"] if eval_set == "val" else split["test_idx"]
    test_grid = full_grid._subset(np.asarray(eval_idx, dtype=int))

    max_train = cfg.get("max_train", None)
    if max_train is not None and int(max_train) < train_grid.n_models:
        idx = np.random.default_rng(0).choice(train_grid.n_models, size=int(max_train), replace=False)
        train_grid = train_grid._subset(np.sort(idx))

    X_train = np.asarray(train_grid.X, dtype=np.float64)
    Y_train = np.asarray(train_grid._Y, dtype=np.float64)
    x_mean, x_std, y_mean, y_std = fit_preprocessors(X_train, Y_train)
    prep = cfg.get("preprocess", {}) or {}
    if str(prep.get("x", "zscore")).lower() == "none":
        x_mean = np.zeros_like(x_mean)
        x_std = np.ones_like(x_std)
    if str(prep.get("y", "zscore")).lower() == "none":
        y_mean = np.zeros_like(y_mean)
        y_std = np.ones_like(y_std)

    X_test = np.asarray(test_grid.X, dtype=np.float64)
    y_true = np.asarray(test_grid._Y, dtype=np.float64)
    y_sig_true = None if test_grid.Y_sig is None else np.asarray(test_grid.Y_sig, dtype=np.float64)
    X_test_z = apply_x_standardizer(X_test, x_mean, x_std)

    print(f"[toy.train] n_train={train_grid.n_models} n_eval={test_grid.n_models} eval_set={eval_set}")

    results: dict[str, Any] = {}
    fitted_params_by_emulator: dict[str, Any] = {}
    for name in cfg["emulators"]:
        t0 = time.perf_counter()
        print(f"[toy.train] emulator={name} train...")

        if name == "pca_gp":
            t_train0 = time.perf_counter()
            val_sig = None if val_grid is None or val_grid.Y_sig is None else np.asarray(val_grid.Y_sig, dtype=np.float64)
            emu, extra = _fit_pca_gp(
                train_grid,
                cfg=cfg["pca_gp"],
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                val_data=(
                    None
                    if val_grid is None
                    else (
                        np.asarray(val_grid.X, dtype=np.float64),
                        np.asarray(val_grid._Y, dtype=np.float64),
                        *(tuple() if val_sig is None else (val_sig,)),
                    )
                ),
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
            diagnostics = {"pca_score_r2": r2, "pca_score_r2_mean": float(np.nanmean(r2))}
            s2 = emu.sigma_xi2
            fitted_params = {
                "kernel": str(emu.kernel),
                "n_components": int(emu.n_components),
                "sigma_xi2_mode": str(emu.sigma_xi2_mode),
                "sigma_xi2": (float(s2) if not isinstance(s2, np.ndarray) else [float(x) for x in s2]),
                "variance": [float(x) for x in emu.variances],
                "lengthscales": [[float(x) for x in row] for row in emu.lengthscales],
            }

        elif name == "gplfr":
            if val_grid is None:
                raise ValueError("gplfr requires val_idx in split NPZ (for optional early stopping).")
            t_train0 = time.perf_counter()
            val_sig = None if val_grid.Y_sig is None else np.asarray(val_grid.Y_sig, dtype=np.float64)
            emu, extra = _fit_gplfr(
                train_grid,
                cfg=cfg["gplfr"],
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                val_data=(
                    np.asarray(val_grid.X, dtype=np.float64),
                    np.asarray(val_grid._Y, dtype=np.float64),
                    *(tuple() if val_sig is None else (val_sig,)),
                ),
            )
            t_train = time.perf_counter() - t_train0

            t_pred0 = time.perf_counter()
            y_pred_z = emu.predict(X_test_z)
            y_pred = invert_y_standardizer(y_pred_z, y_mean, y_std)
            t_pred = time.perf_counter() - t_pred0
            diagnostics = None
            post = getattr(emu, "posterior_samples_", None) or {}
            sigma = None if "sigma" not in post else float(post["sigma"][0].detach().cpu().item())
            ell = None if "ell" not in post else post["ell"][0].detach().cpu().numpy().tolist()
            sigma_f = None if "sigma_f" not in post else post["sigma_f"][0].detach().cpu().numpy().tolist()
            sigma_xi2 = None
            if "sigma_xi2" in post:
                s2 = post["sigma_xi2"][0].detach().cpu().numpy()
                sigma_xi2 = float(s2) if s2.shape == () else s2.tolist()
            Z_T_rms = None
            if "Z_T" in post:
                Z_T = post["Z_T"][0].detach().cpu().numpy()
                Z_T_rms = float(np.sqrt(np.mean(Z_T**2)))
            fitted_params = {
                "tau": float(cfg["gplfr"].get("tau", 1.0)),
                "sigma_xi2_mode": str(cfg["gplfr"].get("sigma_xi2_mode", "float")),
                "sigma_xi2": float(cfg["gplfr"].get("sigma_xi2", 1e-6)) if sigma_xi2 is None else sigma_xi2,
                "sigma": sigma,
                "ell": ell,
                "sigma_f": sigma_f,
                "Z_T_rms": Z_T_rms,
            }

        else:
            raise ValueError(f"Unknown emulator {name!r}")

        met = metrics_mod.evaluate_metrics(np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64), y_sig_true=y_sig_true)
        if diagnostics and "pca_score_r2_mean" in diagnostics:
            met["pca_score_r2_mean"] = float(diagnostics["pca_score_r2_mean"])

        t_total = time.perf_counter() - t0
        print(f"[toy.train] emulator={name} done train={t_train:.3f}s pred={t_pred:.3f}s total={t_total:.3f}s")

        results[name] = {"metrics": met, **extra, **({} if diagnostics is None else {"diagnostics": diagnostics})}
        fitted_params_by_emulator[name] = fitted_params

    (out_dir / "cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    payload = {
        "meta": {
            "data_path": str(data_path),
            "split_path": str(split_path),
            "out_dir": str(out_dir),
            "n_train": int(train_grid.n_models),
            "eval_set": eval_set,
            "n_eval": int(test_grid.n_models),
            "n_pix": int(test_grid.n_pix),
        },
        "results": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "fitted_params.json").write_text(
        json.dumps({"meta": payload["meta"], "fitted_params": fitted_params_by_emulator}, indent=2),
        encoding="utf-8",
    )
    print(f"[toy.train] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
