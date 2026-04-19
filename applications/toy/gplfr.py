r"""GPLFR for toy benchmark: latent GP + linear decoder with collapsed weights.

Assumes *external* preprocessing:
  - X is standardized (z-scored).
  - Y is in the desired training space (for PHOENIX we z-score spectra).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.infer.autoguide.initialization import init_to_median, init_to_sample
import torch
from torch import Tensor

# Register XPU backend if Intel Extension for PyTorch is available
try:
    import intel_extension_for_pytorch as ipex  # pyright: ignore[reportMissingImports]
except ImportError:
    pass

from gplfr.gplfr.kernels import apply_kernel
from gplfr.gplfr.linear_trend import build_design_matrix, fit_ridge
from ._mvn_cholesky import MVNCholesky


PRIOR_CFG = {
    "ell_prior_type": "lognormal",
    "ell_prior_loc": 1.0,
    "ell_prior_scale": 0.3,
    "ell_inv_prior_concentration": 2.0,
    "sigma_prior_scale": 0.5,
    "sigma_f_prior_scale": 1.0,
    "sigma_xi2_prior_scale": 2.0,
}


KernelName = Literal["rbf", "matern32", "matern52"]
EllMode = Literal["shared", "per_latent", "fixed"]
SigmaFMode = Literal["shared", "per_latent", "per_group", "fixed"]
SigmaXi2Mode = Literal["float", "shared", "per_latent"]
EllPriorType = Literal["lognormal", "gamma_inv"]


@dataclass
class GPLFRFitResult:
    num_train_steps: int
    final_loss: float
    loss_curve: list[dict[str, Any]] | None = None


class GPLFR:
    def __init__(
        self,
        *,
        latent_dim: int = 8,
        kernel: KernelName = "matern52",
        ell_mode: EllMode = "shared",
        ell_fixed: float | list[float] | np.ndarray | None = None,
        sigma_f_mode: SigmaFMode | None = "per_latent",
        sigma_f_fixed: float | None = None,
        tau: float = 1.0,
        data_fit_scale: float = 1.0,
        sigma_xi2_mode: SigmaXi2Mode = "float",
        sigma_xi2: float = 1e-6,
        jitter: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if sigma_xi2 < 0:
            raise ValueError("sigma_xi2 must be >= 0")
        if jitter < 0:
            raise ValueError("jitter must be >= 0")
        if tau <= 0:
            raise ValueError("tau must be > 0")
        if data_fit_scale <= 0:
            raise ValueError("data_fit_scale must be > 0")
        if sigma_f_mode == "per_group":
            raise NotImplementedError("sigma_f_mode='per_group' is not implemented (use 'per_latent').")

        self.latent_dim = int(latent_dim)
        self.kernel: KernelName = kernel
        self.ell_mode: EllMode = ell_mode
        self.ell_fixed: float | np.ndarray | None = None
        if ell_fixed is not None:
            e = np.asarray(ell_fixed, dtype=float)
            self.ell_fixed = float(e) if e.shape == () else e.reshape(-1)
        if self.ell_mode == "fixed" and self.ell_fixed is None:
            raise ValueError("ell_fixed is required when ell_mode='fixed'")

        self.sigma_f_mode = None if sigma_f_mode is None else str(sigma_f_mode)
        self.sigma_f_fixed = None if sigma_f_fixed is None else float(sigma_f_fixed)
        if self.sigma_f_mode == "fixed" and self.sigma_f_fixed is None:
            raise ValueError("sigma_f_fixed is required when sigma_f_mode='fixed'")
        self.tau = float(tau)
        self.data_fit_scale = float(data_fit_scale)
        self.sigma_xi2_mode: SigmaXi2Mode = str(sigma_xi2_mode)  # type: ignore[assignment]
        self.sigma_xi2 = float(sigma_xi2)
        self.sigma_xi2_extra = 0.0
        self.jitter = float(jitter)
        self._eps = 1e-12
        if self.sigma_xi2_mode not in ("float", "shared", "per_latent"):
            raise ValueError("sigma_xi2_mode must be one of: float | shared | per_latent")
        if self.sigma_xi2_mode != "float" and not (self.sigma_xi2 > 0):
            raise ValueError("sigma_xi2 must be > 0 when sigma_xi2_mode is learned")

        self.dtype = dtype
        if isinstance(device, str) and device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.X_train_: Tensor | None = None
        self.Y_train_: Tensor | None = None
        self.linear_trend_: dict[str, Any] | None = None
        self.posterior_samples_: dict[str, Tensor] | None = None
        self._cached_state_: list[dict[str, Tensor]] | None = None
        self.fit_result_: GPLFRFitResult | None = None
        self._ell_min: np.ndarray | None = None

    # ----------------------------
    # Public API
    # ----------------------------

    def fit(
        self,
        X: np.ndarray | Tensor,
        Y: np.ndarray | Tensor,
        *,
        num_steps: int = 5000,
        lr_Z: float = 1.0e-2,
        lr_global: float = 1.0e-4,
        lr_gamma: float | None = None,
        frozen_Z_start: float = 0.0,
        log_every: int = 50,
        record_loss_curve: bool = False,
        pca_init: dict[str, Any] | None = None,
        linear_trend_cfg: dict[str, Any] | None = None,
        tau_schedule: dict[str, Any] | None = None,
        homotopy: dict[str, Any] | None = None,
        annealing: dict[str, Any] | None = None,
        val_X: np.ndarray | Tensor | None = None,
        val_Y: np.ndarray | Tensor | None = None,
        val_Y_sig: np.ndarray | Tensor | None = None,
        val_y_std: np.ndarray | Tensor | None = None,
        early_stop_metric: str = "mae_val",
        early_stop_patience_evals: int | None = 10,
        early_stop_min_delta: float = 0.0,
        seed: int = 0,
        verbose: bool = True,
    ) -> GPLFRFitResult:
        pyro.set_rng_seed(int(seed))
        pyro.clear_param_store()
        self._ell_min = None

        X_t = self._as_tensor(X)
        Y_t = self._as_tensor(Y)
        if X_t.ndim != 2 or Y_t.ndim != 2:
            raise ValueError(f"Expected X,Y as 2D arrays; got X={tuple(X_t.shape)} Y={tuple(Y_t.shape)}")
        if X_t.shape[0] != Y_t.shape[0]:
            raise ValueError("X and Y must have the same number of rows")

        self.X_train_ = X_t
        self.Y_train_ = Y_t
        self._ell_min = min_lengthscale_1nn(X_t.detach().cpu().numpy(), factor=1.0)

        # Optional linear trend/mean removal (fit in output space, subtract before GPLFR)
        lt_cfg = linear_trend_cfg or {}
        if lt_cfg.get("enabled", False):
            design_in = lt_cfg.get("design", {}) or {}
            design_cfg = {
                "intercept": design_in.get("intercept", True),
                "inputs": design_in.get("inputs", True),
                "sim_onehot": False,
            }
            lambda_reg = float(lt_cfg.get("lambda", 1.0e-3))
            s0 = X_t.new_zeros((X_t.shape[0],), dtype=torch.long)
            H = build_design_matrix(X_t, s0, n_sim_types=1, design_cfg=design_cfg)
            Y3 = Y_t.unsqueeze(1)
            Gamma = fit_ridge(H, Y3, lambda_reg=lambda_reg, field_mask=None)
            self.Y_train_ = Y_t - torch.einsum("np,paf->naf", H, Gamma).squeeze(1)
            self.linear_trend_ = {"Gamma": Gamma, "design_cfg": design_cfg, "lambda_reg": lambda_reg}
        else:
            self.linear_trend_ = None

        Xv = None if val_X is None else self._as_tensor(val_X)
        Yv = None if val_Y is None else self._as_tensor(val_Y)
        if (Xv is None) != (Yv is None):
            raise ValueError("Pass val_X and val_Y together (or neither).")
        if Xv is not None and (Xv.ndim != 2 or Yv.ndim != 2 or Xv.shape[0] != Yv.shape[0]):
            raise ValueError(f"Invalid val shapes: X={tuple(Xv.shape)} Y={tuple(Yv.shape)}")

        metric = str(early_stop_metric).lower()
        Ys_v = None if val_Y_sig is None else self._as_tensor(val_Y_sig)
        y_std_v = None if val_y_std is None else self._as_tensor(val_y_std)
        if metric != "mae_val":
            if Xv is None or Yv is None:
                raise ValueError("early_stop_metric != 'mae_val' requires val_X and val_Y.")
            if y_std_v is None:
                raise ValueError("early_stop_metric != 'mae_val' requires val_y_std.")
            if y_std_v.ndim != 1 or int(y_std_v.shape[0]) != int(Yv.shape[1]):
                raise ValueError(f"Invalid val_y_std shape: {tuple(y_std_v.shape)} (expected ({int(Yv.shape[1])},))")
            if metric in ("rmse_sig", "rmse_val_sig"):
                if Ys_v is None:
                    raise ValueError("rmse_val_sig requires val_Y_sig.")
                if Ys_v.ndim != 2 or Ys_v.shape != Yv.shape:
                    raise ValueError(f"Invalid val_Y_sig shape: {tuple(Ys_v.shape)} (expected {tuple(Yv.shape)})")

        init_values = self._pca_init_Z(**(pca_init or {})) if pca_init is not None else None
        tau0 = float(self.tau)
        beta0 = float(self.data_fit_scale)
        xi_extra0 = float(self.sigma_xi2_extra)
        sched = self._parse_tau_schedule(tau_schedule, fallback=tau0)
        h = self._parse_homotopy(homotopy, fallback_beta=beta0)
        if h is None and annealing is not None:
            h = {"likelihood_tempering": self._parse_likelihood_tempering(annealing, fallback=beta0), "nugget_schedule": None}
        if sched is not None:
            self.tau = float(sched["start"])
        lt = None if h is None else h.get("likelihood_tempering", None)
        ns = None if h is None else h.get("nugget_schedule", None)
        if lt is not None:
            self.data_fit_scale = float(lt["start"])
        if ns is not None:
            self.sigma_xi2_extra = float(ns["start"])
        guide, final_loss, steps_run, loss_curve = self._fit_map(
            num_steps=num_steps,
            lr_Z=lr_Z,
            lr_global=lr_global,
            lr_gamma=lr_gamma,
            frozen_Z_start=frozen_Z_start,
            log_every=log_every,
            verbose=verbose,
            init_values=init_values,
            record_loss_curve=record_loss_curve,
            tau_schedule=sched,
            likelihood_tempering=lt,
            nugget_schedule=ns,
            val_X=Xv,
            val_Y=Yv,
            val_Y_sig=Ys_v,
            val_y_std=y_std_v,
            early_stop_metric=metric,
            early_stop_patience_evals=early_stop_patience_evals,
            early_stop_min_delta=early_stop_min_delta,
        )
        if sched is None:
            self.tau = tau0
        if lt is None:
            self.data_fit_scale = beta0
        if ns is None:
            self.sigma_xi2_extra = xi_extra0
        self.fit_result_ = GPLFRFitResult(int(steps_run), float(final_loss), loss_curve=loss_curve)

        params = guide(self.Y_train_, self.X_train_)
        samples: dict[str, Tensor] = {k: v.detach().unsqueeze(0) for k, v in params.items() if isinstance(v, Tensor)}
        if "ell_raw" in samples or "inv_ell_raw" in samples:
            if "ell_raw" in samples:
                assert self._ell_min is not None
                ell_min = X_t.new_tensor(self._ell_min)
                samples["ell"] = ell_min + samples["ell_raw"]
                del samples["ell_raw"]
            if "inv_ell_raw" in samples:
                samples["ell"] = (samples["inv_ell_raw"] + self._eps).reciprocal()
                del samples["inv_ell_raw"]
        self.posterior_samples_ = samples
        self._build_cached_state()
        return self.fit_result_

    def _parse_homotopy(self, cfg: dict[str, Any] | None, *, fallback_beta: float) -> dict[str, dict[str, float] | None] | None:
        cfg = cfg or {}
        lt = self._parse_likelihood_tempering(cfg.get("likelihood_tempering", None), fallback=float(fallback_beta))
        ns = self._parse_nugget_schedule(cfg.get("nugget_schedule", None))
        return None if lt is None and ns is None else {"likelihood_tempering": lt, "nugget_schedule": ns}

    def _parse_likelihood_tempering(self, cfg: dict[str, Any] | None, *, fallback: float) -> dict[str, float] | None:
        cfg = cfg or {}
        if not bool(cfg.get("enabled", False)):
            return None
        beta_range = cfg.get("range", None)
        if beta_range is None or not isinstance(beta_range, (list, tuple)) or len(beta_range) != 2:
            raise ValueError("homotopy.likelihood_tempering.range must be a 2-element list [start, end].")
        start, end = float(beta_range[0]), float(beta_range[1])
        if start < 0 or end < 0:
            raise ValueError("homotopy.likelihood_tempering.range values must be >= 0.")
        frac = float(cfg.get("num_steps_fraction", 1.0))
        if frac <= 0:
            raise ValueError("homotopy.likelihood_tempering.num_steps_fraction must be > 0.")
        return {"start": start, "end": end, "frac": frac, "fallback": float(fallback)}

    def _parse_nugget_schedule(self, cfg: dict[str, Any] | None) -> dict[str, float] | None:
        cfg = cfg or {}
        if not bool(cfg.get("enabled", False)):
            return None
        s2_range = cfg.get("range", None)
        if s2_range is None or not isinstance(s2_range, (list, tuple)) or len(s2_range) != 2:
            raise ValueError("homotopy.nugget_schedule.range must be a 2-element list [start, end].")
        start, end = float(s2_range[0]), float(s2_range[1])
        if start <= 0 or end <= 0:
            raise ValueError("homotopy.nugget_schedule.range values must be > 0 (log schedule).")
        frac = float(cfg.get("num_steps_fraction", 1.0))
        if frac <= 0:
            raise ValueError("homotopy.nugget_schedule.num_steps_fraction must be > 0.")
        return {"start": start, "end": end, "frac": frac}

    def _beta_at(self, step: int, total: int, sched: dict[str, float]) -> float:
        start, end, frac = float(sched["start"]), float(sched["end"]), float(sched["frac"])
        n = max(int(total * frac), 1)
        if step >= n:
            return end
        w = float(step) / float(n)
        return (1.0 - w) * start + w * end

    def _s2_extra_at(self, step: int, total: int, sched: dict[str, float]) -> float:
        start, end, frac = float(sched["start"]), float(sched["end"]), float(sched["frac"])
        n = max(int(total * frac), 1)
        if step >= n:
            return end
        w = float(step) / float(n)
        return float(np.exp((1.0 - w) * np.log(start) + w * np.log(end)))

    @torch.no_grad()
    def predict(self, X_new: np.ndarray | Tensor) -> np.ndarray:
        self._require_fitted()
        Xn = self._as_tensor(X_new)
        if Xn.ndim == 1:
            Xn = Xn.unsqueeze(0)

        assert self._cached_state_ is not None
        preds = []
        for state in self._cached_state_:
            Z_mean = self._gp_predict_latents_mean(Xn, state)
            y = Z_mean @ state["mu_W"]
            if self.linear_trend_ is not None:
                s0 = Xn.new_zeros((Xn.shape[0],), dtype=torch.long)
                H = build_design_matrix(Xn, s0, n_sim_types=1, design_cfg=self.linear_trend_["design_cfg"])
                trend = torch.einsum("np,paf->naf", H, self.linear_trend_["Gamma"]).squeeze(1)
                y = y + trend
            preds.append(y)
        return torch.stack(preds, dim=0).mean(dim=0).cpu().numpy()

    # ----------------------------
    # Pyro model (latent GP + collapsed linear decoder)
    # ----------------------------

    def model(self, Y: Tensor, X: Tensor) -> None:
        n_train, d = X.shape
        q = int(self.latent_dim)
        if self.ell_mode == "fixed":
            if self.ell_fixed is None:
                raise RuntimeError("ell_fixed is not set.")
            ell = (
                X.new_full((d,), float(self.ell_fixed))
                if isinstance(self.ell_fixed, float)
                else X.new_tensor(self.ell_fixed, dtype=X.dtype)
            )
        else:
            prior_type = self._ell_prior_type()
            if prior_type == "lognormal":
                if self._ell_min is None:
                    raise RuntimeError("ell_min is not set (call fit first).")
                ell_min = X.new_tensor(self._ell_min)
                ell_raw_loc = self._ell_prior_loc_vector(X, d, default=2.0 * ell_min + self._eps)
                ell_prior_scale = X.new_full((d,), float(PRIOR_CFG["ell_prior_scale"]))
                if self.ell_mode == "shared":
                    ell_raw = pyro.sample("ell_raw", dist.LogNormal(torch.log(ell_raw_loc), ell_prior_scale).to_event(1))
                    ell = ell_min + ell_raw
                elif self.ell_mode == "per_latent":
                    ell_raw = pyro.sample(
                        "ell_raw",
                        dist.LogNormal(torch.log(ell_raw_loc), ell_prior_scale).expand([q, d]).to_event(2),
                    )
                    ell = ell_min + ell_raw
                else:  # pragma: no cover
                    raise ValueError(f"Unknown ell_mode {self.ell_mode!r}")
            else:
                inv_ell_loc = self._ell_prior_loc_vector(X, d, default=X.new_ones((d,)))
                inv_conc = float(PRIOR_CFG["ell_inv_prior_concentration"])
                if inv_conc <= 0:
                    raise ValueError("PRIOR_CFG['ell_inv_prior_concentration'] must be > 0.")
                conc = X.new_full((d,), inv_conc)
                rate = conc / inv_ell_loc
                if self.ell_mode == "shared":
                    inv_ell_raw = pyro.sample("inv_ell_raw", dist.Gamma(conc, rate).to_event(1))
                    ell = (inv_ell_raw + self._eps).reciprocal()
                elif self.ell_mode == "per_latent":
                    inv_ell_raw = pyro.sample(
                        "inv_ell_raw",
                        dist.Gamma(conc, rate).expand([q, d]).to_event(2),
                    )
                    ell = (inv_ell_raw + self._eps).reciprocal()
                else:  # pragma: no cover
                    raise ValueError(f"Unknown ell_mode {self.ell_mode!r}")

        K = apply_kernel(self.kernel, X, ell, X.new_tensor(1.0))
        sf2: Tensor | None = None
        if self.sigma_f_mode == "fixed":
            if self.sigma_f_fixed is None:
                raise RuntimeError("sigma_f_fixed is not set.")
            sf2 = X.new_tensor(self.sigma_f_fixed) ** 2
        elif self.sigma_f_mode == "shared":
            sigma_f = pyro.sample(
                "sigma_f",
                dist.LogNormal(X.new_tensor(0.0), X.new_tensor(float(PRIOR_CFG["sigma_f_prior_scale"]))),
            )
            sf2 = sigma_f**2
        elif self.sigma_f_mode == "per_latent":
            sigma_f = pyro.sample(
                "sigma_f",
                dist.LogNormal(X.new_zeros((q,)), X.new_full((q,), float(PRIOR_CFG["sigma_f_prior_scale"]))).to_event(1),
            )
            sf2 = sigma_f**2

        eye = torch.eye(n_train, device=X.device, dtype=X.dtype)
        if self.sigma_xi2_mode == "float":
            s2 = X.new_tensor(self.sigma_xi2)
        else:
            loc = torch.log(X.new_tensor(self.sigma_xi2 + self._eps))
            scale = X.new_tensor(float(PRIOR_CFG["sigma_xi2_prior_scale"]))
            if self.sigma_xi2_mode == "shared":
                s2 = pyro.sample("sigma_xi2", dist.LogNormal(loc, scale))
            elif self.sigma_xi2_mode == "per_latent":
                s2 = pyro.sample("sigma_xi2", dist.LogNormal(loc, scale).expand([q]).to_event(1))
            else:  # pragma: no cover
                raise ValueError(f"Unknown sigma_xi2_mode {self.sigma_xi2_mode!r}")
        if K.ndim == 2 and s2.ndim == 1:
            K = K.unsqueeze(0).expand(int(s2.shape[0]), -1, -1)
        s2 = s2 + X.new_tensor(float(self.sigma_xi2_extra))
        K = K + eye * s2 if K.ndim == 2 else K + eye.unsqueeze(0) * (s2 if s2.ndim == 0 else s2[:, None, None])
        if sf2 is not None:
            if sf2.ndim != 0 and K.ndim == 2:
                K = K.unsqueeze(0).expand(int(sf2.shape[0]), -1, -1)
            K = K * (sf2 if sf2.ndim == 0 else sf2[:, None, None])
        K = 0.5 * (K + K.transpose(-1, -2))
        L_K = torch.linalg.cholesky(K)

        Z_T = pyro.sample("Z_T", MVNCholesky(loc=X.new_zeros((q, n_train)), scale_tril=L_K).to_event(1))
        Z = Z_T.T

        sigma = pyro.sample("sigma", dist.HalfNormal(X.new_tensor(PRIOR_CFG["sigma_prior_scale"])))
        ll = self._collapsed_loglikelihood(Y, Z, sigma)
        pyro.factor("collapsed_ll", ll * X.new_tensor(float(self.data_fit_scale)))

    # ----------------------------
    # Inference helpers (MAP only)
    # ----------------------------

    def _build_svi(
        self,
        *,
        lr_Z: float,
        lr_global: float,
        lr_gamma: float | None,
        init_values: dict[str, Any] | None,
        guide: AutoDelta | None = None,
    ) -> tuple[AutoDelta, SVI, float]:
        assert self.X_train_ is not None and self.Y_train_ is not None

        def _as_init_values(v: dict[str, Any] | None) -> dict[str, Tensor] | None:
            if v is None:
                return None
            ref = self.X_train_
            return {k: (val if isinstance(val, Tensor) else ref.new_tensor(val)) for k, val in v.items()}

        if guide is None:
            init_t = _as_init_values(init_values)
            init_med = init_to_median()
            init_samp = init_to_sample()

            def init_loc(site: dict[str, Any]) -> Tensor:
                name = site["name"]
                if init_t is not None and name in init_t:
                    return init_t[name]
                return init_samp(site) if name == "Z_T" else init_med(site)

            guide = AutoDelta(self.model, init_loc_fn=init_loc)

        def optim_args(param_name: str) -> dict[str, float]:
            name = param_name.rsplit(".", 1)[-1]
            return {"lr": float(lr_Z)} if name == "Z_T" else {"lr": float(lr_global)}

        optimizer = (
            pyro.optim.ExponentialLR(
                {"optimizer": torch.optim.Adam, "optim_args": optim_args, "gamma": float(lr_gamma)},
            )
            if lr_gamma is not None
            else pyro.optim.Adam(optim_args)
        )
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())
        loss_norm = float(self.Y_train_.shape[0] * self.Y_train_.shape[1])
        return guide, svi, loss_norm

    def _fit_map(
        self,
        *,
        num_steps: int,
        lr_Z: float,
        lr_global: float,
        lr_gamma: float | None,
        frozen_Z_start: float,
        log_every: int,
        verbose: bool,
        init_values: dict[str, Any] | None,
        record_loss_curve: bool,
        tau_schedule: dict[str, Any] | None,
        likelihood_tempering: dict[str, float] | None,
        nugget_schedule: dict[str, float] | None,
        val_X: Tensor | None,
        val_Y: Tensor | None,
        val_Y_sig: Tensor | None,
        val_y_std: Tensor | None,
        early_stop_metric: str,
        early_stop_patience_evals: int | None,
        early_stop_min_delta: float,
    ) -> tuple[AutoDelta, float, int, list[dict[str, Any]] | None]:
        assert self.X_train_ is not None and self.Y_train_ is not None
        store = pyro.get_param_store()
        curve: list[dict[str, Any]] | None = [] if record_loss_curve else None

        total = int(num_steps)
        if total < 1:
            raise ValueError("num_steps must be >= 1")
        frozen_Z_start = float(frozen_Z_start)
        if frozen_Z_start < 0.0 or frozen_Z_start > 1.0:
            raise ValueError("frozen_Z_start must be in [0,1]")

        patience = None if early_stop_patience_evals is None else int(early_stop_patience_evals)
        patience = None if patience is None or patience <= 0 else patience
        min_delta = float(early_stop_min_delta)
        if patience is not None and (val_X is None or val_Y is None):
            print("Warning: early_stop_patience_evals is set but val data is not provided, so early stopping is disabled.")
            patience = None

        best_val_metric = float("inf")
        best_loss = float("inf")
        best_state = None
        best_tau = None
        bad_evals = 0
        lt_steps = None if likelihood_tempering is None else max(int(total * float(likelihood_tempering["frac"])), 1)
        ns_steps = None if nugget_schedule is None else max(int(total * float(nugget_schedule["frac"])), 1)

        def step_lr(svi_obj: SVI) -> None:
            stepper = getattr(svi_obj.optim, "step", None)
            if callable(stepper):
                stepper()

        def eval_val() -> dict[str, float]:
            assert val_X is not None and val_Y is not None
            with torch.no_grad():
                X = self.X_train_
                assert X is not None
                n_train = int(X.shape[0])

                sigma = store.get_param("AutoDelta.sigma")
                ell = (
                    (X.new_full((X.shape[1],), float(self.ell_fixed)) if isinstance(self.ell_fixed, float) else X.new_tensor(self.ell_fixed))
                    if self.ell_mode == "fixed"
                    else (
                        X.new_tensor(self._ell_min) + store.get_param("AutoDelta.ell_raw")
                        if self._ell_prior_type() == "lognormal"
                        else (store.get_param("AutoDelta.inv_ell_raw") + self._eps).reciprocal()
                    )
                )
                sigma_f = (
                    X.new_tensor(self.sigma_f_fixed)
                    if self.sigma_f_mode == "fixed"
                    else (
                        None
                        if self.sigma_f_mode not in ("shared", "per_latent")
                        else store.get_param("AutoDelta.sigma_f")
                    )
                )

                K = apply_kernel(self.kernel, X, ell, X.new_tensor(1.0))
                if sigma_f is not None:
                    sf2 = sigma_f**2
                    if sf2.ndim == 0:
                        K = K * sf2
                    else:
                        K = K.unsqueeze(0) * sf2[:, None, None] if K.ndim == 2 else K * sf2[:, None, None]
                eye = torch.eye(n_train, device=X.device, dtype=X.dtype)
                s2 = X.new_tensor(self.sigma_xi2) if self.sigma_xi2_mode == "float" else store.get_param("AutoDelta.sigma_xi2")
                s2 = s2 + X.new_tensor(float(self.sigma_xi2_extra))
                if K.ndim == 2 and s2.ndim == 1:
                    K = K.unsqueeze(0).expand(int(s2.shape[0]), -1, -1)
                K = K + eye * s2 if K.ndim == 2 else K + eye.unsqueeze(0) * (s2 if s2.ndim == 0 else s2[:, None, None])
                K = 0.5 * (K + K.transpose(-1, -2))
                L_K = torch.linalg.cholesky(K)

                Z_train = store.get_param("AutoDelta.Z_T").T
                mu_W, _ = self._decoder_posterior(Z_train, self.Y_train_, sigma)
                st: dict[str, Tensor] = {"ell": ell, "L_K": L_K, "Z_train": Z_train}
                if sigma_f is not None:
                    st["sigma_f"] = sigma_f
                yhat = self._gp_predict_latents_mean(val_X, st) @ mu_W
                if self.linear_trend_ is not None:
                    s0 = val_X.new_zeros((val_X.shape[0],), dtype=torch.long)
                    H = build_design_matrix(val_X, s0, n_sim_types=1, design_cfg=self.linear_trend_["design_cfg"])
                    trend = torch.einsum("np,paf->naf", H, self.linear_trend_["Gamma"]).squeeze(1)
                    yhat = yhat + trend

                out = {"mae_val": float(torch.mean(torch.abs(yhat - val_Y)))}
                if val_y_std is not None:
                    diff_obs = (yhat - val_Y) * val_y_std[None, :]
                    out["rmse_val_obs"] = float(torch.sqrt(torch.mean(diff_obs * diff_obs)))
                    if val_Y_sig is not None:
                        diff_sig = (yhat - val_Y_sig) * val_y_std[None, :]
                        out["rmse_val_sig"] = float(torch.sqrt(torch.mean(diff_sig * diff_sig)))

                m = str(early_stop_metric).lower()
                if m == "mae_val":
                    out["val_metric"] = out["mae_val"]
                    return out
                if m in ("rmse_val_obs", "rmse_obs", "rmse_obs_val"):
                    if "rmse_val_obs" not in out:
                        raise ValueError("rmse_val_obs requires val_y_std.")
                    out["val_metric"] = out["rmse_val_obs"]
                    return out
                if m in ("rmse_val_sig", "rmse_sig"):
                    if "rmse_val_sig" not in out:
                        raise ValueError("rmse_val_sig requires val_Y_sig and val_y_std.")
                    out["val_metric"] = out["rmse_val_sig"]
                    return out
                raise ValueError(
                    f"Unknown early_stop_metric={early_stop_metric!r} "
                    "(expected 'mae_val' | 'rmse_val_obs' | 'rmse_val_sig' (aliases: rmse_obs, rmse_sig))."
                )

        z_unc = None
        if frozen_Z_start > 0.0:
            if total < 2:
                raise ValueError("num_steps must be >= 2 when frozen_Z_start > 0")
            n1 = max(int(total * frozen_Z_start), 1)
            n2 = max(total - n1, 1)

            guide, svi1, loss_norm = self._build_svi(lr_Z=lr_Z, lr_global=lr_global, lr_gamma=None, init_values=init_values, guide=None)
            guide(self.Y_train_, self.X_train_)  # init params
            _, svi2, _ = self._build_svi(lr_Z=lr_Z, lr_global=lr_global, lr_gamma=lr_gamma, init_values=None, guide=guide)
            z_unc = store.get_param("AutoDelta.Z_T").unconstrained()
            phases: list[tuple[str, int, bool, SVI]] = [("freeze1", n1, False, svi1), ("freeze2", n2, True, svi2)]
        else:
            guide, svi, loss_norm = self._build_svi(lr_Z=lr_Z, lr_global=lr_global, lr_gamma=lr_gamma, init_values=init_values, guide=None)
            phases = [("map", total, True, svi)]

        last_loss = None
        steps_run = 0
        for phase, n_steps_phase, z_trainable, svi_phase in phases:
            phase_label = "frozen_Z" if not z_trainable else None
            if z_unc is not None:
                z_unc.requires_grad_(bool(z_trainable))
            for _ in range(int(n_steps_phase)):
                if tau_schedule is not None:
                    self.tau = self._tau_at(int(steps_run), total, tau_schedule)
                if likelihood_tempering is not None:
                    self.data_fit_scale = self._beta_at(int(steps_run), total, likelihood_tempering)
                if nugget_schedule is not None:
                    self.sigma_xi2_extra = self._s2_extra_at(int(steps_run), total, nugget_schedule)
                loss = svi_phase.step(self.Y_train_, self.X_train_)
                last_loss = loss
                step_lr(svi_phase)
                should_log = steps_run % int(log_every) == 0 or steps_run == total - 1
                val_out = None if not should_log or val_X is None else eval_val()
                val_metric = None if val_out is None else float(val_out["val_metric"])
                if curve is not None and should_log:
                    row: dict[str, Any] = {"step": int(steps_run), "loss": float(loss / loss_norm)}
                    row["tau"] = float(self.tau)
                    if likelihood_tempering is not None:
                        row["beta"] = float(self.data_fit_scale)
                    if nugget_schedule is not None:
                        row["sigma_xi2_extra"] = float(self.sigma_xi2_extra)
                    if phase_label is not None:
                        row["phase"] = phase_label
                    if val_out is not None:
                        for k in ("mae_val", "rmse_val_obs", "rmse_val_sig"):
                            if k in val_out:
                                row[k] = float(val_out[k])
                    if val_metric is not None and str(early_stop_metric).lower() != "mae_val":
                        row["val_metric"] = float(val_metric)
                        row["val_metric_name"] = str(early_stop_metric)
                    curve.append(row)
                if verbose and should_log:
                    msg = f"[GPLFR/toy] step={steps_run}/{total} loss={loss / loss_norm:.6g}"
                    if val_metric is not None:
                        k = "mae_val" if early_stop_metric == "mae_val" else str(early_stop_metric)
                        msg += f" {k}={float(val_metric):.6g}"
                    if likelihood_tempering is not None:
                        msg += f" beta={float(self.data_fit_scale):.3g}"
                    if nugget_schedule is not None:
                        msg += f" xi2_extra={float(self.sigma_xi2_extra):.3g}"
                    print(msg + (f" phase={phase_label}" if phase_label is not None else ""))
                allow_early_stop = (lt_steps is None or int(steps_run) >= int(lt_steps)) and (ns_steps is None or int(steps_run) >= int(ns_steps))
                if should_log and allow_early_stop and val_metric is not None:
                    if float(val_metric) < best_val_metric - min_delta:
                        best_val_metric = float(val_metric)
                        best_loss = float(loss)
                        best_state = deepcopy(store.get_state())
                        best_tau = float(self.tau)
                        bad_evals = 0
                    else:
                        if patience is not None and (frozen_Z_start == 0.0 or z_trainable):
                            bad_evals += 1
                            if bad_evals >= patience:
                                if verbose:
                                    print(f"[GPLFR/EARLY_STOP] stop at step={int(steps_run)} best_val={best_val_metric:.6g} metric={early_stop_metric}")
                                break
                steps_run += 1
            if patience is not None and bad_evals >= patience:
                break

        if best_state is not None:
            store.set_state(best_state)
            if best_tau is not None:
                self.tau = float(best_tau)
        out_loss = float(best_loss) if best_state is not None else (float(last_loss) if last_loss is not None else float("nan"))
        return guide, out_loss, int(steps_run), curve

    def _ell_prior_type(self) -> EllPriorType:
        prior_type = str(PRIOR_CFG.get("ell_prior_type", "lognormal")).lower()
        if prior_type in {"lognormal", "lognormal_ell", "lognormal_ell_raw", "lognormal_raw"}:
            return "lognormal"
        if prior_type in {"gamma_inv", "gamma_inverse", "inv_gamma", "gamma_inverse_ell", "gamma_inv_ell"}:
            return "gamma_inv"
        raise ValueError(f"Unknown PRIOR_CFG['ell_prior_type']={prior_type!r} (expected lognormal|gamma_inv).")

    def _ell_prior_loc_vector(self, ref: Tensor, d: int, *, default: Tensor) -> Tensor:
        loc_cfg = PRIOR_CFG.get("ell_prior_loc", None)
        if loc_cfg is None or (isinstance(loc_cfg, str) and loc_cfg.lower() == "auto"):
            loc = default
        else:
            loc_t = ref.new_tensor(loc_cfg)
            if loc_t.ndim == 0:
                loc = ref.new_full((d,), float(loc_t))
            elif int(loc_t.numel()) == int(d):
                loc = loc_t.reshape(d)
            else:
                raise ValueError("PRIOR_CFG['ell_prior_loc'] must be scalar or length d.")
        if torch.any(loc <= 0):
            raise ValueError("PRIOR_CFG['ell_prior_loc'] must be > 0.")
        return loc

    # ----------------------------
    # Prediction helpers
    # ----------------------------

    def _build_cached_state(self) -> None:
        if self.X_train_ is None or self.Y_train_ is None or self.posterior_samples_ is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        X = self.X_train_
        Y = self.Y_train_
        assert X is not None and Y is not None and self.posterior_samples_ is not None
        n_train = int(X.shape[0])

        n_post = int(self.posterior_samples_["sigma"].shape[0])
        cached: list[dict[str, Tensor]] = []
        for i in range(n_post):
            if "ell" in self.posterior_samples_:
                ell = self.posterior_samples_["ell"][i]
            else:
                if self.ell_fixed is None:
                    raise RuntimeError("ell_fixed is required when ell is not sampled")
                ell = X.new_full((X.shape[1],), float(self.ell_fixed)) if isinstance(self.ell_fixed, float) else X.new_tensor(self.ell_fixed)
            sigma = self.posterior_samples_["sigma"][i]
            sigma_f = (
                X.new_tensor(self.sigma_f_fixed)
                if self.sigma_f_mode == "fixed"
                else (None if "sigma_f" not in self.posterior_samples_ else self.posterior_samples_["sigma_f"][i])
            )

            K = apply_kernel(self.kernel, X, ell, X.new_tensor(1.0))
            eye = torch.eye(n_train, device=X.device, dtype=X.dtype)
            s2 = (
                X.new_tensor(self.sigma_xi2)
                if self.sigma_xi2_mode == "float"
                else self.posterior_samples_["sigma_xi2"][i]  # type: ignore[index]
            )
            s2 = s2 + X.new_tensor(float(self.sigma_xi2_extra))
            if K.ndim == 2 and s2.ndim == 1:
                K = K.unsqueeze(0).expand(int(s2.shape[0]), -1, -1)
            K = K + eye * s2 if K.ndim == 2 else K + eye.unsqueeze(0) * (s2 if s2.ndim == 0 else s2[:, None, None])
            if sigma_f is not None:
                sf2 = sigma_f**2
                if sf2.ndim != 0 and K.ndim == 2:
                    K = K.unsqueeze(0).expand(int(sf2.shape[0]), -1, -1)
                K = K * (sf2 if sf2.ndim == 0 else sf2[:, None, None])
            K = 0.5 * (K + K.transpose(-1, -2))
            L_K = torch.linalg.cholesky(K)

            Z_train = self.posterior_samples_["Z_T"][i].T
            mu_W, Sigma_W = self._decoder_posterior(Z_train, Y, sigma)

            row = {"ell": ell, "sigma": sigma, "L_K": L_K, "Z_train": Z_train, "mu_W": mu_W, "Sigma_W": Sigma_W}
            if sigma_f is not None:
                row["sigma_f"] = sigma_f
            cached.append(row)

        self._cached_state_ = cached

    def _gp_predict_latents_mean(self, X_new: Tensor, state: dict[str, Tensor]) -> Tensor:
        X_train = self.X_train_
        assert X_train is not None

        ell = state["ell"]
        L_K = state["L_K"]
        Z_train = state["Z_train"]
        sigma_f = state.get("sigma_f", None)

        K_star = apply_kernel(self.kernel, X_new, ell, X_new.new_tensor(1.0), X2=X_train)
        if sigma_f is not None:
            sf2 = sigma_f**2
            if sf2.ndim == 0:
                K_star = K_star * sf2
            else:
                K_star = K_star.unsqueeze(0) * sf2[:, None, None] if K_star.ndim == 2 else K_star * sf2[:, None, None]

        if L_K.ndim == 2:
            Kinv_k = torch.cholesky_solve(K_star.T, L_K).T
            return Kinv_k @ Z_train

        if K_star.ndim == 2:
            K_star = K_star.unsqueeze(0).expand(int(L_K.shape[0]), -1, -1)
        Kinv_k = torch.cholesky_solve(K_star.permute(0, 2, 1), L_K).permute(0, 2, 1)
        return torch.einsum("jab,bj->aj", Kinv_k, Z_train)

    # ----------------------------
    # Core math (decoder)
    # ----------------------------

    def _collapsed_loglikelihood(
        self,
        Y: Tensor,
        Z: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        n_train, p = Y.shape
        q = Z.shape[1]

        Z = float(self.tau) * Z
        sigma_sq = sigma**2 + Y.new_tensor(self.jitter)
        inv_sigma_sq = 1.0 / sigma_sq

        Psi = torch.eye(q, device=Y.device, dtype=Y.dtype) + inv_sigma_sq * (Z.T @ Z)
        L_Psi = torch.linalg.cholesky(Psi)
        logdet_Psi = 2.0 * torch.sum(torch.log(torch.diagonal(L_Psi, offset=0, dim1=-2, dim2=-1)))

        term1 = -0.5 * p * n_train * torch.log(Y.new_tensor(2.0 * torch.pi))
        term2 = -0.5 * p * n_train * torch.log(sigma_sq)
        term3 = -0.5 * p * logdet_Psi
        term4 = -0.5 * inv_sigma_sq * torch.sum(Y * Y)

        ZTY = Z.T @ Y
        Psi_inv_ZTY = torch.cholesky_solve(ZTY, L_Psi)
        term5 = 0.5 * inv_sigma_sq**2 * torch.sum(ZTY * Psi_inv_ZTY)
        return term1 + term2 + term3 + term4 + term5

    def _decoder_posterior(
        self,
        Z: Tensor,
        Y: Tensor,
        sigma: Tensor,
    ) -> tuple[Tensor, Tensor]:
        q = Z.shape[1]
        Z = float(self.tau) * Z
        inv_sigma_sq = 1.0 / (sigma**2 + Y.new_tensor(self.jitter) + self._eps)
        Lambda_post = torch.eye(q, device=Y.device, dtype=Y.dtype) + inv_sigma_sq * (Z.T @ Z)
        L = torch.linalg.cholesky(Lambda_post)
        mu_post = inv_sigma_sq * torch.cholesky_solve(Z.T @ Y, L)
        Sigma_post = torch.cholesky_inverse(L)
        return float(self.tau) * mu_post, float(self.tau) ** 2 * Sigma_post

    # ----------------------------
    # Init + small utils
    # ----------------------------

    def _pca_init_Z(self, **cfg: Any) -> dict[str, Tensor]:
        X = self.X_train_
        Y = self.Y_train_
        assert X is not None and Y is not None
        q = int(self.latent_dim)

        from sklearn.decomposition import PCA

        Y_np = Y.detach().cpu().numpy()
        n, p = int(Y_np.shape[0]), int(Y_np.shape[1])
        q_eff = int(min(q, max(n - 1, 0), p))
        if q_eff < 1:
            raise ValueError(f"[GPLFR] PCA init requires n_train>=2 and n_features>=1, got (n={n}, p={p})")

        pca_kwargs = {"svd_solver": "full", **(cfg.get("pca_kwargs", {}) or {})}
        scores = PCA(n_components=q_eff, **pca_kwargs).fit_transform(Y_np)

        score_std = np.maximum(scores.std(axis=0), 1e-12)
        whiten = bool(cfg.get("whiten_scores", True))
        Z0_eff = scores / score_std if whiten else scores
        Z0 = np.zeros((n, q), dtype=Z0_eff.dtype)
        Z0[:, :q_eff] = Z0_eff
        init: dict[str, Tensor] = {"Z_T": X.new_tensor(Z0, dtype=X.dtype).T}

        init_sigma_f = cfg.get("init_sigma_f", None)
        if init_sigma_f is not None:
            init_sigma_f = str(init_sigma_f).lower()
            if init_sigma_f != "pca":
                raise ValueError(f"[GPLFR] Unknown pca_init.init_sigma_f={init_sigma_f!r} (expected null or 'pca').")
            if self.sigma_f_mode != "per_latent":
                raise ValueError("[GPLFR] pca_init.init_sigma_f requires sigma_f_mode='per_latent'.")
            sigma_f0 = np.ones(q, dtype=float)
            sigma_f0[:q_eff] = score_std
            init["sigma_f"] = X.new_tensor(sigma_f0, dtype=X.dtype)

        return init

    @staticmethod
    def _parse_tau_schedule(cfg: dict[str, Any] | None, *, fallback: float) -> dict[str, float | str] | None:
        if not cfg:
            return None
        if not bool(cfg.get("enabled", False)):
            return None
        kind = str(cfg.get("kind", "exp")).lower()
        start = float(cfg.get("start", fallback))
        end = float(cfg.get("end", fallback))
        if kind not in ("linear", "exp"):
            raise ValueError("tau_schedule.kind must be 'linear' or 'exp'")
        if start <= 0 or end <= 0:
            raise ValueError("tau_schedule start/end must be > 0")
        return {"kind": kind, "start": start, "end": end}

    @staticmethod
    def _tau_at(step: int, total_steps: int, sched: dict[str, Any]) -> float:
        if total_steps <= 1:
            return float(sched["end"])
        frac = float(step) / float(total_steps - 1)
        start = float(sched["start"])
        end = float(sched["end"])
        if sched.get("kind") == "linear":
            return start + (end - start) * frac
        return float(np.exp(np.log(start) + (np.log(end) - np.log(start)) * frac))

    def _as_tensor(self, x: np.ndarray | Tensor) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def _require_fitted(self) -> None:
        if self.X_train_ is None or self.Y_train_ is None or self.posterior_samples_ is None or self._cached_state_ is None:
            raise RuntimeError("Model is not fitted; call fit() first.")


def median_1nn_distance(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n = int(X.shape[0])
    if n < 2:  # pragma: no cover
        raise ValueError("Need at least 2 points to compute 1-NN distances")
    s = np.sum(X * X, axis=1)
    d2 = s[:, None] + s[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(np.min(d2, axis=1))))


def min_lengthscale_1nn(X: np.ndarray, *, factor: float = 1.0) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return float(factor) * median_1nn_distance(X) * np.ones((int(X.shape[1]),), dtype=float)
