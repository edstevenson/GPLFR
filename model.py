r"""GPLFR model: latent GP + collapsed linear decoder.

Assumes external preprocessing:
  - X is standardized.
  - Y is centered/scaled in the desired training space.
"""

from __future__ import annotations

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

from .kernels import apply_kernel, stabilize_kernel


PRIOR_CFG = {
    "lengthscale_prior_loc": 0.0,
    "lengthscale_prior_scale": 0.3,
    "amplitude_prior_scale": 1.0,
    "sigma_prior_scale": 0.5,
}


KernelName = Literal["rbf", "matern32", "matern52"]
LengthscaleGrouping = Literal["shared", "per_latent", "fixed"]
AmplitudeGrouping = Literal["shared", "per_latent", "fixed"]


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
        lengthscale_grouping: LengthscaleGrouping = "per_latent",
        lengthscale: float | list[float] | np.ndarray | None = None,
        amplitude_grouping: AmplitudeGrouping = "fixed",
        amplitude: float | None = 1.0,
        inverse_temperature: float = 1.0,
        latent_noise: float = 1.0e-3,
        jitter: float = 1.0e-8,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if inverse_temperature <= 0:
            raise ValueError("inverse_temperature must be > 0")
        if latent_noise < 0:
            raise ValueError("latent_noise must be >= 0")
        if jitter < 0:
            raise ValueError("jitter must be >= 0")

        self.latent_dim = int(latent_dim)
        self.kernel: KernelName = kernel
        self.lengthscale_grouping: LengthscaleGrouping = lengthscale_grouping
        self.lengthscale = None if lengthscale is None else np.asarray(lengthscale, dtype=float)
        if self.lengthscale_grouping == "fixed" and self.lengthscale is None:
            raise ValueError("lengthscale is required when lengthscale_grouping='fixed'")
        if self.lengthscale_grouping != "fixed" and self.lengthscale is not None:
            raise ValueError("lengthscale is only used when lengthscale_grouping='fixed'")

        self.amplitude_grouping: AmplitudeGrouping = amplitude_grouping
        self.amplitude = None if amplitude is None else float(amplitude)
        if self.amplitude_grouping == "fixed" and self.amplitude is None:
            raise ValueError("amplitude is required when amplitude_grouping='fixed'")
        if self.amplitude is not None and self.amplitude <= 0:
            raise ValueError("amplitude must be > 0")

        self.inverse_temperature = float(inverse_temperature)
        self.latent_noise = float(latent_noise)
        self.jitter = float(jitter)
        self._eps = 1.0e-12

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
        self.posterior_samples_: dict[str, Tensor] | None = None
        self._cached_state_: list[dict[str, Tensor]] | None = None
        self.fit_result_: GPLFRFitResult | None = None

    def fit(
        self,
        X: np.ndarray | Tensor,
        Y: np.ndarray | Tensor,
        *,
        num_steps: int = 5000,
        lr_Z: float = 1.0e-2,
        lr_global: float = 1.0e-3,
        log_every: int = 50,
        record_loss_curve: bool = False,
        seed: int = 0,
        verbose: bool = True,
    ) -> GPLFRFitResult:
        pyro.set_rng_seed(int(seed))
        pyro.clear_param_store()

        X_t = self._as_tensor(X)
        Y_t = self._as_tensor(Y)
        if X_t.ndim != 2 or Y_t.ndim != 2:
            raise ValueError(f"Expected X,Y as 2D arrays; got X={tuple(X_t.shape)} Y={tuple(Y_t.shape)}")
        if X_t.shape[0] != Y_t.shape[0]:
            raise ValueError("X and Y must have the same number of rows")

        self.X_train_ = X_t
        self.Y_train_ = Y_t
        guide, final_loss, loss_curve = self._fit_map(
            num_steps=num_steps,
            lr_Z=lr_Z,
            lr_global=lr_global,
            log_every=log_every,
            verbose=verbose,
            record_loss_curve=record_loss_curve,
        )

        self.fit_result_ = GPLFRFitResult(int(num_steps), float(final_loss), loss_curve=loss_curve)
        params = guide(self.Y_train_, self.X_train_)
        samples = {k: v.detach().unsqueeze(0) for k, v in params.items() if isinstance(v, Tensor)}
        if self.lengthscale_grouping == "fixed":
            samples["lengthscale"] = self._fixed_lengthscale_tensor(X_t).unsqueeze(0)
        if self.amplitude_grouping == "fixed":
            samples["amplitude"] = X_t.new_tensor(float(self.amplitude)).view(1)
        self.posterior_samples_ = samples
        self._build_cached_state()
        return self.fit_result_

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
            preds.append(Z_mean @ state["mu_W"])
        return torch.stack(preds, dim=0).mean(dim=0).cpu().numpy()

    def model(self, Y: Tensor, X: Tensor) -> None:
        n_train, d = X.shape
        q = int(self.latent_dim)
        lengthscale = self._sample_lengthscale(X, d)
        amplitude = self._sample_amplitude(X, q)
        K = self._kernel_matrix(X, lengthscale, amplitude)
        L_K = torch.linalg.cholesky(K)

        Z_T = pyro.sample(
            "Z_T",
            dist.MultivariateNormal(
                loc=X.new_zeros((q, n_train)),
                scale_tril=L_K,
                validate_args=False,
            ).to_event(1),
        )
        Z = Z_T.T

        sigma = pyro.sample("sigma", dist.HalfNormal(X.new_tensor(PRIOR_CFG["sigma_prior_scale"])))
        pyro.factor(
            "collapsed_ll",
            self._collapsed_loglikelihood(Y, Z, sigma) * X.new_tensor(float(self.inverse_temperature)),
        )

    def _fit_map(
        self,
        *,
        num_steps: int,
        lr_Z: float,
        lr_global: float,
        log_every: int,
        verbose: bool,
        record_loss_curve: bool,
    ) -> tuple[AutoDelta, float, list[dict[str, Any]] | None]:
        assert self.X_train_ is not None and self.Y_train_ is not None
        total = int(num_steps)
        if total < 1:
            raise ValueError("num_steps must be >= 1")
        log_every = int(log_every)
        if log_every < 1:
            raise ValueError("log_every must be >= 1")

        guide, svi, loss_norm = self._build_svi(lr_Z=lr_Z, lr_global=lr_global)
        curve: list[dict[str, Any]] | None = [] if record_loss_curve else None
        last_loss = float("nan")
        for step in range(total):
            loss = float(svi.step(self.Y_train_, self.X_train_))
            last_loss = loss
            if step % log_every and step != total - 1:
                continue
            row = {"step": step, "loss": loss / loss_norm}
            if curve is not None:
                curve.append(row)
            if verbose:
                print(f"[GPLFR] step={step}/{total} loss={row['loss']:.6g}")
        return guide, last_loss, curve

    def _build_svi(self, *, lr_Z: float, lr_global: float) -> tuple[AutoDelta, SVI, float]:
        init_med = init_to_median()
        init_samp = init_to_sample()

        def init_loc(site: dict[str, Any]) -> Tensor:
            return init_samp(site) if site["name"] == "Z_T" else init_med(site)

        guide = AutoDelta(self.model, init_loc_fn=init_loc)

        def optim_args(param_name: str) -> dict[str, float]:
            name = param_name.rsplit(".", 1)[-1]
            return {"lr": float(lr_Z)} if name == "Z_T" else {"lr": float(lr_global)}

        assert self.Y_train_ is not None
        svi = SVI(self.model, guide, pyro.optim.Adam(optim_args), loss=Trace_ELBO())
        loss_norm = float(self.Y_train_.shape[0] * self.Y_train_.shape[1])
        return guide, svi, loss_norm

    def _sample_lengthscale(self, ref: Tensor, input_dim: int) -> Tensor:
        if self.lengthscale_grouping == "fixed":
            return self._fixed_lengthscale_tensor(ref)
        loc = ref.new_full((input_dim,), float(PRIOR_CFG["lengthscale_prior_loc"]))
        scale = ref.new_full((input_dim,), float(PRIOR_CFG["lengthscale_prior_scale"]))
        if self.lengthscale_grouping == "shared":
            return pyro.sample("lengthscale", dist.LogNormal(loc, scale).to_event(1))
        if self.lengthscale_grouping == "per_latent":
            return pyro.sample("lengthscale", dist.LogNormal(loc, scale).expand([self.latent_dim, input_dim]).to_event(2))
        raise ValueError("lengthscale_grouping must be one of: shared | per_latent | fixed")

    def _sample_amplitude(self, ref: Tensor, latent_dim: int) -> Tensor:
        if self.amplitude_grouping == "fixed":
            return ref.new_tensor(float(self.amplitude))
        scale = ref.new_tensor(float(PRIOR_CFG["amplitude_prior_scale"]))
        if self.amplitude_grouping == "shared":
            return pyro.sample("amplitude", dist.LogNormal(ref.new_tensor(0.0), scale))
        if self.amplitude_grouping == "per_latent":
            return pyro.sample("amplitude", dist.LogNormal(ref.new_zeros((latent_dim,)), ref.new_full((latent_dim,), float(scale))).to_event(1))
        raise ValueError("amplitude_grouping must be one of: shared | per_latent | fixed")

    def _fixed_lengthscale_tensor(self, ref: Tensor) -> Tensor:
        assert self.lengthscale is not None
        lengthscale = ref.new_tensor(self.lengthscale)
        if lengthscale.ndim == 0:
            return lengthscale.expand(ref.shape[1])
        return lengthscale.reshape(-1)

    def _kernel_matrix(self, X: Tensor, lengthscale: Tensor, amplitude: Tensor) -> Tensor:
        K = apply_kernel(self.kernel, X, lengthscale, X.new_tensor(1.0))
        K = self._scale_kernel(K, amplitude)
        eye = torch.eye(int(K.shape[-1]), device=X.device, dtype=X.dtype)
        K = K + eye * self.latent_noise if K.ndim == 2 else K + eye.unsqueeze(0) * self.latent_noise
        return stabilize_kernel(K, self.jitter)

    def _scale_kernel(self, K: Tensor, amplitude: Tensor) -> Tensor:
        amp2 = amplitude**2
        if amp2.ndim == 0:
            return K * amp2
        return K.unsqueeze(0) * amp2[:, None, None] if K.ndim == 2 else K * amp2[:, None, None]

    def _build_cached_state(self) -> None:
        self._require_fitted(skip_cache=True)
        assert self.X_train_ is not None and self.Y_train_ is not None and self.posterior_samples_ is not None
        cached = []
        for i in range(int(self.posterior_samples_["sigma"].shape[0])):
            lengthscale = self.posterior_samples_["lengthscale"][i]
            sigma = self.posterior_samples_["sigma"][i]
            amplitude = self.posterior_samples_["amplitude"][i]
            L_K = torch.linalg.cholesky(self._kernel_matrix(self.X_train_, lengthscale, amplitude))
            Z_train = self.posterior_samples_["Z_T"][i].T
            mu_W, Sigma_W = self._decoder_posterior(Z_train, self.Y_train_, sigma)
            cached.append({"lengthscale": lengthscale, "amplitude": amplitude, "L_K": L_K, "Z_train": Z_train, "mu_W": mu_W, "Sigma_W": Sigma_W})
        self._cached_state_ = cached

    def _gp_predict_latents_mean(self, X_new: Tensor, state: dict[str, Tensor]) -> Tensor:
        assert self.X_train_ is not None
        K_star = apply_kernel(self.kernel, X_new, state["lengthscale"], X_new.new_tensor(1.0), X2=self.X_train_)
        K_star = self._scale_kernel(K_star, state["amplitude"])
        L_K = state["L_K"]
        Z_train = state["Z_train"]
        if L_K.ndim == 2:
            return torch.cholesky_solve(K_star.T, L_K).T @ Z_train
        if K_star.ndim == 2:
            K_star = K_star.unsqueeze(0).expand(int(L_K.shape[0]), -1, -1)
        Kinv_k = torch.cholesky_solve(K_star.permute(0, 2, 1), L_K).permute(0, 2, 1)
        return torch.einsum("qtn,nq->tq", Kinv_k, Z_train)

    def _collapsed_loglikelihood(self, Y: Tensor, Z: Tensor, sigma: Tensor) -> Tensor:
        n_train, output_dim = Y.shape
        q = Z.shape[1]
        sigma_sq = sigma**2 + Y.new_tensor(self.jitter + self._eps)
        inv_sigma_sq = 1.0 / sigma_sq

        Psi = torch.eye(q, device=Y.device, dtype=Y.dtype) + inv_sigma_sq * (Z.T @ Z)
        L_Psi = torch.linalg.cholesky(Psi)
        logdet_Psi = 2.0 * torch.sum(torch.log(torch.diagonal(L_Psi, dim1=-2, dim2=-1)))

        ZTY = Z.T @ Y
        Psi_inv_ZTY = torch.cholesky_solve(ZTY, L_Psi)
        return (
            -0.5 * output_dim * n_train * torch.log(Y.new_tensor(2.0 * torch.pi))
            -0.5 * output_dim * n_train * torch.log(sigma_sq)
            -0.5 * output_dim * logdet_Psi
            -0.5 * inv_sigma_sq * torch.sum(Y * Y)
            +0.5 * inv_sigma_sq**2 * torch.sum(ZTY * Psi_inv_ZTY)
        )

    def _decoder_posterior(self, Z: Tensor, Y: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
        q = Z.shape[1]
        inv_sigma_sq = 1.0 / (sigma**2 + Y.new_tensor(self.jitter + self._eps))
        precision = torch.eye(q, device=Y.device, dtype=Y.dtype) + inv_sigma_sq * (Z.T @ Z)
        L = torch.linalg.cholesky(precision)
        mu_post = inv_sigma_sq * torch.cholesky_solve(Z.T @ Y, L)
        return mu_post, torch.cholesky_inverse(L)

    def _as_tensor(self, x: np.ndarray | Tensor) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def _require_fitted(self, *, skip_cache: bool = False) -> None:
        if self.X_train_ is None or self.Y_train_ is None or self.posterior_samples_ is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        if not skip_cache and self._cached_state_ is None:
            raise RuntimeError("Model prediction cache is missing.")
