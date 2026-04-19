"""PCA + multitask GP baseline for reflectance curves (ICM coregionalization over g).

Important: this class does **not** apply log transforms or standardization. It
expects inputs/targets to already be preprocessed (e.g. log10(reflectance) then
z-scored per radius bin).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.decomposition import PCA
import torch

from .lengthscale import min_lengthscale_1nn

log = logging.getLogger(__name__)

KernelName = Literal["rbf", "matern32", "matern52"]
TorchDTypeName = Literal["float32", "float64"]

PCA_KWARGS: dict[str, Any] = {"svd_solver": "full"}


def _resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    out = {"float32": torch.float32, "float64": torch.float64}.get(str(dtype).lower())
    if out is None:
        raise ValueError(f"Unknown dtype {dtype!r} (expected 'float32' or 'float64')")
    return out


def _resolve_torch_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    device = str(device).lower()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # pragma: no cover
            return torch.device("xpu")
        return torch.device("cpu")
    return torch.device(device)


def _scaled_sqdist_t(delta2: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    return (delta2 / (lengthscale**2)).sum(dim=-1)


def _kernel_from_sqdist_t(kernel: KernelName, variance: torch.Tensor, sq_dist: torch.Tensor) -> torch.Tensor:
    if kernel == "rbf":
        return variance * torch.exp(-0.5 * sq_dist)
    r = torch.sqrt(torch.clamp(sq_dist, min=0.0) + 1e-12)
    if kernel == "matern32":
        ar = (3.0**0.5) * r
        return variance * (1.0 + ar) * torch.exp(-ar)
    if kernel == "matern52":
        ar = (5.0**0.5) * r
        return variance * (1.0 + ar + (5.0 / 3.0) * sq_dist) * torch.exp(-ar)
    raise ValueError(f"Unknown kernel {kernel!r} (expected 'rbf', 'matern32', or 'matern52')")


def _normalize_coreg(B: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    d = torch.sqrt(torch.clamp(torch.diagonal(B, dim1=-2, dim2=-1), min=eps))
    return B / (d[:, None] * d[None, :])


@dataclass
class PCAICM:
    """PCA+GP with ICM-style coregionalization over a discrete task label s."""

    grid_points: np.ndarray  # (N,2) continuous inputs, preprocessed
    tasks: np.ndarray  # (N,) int in [0,S)
    axis: np.ndarray  # (P,) radius axis (unused by model, kept for provenance)
    eigencurves: np.ndarray  # (M,P)
    pca_mean: np.ndarray  # (P,)
    weights: np.ndarray  # (N,M)
    kernel: KernelName
    per_latent_kernel: bool
    per_latent_noise: bool
    ell_min: np.ndarray
    coreg_L: np.ndarray  # (S,S) lower-triangular (unconstrained); B = L L^T then normalized to unit diag
    hyperparams: dict[str, float]  # log-params: sigma_z2 and (variance/lengthscales)

    dtype: str | torch.dtype = "float64"
    device: str | torch.device = "auto"

    _cache: dict[str, torch.Tensor] | None = None
    torch_dtype: torch.dtype = field(init=False, repr=False)
    torch_device: torch.device = field(init=False, repr=False)
    _X_t: torch.Tensor = field(init=False, repr=False)
    _s_t: torch.Tensor = field(init=False, repr=False)
    _W_t: torch.Tensor = field(init=False, repr=False)  # (N,M)
    _Phi_t: torch.Tensor = field(init=False, repr=False)  # (M,P)
    _pca_mean_t: torch.Tensor = field(init=False, repr=False)
    _delta2_train_t: torch.Tensor = field(init=False, repr=False)  # (N,N,2)
    _I_t: torch.Tensor = field(init=False, repr=False)  # (N,N)

    def __post_init__(self) -> None:
        self.torch_dtype = _resolve_torch_dtype(self.dtype)
        self.torch_device = _resolve_torch_device(self.device)
        self._X_t = torch.as_tensor(self.grid_points, dtype=self.torch_dtype, device=self.torch_device)
        self._s_t = torch.as_tensor(self.tasks, dtype=torch.long, device=self.torch_device)
        self._W_t = torch.as_tensor(self.weights, dtype=self.torch_dtype, device=self.torch_device)
        self._Phi_t = torch.as_tensor(self.eigencurves, dtype=self.torch_dtype, device=self.torch_device)
        self._pca_mean_t = torch.as_tensor(self.pca_mean, dtype=self.torch_dtype, device=self.torch_device)
        self._delta2_train_t = (self._X_t[:, None, :] - self._X_t[None, :, :]) ** 2
        self._I_t = torch.eye(self._X_t.shape[0], dtype=self.torch_dtype, device=self.torch_device)

    @classmethod
    def from_curves(
        cls,
        *,
        grid_points: np.ndarray,
        tasks: np.ndarray,
        axis: np.ndarray,
        curves: np.ndarray,
        n_components: int = 25,
        kernel: KernelName = "rbf",
        per_latent_kernel: bool = False,
        per_latent_noise: bool = True,
        sigma_z2: float = 1e-4,
        max_matrix_bytes: int = 2_000_000_000,
        dtype: TorchDTypeName | torch.dtype = "float64",
        device: str | torch.device = "auto",
    ) -> "PCAICM":
        grid_points = np.asarray(grid_points, dtype=float)
        tasks = np.asarray(tasks, dtype=int).reshape(-1)
        axis = np.asarray(axis, dtype=float)
        curves = np.asarray(curves, dtype=float)

        if grid_points.ndim != 2 or grid_points.shape[1] != 2:
            raise ValueError("grid_points must have shape (N,2) for (mua,musr)")
        if curves.ndim != 2 or curves.shape[0] != grid_points.shape[0]:
            raise ValueError("curves must have shape (N,P) matching grid_points")
        if tasks.shape[0] != grid_points.shape[0]:
            raise ValueError("tasks must have shape (N,) matching grid_points")
        if axis.ndim != 1 or axis.shape[0] != curves.shape[1]:
            raise ValueError("axis must have shape (P,) matching curves")
        if kernel not in ("rbf", "matern32", "matern52"):
            raise ValueError(f"Unknown kernel {kernel!r}")
        if sigma_z2 <= 0:
            raise ValueError("sigma_z2 must be positive")

        pca = PCA(n_components=n_components, **PCA_KWARGS)
        weights = pca.fit_transform(curves)  # (N,M)
        eigencurves = pca.components_  # (M,P)
        pca_mean = pca.mean_
        M = int(eigencurves.shape[0])

        S = int(tasks.max()) + 1
        coreg_L = np.eye(S, dtype=float)

        ell_min = np.asarray(min_lengthscale_1nn(grid_points, factor=1.0), dtype=float)
        ell0 = 3.0 * ell_min

        hyperparams: dict[str, float] = {}
        if per_latent_noise:
            for m in range(M):
                hyperparams[f"log_sigma_z2:{m}"] = float(np.log(sigma_z2))
        else:
            hyperparams["log_sigma_z2"] = float(np.log(sigma_z2))
        if per_latent_kernel:
            for m in range(M):
                hyperparams[f"log_variance:{m}"] = float(np.log(1.0))
                for d in range(2):
                    hyperparams[f"log_lengthscale:{m}:{d}"] = float(np.log(max(ell0[d] - ell_min[d], 1e-12)))
        else:
            hyperparams["log_variance"] = float(np.log(1.0))
            for d in range(2):
                hyperparams[f"log_lengthscale:{d}"] = float(np.log(max(ell0[d] - ell_min[d], 1e-12)))

        torch_dtype = _resolve_torch_dtype(dtype)
        bytes_needed = int(M) * int(grid_points.shape[0]) ** 2 * torch.tensor([], dtype=torch_dtype).element_size()
        if bytes_needed > max_matrix_bytes:
            raise MemoryError(f"Would cache ~{bytes_needed/1e9:.2f}GB dense GP state for M={M}, N={grid_points.shape[0]}.")

        return cls(
            grid_points=grid_points,
            tasks=tasks,
            axis=axis,
            eigencurves=eigencurves,
            pca_mean=pca_mean,
            weights=weights,
            kernel=kernel,
            per_latent_kernel=per_latent_kernel,
            per_latent_noise=per_latent_noise,
            ell_min=ell_min,
            coreg_L=coreg_L,
            hyperparams=hyperparams,
            dtype=dtype,
            device=device,
        )

    @property
    def n_components(self) -> int:
        return int(self.eigencurves.shape[0])

    @property
    def n_tasks(self) -> int:
        return int(self.coreg_L.shape[0])

    def _invalidate_cache(self) -> None:
        self._cache = None

    def _sigma_z2_t(self) -> torch.Tensor:
        if self.per_latent_noise:
            return torch.exp(
                torch.stack([self._X_t.new_tensor(self.hyperparams[f"log_sigma_z2:{m}"]) for m in range(self.n_components)])
            )
        return torch.exp(self._X_t.new_tensor(self.hyperparams["log_sigma_z2"])).expand(int(self.n_components))

    def _var_ell_t(self) -> tuple[torch.Tensor, torch.Tensor]:
        ell_min_t = self._X_t.new_tensor(self.ell_min)
        if self.per_latent_kernel:
            var = torch.exp(
                torch.stack([self._X_t.new_tensor(self.hyperparams[f"log_variance:{m}"]) for m in range(self.n_components)])
            )
            ell_excess = torch.exp(
                torch.stack(
                    [
                        torch.stack(
                            [self._X_t.new_tensor(self.hyperparams[f"log_lengthscale:{m}:{d}"]) for d in range(2)], dim=0
                        )
                        for m in range(self.n_components)
                    ],
                    dim=0,
                )
            )
            return var, ell_min_t[None, :] + ell_excess

        var = torch.exp(self._X_t.new_tensor(self.hyperparams["log_variance"]))[None]
        ell_excess = torch.exp(
            torch.stack([self._X_t.new_tensor(self.hyperparams[f"log_lengthscale:{d}"]) for d in range(2)], dim=0)
        )[None, :]
        return var, ell_min_t[None, :] + ell_excess

    def _B_t(self) -> torch.Tensor:
        L = torch.tril(torch.as_tensor(self.coreg_L, dtype=self.torch_dtype, device=self.torch_device))
        return _normalize_coreg(L @ L.T)

    def _ensure_cache(self) -> None:
        if self._cache is not None:
            return

        B = self._B_t()
        Bs = B[self._s_t[:, None], self._s_t[None, :]]
        sigma = self._sigma_z2_t()
        var, ell = self._var_ell_t()
        M = int(self.n_components)

        if self.per_latent_kernel:
            sq = _scaled_sqdist_t(self._delta2_train_t[None, :, :, :], ell[:, None, None, :])
            Kx = _kernel_from_sqdist_t(self.kernel, var[:, None, None], sq)
        else:
            sq = _scaled_sqdist_t(self._delta2_train_t, ell[0])
            Kx0 = _kernel_from_sqdist_t(self.kernel, var[0], sq)
            Kx = Kx0[None, :, :].expand(M, -1, -1)

        Ky = Kx * Bs[None, :, :] + (sigma[:, None, None] + 1e-6) * self._I_t[None, :, :]
        Lc, info = torch.linalg.cholesky_ex(Ky)
        if torch.any(info != 0):
            raise ValueError("Cholesky failed while building cache (kernel/coreg/noise produced non-PD matrix)")

        alpha = torch.cholesky_solve(self._W_t.T.unsqueeze(-1), Lc).squeeze(-1)
        self._cache = {"B": B, "L": Lc, "alpha": alpha}

    def log_likelihood(self) -> float:
        self._ensure_cache()
        L = self._cache["L"]  # type: ignore[index]
        alpha = self._cache["alpha"]  # type: ignore[index]
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        logdet = 2.0 * torch.sum(torch.log(diag), dim=-1)
        quad = torch.sum(self._W_t.T * alpha, dim=-1)
        ll = -0.5 * torch.sum(logdet + quad)
        return float(ll.detach().cpu().item())

    def train(
        self,
        *,
        num_steps: int = 200,
        lr: float = 5e-2,
        seed: int = 0,
        min_sigma_z2: float = 1e-6,
        min_variance: float = 1e-12,
        min_lengthscale: float | np.ndarray | None = None,
        jitter: float = 1e-6,
        val_X: np.ndarray | None = None,
        val_tasks: np.ndarray | None = None,
        val_Y: np.ndarray | None = None,
        val_y_std: np.ndarray | None = None,
        early_stop_metric: str = "mae_val",
        early_stop_patience_evals: int | None = None,
        early_stop_min_delta: float = 0.0,
        log_every: int = 100,
        verbose: bool = False,
    ) -> None:
        torch.manual_seed(int(seed))
        M = int(self.n_components)

        patience = None if early_stop_patience_evals is None else int(early_stop_patience_evals)
        patience = None if patience is None or patience <= 0 else patience
        min_delta = float(early_stop_min_delta)

        Xv_t = sv_t = Yv_t = y_std_t = delta2_tv = None
        if patience is not None:
            if val_X is None or val_tasks is None or val_Y is None or val_y_std is None:
                raise ValueError("Early stopping requires val_X, val_tasks, val_Y, and val_y_std.")
            Xv = np.asarray(val_X, dtype=float)
            sv = np.asarray(val_tasks, dtype=int).reshape(-1)
            Yv = np.asarray(val_Y, dtype=float)
            if Xv.ndim != 2 or Xv.shape[1] != 2:
                raise ValueError("val_X must have shape (Q,2)")
            if sv.shape[0] != Xv.shape[0] or Yv.shape[0] != Xv.shape[0]:
                raise ValueError("val_tasks and val_Y must match val_X rows")
            if Yv.shape[1] != self._pca_mean_t.shape[0]:
                raise ValueError("val_Y must have the same number of curve samples as training curves")

            Xv_t = torch.as_tensor(Xv, dtype=self.torch_dtype, device=self.torch_device)
            sv_t = torch.as_tensor(sv, dtype=torch.long, device=self.torch_device)
            Yv_t = torch.as_tensor(Yv, dtype=self.torch_dtype, device=self.torch_device)
            y_std_v = np.asarray(val_y_std, dtype=float).reshape(-1)
            if y_std_v.shape[0] != Yv.shape[1]:
                raise ValueError("val_y_std must have shape (P,) matching val_Y columns")
            y_std_t = torch.as_tensor(y_std_v, dtype=self.torch_dtype, device=self.torch_device)
            delta2_tv = (self._X_t[:, None, :] - Xv_t[None, :, :]) ** 2

        log_sigma = torch.nn.Parameter(
            (
                torch.tensor([self.hyperparams[f"log_sigma_z2:{m}"] for m in range(M)], dtype=self.torch_dtype, device=self.torch_device)
                if self.per_latent_noise
                else torch.tensor(float(self.hyperparams["log_sigma_z2"]), dtype=self.torch_dtype, device=self.torch_device)
            )
        )
        if self.per_latent_kernel:
            log_var = torch.nn.Parameter(
                torch.tensor([self.hyperparams[f"log_variance:{m}"] for m in range(M)], dtype=self.torch_dtype, device=self.torch_device)
            )
            log_ell = torch.nn.Parameter(
                torch.tensor(
                    [[self.hyperparams[f"log_lengthscale:{m}:{d}"] for d in range(2)] for m in range(M)],
                    dtype=self.torch_dtype,
                    device=self.torch_device,
                )
            )
        else:
            log_var = torch.nn.Parameter(torch.tensor(float(self.hyperparams["log_variance"]), dtype=self.torch_dtype, device=self.torch_device))
            log_ell = torch.nn.Parameter(
                torch.tensor([self.hyperparams[f"log_lengthscale:{d}"] for d in range(2)], dtype=self.torch_dtype, device=self.torch_device)
            )

        L_raw = torch.nn.Parameter(torch.as_tensor(self.coreg_L, dtype=self.torch_dtype, device=self.torch_device))
        opt = torch.optim.Adam([log_sigma, log_var, log_ell, L_raw], lr=float(lr))

        ell_min = np.asarray(self.ell_min if min_lengthscale is None else min_lengthscale, dtype=float)
        if ell_min.ndim == 0:
            ell_min = np.full((2,), float(ell_min))
        if ell_min.shape != (2,) or np.any(ell_min <= 0):
            raise ValueError("min_lengthscale must be positive with shape (2,)")
        self.ell_min = ell_min
        ell_min_t = self._X_t.new_tensor(self.ell_min)

        min_log_sigma = float(np.log(min_sigma_z2))
        min_log_var = float(np.log(min_variance))

        best_mae_val = float("inf")
        best_state = None
        bad_evals = 0

        metric = str(early_stop_metric).lower()
        metric = "mae_val" if metric in ("mae", "mae_val") else ("rmse_val" if metric in ("rmse", "rmse_val") else metric)
        if metric not in ("mae_val", "rmse_val"):
            raise ValueError("early_stop_metric must be 'mae_val' or 'rmse_val'")

        def eval_val_metric() -> float:
            assert Xv_t is not None and sv_t is not None and Yv_t is not None and y_std_t is not None and delta2_tv is not None
            with torch.no_grad():
                L = torch.tril(L_raw)
                B = _normalize_coreg(L @ L.T)
                Bs = B[self._s_t[:, None], self._s_t[None, :]]
                Bs_star = B[self._s_t[:, None], sv_t[None, :]]

                sigma = torch.exp(log_sigma).expand(M) if not self.per_latent_noise else torch.exp(log_sigma)
                if self.per_latent_kernel:
                    var = torch.exp(log_var)
                    ell = ell_min_t + torch.exp(log_ell)
                    sq = _scaled_sqdist_t(self._delta2_train_t[None, :, :, :], ell[:, None, None, :])
                    Kx = _kernel_from_sqdist_t(self.kernel, var[:, None, None], sq)
                    sq_star = _scaled_sqdist_t(delta2_tv[None, :, :, :], ell[:, None, None, :])
                    K_star = _kernel_from_sqdist_t(self.kernel, var[:, None, None], sq_star)
                else:
                    var0 = torch.exp(log_var)
                    ell0 = ell_min_t + torch.exp(log_ell)
                    sq = _scaled_sqdist_t(self._delta2_train_t, ell0)
                    Kx0 = _kernel_from_sqdist_t(self.kernel, var0, sq)
                    Kx = Kx0[None, :, :].expand(M, -1, -1)
                    sq_star = _scaled_sqdist_t(delta2_tv, ell0)
                    K_star0 = _kernel_from_sqdist_t(self.kernel, var0, sq_star)
                    K_star = K_star0[None, :, :].expand(M, -1, -1)

                Ky = Kx * Bs[None, :, :] + (sigma[:, None, None] + float(jitter)) * self._I_t[None, :, :]
                Lc, info = torch.linalg.cholesky_ex(Ky)
                if torch.any(info != 0):
                    raise ValueError("Cholesky failed during val eval")
                alpha = torch.cholesky_solve(self._W_t.T.unsqueeze(-1), Lc).squeeze(-1)

                mu_w = torch.sum(K_star * Bs_star[None, :, :] * alpha[:, :, None], dim=1).T
                y_pred = mu_w @ self._Phi_t + self._pca_mean_t[None, :]
                err = (y_pred - Yv_t) * y_std_t.unsqueeze(0)
                if metric == "mae_val":
                    return float(torch.mean(torch.abs(err)).cpu().item())
                return float(torch.sqrt(torch.mean(err * err)).cpu().item())

        best_val = float("inf")

        for step in range(int(num_steps)):
            opt.zero_grad(set_to_none=True)
            L = torch.tril(L_raw)
            B = _normalize_coreg(L @ L.T)
            Bs = B[self._s_t[:, None], self._s_t[None, :]]

            sigma = torch.exp(log_sigma).expand(M) if not self.per_latent_noise else torch.exp(log_sigma)
            if self.per_latent_kernel:
                var = torch.exp(log_var)
                ell = ell_min_t + torch.exp(log_ell)
                sq = _scaled_sqdist_t(self._delta2_train_t[None, :, :, :], ell[:, None, None, :])
                Kx = _kernel_from_sqdist_t(self.kernel, var[:, None, None], sq)
            else:
                var0 = torch.exp(log_var)
                ell0 = ell_min_t + torch.exp(log_ell)
                sq = _scaled_sqdist_t(self._delta2_train_t, ell0)
                Kx0 = _kernel_from_sqdist_t(self.kernel, var0, sq)
                Kx = Kx0[None, :, :].expand(M, -1, -1)

            Ky = Kx * Bs[None, :, :] + (sigma[:, None, None] + float(jitter)) * self._I_t[None, :, :]
            Lc, info = torch.linalg.cholesky_ex(Ky)
            if torch.any(info != 0):
                raise ValueError("Cholesky failed during training (try larger noise or longer lengthscales)")

            alpha = torch.cholesky_solve(self._W_t.T.unsqueeze(-1), Lc).squeeze(-1)
            diag = torch.diagonal(Lc, dim1=-2, dim2=-1)
            logdet = 2.0 * torch.sum(torch.log(diag), dim=-1)
            quad = torch.sum(self._W_t.T * alpha, dim=-1)
            loss = 0.5 * torch.sum(logdet + quad)
            loss.backward()
            opt.step()

            with torch.no_grad():
                log_sigma.clamp_(min=min_log_sigma)
                if self.per_latent_kernel:
                    log_var.clamp_(min=min_log_var)
                else:
                    log_var.clamp_(min=min_log_var)

            should_log = patience is not None and (step % int(log_every) == 0 or step == int(num_steps) - 1)
            if should_log:
                val_metric = eval_val_metric()
                if verbose:
                    print(f"[PCAICM/pyxopto] step={step}/{int(num_steps)} loss={float(loss.detach().cpu()):.6g} {metric}={val_metric:.6g}")
                if val_metric < best_val - min_delta:
                    best_val = val_metric
                    best_state = (
                        log_sigma.detach().clone(),
                        log_var.detach().clone(),
                        log_ell.detach().clone(),
                        L_raw.detach().clone(),
                    )
                    bad_evals = 0
                else:
                    bad_evals += 1
                    if bad_evals >= patience:
                        if verbose:
                            print(f"[PCAICM/EARLY_STOP] stop at step={step} best_{metric}={best_val:.6g}")
                        break

        if best_state is not None:
            log_sigma.data.copy_(best_state[0])
            log_var.data.copy_(best_state[1])
            log_ell.data.copy_(best_state[2])
            L_raw.data.copy_(best_state[3])

        for m in range(M):
            if self.per_latent_noise:
                self.hyperparams[f"log_sigma_z2:{m}"] = float(log_sigma.detach().cpu().numpy()[m])
        if not self.per_latent_noise:
            self.hyperparams["log_sigma_z2"] = float(log_sigma.detach().cpu().item())
        if self.per_latent_kernel:
            lv = log_var.detach().cpu().numpy()
            le = log_ell.detach().cpu().numpy()
            for m in range(M):
                self.hyperparams[f"log_variance:{m}"] = float(lv[m])
                for d in range(2):
                    self.hyperparams[f"log_lengthscale:{m}:{d}"] = float(le[m, d])
        else:
            self.hyperparams["log_variance"] = float(log_var.detach().cpu().item())
            le = log_ell.detach().cpu().numpy()
            for d in range(2):
                self.hyperparams[f"log_lengthscale:{d}"] = float(le[d])

        self.coreg_L = torch.tril(L_raw).detach().cpu().numpy()
        self._invalidate_cache()

    def predict_weights(self, X: np.ndarray, tasks: np.ndarray) -> np.ndarray:
        self._ensure_cache()
        Xq = np.asarray(X, dtype=float)
        sq = np.asarray(tasks, dtype=int).reshape(-1)
        if Xq.ndim != 2 or Xq.shape[1] != 2:
            raise ValueError("X must have shape (Q,2)")
        if sq.shape[0] != Xq.shape[0]:
            raise ValueError("tasks must have shape (Q,) matching X")

        Xq_t = torch.as_tensor(Xq, dtype=self.torch_dtype, device=self.torch_device)
        sq_t = torch.as_tensor(sq, dtype=torch.long, device=self.torch_device)

        B = self._cache["B"]  # type: ignore[index]
        alpha = self._cache["alpha"]  # type: ignore[index]
        var, ell = self._var_ell_t()
        M = int(self.n_components)

        delta2 = (self._X_t[:, None, :] - Xq_t[None, :, :]) ** 2
        Bs_star = B[self._s_t[:, None], sq_t[None, :]]

        if self.per_latent_kernel:
            sqd = _scaled_sqdist_t(delta2[None, :, :, :], ell[:, None, None, :])
            kx = _kernel_from_sqdist_t(self.kernel, var[:, None, None], sqd)
        else:
            sqd = _scaled_sqdist_t(delta2, ell[0])
            k0 = _kernel_from_sqdist_t(self.kernel, var[0], sqd)
            kx = k0[None, :, :].expand(M, -1, -1)

        mu = torch.sum(kx * Bs_star[None, :, :] * alpha[:, :, None], dim=1).T
        return mu.detach().cpu().numpy().squeeze()

    def predict_curve(self, X: np.ndarray, tasks: np.ndarray) -> np.ndarray:
        mu_w = self.predict_weights(X, tasks)
        mu_t = torch.as_tensor(np.atleast_2d(mu_w), dtype=self.torch_dtype, device=self.torch_device)
        y = mu_t @ self._Phi_t + self._pca_mean_t[None, :]
        return y.detach().cpu().numpy().squeeze()
