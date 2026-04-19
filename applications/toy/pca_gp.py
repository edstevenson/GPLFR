"""Generic PCA + GP regression baseline (Starfish-style, simplified).

NOTE: This file is a local copy for the toy benchmark so it can evolve
independently of other benchmarks.

This is a reusable PCA+GP emulator that maps 3D inputs X->Y where Y is a dense
vector (e.g., a spectrum, a radial profile, or any fixed-length signal).

Important: this class does **not** standardize X or Y for you. Callers should
apply any desired preprocessing (e.g., z-scoring of X and per-dimension scaling
or log transforms of Y) before calling :meth:`from_spectra`.

Method summary:
1) Fit a PCA basis on the provided (already-preprocessed) Y.
2) Model each PCA weight as an independent GP over the 3 input parameters.
3) Predict weights and reconstruct Y in the same preprocessed space.

Practical note on scaling: this Variant-A implementation fits M independent
GPs on N training points, with cost ~O(M*N^3) if refactorizing per component.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import torch

# Register XPU backend if Intel Extension for PyTorch is available
try:
    import intel_extension_for_pytorch as ipex  # pyright: ignore[reportMissingImports]
except ImportError:
    pass

log = logging.getLogger(__name__)

KernelName = Literal["rbf", "matern32", "matern52"]
TorchDTypeName = Literal["float32", "float64"]
SigmaXi2Mode = Literal["shared", "per-latent", "float"]

PCA_KWARGS: dict[str, Any] = {"svd_solver": "full"}


def _median_1nn_distance(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, d = (int(X.shape[0]), int(X.shape[1]))
    if n < 2:
        raise ValueError("Need at least 2 points to compute 1-NN distances")
    out = np.empty((d,), dtype=float)
    for k in range(d):
        u, inv = np.unique(X[:, k], return_inverse=True)
        if u.size < 2:
            raise ValueError(f"Need at least 2 unique values in dim={k} to compute 1-NN distances")
        diffs = np.diff(u)
        prev = np.concatenate(([np.inf], diffs))
        nxt = np.concatenate((diffs, [np.inf]))
        d_unique = np.minimum(prev, nxt)
        d_point = d_unique[inv]
        d_point = d_point[np.isfinite(d_point)]
        if d_point.size == 0:
            raise ValueError(f"No finite neighbor distances found in dim={k}")
        out[k] = float(np.median(d_point))
    return out


def min_lengthscale_1nn(X: np.ndarray, *, factor: float = 1.0) -> np.ndarray:
    return float(factor) * _median_1nn_distance(X)


def _resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = str(dtype).lower()
    out = {"float32": torch.float32, "float64": torch.float64}.get(dtype)
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
    """ARD-scaled squared distances from precomputed squared diffs.

    delta2: (..., D) where D=3
    lengthscale: (..., D) broadcastable to delta2's leading dims
    """
    return (delta2 / (lengthscale**2)).sum(dim=-1)


def _kernel_from_sqdist_t(
    kernel: KernelName, variance: torch.Tensor, sq_dist: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return (K, ar, exp_term) given squared distances in ARD-scaled space."""
    if kernel == "rbf":
        return variance * torch.exp(-0.5 * sq_dist), None, None

    r = torch.sqrt(torch.clamp(sq_dist, min=0.0))
    if kernel == "matern32":
        ar = (3.0**0.5) * r
        exp_term = torch.exp(-ar)
        return variance * (1.0 + ar) * exp_term, ar, exp_term
    if kernel == "matern52":
        ar = (5.0**0.5) * r
        exp_term = torch.exp(-ar)
        return variance * (1.0 + ar + (5.0 / 3.0) * sq_dist) * exp_term, ar, exp_term
    raise ValueError(f"Unknown kernel {kernel!r} (expected 'rbf', 'matern32', or 'matern52')")


def kernel_matrix(kernel: KernelName, X: np.ndarray, Z: np.ndarray, variance: float, lengthscale: np.ndarray) -> np.ndarray:
    """Stationary ARD kernels (rbf/matern32/matern52) with k(x,x)=variance."""
    X_t = torch.as_tensor(np.asarray(X, dtype=float))
    Z_t = torch.as_tensor(np.asarray(Z, dtype=float))
    ell_t = torch.as_tensor(np.asarray(lengthscale, dtype=float))
    if ell_t.ndim == 0:
        ell_t = ell_t.expand(X_t.shape[1])
    delta2 = (X_t[:, None, :] - Z_t[None, :, :]) ** 2
    sq_dist = _scaled_sqdist_t(delta2, ell_t)
    K, _, _ = _kernel_from_sqdist_t(kernel, torch.as_tensor(float(variance), dtype=X_t.dtype), sq_dist)
    return K.detach().cpu().numpy()


@dataclass
class PCAGP:
    """A Starfish-style PCA+GP emulator.

    Parameters
    ----------
    grid_points
        Training inputs, shape (N, 3) with columns (Teff, logg, [Fe/H]).
    wavelength_nm
        Wavelength grid in nm.
    eigenspectra
        PCA components, shape (M, n_pix). (Rows are principal axes.)
    pca_mean
        Mean vector used internally by sklearn PCA on the standardized spectra.
        (Typically ~0 here, but stored for correctness.)
    flux_mean
        Mean of normalized fluxes, shape (n_pix,).
    flux_std
        Std of *normalized* fluxes, shape (n_pix,).
    weights
        PCA scores for each training spectrum, shape (N, M).
    hyperparams
        Hyperparameters stored in log space:
          - log_sigma_xi2 (shared weight observation-noise variance)
          - log_variance:{m}
          - log_lengthscale:{m}:{d}
    """

    # Preprocessed (z-scored) training inputs: shape (N, 3).
    grid_points: np.ndarray
    wavelength_nm: np.ndarray
    eigenspectra: np.ndarray
    pca_mean: np.ndarray
    weights: np.ndarray
    kernel: KernelName

    hyperparams: dict[str, float]

    sigma_xi2_mode: SigmaXi2Mode = "shared"
    sigma_xi2_fixed: float | None = None

    dtype: str | torch.dtype = "float64"
    device: str | torch.device = "auto"

    # Cached per-component GP factorizations:
    # cache contains batched Cholesky factors and alpha vectors on torch_device.
    # - L: (M, N, N) lower-triangular where Ky = L L^T
    # - alpha: (M, N) where alpha_m = (Km + sigma_xi2 I)^-1 w_m
    _cache: dict[str, torch.Tensor] | None = None

    # Torch mirrors (for GPU-friendly math)
    torch_dtype: torch.dtype = field(init=False, repr=False)
    torch_device: torch.device = field(init=False, repr=False)
    _X_t: torch.Tensor = field(init=False, repr=False)
    _W_t: torch.Tensor = field(init=False, repr=False)  # (N,M)
    _Phi_t: torch.Tensor = field(init=False, repr=False)  # (M,P)
    _pca_mean_t: torch.Tensor = field(init=False, repr=False)  # (P,)
    _delta2_train_t: torch.Tensor = field(init=False, repr=False)  # (N,N,3)
    _I_t: torch.Tensor = field(init=False, repr=False)  # (N,N)

    def __post_init__(self) -> None:
        self.torch_dtype = _resolve_torch_dtype(self.dtype)
        self.torch_device = _resolve_torch_device(self.device)
        self._X_t = torch.as_tensor(self.grid_points, dtype=self.torch_dtype, device=self.torch_device)
        self._W_t = torch.as_tensor(self.weights, dtype=self.torch_dtype, device=self.torch_device)
        self._Phi_t = torch.as_tensor(self.eigenspectra, dtype=self.torch_dtype, device=self.torch_device)
        self._pca_mean_t = torch.as_tensor(self.pca_mean, dtype=self.torch_dtype, device=self.torch_device)
        self._delta2_train_t = (self._X_t[:, None, :] - self._X_t[None, :, :]) ** 2
        self._I_t = torch.eye(self._X_t.shape[0], dtype=self.torch_dtype, device=self.torch_device)

    @classmethod
    def from_spectra(
        cls,
        *,
        grid_points: np.ndarray,
        wavelength_nm: np.ndarray,
        fluxes: np.ndarray,
        n_components: int = 25,
        kernel: KernelName = "rbf",
        sigma_xi2_mode: SigmaXi2Mode = "shared",
        sigma_xi2: float | np.ndarray = 1.0e-6,
        variances: np.ndarray | None = None,
        lengthscales: np.ndarray | None = None,
        max_matrix_bytes: int = 2_000_000_000,
        dtype: TorchDTypeName | torch.dtype = "float64",
        device: str | torch.device = "auto",
    ) -> "PCAGP":
        """Train PCA basis and initialize GP hyperparameters.

        Parameters
        ----------
        grid_points
            Shape (N, 3), **already z-scored** (Teff/logg/FeH).
        wavelength_nm
            Shape (n_pix,).
        fluxes
            Shape (N, n_pix), **already z-scored per pixel**.
        n_components
            Number of PCA components.
        sigma_xi2
            Shared observation-noise variance on the PCA weights (Variant A).
            This corresponds to Starfish's truncation/nugget term under orthonormal PCA.
        variances
            Initial GP variances, shape (M,). If None, uses 1e4.
        lengthscales
            Initial GP ARD lengthscales, shape (M, 3). If None, uses 3x a
            heuristic grid separation (Starfish-style).
        max_matrix_bytes
            Safety guard for caching all per-component (N x N) Cholesky factors.
            Approximate usage is ~ M * N^2 * 8 bytes.

        Returns
        -------
        PCAGP
        """
        # Validate inputs
        grid_points = np.asarray(grid_points, dtype=float)
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        fluxes = np.asarray(fluxes, dtype=float)

        if grid_points.ndim != 2 or grid_points.shape[1] != 3:
            raise ValueError("grid_points must have shape (N,3) (raw Teff/logg/FeH)")
        if fluxes.ndim != 2:
            raise ValueError("fluxes must have shape (N,n_pix)")
        if fluxes.shape[0] != grid_points.shape[0]:
            raise ValueError("fluxes and grid_points must have same N")

        sigma_xi2_mode = str(sigma_xi2_mode).lower()  # type: ignore[assignment]
        if sigma_xi2_mode not in ("shared", "per-latent", "float"):
            raise ValueError("sigma_xi2_mode must be one of: shared | per-latent | float")
        if kernel not in ("rbf", "matern32", "matern52"):
            raise ValueError(f"Unknown kernel {kernel!r} (expected 'rbf', 'matern32', or 'matern52')")

        X_gp = grid_points
        N = X_gp.shape[0]
        D = 3

        # Fit PCA basis
        pca = PCA(n_components=n_components, **PCA_KWARGS)
        weights = pca.fit_transform(fluxes)  # (N, M)
        eigenspectra = pca.components_  # (M, n_pix)
        pca_mean = pca.mean_  # (n_pix,)
        M = eigenspectra.shape[0]

        # Initialize GP hyperparameters (Starfish-like defaults)
        if variances is None:
            variances = 1e4 * np.ones(M)
        variances = np.asarray(variances, dtype=float)
        if variances.shape != (M,):
            raise ValueError(f"variances must have shape ({M},)")

        if lengthscales is None:
            # The original Starfish-style "grid separation" heuristic assumes a structured grid.
            # For the toy benchmark (random X), it oversmooths badly. Use a 1-NN-based scale as
            # a reference but default to O(1) lengthscales in standardized X space.
            ell_min = np.asarray(min_lengthscale_1nn(X_gp, factor=1.0), dtype=float)
            lengthscales = np.tile(np.maximum(1.0, 10.0 * ell_min), (M, 1))
        lengthscales = np.asarray(lengthscales, dtype=float)
        if lengthscales.shape != (M, D):
            raise ValueError(f"lengthscales must have shape ({M},{D})")

        # Memory guard for caching per-component N×N factors
        torch_dtype = _resolve_torch_dtype(dtype)
        bytes_needed = int(M) * int(N) * int(N) * torch.tensor([], dtype=torch_dtype).element_size()
        if bytes_needed > max_matrix_bytes:
            raise MemoryError(
                "PCA+GP emulator (Variant A) would cache too much dense state. "
                f"Would cache ~ {bytes_needed/1e9:.2f} GB for M={M}, N={N}. "
                "Reduce n_components and/or max_train, or avoid caching / use an approximate GP."
            )

        # Build hyperparams dict in log space
        hyperparams: dict[str, float] = {}
        sigma_xi2_fixed = None
        if sigma_xi2_mode == "shared":
            sigma0 = float(np.asarray(sigma_xi2, dtype=float).reshape(-1)[0])
            if sigma0 <= 0:
                raise ValueError("sigma_xi2 must be positive")
            hyperparams["log_sigma_xi2"] = float(np.log(sigma0))
        elif sigma_xi2_mode == "per-latent":
            s0 = np.asarray(sigma_xi2, dtype=float)
            s0 = np.full(M, float(s0.reshape(-1)[0])) if s0.size == 1 else s0.reshape(-1)
            if s0.shape != (M,):
                raise ValueError(f"sigma_xi2 must have shape ({M},) for sigma_xi2_mode='per-latent'")
            if not np.all(s0 > 0):
                raise ValueError("sigma_xi2 entries must be positive")
            for i in range(M):
                hyperparams[f"log_sigma_xi2:{i}"] = float(np.log(s0[i]))
        else:  # float
            sigma0 = float(np.asarray(sigma_xi2, dtype=float).reshape(-1)[0])
            if sigma0 <= 0:
                raise ValueError("sigma_xi2 must be positive")
            sigma_xi2_fixed = sigma0
        for i in range(M):
            hyperparams[f"log_variance:{i}"] = float(np.log(variances[i]))
            for j in range(D):
                hyperparams[f"log_lengthscale:{i}:{j}"] = float(np.log(lengthscales[i, j]))

        # Create emulator instance
        emu = cls(
            grid_points=X_gp,
            wavelength_nm=wavelength_nm,
            eigenspectra=eigenspectra,
            pca_mean=pca_mean,
            weights=weights,
            kernel=kernel,
            sigma_xi2_mode=sigma_xi2_mode,  # type: ignore[arg-type]
            sigma_xi2_fixed=sigma_xi2_fixed,
            hyperparams=hyperparams,
            dtype=dtype,
            device=device,
        )

        return emu

    @property
    def n_components(self) -> int:
        return int(self.eigenspectra.shape[0])

    @property
    def sigma_xi2(self) -> float | np.ndarray:
        """Observation-noise variance on PCA weights (shared or per-latent)."""
        if self.sigma_xi2_mode == "float":
            assert self.sigma_xi2_fixed is not None
            return float(self.sigma_xi2_fixed)
        if self.sigma_xi2_mode == "shared":
            return float(np.exp(self.hyperparams["log_sigma_xi2"]))
        return np.exp(np.array([self.hyperparams[f"log_sigma_xi2:{i}"] for i in range(self.n_components)], dtype=float))

    @sigma_xi2.setter
    def sigma_xi2(self, value: float | np.ndarray) -> None:
        if self.sigma_xi2_mode == "float":
            v = float(np.asarray(value, dtype=float).reshape(-1)[0])
            if v <= 0:
                raise ValueError("sigma_xi2 must be positive")
            self.sigma_xi2_fixed = v
            self._invalidate_cache()
            return

        if self.sigma_xi2_mode == "shared":
            v = float(np.asarray(value, dtype=float).reshape(-1)[0])
            if v <= 0:
                raise ValueError("sigma_xi2 must be positive")
            self.hyperparams["log_sigma_xi2"] = float(np.log(v))
            self._invalidate_cache()
            return

        v = np.asarray(value, dtype=float)
        v = np.full(self.n_components, float(v.reshape(-1)[0])) if v.size == 1 else v.reshape(-1)
        if v.shape != (self.n_components,):
            raise ValueError("sigma_xi2 has wrong shape")
        if not np.all(v > 0):
            raise ValueError("sigma_xi2 entries must be positive")
        for i, s2 in enumerate(v):
            self.hyperparams[f"log_sigma_xi2:{i}"] = float(np.log(s2))
        self._invalidate_cache()

    @property
    def variances(self) -> np.ndarray:
        return np.exp(np.array([self.hyperparams[f"log_variance:{i}"] for i in range(self.n_components)], dtype=float))

    @variances.setter
    def variances(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if values.shape != (self.n_components,):
            raise ValueError("variances has wrong shape")
        for i, v in enumerate(values):
            self.hyperparams[f"log_variance:{i}"] = float(np.log(v))
        self._invalidate_cache()

    @property
    def lengthscales(self) -> np.ndarray:
        out = np.empty((self.n_components, 3), dtype=float)
        for i in range(self.n_components):
            for j in range(3):
                out[i, j] = np.exp(self.hyperparams[f"log_lengthscale:{i}:{j}"])
        return out

    @lengthscales.setter
    def lengthscales(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if values.shape != (self.n_components, 3):
            raise ValueError("lengthscales has wrong shape")
        for i in range(self.n_components):
            for j in range(3):
                self.hyperparams[f"log_lengthscale:{i}:{j}"] = float(np.log(values[i, j]))
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._cache = None

    def _variances_t(self) -> torch.Tensor:
        return torch.exp(
            torch.stack(
                [self._X_t.new_tensor(self.hyperparams[f"log_variance:{i}"]) for i in range(self.n_components)],
                dim=0,
            )
        )

    def _lengthscales_t(self) -> torch.Tensor:
        return torch.exp(
            torch.stack(
                [
                    torch.stack(
                        [self._X_t.new_tensor(self.hyperparams[f"log_lengthscale:{i}:{j}"]) for j in range(3)], dim=0
                    )
                    for i in range(self.n_components)
                ],
                dim=0,
            )
        )

    def _sigma_xi2_t(self) -> torch.Tensor:
        if self.sigma_xi2_mode == "float":
            assert self.sigma_xi2_fixed is not None
            return self._X_t.new_full((self.n_components,), float(self.sigma_xi2_fixed))
        if self.sigma_xi2_mode == "shared":
            return torch.exp(self._X_t.new_full((self.n_components,), float(self.hyperparams["log_sigma_xi2"])))
        return torch.exp(
            torch.stack(
                [self._X_t.new_tensor(self.hyperparams[f"log_sigma_xi2:{i}"]) for i in range(self.n_components)], dim=0
            )
        )

    def _ensure_cache(self) -> None:
        """Build per-component Cholesky factors and alpha vectors for current hyperparams."""
        if self._cache is not None:
            return

        X = self._X_t  # (N,3)
        W = self._W_t  # (N,M)
        ell = self._lengthscales_t()  # (M,3)
        var = self._variances_t()  # (M,)
        s2 = self._sigma_xi2_t()  # (M,)

        sq_dist = _scaled_sqdist_t(self._delta2_train_t[None, :, :, :], ell[:, None, None, :])  # (M,N,N)
        K, _, _ = _kernel_from_sqdist_t(self.kernel, var[:, None, None], sq_dist)
        Ky = K + s2[:, None, None] * self._I_t[None, :, :]
        L, info = torch.linalg.cholesky_ex(Ky)
        if torch.any(info != 0):
            raise np.linalg.LinAlgError(f"Cholesky failed for {int(torch.sum(info != 0))} components (non-PD Ky).")

        alpha = torch.cholesky_solve(W.T.unsqueeze(-1), L).squeeze(-1)  # (M,N)
        self._cache = {"L": L, "alpha": alpha}

    # Core GP prediction

    def _predict_weights_t(self, params: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        params_t = torch.as_tensor(np.asarray(params, dtype=float), dtype=self.torch_dtype, device=self.torch_device)
        if params_t.ndim == 1:
            params_t = params_t[None, :]
        if params_t.shape[1] != 3:
            raise ValueError("params must have shape (Q,3)")

        self._ensure_cache()
        L = self._cache["L"]  # type: ignore[index]
        alpha = self._cache["alpha"]  # type: ignore[index]

        ell = self._lengthscales_t()  # (M,3)
        var = self._variances_t()  # (M,)

        delta2 = (self._X_t[:, None, :] - params_t[None, :, :]) ** 2  # (N,Q,3)
        sq_dist = _scaled_sqdist_t(delta2[None, :, :, :], ell[:, None, None, :])  # (M,N,Q)
        k_star, _, _ = _kernel_from_sqdist_t(self.kernel, var[:, None, None], sq_dist)

        mu = torch.sum(k_star * alpha[:, :, None], dim=1).T  # (Q,M)
        v = torch.cholesky_solve(k_star, L)  # (M,N,Q)
        var_w = (var[:, None] - torch.sum(k_star * v, dim=1)).clamp_min(0.0).T  # (Q,M)
        return mu, var_w

    def __call__(
        self,
        params: Sequence[float] | np.ndarray,
        *,
        full_cov: bool = False,
        reinterpret_batch: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict PCA weight distribution at ``params``.

        Returns the GP predictive distribution over *latent* weights w(x), not
        including the observation-noise variance sigma_xi2.

        Parameters
        ----------
        params
            Shape (3,) or (Q,3).
        full_cov
            If True, return the full covariance matrix in the stacked representation
            (size (M*Q)×(M*Q)). If False, return only the diagonal variances.
        reinterpret_batch
            If True, reshape outputs to (Q, M).

        Returns
        -------
        mu
            Predictive mean of weights (Q, M) if reinterpret_batch else (M*Q,).
        cov
            Predictive variance per weight (Q, M) if not full_cov and reinterpret_batch.
            If full_cov, a (M*Q, M*Q) block-diagonal matrix.
        """
        # Validate inputs
        params_raw = np.atleast_2d(np.asarray(params, dtype=float))
        if params_raw.shape[1] != 3:
            raise ValueError("params must have shape (Q,3)")
        params = params_raw

        if full_cov and reinterpret_batch:
            raise ValueError("Cannot reinterpret_batch when full_cov=True (matches Starfish behavior).")

        # NOTE: We intentionally allow extrapolation outside the training bounds.
        # This differs from Starfish's stricter behavior, but is required for
        # learning-curve sweeps where small training subsets may not cover the
        # fixed test set parameter range.

        mu_t, var_t = self._predict_weights_t(params)
        Q = int(params.shape[0])
        M = int(self.n_components)

        if full_cov:
            ell = self._lengthscales_t()
            var = self._variances_t()
            params_t = torch.as_tensor(params, dtype=self.torch_dtype, device=self.torch_device)
            delta2_ss = (params_t[:, None, :] - params_t[None, :, :]) ** 2  # (Q,Q,3)
            sq_dist_ss = _scaled_sqdist_t(delta2_ss[None, :, :, :], ell[:, None, None, :])  # (M,Q,Q)
            k_ss, _, _ = _kernel_from_sqdist_t(self.kernel, var[:, None, None], sq_dist_ss)  # (M,Q,Q)

            self._ensure_cache()
            L = self._cache["L"]  # type: ignore[index]
            alpha = self._cache["alpha"]  # type: ignore[index]
            delta2 = (self._X_t[:, None, :] - params_t[None, :, :]) ** 2  # (N,Q,3)
            sq_dist = _scaled_sqdist_t(delta2[None, :, :, :], ell[:, None, None, :])  # (M,N,Q)
            k_star, _, _ = _kernel_from_sqdist_t(self.kernel, var[:, None, None], sq_dist)  # (M,N,Q)
            v = torch.cholesky_solve(k_star, L)  # (M,N,Q)
            cov_blocks_t = k_ss - torch.bmm(k_star.transpose(1, 2), v)  # (M,Q,Q)

            cov = np.zeros((M * Q, M * Q), dtype=float)
            blocks = cov_blocks_t.detach().cpu().numpy()
            for m in range(M):
                cov[m * Q : (m + 1) * Q, m * Q : (m + 1) * Q] = blocks[m]
            mu = mu_t.T.reshape(M * Q, order="F").detach().cpu().numpy()
            return mu, cov

        mu_np = mu_t.detach().cpu().numpy()
        var_np = var_t.detach().cpu().numpy()
        if reinterpret_batch:
            return mu_np.squeeze(), var_np.squeeze()
        mu = mu_np.T.reshape(M * Q, order="F")
        var = var_np.T.reshape(M * Q, order="F")
        return mu, var

    # Spectrum interface

    def predict_flux(
        self,
        params: Sequence[float] | np.ndarray,
        *,
        return_std: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict the mean spectrum at ``params``.

        Parameters
        ----------
        params
            Shape (3,) or (Q,3), **already z-scored**.
        return_std
            If True, also return an approximate per-pixel standard deviation derived
            from the (diagonal) weight covariance. This ignores cross-wavelength covariance.

        Returns
        -------
        flux_mean : np.ndarray
            Shape (Q, n_pix).
        flux_std : np.ndarray, optional
            Shape (Q, n_pix).
        """
        mu_w_t, var_w_t = self._predict_weights_t(np.asarray(params, dtype=float))
        flux_stdzd_t = mu_w_t @ self._Phi_t + self._pca_mean_t[None, :]
        if not return_std:
            return flux_stdzd_t.detach().cpu().numpy().squeeze()

        flux_var_t = var_w_t @ (self._Phi_t**2)
        flux_std_t = torch.sqrt(torch.clamp(flux_var_t, min=0.0))
        return flux_stdzd_t.detach().cpu().numpy().squeeze(), flux_std_t.detach().cpu().numpy().squeeze()

    # Training / scoring

    def log_likelihood(self) -> float:
        """Sum of independent GP log-likelihoods over PCA weight dimensions (Variant A)."""
        self._ensure_cache()
        L = self._cache["L"]  # type: ignore[index]
        alpha = self._cache["alpha"]  # type: ignore[index]
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        logdet = 2.0 * torch.sum(torch.log(diag), dim=-1)  # (M,)
        quad = torch.sum(self._W_t.T * alpha, dim=-1)  # (M,)
        ll = -0.5 * torch.sum(logdet + quad)
        return float(ll.detach().cpu().item())

    def _nll_and_grad(self, vec: np.ndarray, *, delta2: list[np.ndarray], I: np.ndarray) -> tuple[float, np.ndarray]:
        vec = np.asarray(vec, dtype=float)
        if np.any(~np.isfinite(vec)):
            return np.inf, np.zeros_like(vec)
        # NOTE: delta2 and I are passed for compatibility with the previous CPU implementation;
        # we ignore them and use precomputed torch tensors on the configured device.
        self.set_param_vector(vec)

        ell = self._lengthscales_t()  # (M,3)
        var = self._variances_t()  # (M,)
        s2 = self._sigma_xi2_t()  # (M,)

        sq_dist = _scaled_sqdist_t(self._delta2_train_t[None, :, :, :], ell[:, None, None, :])  # (M,N,N)
        K, ar, exp_term = _kernel_from_sqdist_t(self.kernel, var[:, None, None], sq_dist)
        Ky = K + s2[:, None, None] * self._I_t[None, :, :]

        L, info = torch.linalg.cholesky_ex(Ky)
        if torch.any(info != 0):
            # Robust behavior for optimizer: signal invalid region without crashing.
            return np.inf, np.zeros_like(vec)

        alpha = torch.cholesky_solve(self._W_t.T.unsqueeze(-1), L).squeeze(-1)  # (M,N)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        logdet = 2.0 * torch.sum(torch.log(diag), dim=-1)  # (M,)
        quad = torch.sum(self._W_t.T * alpha, dim=-1)  # (M,)
        nll_t = 0.5 * torch.sum(logdet + quad)

        # NOTE: torch.cholesky_inverse is not batched on some torch builds (e.g. older/cluster builds).
        # Use cholesky_solve against I instead; same math, supports batching.
        Kinv = torch.cholesky_solve(self._I_t[None, :, :].expand(L.shape[0], -1, -1), L)  # (M,N,N)
        A = alpha[:, :, None] * alpha[:, None, :] - Kinv  # (M,N,N)

        traceA = torch.sum(torch.diagonal(A, dim1=-2, dim2=-1), dim=-1)  # (M,)
        g_log_s2_vec = -0.5 * s2 * traceA  # (M,)
        g_log_var = -0.5 * torch.sum(A * K, dim=(-2, -1))  # (M,)

        base = self._delta2_train_t[None, :, :, :] / (ell[:, None, None, :] ** 2)  # (M,N,N,3)
        if self.kernel == "rbf":
            dK = K[:, :, :, None] * base
        elif self.kernel == "matern32":
            assert exp_term is not None
            dK = var[:, None, None, None] * 3.0 * exp_term[:, :, :, None] * base
        else:  # matern52
            assert ar is not None and exp_term is not None
            dK = var[:, None, None, None] * (5.0 / 3.0) * (1.0 + ar)[:, :, :, None] * exp_term[:, :, :, None] * base
        g_log_ell = -0.5 * torch.sum(A[:, :, :, None] * dK, dim=(-3, -2))  # (M,3)

        keys = list(self.hyperparams.keys())
        key_to_idx = {k: i for i, k in enumerate(keys)}
        grad = np.zeros_like(vec)
        if self.sigma_xi2_mode == "shared":
            grad[key_to_idx["log_sigma_xi2"]] = float(torch.sum(g_log_s2_vec).detach().cpu().item())
        elif self.sigma_xi2_mode == "per-latent":
            gs2 = g_log_s2_vec.detach().cpu().numpy()
            for i in range(self.n_components):
                grad[key_to_idx[f"log_sigma_xi2:{i}"]] = float(gs2[i])
        gv = g_log_var.detach().cpu().numpy()
        ge = g_log_ell.detach().cpu().numpy()
        for m in range(self.n_components):
            grad[key_to_idx[f"log_variance:{m}"]] = float(gv[m])
            for d in range(3):
                grad[key_to_idx[f"log_lengthscale:{m}:{d}"]] = float(ge[m, d])

        return float(nll_t.detach().cpu().item()), grad

    def train(self, **opt_kwargs: Any) -> None:
        """Optimize hyperparameters with ``scipy.optimize.minimize`` (Starfish style)."""
        ell_lb = np.asarray(min_lengthscale_1nn(self.grid_points, factor=2.0), dtype=float)

        X = self.grid_points
        delta2 = [(X[:, d][:, None] - X[:, d][None, :]) ** 2 for d in range(3)]
        I = np.eye(X.shape[0])

        keys = list(self.hyperparams.keys())
        key_to_idx = {k: i for i, k in enumerate(keys)}
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * len(keys)
        if self.sigma_xi2_mode == "shared":
            bounds[key_to_idx["log_sigma_xi2"]] = (float(np.log(1e-8)), None)
        elif self.sigma_xi2_mode == "per-latent":
            for i in range(self.n_components):
                bounds[key_to_idx[f"log_sigma_xi2:{i}"]] = (float(np.log(1e-8)), None)
        for m in range(self.n_components):
            bounds[key_to_idx[f"log_variance:{m}"]] = (float(np.log(1e-12)), None)
            for d in range(3):
                bounds[key_to_idx[f"log_lengthscale:{m}:{d}"]] = (float(np.log(ell_lb[d])), None)

        def objective(P: np.ndarray) -> tuple[float, np.ndarray]:
            return self._nll_and_grad(P, delta2=delta2, I=I)

        patience = opt_kwargs.pop("early_stop_patience_evals", None)
        patience = None if patience is None or int(patience) <= 0 else int(patience)
        min_delta = float(opt_kwargs.pop("early_stop_min_delta", 0.0))
        early_stop_metric = str(opt_kwargs.pop("early_stop_metric", "rmse_val_obs")).lower()
        log_every = max(int(opt_kwargs.pop("log_every", 100)), 1)
        verbose = bool(opt_kwargs.pop("verbose", False))
        val_X = opt_kwargs.pop("val_X", None)
        val_Y = opt_kwargs.pop("val_Y", None)
        val_Y_sig = opt_kwargs.pop("val_Y_sig", None)
        val_y_std = opt_kwargs.pop("val_y_std", None)

        val_X_np = val_Y_np = val_Y_sig_np = val_y_std_np = None
        if patience is not None:
            if val_X is None or val_Y is None:
                raise ValueError("pca_gp early stopping requires val_X and val_Y.")
            val_X_np = np.asarray(val_X, dtype=float)
            val_Y_np = np.asarray(val_Y, dtype=float)
            if val_X_np.ndim != 2 or val_X_np.shape[1] != 3:
                raise ValueError("val_X must have shape (Q,3)")
            if val_Y_np.ndim != 2:
                raise ValueError("val_Y must have shape (Q,P)")
            if val_Y_np.shape[0] != val_X_np.shape[0]:
                raise ValueError("val_Y must match val_X rows")
            if val_Y_np.shape[1] != int(self.pca_mean.shape[0]):
                raise ValueError("val_Y must have same P as training fluxes")
            if val_Y_sig is not None:
                val_Y_sig_np = np.asarray(val_Y_sig, dtype=float)
                if val_Y_sig_np.shape != val_Y_np.shape:
                    raise ValueError("val_Y_sig must match val_Y shape")
            if val_y_std is not None:
                val_y_std_np = np.asarray(val_y_std, dtype=float).reshape(-1)
                if val_y_std_np.shape[0] != val_Y_np.shape[1]:
                    raise ValueError("val_y_std must have shape (P,) matching val_Y columns")

        def eval_val_metric(P: np.ndarray) -> tuple[float, dict[str, float]]:
            assert val_X_np is not None and val_Y_np is not None
            self.set_param_vector(np.asarray(P, dtype=float))
            yhat = np.asarray(self.predict_flux(val_X_np), dtype=float)
            if yhat.ndim == 1:
                yhat = yhat[None, :]
            out = {"mae_val": float(np.mean(np.abs(yhat - val_Y_np)))}
            if val_y_std_np is not None:
                diff_obs = (yhat - val_Y_np) * val_y_std_np[None, :]
                out["rmse_val_obs"] = float(np.sqrt(np.mean(diff_obs * diff_obs)))
                if val_Y_sig_np is not None:
                    diff_sig = (yhat - val_Y_sig_np) * val_y_std_np[None, :]
                    out["rmse_val_sig"] = float(np.sqrt(np.mean(diff_sig * diff_sig)))
            if early_stop_metric == "mae_val":
                return out["mae_val"], out
            if early_stop_metric in ("rmse_val_obs", "rmse_obs", "rmse_obs_val"):
                if "rmse_val_obs" not in out:
                    raise ValueError("rmse_val_obs requires val_y_std.")
                return out["rmse_val_obs"], out
            if early_stop_metric in ("rmse_val_sig", "rmse_sig"):
                if "rmse_val_sig" not in out:
                    raise ValueError("rmse_val_sig requires val_Y_sig and val_y_std.")
                return out["rmse_val_sig"], out
            raise ValueError(
                f"Unknown early_stop_metric={early_stop_metric!r} "
                "(expected 'mae_val' | 'rmse_val_obs' | 'rmse_val_sig' (aliases: rmse_obs, rmse_sig))."
            )

        class _EarlyStop(Exception):
            def __init__(self, x: np.ndarray, step: int, best_metric: float):
                self.x = np.asarray(x, dtype=float).copy()
                self.step = int(step)
                self.best_metric = float(best_metric)

        maxiter = opt_kwargs.pop("maxiter", None)
        if maxiter is not None:
            maxiter = int(maxiter)
            options = dict(opt_kwargs.get("options") or {})
            options["maxiter"] = maxiter
            opt_kwargs["options"] = options

        n_restarts = int(opt_kwargs.pop("n_restarts", 0) or 0)
        restart_jitter = float(opt_kwargs.pop("restart_jitter", 0.1) or 0.1)
        restart_seed = int(opt_kwargs.pop("restart_seed", 0) or 0)

        P0 = self.get_param_vector()
        default_kwargs = {"method": "L-BFGS-B", "jac": True, "bounds": bounds, "options": {"maxiter": 200}}
        default_kwargs.update(opt_kwargs)

        rng = np.random.default_rng(restart_seed)

        def _clip_to_bounds(P: np.ndarray) -> np.ndarray:
            P = np.asarray(P, dtype=float).copy()
            for i, (lb, ub) in enumerate(bounds):
                if lb is not None:
                    P[i] = max(P[i], lb)
                if ub is not None:
                    P[i] = min(P[i], ub)
            return P

        starts = [P0] + [P0 + restart_jitter * rng.standard_normal(P0.shape) for _ in range(n_restarts)]

        best_x = best_fun = best_success = best_message = None
        early_stop_count = 0
        early_stop_steps: list[int] = []
        for restart_idx, P_start in enumerate(starts):
            bad_evals = 0
            iter_count = 0
            best_val = float("inf")
            best_x_val = None

            def callback(xk: np.ndarray) -> None:
                nonlocal bad_evals, iter_count, best_val, best_x_val
                iter_count += 1
                if patience is None or iter_count % log_every != 0:
                    return
                val_metric, _ = eval_val_metric(xk)
                if verbose:
                    print(f"[PCAGP/EARLY_STOP] iter={iter_count} {early_stop_metric}={val_metric:.6g}")
                if val_metric < best_val - min_delta:
                    best_val = float(val_metric)
                    best_x_val = np.asarray(xk, dtype=float).copy()
                    bad_evals = 0
                else:
                    bad_evals += 1
                    if bad_evals >= patience:
                        raise _EarlyStop(xk, step=iter_count, best_metric=best_val)

            run_kwargs = dict(default_kwargs)
            run_kwargs["callback"] = callback
            try:
                soln = minimize(objective, _clip_to_bounds(P_start), **run_kwargs)
                cand_x = np.asarray(soln.x, dtype=float)
                cand_message = str(soln.message)
                cand_success = bool(soln.success)
                if patience is not None:
                    val_metric, _ = eval_val_metric(cand_x)
                    if val_metric < best_val - min_delta:
                        best_val = float(val_metric)
                        best_x_val = cand_x.copy()
                    if best_x_val is not None:
                        cand_x = best_x_val.copy()
                        cand_success = True
                        cand_message = f"early-stop-best-{early_stop_metric}={best_val:.6g}"
            except _EarlyStop as stop:
                early_stop_count += 1
                early_stop_steps.append(int(stop.step))
                cand_x = stop.x.copy()
                cand_success = True
                cand_message = f"early-stop-{early_stop_metric}={stop.best_metric:.6g}@iter{stop.step}"
                if best_x_val is not None:
                    cand_x = best_x_val.copy()
                print(
                    f"[PCAGP/EARLY_STOP] restart={restart_idx + 1}/{len(starts)} "
                    f"stop_iter={stop.step} best_{early_stop_metric}={stop.best_metric:.6g}"
                )

            cand_fun, _ = objective(cand_x)
            if not np.isfinite(cand_fun):
                continue
            if best_x is None or float(cand_fun) < float(best_fun):
                best_x = cand_x.copy()
                best_fun = float(cand_fun)
                best_success = bool(cand_success)
                best_message = str(cand_message)

        if best_x is None:
            log.warning("PCA+GP hyperparameter optimization did not succeed: all restarts failed")
            return
        if not bool(best_success):
            log.warning("PCA+GP hyperparameter optimization did not succeed: %s", best_message)
        if patience is not None:
            print(
                f"[PCAGP/EARLY_STOP] summary restarts={len(starts)} early_stops={early_stop_count} "
                f"steps={early_stop_steps}"
            )
        self.set_param_vector(np.asarray(best_x, dtype=float))

    # Parameter vector interface (for optimizers)
    def get_param_vector(self) -> np.ndarray:
        keys = list(self.hyperparams.keys())  # preserve insertion order
        return np.array([self.hyperparams[k] for k in keys], dtype=float)

    def set_param_vector(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=float)
        keys = list(self.hyperparams.keys())
        if vec.shape[0] != len(keys):
            raise ValueError("Parameter vector has wrong length")
        self.hyperparams = {k: float(v) for k, v in zip(keys, vec)}
        self._invalidate_cache()
