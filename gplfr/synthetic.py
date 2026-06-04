"""Synthetic GPLFR data with spatially-correlated nuisance fields.

Forward model (per sample i):
  y_i = W_sig z_sig(x_i) + y_nuis,i + eps_i
  z_sig^(q)(·) ~ GP(0, sigma_sig^2 k_x)  for q=1..D_sig
  y_nuis,i ~ N(0, Sigma_nuis)            (RBF over 2D output grid, separable/Kronecker)
  eps_i ~ N(0, sigma_eps^2 I)

Output keys:
  - X: (N, Dx)
  - Y: (N, Dy)
  - Y_sig: (N, Dy) true conditional mean (predictable component)
  - Y_nuis: (N, Dy) nuisance random fields
  - W_sig: (Dy, D_sig)
  - blob_params: (D_sig, 3) columns [mu_u, mu_v, s]
  - ell_mode: scalar string ("fixed" or "uniform")
  - ell_sig: (D_sig,) per-latent lengthscales (only if ell_mode="uniform")
  - H, W, Dx, D_sig: scalars
"""

from __future__ import annotations

import numpy as np
import torch

from .kernels import apply_kernel, stabilize_kernel


def _blob_dictionary(
    H: int,
    W: int,
    D_sig: int,
    *,
    sigma_min: float,
    sigma_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    mu_u = rng.uniform(0.0, float(H - 1), size=D_sig)
    mu_v = rng.uniform(0.0, float(W - 1), size=D_sig)
    s = rng.uniform(float(sigma_min), float(sigma_max), size=D_sig)
    cols = [np.exp(-((u - mu_u[d]) ** 2 + (v - mu_v[d]) ** 2) / (2.0 * s[d] ** 2)).reshape(H * W) for d in range(D_sig)]
    return np.stack(cols, axis=1), np.stack([mu_u, mu_v, s], axis=1)


def _rbf_gram_1d(n: int, ell: float) -> np.ndarray:
    x = np.arange(int(n), dtype=np.float64)
    diff = x[:, None] - x[None, :]
    return np.exp(-0.5 * (diff / float(ell)) ** 2)


def _sample_nuisance_fields(
    N: int,
    H: int,
    W: int,
    *,
    ell_nuis: float,
    sigma_nuis: float,
    jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    Ku = _rbf_gram_1d(H, float(ell_nuis)) + float(jitter) * np.eye(int(H), dtype=np.float64)
    Kv = _rbf_gram_1d(W, float(ell_nuis)) + float(jitter) * np.eye(int(W), dtype=np.float64)
    Lu = np.linalg.cholesky(Ku)
    Lv = np.linalg.cholesky(Kv)
    E = rng.normal(size=(int(N), int(H), int(W))).astype(np.float64)
    tmp = np.einsum("uh,nhw->nuw", Lu, E, optimize=True)
    img = np.einsum("nuw,vw->nuv", tmp, Lv, optimize=True)
    return (float(sigma_nuis) * img).reshape(int(N), int(H) * int(W))

def create_synthetic_data(
    *,
    N: int = 1000,
    Dx: int = 3,
    H: int = 16,
    W: int = 16,
    D_sig: int = 4,
    kernel: str = "rbf",
    ell: float = 1.0,
    ell_mode: str = "fixed",
    ell_min: float | None = None,
    ell_max: float | None = None,
    sigma_sig: float = 0.2,
    ell_nuis: float = 2.0,
    sigma_nuis: float = 1.0,
    sigma_eps: float = 0.02,
    jitter: float = 1.0e-6,
    blob_sigma_min: float = 0.8,
    blob_sigma_max: float = 2.2,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    Dy = int(H) * int(W)

    # Inputs
    X = rng.normal(size=(int(N), int(Dx))).astype(np.float64)

    # Output dictionary (W_sig): localized blobs
    W_sig, blob_params = _blob_dictionary(H, W, D_sig, sigma_min=blob_sigma_min, sigma_max=blob_sigma_max, rng=rng)

    # Predictable latents: independent GP draws evaluated at inputs
    X_t = torch.as_tensor(X, dtype=torch.float64)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(int(seed))
    ell_mode = str(ell_mode).lower().strip()
    if ell_mode not in ("fixed", "uniform"):
        raise ValueError("ell_mode must be 'fixed' or 'uniform'")
    ell_sig = None
    if ell_mode == "fixed":
        Kx = apply_kernel(str(kernel), X_t, X_t.new_full((int(Dx),), float(ell)), X_t.new_tensor(float(sigma_sig)))
        Lx = torch.linalg.cholesky(stabilize_kernel(Kx, float(jitter)))
        Z_sig = (Lx @ torch.randn((int(N), int(D_sig)), dtype=X_t.dtype, device=X_t.device, generator=torch_rng)).cpu().numpy()
    else:
        if ell_min is None or ell_max is None:
            raise ValueError("ell_mode='uniform' requires ell_min and ell_max")
        if not (float(ell_min) > 0.0 and float(ell_max) > 0.0 and float(ell_min) < float(ell_max)):
            raise ValueError(f"Expected 0 < ell_min < ell_max, got ell_min={ell_min} ell_max={ell_max}")
        ell_sig = rng.uniform(float(ell_min), float(ell_max), size=int(D_sig)).astype(np.float64)
        ell_t = X_t.new_tensor(ell_sig)[:, None].expand(int(D_sig), int(Dx))
        Kx = apply_kernel(str(kernel), X_t, ell_t, X_t.new_tensor(float(sigma_sig)))
        Lx = torch.linalg.cholesky(stabilize_kernel(Kx, float(jitter)))
        E = torch.randn((int(D_sig), int(N)), dtype=X_t.dtype, device=X_t.device, generator=torch_rng)
        Z_sig = torch.einsum("qij,qj->qi", Lx, E).T.cpu().numpy()
    Y_sig = (Z_sig @ W_sig.T).astype(np.float64)

    # Nuisance random fields: iid across samples, spatially correlated across output pixels
    Y_nuis = _sample_nuisance_fields(N, H, W, ell_nuis=ell_nuis, sigma_nuis=sigma_nuis, jitter=jitter, rng=rng)

    # Add iid observation noise
    Y = Y_sig + Y_nuis + float(sigma_eps) * rng.normal(size=(int(N), Dy))

    data = {
        "X": X.astype(np.float32),
        "Y": Y.astype(np.float32),
        "Y_sig": Y_sig.astype(np.float32),
        "Y_nuis": Y_nuis.astype(np.float32),
        "W_sig": W_sig.astype(np.float32),
        "blob_params": blob_params.astype(np.float32),
        "H": np.array(int(H)),
        "W": np.array(int(W)),
        "Dx": np.array(int(Dx)),
        "D_sig": np.array(int(D_sig)),
        "kernel": np.array(str(kernel)),
        "ell": np.array(float(ell)),
        "ell_mode": np.array(str(ell_mode)),
        "ell_min": np.array(float(ell_min) if ell_min is not None else float("nan")),
        "ell_max": np.array(float(ell_max) if ell_max is not None else float("nan")),
        "sigma_sig": np.array(float(sigma_sig)),
        "ell_nuis": np.array(float(ell_nuis)),
        "sigma_nuis": np.array(float(sigma_nuis)),
        "sigma_eps": np.array(float(sigma_eps)),
        "jitter": np.array(float(jitter)),
        "blob_sigma_min": np.array(float(blob_sigma_min)),
        "blob_sigma_max": np.array(float(blob_sigma_max)),
        "seed": np.array(int(seed)),
    }
    if ell_sig is not None:
        data["ell_sig"] = ell_sig.astype(np.float32)
    return data
