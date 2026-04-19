"""Create a mechanism-toy dataset NPZ with spatially-correlated nuisance fields.

Forward model (per sample i):
  y_i = W_sig z_sig(x_i) + y_nuis,i + eps_i
  z_sig^(q)(·) ~ GP(0, sigma_sig^2 k_x)  for q=1..D_sig
  y_nuis,i ~ N(0, Sigma_nuis)            (RBF over 2D output grid, separable/Kronecker)
  eps_i ~ N(0, sigma_eps^2 I)

Output keys:
  - X: (N, Dx)
  - axis: (Dy,) (dummy 1D axis; Dy = H*W)
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

from pathlib import Path

import numpy as np
import torch
import yaml

from gplfr.shared.kernels import apply_kernel, stabilize_kernel


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

def create_dataset_gpfield_npz(
    out_path: str | Path,
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
) -> None:
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = (Path(__file__).parent / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path = out_path.with_suffix(".yaml")

    rng = np.random.default_rng(int(seed))
    Dy = int(H) * int(W)
    axis = np.linspace(0.0, 1.0, Dy, dtype=np.float64)

    # Inputs
    X = rng.normal(size=(int(N), int(Dx))).astype(np.float64)

    # Output dictionary (W_sig): localized blobs
    W_sig, blob_params = _blob_dictionary(H, W, D_sig, sigma_min=blob_sigma_min, sigma_max=blob_sigma_max, rng=rng)

    # Predictable latents: independent GP draws evaluated at inputs
    X_t = torch.as_tensor(X, dtype=torch.float64)
    ell_mode = str(ell_mode).lower().strip()
    if ell_mode not in ("fixed", "uniform"):
        raise ValueError("ell_mode must be 'fixed' or 'uniform'")
    ell_sig = None
    if ell_mode == "fixed":
        Kx = apply_kernel(str(kernel), X_t, X_t.new_full((int(Dx),), float(ell)), X_t.new_tensor(float(sigma_sig)))
        Lx = torch.linalg.cholesky(stabilize_kernel(Kx, float(jitter)))
        Z_sig = (Lx @ torch.randn((int(N), int(D_sig)), dtype=X_t.dtype, device=X_t.device)).cpu().numpy()
    else:
        if ell_min is None or ell_max is None:
            raise ValueError("ell_mode='uniform' requires ell_min and ell_max")
        if not (float(ell_min) > 0.0 and float(ell_max) > 0.0 and float(ell_min) < float(ell_max)):
            raise ValueError(f"Expected 0 < ell_min < ell_max, got ell_min={ell_min} ell_max={ell_max}")
        ell_sig = rng.uniform(float(ell_min), float(ell_max), size=int(D_sig)).astype(np.float64)
        ell_t = X_t.new_tensor(ell_sig)[:, None].expand(int(D_sig), int(Dx))
        Kx = apply_kernel(str(kernel), X_t, ell_t, X_t.new_tensor(float(sigma_sig)))
        Lx = torch.linalg.cholesky(stabilize_kernel(Kx, float(jitter)))
        E = torch.randn((int(D_sig), int(N)), dtype=X_t.dtype, device=X_t.device)
        Z_sig = torch.einsum("qij,qj->qi", Lx, E).T.cpu().numpy()
    Y_sig = (Z_sig @ W_sig.T).astype(np.float64)

    # Nuisance random fields: iid across samples, spatially correlated across output pixels
    Y_nuis = _sample_nuisance_fields(N, H, W, ell_nuis=ell_nuis, sigma_nuis=sigma_nuis, jitter=jitter, rng=rng)

    # Add iid observation noise
    Y = Y_sig + Y_nuis + float(sigma_eps) * rng.normal(size=(int(N), Dy))

    out = {
        "X": X.astype(np.float32),
        "axis": axis,
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
        out["ell_sig"] = ell_sig.astype(np.float32)
    np.savez_compressed(out_path, **out)
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "out_path": str(out_path),
                "N": int(N),
                "Dx": int(Dx),
                "H": int(H),
                "W": int(W),
                "Dy": int(Dy),
                "D_sig": int(D_sig),
                "kernel": str(kernel),
                "ell": float(ell),
                "ell_mode": str(ell_mode),
                "ell_min": (None if ell_min is None else float(ell_min)),
                "ell_max": (None if ell_max is None else float(ell_max)),
                "sigma_sig": float(sigma_sig),
                "ell_nuis": float(ell_nuis),
                "sigma_nuis": float(sigma_nuis),
                "sigma_eps": float(sigma_eps),
                "jitter": float(jitter),
                "blob_sigma_min": float(blob_sigma_min),
                "blob_sigma_max": float(blob_sigma_max),
                "seed": int(seed),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved {out_path} (N={N}, Dx={Dx}, Dy={Dy}, D_sig={D_sig})")
    print(f"Saved {yaml_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Create mechanism-toy dataset NPZ (GP signal + spatial nuisance fields)")
    p.add_argument("out_path", help="Output NPZ path (relative to this folder if not absolute)")
    p.add_argument("--N", type=int, default=3000)
    p.add_argument("--Dx", type=int, default=3)
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--W", type=int, default=16)
    p.add_argument("--D-sig", type=int, default=4)
    p.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "matern32", "matern52"])
    p.add_argument("--ell", type=float, default=2.0)
    p.add_argument("--ell-mode", type=str, default="fixed", choices=["fixed", "uniform"])
    p.add_argument("--ell-min", type=float, default=None, help="Used when --ell-mode=uniform")
    p.add_argument("--ell-max", type=float, default=None, help="Used when --ell-mode=uniform")
    p.add_argument("--sigma-sig", type=float, default=1.0)
    p.add_argument("--ell-nuis", type=float, default=2.0)
    p.add_argument("--sigma-nuis", type=float, default=1.0)
    p.add_argument("--sigma-eps", type=float, default=0.01)
    p.add_argument("--jitter", type=float, default=1.0e-6)
    p.add_argument("--blob-sigma-min", type=float, default=1.2)
    p.add_argument("--blob-sigma-max", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    create_dataset_gpfield_npz(
        args.out_path,
        N=args.N,
        Dx=args.Dx,
        H=args.H,
        W=args.W,
        D_sig=args.D_sig,
        kernel=args.kernel,
        ell=args.ell,
        ell_mode=args.ell_mode,
        ell_min=args.ell_min,
        ell_max=args.ell_max,
        sigma_sig=args.sigma_sig,
        ell_nuis=args.ell_nuis,
        sigma_nuis=args.sigma_nuis,
        sigma_eps=args.sigma_eps,
        jitter=args.jitter,
        blob_sigma_min=args.blob_sigma_min,
        blob_sigma_max=args.blob_sigma_max,
        seed=args.seed,
    )
