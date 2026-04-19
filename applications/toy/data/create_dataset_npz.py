"""Create a mechanism-toy dataset NPZ.

Output keys:
  - X: (N, Dx)
  - axis: (Dy,) (dummy 1D axis; Dy = H*W)
  - Y: (N, Dy)
  - Y_sig: (N, Dy) true conditional mean (predictable component)
  - W_sig: (Dy, D_sig)
  - W_nuis: (Dy, D_nuis)
  - blob_params: (D_sig, 3) columns [mu_u, mu_v, s]
  - dct_modes: (D_nuis, 2) columns [k, l]
  - H, W: scalars
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from gplfr.gplfr.kernels import apply_kernel, stabilize_kernel


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
    cols = []
    for d in range(D_sig):
        img = np.exp(-((u - mu_u[d]) ** 2 + (v - mu_v[d]) ** 2) / (2.0 * s[d] ** 2))
        cols.append((rng.choice([-1.0, 1.0]) * img).reshape(H * W))
    return np.stack(cols, axis=1), np.stack([mu_u, mu_v, s], axis=1)


def _dct_dictionary(H: int, W: int, D_nuis: int, *, k_max: int, l_max: int) -> tuple[np.ndarray, np.ndarray]:
    u = (np.arange(H, dtype=np.float64) + 0.5) / float(H)
    v = (np.arange(W, dtype=np.float64) + 0.5) / float(W)
    pairs = [(k, l) for k in range(k_max + 1) for l in range(l_max + 1) if not (k == 0 and l == 0)]
    pairs = sorted(pairs, key=lambda p: (p[0] ** 2 + p[1] ** 2, p[0], p[1]))[:D_nuis]
    cols = []
    for k, l in pairs:
        cu = np.cos(np.pi * k * u)
        cv = np.cos(np.pi * l * v)
        cols.append(np.outer(cu, cv).reshape(H * W))
    return np.stack(cols, axis=1), np.asarray(pairs, dtype=int)


def create_dataset_npz(
    out_path: str | Path,
    *,
    N: int = 1000,
    Dx: int = 3,
    H: int = 16,
    W: int = 16,
    D_sig: int = 4,
    D_nuis: int = 4,
    kernel: str = "rbf",
    ell: float = 1.0,
    sigma_sig: float = 0.2,
    sigma_nuis: float = 2.0,
    sigma_eps: float = 0.02,
    jitter: float = 1.0e-6,
    blob_sigma_min: float = 0.8,
    blob_sigma_max: float = 2.2,
    dct_k_max: int = 3,
    dct_l_max: int = 3,
    seed: int = 0,
) -> None:
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = (Path(__file__).parent / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path = out_path.with_suffix(".yaml")

    rng = np.random.default_rng(seed)
    Dy = int(H) * int(W)
    axis = np.linspace(0.0, 1.0, Dy, dtype=np.float64)

    # Inputs
    X = rng.normal(size=(int(N), int(Dx))).astype(np.float64)

    # Structured dictionaries -> single QR -> orthonormal bases.
    # NOTE: We orthonormalize with nuisance-first ordering so the nuisance basis remains
    # low-frequency/global; the signal basis becomes "blob-like but orthogonal to nuisance".
    W_sig_tilde, blob_params = _blob_dictionary(H, W, D_sig, sigma_min=blob_sigma_min, sigma_max=blob_sigma_max, rng=rng)
    W_nuis_tilde, dct_modes = _dct_dictionary(H, W, D_nuis, k_max=int(dct_k_max), l_max=int(dct_l_max))
    Q, _ = np.linalg.qr(np.concatenate([W_nuis_tilde, W_sig_tilde], axis=1), mode="reduced")
    W_nuis = Q[:, :D_nuis].astype(np.float64)
    W_sig = Q[:, D_nuis : D_nuis + D_sig].astype(np.float64)

    # Predictable latents: GP draws evaluated at training inputs
    X_t = torch.as_tensor(X, dtype=torch.float64)
    K = apply_kernel(str(kernel), X_t, X_t.new_full((Dx,), float(ell)), X_t.new_tensor(float(sigma_sig)))
    L = torch.linalg.cholesky(stabilize_kernel(K, float(jitter)))
    Z_sig = (L @ torch.randn((int(N), int(D_sig)), dtype=X_t.dtype, device=X_t.device)).cpu().numpy()

    # Nuisance latents: iid across samples
    Z_nuis = (float(sigma_nuis) * rng.normal(size=(int(N), int(D_nuis)))).astype(np.float64)

    # Compose outputs
    Y_sig = (Z_sig @ W_sig.T).astype(np.float64)
    Y_nuis = (Z_nuis @ W_nuis.T).astype(np.float64)
    Y = Y_sig + Y_nuis + float(sigma_eps) * rng.normal(size=(int(N), Dy))

    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        axis=axis,
        Y=Y.astype(np.float32),
        Y_sig=Y_sig.astype(np.float32),
        W_sig=W_sig.astype(np.float32),
        W_nuis=W_nuis.astype(np.float32),
        blob_params=blob_params.astype(np.float32),
        dct_modes=dct_modes.astype(int),
        H=np.array(int(H)),
        W=np.array(int(W)),
        Dx=np.array(int(Dx)),
        D_sig=np.array(int(D_sig)),
        D_nuis=np.array(int(D_nuis)),
        kernel=np.array(str(kernel)),
        ell=np.array(float(ell)),
        sigma_sig=np.array(float(sigma_sig)),
        sigma_nuis=np.array(float(sigma_nuis)),
        sigma_eps=np.array(float(sigma_eps)),
        jitter=np.array(float(jitter)),
        seed=np.array(int(seed)),
    )
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
                "D_nuis": int(D_nuis),
                "kernel": str(kernel),
                "ell": float(ell),
                "sigma_sig": float(sigma_sig),
                "sigma_nuis": float(sigma_nuis),
                "sigma_eps": float(sigma_eps),
                "jitter": float(jitter),
                "blob_sigma_min": float(blob_sigma_min),
                "blob_sigma_max": float(blob_sigma_max),
                "dct_k_max": int(dct_k_max),
                "dct_l_max": int(dct_l_max),
                "orthonormalize_order": "nuisance_first",
                "seed": int(seed),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved {out_path} (N={N}, Dx={Dx}, Dy={Dy}, D_sig={D_sig}, D_nuis={D_nuis})")
    print(f"Saved {yaml_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Create mechanism-toy dataset NPZ")
    p.add_argument("out_path", help="Output NPZ path (relative to this folder if not absolute)")
    p.add_argument("--N", type=int, default=3000)
    p.add_argument("--Dx", type=int, default=3)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--D-sig", type=int, default=4)
    p.add_argument("--D-nuis", type=int, default=4)
    p.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "matern32", "matern52"])
    p.add_argument("--ell", type=float, default=0.5)
    p.add_argument("--sigma-sig", type=float, default=0.5)
    p.add_argument("--sigma-nuis", type=float, default=1.5)
    p.add_argument("--sigma-eps", type=float, default=0.01)
    p.add_argument("--jitter", type=float, default=1.0e-6)
    p.add_argument("--blob-sigma-min", type=float, default=1.2)
    p.add_argument("--blob-sigma-max", type=float, default=2.5)
    p.add_argument("--dct-k-max", type=int, default=4)
    p.add_argument("--dct-l-max", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    create_dataset_npz(
        args.out_path,
        N=args.N,
        Dx=args.Dx,
        H=args.H,
        W=args.W,
        D_sig=args.D_sig,
        D_nuis=args.D_nuis,
        kernel=args.kernel,
        ell=args.ell,
        sigma_sig=args.sigma_sig,
        sigma_nuis=args.sigma_nuis,
        sigma_eps=args.sigma_eps,
        jitter=args.jitter,
        blob_sigma_min=args.blob_sigma_min,
        blob_sigma_max=args.blob_sigma_max,
        dct_k_max=args.dct_k_max,
        dct_l_max=args.dct_l_max,
        seed=args.seed,
    )
