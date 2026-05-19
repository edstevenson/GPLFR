from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gplfr import apply_kernel, stabilize_kernel


HERE = Path(__file__).resolve().parent


def main() -> None:
    args = parse_args()
    cfg = json.loads((HERE / args.config).read_text())
    data_path = HERE / cfg["data_path"]
    split_path = HERE / cfg["split_path"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    create_dataset(data_path, cfg["dataset"])
    create_split(data_path, split_path, cfg["split"])
    (data_path.parent / "data_generation.json").write_text(json.dumps({"mode": "generated_from_config", "config": cfg["dataset"], "split": cfg["split"]}, indent=2) + "\n")
    print(f"wrote {data_path}")
    print(f"wrote {split_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    return p.parse_args()


def create_dataset(out_path: Path, cfg: dict[str, Any]) -> None:
    rng = np.random.default_rng(int(cfg["seed"]))
    torch.manual_seed(int(cfg["torch_seed"]))
    n, dx, h, w, d_sig = (int(cfg[k]) for k in ("N", "Dx", "H", "W", "D_sig"))
    x = rng.normal(size=(n, dx)).astype(np.float64)
    w_sig, blob_params = blob_dictionary(h, w, d_sig, float(cfg["blob_sigma_min"]), float(cfg["blob_sigma_max"]), rng)
    ell_sig = rng.uniform(float(cfg["ell_min"]), float(cfg["ell_max"]), size=d_sig).astype(np.float64)
    x_t = torch.as_tensor(x, dtype=torch.float64)
    ell_t = x_t.new_tensor(ell_sig)[:, None].expand(d_sig, dx)
    kx = apply_kernel(cfg["kernel"], x_t, ell_t, x_t.new_tensor(float(cfg["sigma_sig"])))
    z_sig = torch.einsum("dij,dj->id", torch.linalg.cholesky(stabilize_kernel(kx, float(cfg["jitter"]))), torch.randn((d_sig, n), dtype=x_t.dtype)).numpy()
    y_sig = (z_sig @ w_sig.T).astype(np.float64)
    y_nuis = sample_nuisance(n, h, w, float(cfg["ell_nuis"]), float(cfg["sigma_nuis"]), float(cfg["jitter"]), rng)
    y = y_sig + y_nuis + float(cfg["sigma_eps"]) * rng.normal(size=(n, h * w))
    np.savez_compressed(
        out_path,
        X=x.astype(np.float32),
        axis=np.linspace(0.0, 1.0, h * w, dtype=np.float64),
        Y=y.astype(np.float32),
        Y_sig=y_sig.astype(np.float32),
        Y_nuis=y_nuis.astype(np.float32),
        W_sig=w_sig.astype(np.float32),
        blob_params=blob_params.astype(np.float32),
        H=np.array(h),
        W=np.array(w),
        Dx=np.array(dx),
        D_sig=np.array(d_sig),
        kernel=np.array(str(cfg["kernel"])),
        ell=np.array(float(cfg["ell"])),
        ell_mode=np.array(str(cfg["ell_mode"])),
        ell_min=np.array(float(cfg["ell_min"])),
        ell_max=np.array(float(cfg["ell_max"])),
        sigma_sig=np.array(float(cfg["sigma_sig"])),
        ell_nuis=np.array(float(cfg["ell_nuis"])),
        sigma_nuis=np.array(float(cfg["sigma_nuis"])),
        sigma_eps=np.array(float(cfg["sigma_eps"])),
        jitter=np.array(float(cfg["jitter"])),
        blob_sigma_min=np.array(float(cfg["blob_sigma_min"])),
        blob_sigma_max=np.array(float(cfg["blob_sigma_max"])),
        seed=np.array(int(cfg["seed"])),
        ell_sig=ell_sig.astype(np.float32),
    )


def create_split(data_path: Path, out_path: Path, cfg: dict[str, Any]) -> None:
    n = int(np.load(data_path)["X"].shape[0])
    perm = np.random.default_rng(int(cfg["seed"])).permutation(n)
    n_pool, n_val, n_test = (int(cfg[k]) for k in ("n_train_pool", "n_val", "n_test"))
    np.savez(out_path, train_idx=np.sort(perm[:n_pool]), val_idx=np.sort(perm[n_pool : n_pool + n_val]), test_idx=np.sort(perm[n_pool + n_val : n_pool + n_val + n_test]))


def blob_dictionary(h: int, w: int, d_sig: int, sigma_min: float, sigma_max: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    mu_u, mu_v, s = rng.uniform(0.0, h - 1, d_sig), rng.uniform(0.0, w - 1, d_sig), rng.uniform(sigma_min, sigma_max, d_sig)
    return np.stack([np.exp(-((u - mu_u[d]) ** 2 + (v - mu_v[d]) ** 2) / (2.0 * s[d] ** 2)).reshape(h * w) for d in range(d_sig)], axis=1), np.stack([mu_u, mu_v, s], axis=1)


def sample_nuisance(n: int, h: int, w: int, ell: float, sigma: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    grid_h, grid_w = np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64)
    ku = np.exp(-0.5 * ((grid_h[:, None] - grid_h[None, :]) / ell) ** 2) + jitter * np.eye(h)
    kv = np.exp(-0.5 * ((grid_w[:, None] - grid_w[None, :]) / ell) ** 2) + jitter * np.eye(w)
    e = rng.normal(size=(n, h, w)).astype(np.float64)
    return (sigma * np.einsum("uh,nhw,vw->nuv", np.linalg.cholesky(ku), e, np.linalg.cholesky(kv), optimize=True)).reshape(n, h * w)


if __name__ == "__main__":
    main()
