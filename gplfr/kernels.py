from __future__ import annotations

import math
from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype)
def rbf_kernel(
    X1: Float[Tensor, "n1 d"],
    ell: Float[Tensor, "*batch d"],
    sigma_x: Float[Tensor, ""],
    X2: Float[Tensor, "n2 d"] | None = None,
) -> Float[Tensor, "*batch n1 n2"]:
    if X2 is None:
        X2 = X1
    diff = X1[:, None, :] - X2[None, :, :]
    if ell.dim() == 1:
        r2 = torch.sum((diff / ell) ** 2, dim=-1)
    else:
        r2 = torch.sum((diff.unsqueeze(0) / ell[:, None, None, :]) ** 2, dim=-1)
    return sigma_x**2 * torch.exp(-0.5 * r2)


@jaxtyped(typechecker=beartype)
def matern32_kernel(
    X1: Float[Tensor, "n1 d"],
    ell: Float[Tensor, "*batch d"],
    sigma_x: Float[Tensor, ""],
    X2: Float[Tensor, "n2 d"] | None = None,
) -> Float[Tensor, "*batch n1 n2"]:
    if X2 is None:
        X2 = X1
    diff = X1[:, None, :] - X2[None, :, :]
    if ell.dim() == 1:
        r2 = torch.sum((diff / ell) ** 2, dim=-1)
    else:
        r2 = torch.sum((diff.unsqueeze(0) / ell[:, None, None, :]) ** 2, dim=-1)
    r = torch.sqrt(r2 + torch.finfo(r2.dtype).eps)
    sqrt3_r = math.sqrt(3.0) * r
    return sigma_x**2 * (1.0 + sqrt3_r) * torch.exp(-sqrt3_r)


@jaxtyped(typechecker=beartype)
def matern52_kernel(
    X1: Float[Tensor, "n1 d"],
    ell: Float[Tensor, "*batch d"],
    sigma_x: Float[Tensor, ""],
    X2: Float[Tensor, "n2 d"] | None = None,
) -> Float[Tensor, "*batch n1 n2"]:
    if X2 is None:
        X2 = X1
    diff = X1[:, None, :] - X2[None, :, :]
    if ell.dim() == 1:
        r2 = torch.sum((diff / ell) ** 2, dim=-1)
    else:
        r2 = torch.sum((diff.unsqueeze(0) / ell[:, None, None, :]) ** 2, dim=-1)
    r = torch.sqrt(r2 + torch.finfo(r2.dtype).eps)
    sqrt5_r = math.sqrt(5.0) * r
    return sigma_x**2 * (1.0 + sqrt5_r + (5.0 / 3.0) * r2) * torch.exp(-sqrt5_r)


@jaxtyped(typechecker=beartype)
def compute_sim_type_kernel(
    L_corr: Float[Tensor, "n_types n_types"],
    r_sim: Float[Tensor, "n_types"],
) -> Float[Tensor, "n_types n_types"]:
    R = L_corr @ L_corr.T
    return r_sim.unsqueeze(1) * r_sim.unsqueeze(0) * R


@jaxtyped(typechecker=beartype)
def apply_kernel(
    kernel: Literal["rbf", "matern32", "matern52"],
    X1: Float[Tensor, "n1 d"],
    ell: Float[Tensor, "*batch d"],
    sigma_x: Float[Tensor, ""],
    X2: Float[Tensor, "n2 d"] | None = None,
) -> Float[Tensor, "*batch n1 n2"]:
    if kernel == "rbf":
        return rbf_kernel(X1, ell, sigma_x, X2=X2)
    if kernel == "matern32":
        return matern32_kernel(X1, ell, sigma_x, X2=X2)
    if kernel == "matern52":
        return matern52_kernel(X1, ell, sigma_x, X2=X2)
    raise ValueError(f"Unknown kernel '{kernel}'. Expected 'rbf', 'matern32', or 'matern52'.")


def stabilize_kernel(K: Tensor, jitter: float) -> Tensor:
    eye = torch.eye(int(K.shape[-1]), device=K.device, dtype=K.dtype)
    K = K + eye * float(jitter) if K.dim() == 2 else K + eye.unsqueeze(0) * float(jitter)
    return 0.5 * (K + K.transpose(-1, -2))
