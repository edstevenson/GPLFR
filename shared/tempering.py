from __future__ import annotations

import warnings

import torch


def beta_independent(y: torch.Tensor) -> float:
    y = y.to(dtype=torch.float64)
    N, D = y.shape
    trR2_star = _debias_tr_r2(_tr_r2_from_data(y), D=int(D), N=int(N))
    return float(D / trR2_star)


def beta_rank1_diag(y: torch.Tensor) -> float:
    y = y.to(dtype=torch.float64)
    N, D = y.shape
    Z = _zscore_cols(y)
    G = Z @ Z.T
    trR2_star = _debias_tr_r2(float((G * G).sum().item() / (N - 1) ** 2), D=int(D), N=int(N))
    lam1 = float(torch.linalg.eigvalsh(G / (N - 1))[-1].item())
    tail = (D - lam1) ** 2 / (trR2_star - lam1**2)
    return float((1.0 + tail) / D)


def beta_independent_xce(
    Y: torch.Tensor,
    *,
    field_mask: torch.Tensor | None = None,
    sh_mask: torch.Tensor | None = None,
) -> float:
    return beta_independent(_flatten_outputs(_subset_full_coverage(Y, field_mask=field_mask)[0], sh_mask=sh_mask))


def beta_rank1_diag_xce(
    Y: torch.Tensor,
    *,
    field_mask: torch.Tensor | None = None,
    sh_mask: torch.Tensor | None = None,
) -> float:
    return beta_rank1_diag(_flatten_outputs(_subset_full_coverage(Y, field_mask=field_mask)[0], sh_mask=sh_mask))


def beta_structured_field_xce(
    Y: torch.Tensor,
    *,
    field_mask: torch.Tensor | None = None,
    sh_mask: torch.Tensor | None = None,
) -> float:
    Yv, N = _subset_full_coverage(Y, field_mask=field_mask)
    Yc, _, A_common = _common_coeffs(Yv, sh_mask=sh_mask)

    beta_within, A_eff = _beta_within_fields(Yc, N=N, return_A_eff=True)
    Zf = _zscore_over_examples_per_coord(Yc).reshape(N * A_common, Yc.shape[-1]).to(dtype=torch.float64)
    M_eff = max(2, int(round(N * A_eff)))
    Rf = (Zf.T @ Zf) / (Zf.shape[0] - 1)
    trR2_F_star = _debias_tr_r2(float((Rf * Rf).sum().item()), D=int(Rf.shape[0]), N=M_eff)
    lam1_F = float(torch.linalg.eigvalsh(Rf)[-1].item())
    F = int(Rf.shape[0])
    tail_F = (F - lam1_F) ** 2 / (trR2_F_star - lam1_F**2)
    return float(((1.0 + tail_F) / F) * beta_within)


def _subset_full_coverage(Y: torch.Tensor, *, field_mask: torch.Tensor | None) -> tuple[torch.Tensor, int]:
    if field_mask is None:
        return Y, int(Y.shape[0])
    keep = field_mask.to(dtype=torch.bool).all(dim=1)
    Yv = Y[keep]
    if int(Yv.shape[0]) < 100:
        msg = f"Full-coverage subset has N={int(Yv.shape[0])} (<100); tempering estimate is likely too noisy."
        warnings.warn(msg)
        raise ValueError(msg)
    return Yv, int(Yv.shape[0])


def _flatten_outputs(Y: torch.Tensor, *, sh_mask: torch.Tensor | None) -> torch.Tensor:
    N, A, F = Y.shape
    if sh_mask is None:
        return Y.reshape(N, A * F).to(dtype=torch.float64)
    return Y.reshape(N, A * F)[:, sh_mask.to(dtype=torch.bool).reshape(-1)].to(dtype=torch.float64)


def _zscore_cols(X: torch.Tensor) -> torch.Tensor:
    X = X.to(dtype=torch.float64)
    Xc = X - X.mean(dim=0, keepdim=True)
    return Xc / torch.sqrt((Xc * Xc).sum(dim=0) / (X.shape[0] - 1))


def _tr_r2_from_data(X: torch.Tensor) -> float:
    Z = _zscore_cols(X)
    G = Z @ Z.T
    return float((G * G).sum().item() / (Z.shape[0] - 1) ** 2)


def _debias_tr_r2(trR2: float, *, D: int, N: int) -> float:
    return float(trR2 - (D * (D - 1)) / (N - 1))


def _common_coeffs(Y: torch.Tensor, *, sh_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    if sh_mask is None:
        return Y, None, int(Y.shape[1])
    common = sh_mask.to(dtype=torch.bool).all(dim=1)
    return Y[:, common, :], sh_mask[common], int(common.sum().item())


def _zscore_over_examples_per_coord(Y: torch.Tensor) -> torch.Tensor:
    Y = Y.to(dtype=torch.float64)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    return Yc / torch.sqrt((Yc * Yc).sum(dim=0) / (Y.shape[0] - 1))


def _beta_within_fields(Y: torch.Tensor, *, N: int, return_A_eff: bool = False) -> float | tuple[float, float]:
    Z = _zscore_over_examples_per_coord(Y)
    _, A, F = Z.shape
    trR2_avg = sum(float(((Z[:, :, j] @ Z[:, :, j].T) ** 2).sum().item() / (N - 1) ** 2) for j in range(F)) / F
    trR2_star = _debias_tr_r2(trR2_avg, D=A, N=N)
    D_eff_star = (A * A) / trR2_star
    beta = float(D_eff_star / A)
    if not return_A_eff:
        return beta
    return beta, float(max(1.0, min(float(A), float(D_eff_star))))
