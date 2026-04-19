from __future__ import annotations

import torch
from torch import Tensor


def build_design_matrix(X: Tensor, s: Tensor, *, n_sim_types: int, design_cfg: dict) -> Tensor:
    cols = []
    if design_cfg.get("intercept", True):
        cols.append(X.new_ones((X.shape[0], 1)))
    if design_cfg.get("inputs", True):
        cols.append(X)
    if design_cfg.get("sim_onehot", False):
        oh = torch.nn.functional.one_hot(s.long(), num_classes=n_sim_types).to(dtype=X.dtype, device=X.device)
        if design_cfg.get("intercept", True):
            oh = oh[:, 1:]
        if oh.numel():
            cols.append(oh)
    if not cols:
        raise ValueError("Design matrix has no columns (all design flags are false).")
    return torch.cat(cols, dim=1)


def fit_ridge(
    H: Tensor,
    Y: Tensor,
    *,
    lambda_reg: float,
    field_mask: Tensor | None,
    coeff_mask: Tensor | None = None,
    sh_mask: Tensor | None = None,
) -> Tensor:
    lamI = H.new_tensor(lambda_reg) * torch.eye(H.shape[1], device=H.device, dtype=H.dtype)
    if field_mask is None and coeff_mask is None and sh_mask is None:
        gamma = torch.linalg.solve(H.T @ H + lamI, H.T @ Y.reshape(Y.shape[0], -1))
        return gamma.reshape(H.shape[1], Y.shape[1], Y.shape[2])

    N, A, F = Y.shape
    obs = torch.ones((N, A, F), dtype=torch.bool, device=Y.device)
    if field_mask is not None:
        obs = obs & field_mask.to(device=Y.device, dtype=torch.bool).unsqueeze(1)
    if coeff_mask is not None:
        obs = obs & coeff_mask.to(device=Y.device, dtype=torch.bool).unsqueeze(-1)
    if sh_mask is not None:
        obs = obs & sh_mask.to(device=Y.device, dtype=torch.bool).unsqueeze(0)

    Y_flat = Y.reshape(N, -1)
    obs_flat = obs.reshape(N, -1)
    Gamma_flat = H.new_zeros((H.shape[1], A * F))

    pattern_map: dict[bytes, tuple[Tensor, list[int]]] = {}
    for j in range(obs_flat.shape[1]):
        m = obs_flat[:, j]
        key = m.to(dtype=torch.uint8).cpu().numpy().tobytes()
        pattern_map.setdefault(key, (m, []))[1].append(j)

    for m, idxs_list in pattern_map.values():
        if not bool(m.any()):
            continue
        idxs = H.new_tensor(idxs_list, dtype=torch.long)
        Hm = H[m]
        Ym = Y_flat[m].index_select(1, idxs)
        gamma = torch.linalg.solve(Hm.T @ Hm + lamI, Hm.T @ Ym)
        Gamma_flat.index_copy_(1, idxs, gamma)

    return Gamma_flat.reshape(H.shape[1], A, F)
