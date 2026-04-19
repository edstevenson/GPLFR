from __future__ import annotations

import torch

import gplfr as core


def test_core_public_surface_and_kernels_smoke() -> None:
    assert {"apply_kernel", "build_design_matrix", "save_json"} <= set(core.__all__)
    assert {"beta_independent_xce", "beta_rank1_diag_xce", "beta_structured_field_xce"} <= set(dir(core))
    X = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    K = core.apply_kernel("matern52", X, torch.ones(2), torch.tensor(1.5))
    assert K.shape == (2, 2)
    assert torch.allclose(K, K.T)
    assert torch.allclose(core.stabilize_kernel(K, 1.0e-4), core.stabilize_kernel(K, 1.0e-4).T)
    sim = core.compute_sim_type_kernel(torch.eye(2), torch.tensor([1.0, 2.0]))
    assert torch.equal(sim.diag(), torch.tensor([1.0, 4.0]))


def test_linear_trend_utils_and_sampling_smoke(tmp_path) -> None:
    H = core.build_design_matrix(
        torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        torch.tensor([0, 1]),
        n_sim_types=2,
        design_cfg={"intercept": True, "inputs": True, "sim_onehot": True},
    )
    gamma = core.fit_ridge(H, torch.arange(8, dtype=torch.float32).reshape(2, 2, 2), lambda_reg=1.0e-3, field_mask=None)
    assert H.shape == (2, 3)
    assert gamma.shape == (3, 2, 2)
    ref = torch.zeros(1, dtype=torch.float32)
    assert torch.equal(core.sample_randn(ref, (3,), core.make_generator(ref, 7)), core.sample_randn(ref, (3,), core.make_generator(ref, 7)))
    assert len(torch.unique(core.sample_randint(10, (4,), device=ref.device, generator=core.make_generator(ref, 7), replace=False))) == 4
    assert core.resolve_precision_dtype("fp64") is torch.float64
    path = tmp_path / "artifact.json"
    core.save_json(path, {"ok": True})
    assert path.exists()


def test_tempering_helpers_return_finite_scalars() -> None:
    y = torch.randn(140, 5, dtype=torch.float64)
    Y = torch.randn(140, 4, 3, dtype=torch.float64)
    mask = torch.ones(140, 3, dtype=torch.bool)
    sh_mask = torch.ones(4, 3, dtype=torch.bool)
    betas = [
        core.beta_independent(y),
        core.beta_rank1_diag(y),
        core.beta_independent_xce(Y, field_mask=mask, sh_mask=sh_mask),
        core.beta_rank1_diag_xce(Y, field_mask=mask, sh_mask=sh_mask),
        core.beta_structured_field_xce(Y, field_mask=mask, sh_mask=sh_mask),
    ]
    assert all(beta > 0 for beta in betas)
