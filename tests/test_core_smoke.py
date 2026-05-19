import torch

import gplfr as core


def test_core_public_surface_and_kernels_smoke() -> None:
    assert {"GPLFR", "apply_kernel", "create_synthetic_data"} <= set(core.__all__)
    X = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    K = core.apply_kernel("matern52", X, torch.ones(2), torch.tensor(1.5))
    assert K.shape == (2, 2)
    assert torch.allclose(K, K.T)
    assert torch.allclose(core.stabilize_kernel(K, 1.0e-4), core.stabilize_kernel(K, 1.0e-4).T)


def test_synthetic_data_smoke() -> None:
    data = core.create_synthetic_data(N=12, Dx=2, H=4, W=4, D_sig=2, seed=0)
    assert data["X"].shape == (12, 2)
    assert data["Y"].shape == (12, 16)
    assert data["Y_sig"].shape == (12, 16)
