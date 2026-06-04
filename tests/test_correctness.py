"""Numerical correctness tests for the equations the model hinges on.

Unlike the smoke tests (shapes/finiteness), these pin the collapsed likelihood
and the predictive mean against independent dense references.
"""

from __future__ import annotations

import torch
import torch.distributions as dist

from gplfr import GPLFR, apply_kernel, create_synthetic_data, stabilize_kernel


def test_collapsed_loglikelihood_matches_dense_mvn() -> None:
    """_collapsed_loglikelihood == sum_j N(y_j; 0, Z Z^T + sigma^2 I) over output columns."""
    torch.manual_seed(0)
    n, q, output_dim = 9, 3, 5
    Z = torch.randn(n, q, dtype=torch.float64)
    Y = torch.randn(n, output_dim, dtype=torch.float64)
    sigma = torch.tensor(0.4, dtype=torch.float64)

    model = GPLFR(latent_dim=q, jitter=0.0, device="cpu")  # jitter=0 -> exact match to the dense form
    collapsed = model._collapsed_loglikelihood(Y, Z, sigma)

    cov = Z @ Z.T + sigma**2 * torch.eye(n, dtype=torch.float64)
    mvn = dist.MultivariateNormal(torch.zeros(n, dtype=torch.float64), covariance_matrix=cov)
    dense = mvn.log_prob(Y.T).sum()  # Y.T: one row per output column y_j

    assert torch.allclose(collapsed, dense, atol=1e-6)


def test_predict_matches_dense_gp_conditional() -> None:
    """predict() == (dense per-latent GP posterior mean) @ (decoder posterior mean)."""
    data = create_synthetic_data(N=40, Dx=2, H=4, W=4, D_sig=2, seed=0)
    X_train, Y_train, X_test = data["X"][:30], data["Y"][:30], data["X"][30:]
    model = GPLFR(latent_dim=2, kernel="rbf", lengthscale_grouping="per_latent", amplitude_grouping="fixed", amplitude=1.0, device="cpu")
    model.fit(X_train, Y_train, num_steps=30, verbose=False, seed=0)
    pred = torch.from_numpy(model.predict(X_test))

    state = model._state_
    Xtr, Xte = model.X_train_, model._as_tensor(X_test)
    n_train = Xtr.shape[0]
    eye = torch.eye(n_train, dtype=Xtr.dtype)
    latent_means = []
    for j in range(model.latent_dim):
        ell = state["lengthscale"][j] if state["lengthscale"].ndim == 2 else state["lengthscale"]
        K = state["amplitude"] ** 2 * apply_kernel("rbf", Xtr, ell, Xtr.new_tensor(1.0)) + model.latent_noise * eye
        K = stabilize_kernel(K, model.jitter)
        K_star = state["amplitude"] ** 2 * apply_kernel("rbf", Xte, ell, Xte.new_tensor(1.0), X2=Xtr)
        latent_means.append(K_star @ torch.linalg.solve(K, state["Z_train"][:, j]))
    dense_pred = torch.stack(latent_means, dim=1) @ state["mu_W"]

    assert torch.allclose(pred, dense_pred, atol=1e-8)
