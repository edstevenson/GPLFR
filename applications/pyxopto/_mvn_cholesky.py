"""Small wrapper for a MultivariateNormal parameterized by a Cholesky factor."""

from __future__ import annotations

import pyro.distributions as dist
import torch


class MVNCholesky(dist.MultivariateNormal):
    def __init__(self, *, loc: torch.Tensor, scale_tril: torch.Tensor) -> None:
        super().__init__(loc=loc, scale_tril=scale_tril, validate_args=False)
