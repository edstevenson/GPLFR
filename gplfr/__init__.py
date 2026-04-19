"""GPLFR core package scaffold."""

from .kernels import apply_kernel, compute_sim_type_kernel, matern32_kernel, matern52_kernel, rbf_kernel, stabilize_kernel
from .linear_trend import build_design_matrix, fit_ridge
from .tempering import beta_independent, beta_independent_xce, beta_rank1_diag, beta_rank1_diag_xce, beta_structured_field_xce
from .utils import make_generator, resolve_precision_dtype, sample_randint, sample_randn, save_json

__all__ = [
    "apply_kernel",
    "beta_independent",
    "beta_rank1_diag",
    "build_design_matrix",
    "compute_sim_type_kernel",
    "fit_ridge",
    "make_generator",
    "matern32_kernel",
    "matern52_kernel",
    "rbf_kernel",
    "resolve_precision_dtype",
    "sample_randint",
    "sample_randn",
    "save_json",
    "stabilize_kernel",
]
