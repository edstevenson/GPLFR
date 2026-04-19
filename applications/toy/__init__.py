"""Mechanism toy benchmark: predictable low-variance vs unpredictable high-variance structure.

This benchmark is designed as a didactic example for PCA+GP vs GPLFR:
- PCA+GP prioritizes variance (unsupervised compression).
- GPLFR prioritizes predictability (latent learning under a GP prior).
"""

from .grid import MechanismToyGrid
from .pca_gp import PCAGP
try:
    from .gplfr import GPLFR  # type: ignore
except Exception:  # pragma: no cover
    GPLFR = None  # type: ignore
from . import metrics

__all__ = ["MechanismToyGrid", "PCAGP", "GPLFR", "metrics"]
