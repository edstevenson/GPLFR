# GPLFR (Gaussian Process Latent Factor Regression)

Lightweight reference implementation of Gaussian Process Latent Factor Regression (GPLFR), accompanying the [paper](https://doi.org/10.48550/arXiv.2606.06576)

## Install

```bash
pip install -e ".[dev]"
```

This installs GPLFR plus the dependencies needed for demos and tests.

## Quickstart

Start with `demos/quickstart.ipynb`, which generates a small synthetic problem, fits GPLFR, and visualizes held-out predictions.

`predict` returns the posterior predictive mean; pass `return_std=True` for the predictive standard deviation (add `include_noise=True` to include observation noise), or call `sample(X_new, n_samples)` to draw from the predictive distribution.

For a script-style reproduction of the synthetic learning-curve experiment, see `demos/synthetic_learning_curve/`; the plot is written to `demos/synthetic_learning_curve/learning_curve.png`.

## Source files

| Path | Role |
| --- | --- |
| `model.py` | `GPLFR` model |
| `synthetic.py` | synthetic data generator |
| `kernels.py` | covariance kernels |
| `demos/` | quickstart and synthetic data learning-curve example |
| `tests/` | smoke tests plus numerical-correctness tests (collapsed likelihood, predictive mean) |

## Relation to exoplanet climate prediction / ThousandWorlds

GPLFR was motivated by the exoplanet climate prediction problem.
This repository is a lightweight reference implementation of GPLFR.
For an expanded version of the exoclimate dataset, more thorough benchmarking, and the problem-specific adaptation of GPLFR, see [ThousandWorlds](https://github.com/edstevenson/ThousandWorlds) and the associated paper (coming soon!).

## Citation

```bibtex
@article{gplfr2026,
  title   = {Gaussian Process Latent Factor Regression for Low-Data, High-Dimensional Output Problems},
  author  = {Stevenson, Edward T. and Wolf, Eric T. and Mak, Mei Ting and Mayne, N. J. and Cranmer, Miles},
  journal = {arXiv preprint arXiv:2606.06576},
  year    = {2026},
  url     = {https://doi.org/10.48550/arXiv.2606.06576}
}
```
