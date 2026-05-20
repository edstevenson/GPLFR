# GPLFR (Gaussian Process Latent Factor Regression)

Lightweight reference implementation of Gaussian Process Latent Factor Regression (GPLFR), accompanying the [paper](TODO gplfr paper url)

## Install

```bash
pip install -e ".[dev]"
```

This installs GPLFR plus the dependencies needed for demos and tests.

## Quickstart

Start with `demos/quickstart.ipynb`, which generates a small synthetic problem, fits GPLFR, and visualizes held-out predictions.

For a script-style reproduction of the synthetic learning-curve experiment, see `demos/synthetic_learning_curve/`; the plot is written to `demos/synthetic_learning_curve/learning_curve.png`.

## Source files

| Path | Role |
| --- | --- |
| `model.py` | `GPLFR` model |
| `synthetic.py` | synthetic data generator |
| `kernels.py` | covariance kernels |
| `demos/` | quickstart and synthetic data learning-curve example |

## Relation to exoplanet climate prediction / ThousandWorlds

GPLFR was motivated by the exoplanet climate prediction problem.
This repository is a lightweight reference implementation of GPLFR.
For the exoclimate dataset, more thorough benchmarking, and the problem-specific adaptation of GPLFR, see [ThousandWorlds](https://github.com/edstevenson/ThousandWorlds) and the associated paper (TODO: add paper link).

## Citation

```bibtex
@article{gplfr2026,
  title   = {GPLFR: Gaussian Process Latent Factor Regression},
  author  = {Stevenson, Ed and [coauthors]},
  journal = {[venue]},
  year    = {2026},
  url     = {TODO: add GPLFR paper URL}
}
```
