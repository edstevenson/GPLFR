# GPLFR (Gaussian Process Latent Factor Regression)

Lightweight reference implementation of Gaussian Process Latent Factor Regression (GPLFR). This repository is meant for understanding the method, fitting small synthetic problems, and adapting the core model to related structured-output regression tasks.

This is not a frozen paper-reproduction bundle. Paper-scale experiments may depend on datasets, preprocessing, and orchestration code that are outside this lean package.

## Install

```bash
pip install -e .
pip install -e ".[notebooks]"  # optional notebook support
pip install -e ".[dev]"        # optional test support
```

## Quickstart

```python
from gplfr import GPLFR, create_synthetic_data

data = create_synthetic_data(N=64, Dx=2, H=6, W=6, D_sig=3, sigma_nuis=0.3, sigma_eps=0.05, seed=0)
model = GPLFR(
    latent_dim=3,
    kernel="rbf",
    lengthscale_grouping="per_latent",
    amplitude_grouping="fixed",
    amplitude=1.0,
)
fit = model.fit(data["X"][:48], data["Y"][:48], num_steps=200, verbose=False, seed=0)
pred = model.predict(data["X"][48:])
print(fit.final_loss, pred.shape)
```

Or open the notebook:

```bash
jupyter lab demos/quickstart.ipynb
```

## Source files

| Path | Role |
| --- | --- |
| `model.py` | `GPLFR` model and fit/predict API |
| `synthetic.py` | in-memory GP-field synthetic data generator |
| `kernels.py` | covariance kernels and stabilization |
| `demos/` | quickstart and synthetic learning-curve examples |

## Relation to ThousandWorlds

GPLFR was motivated by exoplanet climate emulation work. This repository is the method-level reference package: it keeps the model API and synthetic examples small and reusable. ThousandWorlds is the application-facing exoplanet climate benchmark/dataset project; use ThousandWorlds for problem-specific data pipelines, benchmarking, and reproduction details when those materials are available.

## Citation

```bibtex
@article{gplfr2026,
  title   = {GPLFR: Gaussian Process Latent Factor Regression},
  author  = {Stevenson, Ed and [coauthors]},
  journal = {[venue]},
  year    = {2026},
}
```

See [LICENSE](LICENSE) for usage terms.
