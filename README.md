# **GPLFR** (Gaussian Process Latent Factor Regression)

Reference implementation accompanying the methods paper. Three worked examples ship in-tree: 
a synthetic toy problem, a biomedical optics (PyXOpto) emulation problem, and an exoplanet climate emulation problem.

## Install

```bash
pip install -e .
```

For notebooks:

```bash
pip install -e '.[notebooks]'
```

For tests:

```bash
pip install -e '.[dev]'
```

Python 3.10+, PyTorch, and Pyro are required. The exoclimate application expects
the public `exoworldsbench` package as a peer dependency.

## Quickstart

```bash
jupyter lab examples/quickstart.ipynb
```

A small synthetic demo that generates data, fits `applications.toy.GPLFR`, and
plots predictions.

## Worked examples

| Path | What it shows | Scale |
| --- | --- | --- |
| `examples/quickstart.ipynb` | `fit -> predict -> plot` on synthetic data | <1 min |
| `applications/toy/` | Paper toy experiments: sweeps, learning curves, compression curves | minutes-hours |
| `applications/pyxopto/` | Paper PyXOpto reflectance experiments | hours |
| `applications/exoclimate/` | ExoWorldsBench climate-emulation application | depends on dataset |

Each application ships its own `gplfr.py` with a domain-adapted model class on
top of the shared primitives in `gplfr.{kernels,linear_trend,tempering,utils}`.

## CLI

```bash
python -m gplfr.scripts.train config=applications/exoclimate/configs/ewb-baseline.yaml
python -m gplfr.scripts.predict config=applications/exoclimate/configs/ewb-baseline.yaml
```

`train` and `predict` are the supported public entrypoints in this mirror.

<!-- TODO at first gplfr push: insert a "## Reproducing the paper's exoclimate flagship" section
     here, linking the v0.1.0-paper GitHub Release that attaches the reviewer-submission zip.
     See docs/superpowers/plans/2026-04-18-gplfr-public-mirror-tier2.md for the release-time checklist. -->

## Citation

If you use GPLFR in your work, please cite:

```bibtex
@article{gplfr2026,
  title   = {GPLFR: Gaussian Process Latent Factor Regression},
  author  = {Stevenson, Ed and [coauthors]},
  journal = {[venue]},
  year    = {2026},
}
```

Update the BibTeX block with the final venue and coauthor list before the first
public push.

## License

MIT. See [LICENSE](LICENSE).
