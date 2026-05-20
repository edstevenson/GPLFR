# GPLFR Synthetic Learning-Curve Demo

Runs GPLFR on generated synthetic fields for several training-set sizes.

Fits the settings in `config.json`, aggregates the completed runs, and writes a learning-curve plot.

The synthetic problem uses $D_y=16^2=256$, $D_x=3$, and $D_\text{sig}=6$. Each signal latent uses an RBF kernel with a scalar lengthscale sampled independently as $\ell_q \sim \mathcal{U}(1, 3)$, shared across input dimensions. The nuisance field uses $\ell_\text{nuis}=2$. The output dictionary uses localized blobs with $s_q\sim\mathcal{U}(1, 2)$, $\bar{u}_q\sim\mathcal{U}(0,H-1)$, and $\bar{v}_q\sim\mathcal{U}(0,W-1)$.

From the repository root:

```bash
python demos/synthetic_learning_curve/generate_data.py
python demos/synthetic_learning_curve/run_learning_curve.py --all
python demos/synthetic_learning_curve/aggregate_results.py
python demos/synthetic_learning_curve/plot_learning_curve.py
```

The final plot is written to:

`demos/synthetic_learning_curve/learning_curve.png`
