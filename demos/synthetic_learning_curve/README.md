# GPLFR Synthetic Learning-Curve Demo

This directory runs a small synthetic GPLFR training-set-size curve at constant `inverse_temperature=0.1`, `latent_noise=1e-5`, and `latent_dim=6`.

The data files are generated from `config.json`; generated `.npz` files, run outputs, plots, and Slurm logs are local artifacts rather than files to commit to the repository.

## Local Commands

From the repository root, after installing the package and dependencies:

```bash
python demos/synthetic_learning_curve/generate_data.py
python demos/synthetic_learning_curve/run_learning_curve.py --all
python demos/synthetic_learning_curve/aggregate_results.py
python demos/synthetic_learning_curve/plot_learning_curve.py
```

The local runner is intended for small demos. Larger sweeps can run through Slurm on a GPU node.

## Slurm Commands

Edit `run_array.slurm` for your cluster account, partition, environment activation, and GPU constraints before submitting.

```bash
python demos/synthetic_learning_curve/generate_data.py
sbatch demos/synthetic_learning_curve/run_array.slurm
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,NodeList --parsable2
python demos/synthetic_learning_curve/aggregate_results.py
python demos/synthetic_learning_curve/plot_learning_curve.py
```

Outputs:

- per-task results: `results/n_train_*/seed_*/`
- aggregate: `aggregate.json`
- simple plot inputs: `plot_inputs.csv`, `plot_inputs.json`
- final plots: `plots/learning_curve_rmse_sig_beta0p1_z6.{png,pdf}`
