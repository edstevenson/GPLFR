"""Plot predicted vs true reflectance for saved model checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

from .grid import PyXOptoGrid
from .utils import (
    apply_x_standardizer,
    inv_log10_transform,
    invert_y_standardizer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load((args.run_dir / "cfg.yaml").read_text())

    # Load model bundle (remap GPU tensors to CPU)
    import io, pickle as _pickle, torch
    model_files = list(args.run_dir.glob("*_model.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No *_model.pkl found in {args.run_dir}")

    class _CPUUnpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
            return super().find_class(module, name)

    with open(model_files[0], "rb") as f:
        bundle = _CPUUnpickler(f).load()

    ## Force emulator to CPU if it was trained on GPU/XPU
    emu_obj = bundle["emulator"]
    if hasattr(emu_obj, "device"):
        emu_obj.device = torch.device("cpu")

    # Load eval grid
    grid = PyXOptoGrid.load(cfg["data_path"])
    split = np.load(cfg["split_path"])
    eval_key = "val_idx" if cfg.get("eval_set") == "val" else "test_idx"
    test_grid = grid._subset(split[eval_key].astype(int))

    # Select random examples
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(test_grid.n_models, size=min(args.n, test_grid.n_models), replace=False)

    # Predict
    emu = bundle["emulator"]
    name = bundle["emulator_name"]
    eps = bundle["eps"]
    pre = bundle["preprocessors"]

    if name == "simplex":
        y_pred_all = emu.predict(test_grid._X_full)
    else:
        X_test, s_test, _, _ = test_grid.load_matrix(dtype=np.float64)
        X_test_z = apply_x_standardizer(X_test, pre["x_mean"], pre["x_std"])
        if name == "pca_icm":
            y_pred_log_z = emu.predict_curve(X_test_z, s_test)
        else:  # gplfr
            y_pred_log_z = emu.predict(X_test_z, s_test)
        y_pred_all = inv_log10_transform(
            invert_y_standardizer(y_pred_log_z, pre["y_mean"], pre["y_std"]), eps=eps
        )

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in idx:
        x = test_grid._X_full[i]
        label = f"g={x[0]:.2f} \u03bca={x[1]:.3f} \u03bcs'={x[2]:.1f}"
        ax.plot(test_grid.r_mm, np.log10(test_grid._Y[i] + eps), c=f"C{i % 10}", lw=1.2, label=label)
        ax.plot(test_grid.r_mm, np.log10(y_pred_all[i] + eps), c=f"C{i % 10}", lw=1.2, ls="--")
    ax.set_xlabel("r (mm)")
    ax.set_ylabel("log10 reflectance")
    ax.set_title(f"PyXOpto: {name} predictions (solid=true, dashed=pred)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = args.run_dir / "pred_examples.png"
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
