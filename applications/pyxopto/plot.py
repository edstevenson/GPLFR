"""Quick plot: log10 reflectance vs radius for a few examples."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .grid import PyXOptoGrid

DATA = Path(__file__).parent / "data" / "refl.npz"
OUT = Path(__file__).parent / "plot_examples.pdf"

grid = PyXOptoGrid.load(DATA)
rng = np.random.default_rng(0)
idx = rng.choice(grid.n_models, size=8, replace=False)

fig, ax = plt.subplots(figsize=(7.2, 4))
for i in idx:
    x = grid._X_full[i]
    label = rf"$g={x[0]:.2f},\ \mu_a={x[1]:.3f},\ \mu_s'={x[2]:.1f}$"
    ax.plot(grid.r_mm, np.log10(grid._Y[i] + 1e-30), lw=0.9, label=label)

ax.set_xlabel(r"$r\ /\ \mathrm{mm}$", fontsize=16)
ax.set_ylabel(r"$\log_{10}\ \mathrm{reflectance}$", fontsize=16)
ax.legend(fontsize=11, ncol=1, frameon=True, loc="center left", bbox_to_anchor=(1.02, 0.5))
ax.grid(True, alpha=0.35, linewidth=0.8)
ax.tick_params(axis="both", labelsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
