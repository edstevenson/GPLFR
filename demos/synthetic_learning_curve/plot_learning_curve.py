from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader((HERE / "plot_inputs.csv").open())]
    by_seed, by_n = defaultdict(list), defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
        by_n[int(row["n_train"])].append(row["rmse_sig"])
    xs = sorted(by_n)
    color = "#0072B2"
    fig, ax = plt.subplots(figsize=(3.4, 2.45), constrained_layout=True)
    for seed, seed_rows in sorted(by_seed.items()):
        seed_rows = sorted(seed_rows, key=lambda r: r["n_train"])
        ax.plot([r["n_train"] for r in seed_rows], [r["rmse_sig"] for r in seed_rows], color=color, alpha=0.22, lw=1.0)
    ax.plot(xs, [median(by_n[x]) for x in xs], color=color, marker="o", ms=4.0, lw=2.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("Training set size")
    ax.set_ylabel("RMSE to predictable signal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color="0.88", lw=0.8)
    out_path = HERE / "learning_curve.png"
    fig.savefig(out_path, dpi=300)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
