"""Plot example fields from a toy dataset NPZ."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_examples(
    data_path: str | Path,
    *,
    n: int = 6,
    seed: int = 0,
    indices: list[int] | None = None,
    out: str | Path | None = None,
) -> Path:
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = (Path(__file__).parent / data_path).resolve()
    z = np.load(data_path)
    H = int(z["H"])
    W = int(z["W"])
    Y = np.asarray(z["Y"], dtype=np.float64)
    Y_sig = np.asarray(z["Y_sig"], dtype=np.float64) if "Y_sig" in z.files else None
    Y_nuis = None if Y_sig is None else (Y - Y_sig)

    N = int(Y.shape[0])
    if indices is None:
        idx = np.random.default_rng(seed).choice(N, size=min(int(n), N), replace=False).tolist()
    else:
        idx = [int(i) for i in indices]

    cols = 3 if Y_sig is not None else 1
    fig, axes = plt.subplots(len(idx), cols, figsize=(3.6 * cols, 3.2 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        y = Y[i].reshape(H, W)
        v = float(np.max(np.abs(y))) + 1e-12
        ax = axes[r, 0]
        ax.imshow(y, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"Y (i={i})")
        ax.set_xticks([])
        ax.set_yticks([])

        if Y_sig is not None:
            ys = Y_sig[i].reshape(H, W)
            vn = float(np.max(np.abs(Y_nuis[i]))) + 1e-12  # type: ignore[index]
            ax = axes[r, 1]
            ax.imshow(ys, cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title("Y_sig")
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[r, 2]
            ax.imshow(Y_nuis[i].reshape(H, W), cmap="RdBu_r", vmin=-vn, vmax=vn)  # type: ignore[index]
            ax.set_title("Y - Y_sig")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    if out is None:
        out = data_path.with_suffix("").as_posix() + "_examples.png"
    out_path = Path(out).expanduser().resolve()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[toy.plot_examples] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Plot example 2D fields from toy dataset NPZ")
    p.add_argument("data_path", nargs="?", default="data.npz", help="Input NPZ (relative to this folder if not absolute)")
    p.add_argument("--n", type=int, default=5, help="Number of examples to plot")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for selecting examples")
    p.add_argument("--idx", type=str, default=None, help="Comma-separated indices (overrides --n/--seed)")
    p.add_argument("--out", type=str, default=None, help="Output image path (default: alongside NPZ)")
    args = p.parse_args()

    indices = None if args.idx is None else [int(x.strip()) for x in args.idx.split(",") if x.strip()]
    plot_examples(args.data_path, n=args.n, seed=args.seed, indices=indices, out=args.out)

