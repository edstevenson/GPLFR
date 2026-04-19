"""Plot RMSE_sig vs sigma_nuis for toy1e sigma_nuis scans."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from ._style import use_paper_style


def _parse_sigma_from_name(name: str) -> float:
    m = re.search(r"nuis([0-9p]+)-n400-800-z6", name)
    if m is None:
        raise ValueError(f"Could not parse sigma_nuis from {name!r}")
    return float(m.group(1).replace("p", "."))


def _series_key(name: str) -> str | None:
    if name.startswith("pcagp-sigscan-"):
        return "pcagp"
    if name.startswith("gplfr-sigscan-"):
        return "gplfr_pi" if "-pcainit-fz0p3" in name else "gplfr"
    return None


def _load_rows(root: Path, *, exclude_sigma_nuis: list[float] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exclude_sigma_nuis = [] if exclude_sigma_nuis is None else [float(v) for v in exclude_sigma_nuis]
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        sk = _series_key(d.name)
        if sk is None:
            continue
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        sigma_nuis = _parse_sigma_from_name(d.name)
        if any(abs(sigma_nuis - x) <= 1.0e-12 for x in exclude_sigma_nuis):
            continue
        payload = json.loads(mp.read_text(encoding="utf-8"))
        variants = payload.get("variants", [])
        for v in variants:
            stats = v["metrics"]["rmse_sig"]
            rows.append(
                {
                    "series": sk,
                    "sigma_nuis": sigma_nuis,
                    "n_train": int(v["n_train"]),
                    "median": float(stats["median"]),
                    "q25": float(stats["q25"]),
                    "q75": float(stats["q75"]),
                }
            )
    return rows


def _plot(
    rows: list[dict[str, Any]],
    *,
    n_train: int,
    keep_series: list[str],
    out: Path,
    title: str,
    log_y: bool,
) -> None:
    label = {
        "pcagp": "PCA-GP",
        "gplfr": "GPLFR",
        "gplfr_pi": "GPLFR pca_init+fz0.3",
    }
    marker = {"pcagp": "s", "gplfr": "o", "gplfr_pi": "D"}

    use_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for sk in keep_series:
        pts = sorted([r for r in rows if r["series"] == sk and r["n_train"] == int(n_train)], key=lambda r: r["sigma_nuis"])
        if not pts:
            continue
        x = np.asarray([p["sigma_nuis"] for p in pts], dtype=float)
        y = np.asarray([p["median"] for p in pts], dtype=float)
        lo = np.asarray([p["q25"] for p in pts], dtype=float)
        hi = np.asarray([p["q75"] for p in pts], dtype=float)
        line = ax.plot(x, y, marker=marker[sk], linewidth=1.5, label=label[sk])[0]
        ax.fill_between(x, lo, hi, alpha=0.20, color=line.get_color(), linewidth=0)

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma_{\mathrm{nuis}}$")
    ax.set_ylabel(r"RMSE$_{\mathrm{sig}}$")
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)

    xt = sorted({float(r["sigma_nuis"]) for r in rows if r["n_train"] == int(n_train)})
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:g}" for v in xt])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.legend(fontsize="small")
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[toy.sigma_nuis_scan_plots] wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Create sigma_nuis plots for toy1e scans")
    p.add_argument("root", type=str, help="Directory containing merged sweep outputs")
    p.add_argument("--out-dir", type=str, default=None, help="Output figure directory")
    p.add_argument("--exclude-sigma-nuis", type=str, default=None, help="Comma-separated sigma_nuis values to exclude")
    p.add_argument("--log-y", action="store_true", help="Use log y-axis and add _logy suffix to filenames")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "_figures")
    exclude = [float(v.strip()) for v in args.exclude_sigma_nuis.split(",")] if args.exclude_sigma_nuis else None
    rows = _load_rows(root, exclude_sigma_nuis=exclude)
    if not rows:
        raise ValueError(f"No merged metrics.json rows found under {root}")

    suffix = "_logy" if args.log_y else ""
    _plot(rows, n_train=400, keep_series=["pcagp", "gplfr", "gplfr_pi"], out=out_dir / f"sigscan_n400_all{suffix}.png", title="n=400", log_y=bool(args.log_y))
    _plot(rows, n_train=800, keep_series=["pcagp", "gplfr", "gplfr_pi"], out=out_dir / f"sigscan_n800_all{suffix}.png", title="n=800", log_y=bool(args.log_y))
    _plot(rows, n_train=400, keep_series=["pcagp"], out=out_dir / f"sigscan_pcagp_n400{suffix}.png", title="PCA-GP n=400", log_y=bool(args.log_y))
    _plot(rows, n_train=800, keep_series=["pcagp"], out=out_dir / f"sigscan_pcagp_n800{suffix}.png", title="PCA-GP n=800", log_y=bool(args.log_y))
    _plot(rows, n_train=400, keep_series=["gplfr"], out=out_dir / f"sigscan_gplfr_n400{suffix}.png", title="GPLFR n=400", log_y=bool(args.log_y))
    _plot(rows, n_train=800, keep_series=["gplfr"], out=out_dir / f"sigscan_gplfr_n800{suffix}.png", title="GPLFR n=800", log_y=bool(args.log_y))
    _plot(rows, n_train=400, keep_series=["gplfr_pi"], out=out_dir / f"sigscan_gplfrpi_n400{suffix}.png", title="GPLFR pca_init+fz0.3 n=400", log_y=bool(args.log_y))
    _plot(rows, n_train=800, keep_series=["gplfr_pi"], out=out_dir / f"sigscan_gplfrpi_n800{suffix}.png", title="GPLFR pca_init+fz0.3 n=800", log_y=bool(args.log_y))


if __name__ == "__main__":
    main()
