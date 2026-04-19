"""Plot RMSE_sig vs sigma_nuis for toy dataset variants.

This mirrors the visual style of `gplfr.experiments.pyxopto.compression_curve` but
uses sigma_nuis (parsed from sweep directory names) as the x-axis.
"""

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


def _parse_float_tag(s: str) -> float:
    return float(s.replace("p", "."))


def _parse_dir_name(name: str) -> dict[str, Any]:
    emu = "gplfr" if name.startswith("gplfr") else ("pca_gp" if name.startswith("pcagp") else None)
    m_nuis = re.search(r"nuis([0-9p]+)", name)
    m_eps = re.search(r"eps([0-9p]+)", name)
    if emu is None or m_nuis is None or m_eps is None:
        raise ValueError(f"Could not parse emulator/nuis/eps from {name!r}")
    return {"emulator": emu, "sigma_nuis": _parse_float_tag(m_nuis.group(1)), "sigma_eps": _parse_float_tag(m_eps.group(1))}


def _load_metric(p: Path, *, metric: str, centre: str, bounds: str) -> dict[str, float]:
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d["variants"]
    if len(rows) != 1:
        raise ValueError(f"Expected a single variant in {p}; got {len(rows)}")
    stats = rows[0]["metrics"][metric]
    y = float(stats[centre])
    if bounds == "none":
        lo = hi = np.nan
    elif bounds == "std":
        std = float(stats.get("std", 0.0))
        lo, hi = y - std, y + std
    elif bounds == "iqr":
        lo, hi = float(stats["q25"]), float(stats["q75"])
    elif bounds == "minmax":
        lo, hi = float(stats["min"]), float(stats["max"])
    else:
        raise ValueError(f"Unknown bounds={bounds!r}")
    return {"y": y, "lo": lo, "hi": hi}


def _parse_variant_id(variant_id: str) -> dict[str, str]:
    if not variant_id:
        return {}
    out: dict[str, str] = {}
    for part in [x.strip() for x in variant_id.split(",") if x.strip()]:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def _pick_variant(
    variants: list[dict[str, Any]],
    *,
    emulator: str,
    n_train: int,
    z: int,
) -> dict[str, Any]:
    for v in variants:
        if str(v.get("emulator")) != emulator:
            continue
        if int(v.get("n_train", -1)) != int(n_train):
            continue
        parsed = _parse_variant_id(str(v.get("variant_id", "")))
        vz = parsed.get("latent_dim") or parsed.get("n_components") or parsed.get("z")
        if vz is None:
            continue
        if int(vz) == int(z):
            return v
    raise ValueError(f"Could not find emulator={emulator} n_train={n_train} z={z} in variants payload")


def _stats_to_band(stats: dict[str, Any], *, centre: str, bounds: str) -> dict[str, float]:
    y = float(stats[centre])
    if bounds == "none":
        lo = hi = np.nan
    elif bounds == "std":
        std = float(stats.get("std", 0.0))
        lo, hi = y - std, y + std
    elif bounds == "iqr":
        lo, hi = float(stats["q25"]), float(stats["q75"])
    elif bounds == "minmax":
        lo, hi = float(stats["min"]), float(stats["max"])
    else:
        raise ValueError(f"Unknown bounds={bounds!r}")
    return {"y": y, "lo": lo, "hi": hi}


def plot_rmse_vs_sigma_nuis(
    root: Path,
    *,
    sigma_eps: float,
    exclude_sigma_nuis: list[float] | None = None,
    toy1e_gplfr_json: Path | None = None,
    toy1e_pcagp_json: Path | None = None,
    toy1e_sigma_nuis: float = 1.0,
    toy1e_z: int = 6,
    toy1e_n_train: int = 800,
    baseline_json: Path | None = None,
    baseline_key: str = "rmse_sig_trainmeanY",
    metric: str = "rmse_sig",
    centre: str = "median",
    bounds: str = "iqr",
    log_x: bool = False,
    title: str | None = None,
    out: Path | None = None,
) -> Path:
    use_paper_style()

    exclude_sigma_nuis = [] if exclude_sigma_nuis is None else [float(v) for v in exclude_sigma_nuis]

    rows: list[dict[str, Any]] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        meta = _parse_dir_name(d.name)
        if abs(float(meta["sigma_eps"]) - float(sigma_eps)) > 1.0e-12:
            continue
        if any(abs(float(meta["sigma_nuis"]) - x) <= 1.0e-12 for x in exclude_sigma_nuis):
            continue
        y = _load_metric(mp, metric=metric, centre=centre, bounds=bounds)
        rows.append({**meta, **y})

    if not rows:
        raise ValueError(f"No sweep dirs with metrics.json under {root} matched sigma_eps={sigma_eps}")

    by_emu: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_emu.setdefault(str(r["emulator"]), []).append(r)
    for emu in by_emu:
        by_emu[emu] = sorted(by_emu[emu], key=lambda d: float(d["sigma_nuis"]))

    if toy1e_gplfr_json is not None:
        payload = json.loads(toy1e_gplfr_json.read_text(encoding="utf-8"))
        variants = payload["variants"] if isinstance(payload, dict) and isinstance(payload.get("variants"), list) else payload
        if not isinstance(variants, list):
            raise ValueError(f"{toy1e_gplfr_json} is not a recognized metrics JSON (expected dict with variants list).")
        chosen = _pick_variant(variants, emulator="gplfr", n_train=toy1e_n_train, z=toy1e_z)
        band = _stats_to_band(chosen["metrics"][metric], centre=centre, bounds=bounds)
        by_emu.setdefault("gplfr", []).append(
            {"emulator": "gplfr", "sigma_nuis": float(toy1e_sigma_nuis), "sigma_eps": sigma_eps, **band}
        )
        by_emu["gplfr"] = sorted(by_emu["gplfr"], key=lambda d: float(d["sigma_nuis"]))

    if toy1e_pcagp_json is not None:
        payload = json.loads(toy1e_pcagp_json.read_text(encoding="utf-8"))
        variants = payload["variants"] if isinstance(payload, dict) and isinstance(payload.get("variants"), list) else payload
        if not isinstance(variants, list):
            raise ValueError(f"{toy1e_pcagp_json} is not a recognized metrics JSON (expected dict with variants list).")
        chosen = _pick_variant(variants, emulator="pca_gp", n_train=toy1e_n_train, z=toy1e_z)
        band = _stats_to_band(chosen["metrics"][metric], centre=centre, bounds=bounds)
        by_emu.setdefault("pca_gp", []).append(
            {"emulator": "pca_gp", "sigma_nuis": float(toy1e_sigma_nuis), "sigma_eps": sigma_eps, **band}
        )
        by_emu["pca_gp"] = sorted(by_emu["pca_gp"], key=lambda d: float(d["sigma_nuis"]))

    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "d", "p", "8"]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    label_map = {"gplfr": "GPLFR", "pca_gp": "PCA-GP"}
    for i, (emu, rs) in enumerate(sorted(by_emu.items())):
        x = np.asarray([float(r["sigma_nuis"]) for r in rs])
        y = np.asarray([float(r["y"]) for r in rs])
        if bounds == "none":
            lo = hi = None
        else:
            lo = np.asarray([float(r["lo"]) for r in rs])
            hi = np.asarray([float(r["hi"]) for r in rs])
        line = ax.plot(x, y, marker=markers[i % len(markers)], linewidth=1.5, label=label_map.get(emu, emu))[0]
        if bounds != "none":
            ax.fill_between(x, lo, hi, alpha=0.20, color=line.get_color(), linewidth=0)

    if baseline_json is not None:
        d = json.loads(baseline_json.read_text(encoding="utf-8"))
        ax.axhline(float(d[baseline_key]), color="k", linestyle="--", linewidth=1.3, label="Train-Mean Baseline")

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_{\mathrm{nuis}}$")
    ax.set_ylabel(r"RMSE$_{\mathrm{sig}}$")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)
    if title is not None:
        ax.set_title(title)

    xticks = sorted({float(r["sigma_nuis"]) for rs in by_emu.values() for r in rs})
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:g}" for v in xticks])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.legend(fontsize="small")
    fig.tight_layout()

    out = out or (root / f"rmse_sig_vs_sigma_nuis_eps{sigma_eps:g}_{'logx' if log_x else 'linx'}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[toy.nuisance_sweep_plot] wrote {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot RMSE_sig vs sigma_nuis from toy1e-variants sweep outputs")
    p.add_argument("root", type=str, help="Root directory containing sweep output dirs with metrics.json")
    p.add_argument("--sigma-eps", type=float, default=0.01, help="Filter to this sigma_eps (default: 0.01)")
    p.add_argument("--exclude-sigma-nuis", type=str, default=None, help="Comma-separated sigma_nuis values to exclude (e.g. 0.01)")
    p.add_argument("--toy1e-gplfr-json", type=str, default=None, help="Optional toy1e GPLFR metrics.json to add at sigma_nuis=1.0")
    p.add_argument("--toy1e-pcagp-json", type=str, default=None, help="Optional toy1e PCA-GP metrics.json to add at sigma_nuis=1.0")
    p.add_argument("--toy1e-sigma-nuis", type=float, default=1.0, help="x-position for toy1e add-on points (default: 1.0)")
    p.add_argument("--toy1e-z", type=int, default=6, help="latent dim / n_components to select from toy1e metrics (default: 6)")
    p.add_argument("--toy1e-n-train", type=int, default=800, help="n_train to select from toy1e metrics (default: 800)")
    p.add_argument("--baseline-json", type=str, default=None, help="Optional baselines.json path to add train-mean baseline")
    p.add_argument("--baseline-key", type=str, default="rmse_sig_trainmeanY", help="Baseline key (default: rmse_sig_trainmeanY)")
    p.add_argument("--out", type=str, default=None, help="Optional output PNG path")
    p.add_argument("--centre", type=str, default="median", choices=["mean", "median"])
    p.add_argument("--bounds", type=str, default="iqr", choices=["std", "iqr", "minmax", "none"])
    p.add_argument("--log-x", action="store_true", help="Use a log x-axis (default: linear x-axis).")
    p.add_argument("--title", type=str, default=None)
    args = p.parse_args()

    plot_rmse_vs_sigma_nuis(
        Path(args.root).expanduser().resolve(),
        sigma_eps=float(args.sigma_eps),
        exclude_sigma_nuis=([float(x.strip()) for x in args.exclude_sigma_nuis.split(",")] if args.exclude_sigma_nuis else None),
        toy1e_gplfr_json=(Path(args.toy1e_gplfr_json).expanduser().resolve() if args.toy1e_gplfr_json else None),
        toy1e_pcagp_json=(Path(args.toy1e_pcagp_json).expanduser().resolve() if args.toy1e_pcagp_json else None),
        toy1e_sigma_nuis=float(args.toy1e_sigma_nuis),
        toy1e_z=int(args.toy1e_z),
        toy1e_n_train=int(args.toy1e_n_train),
        baseline_json=(Path(args.baseline_json).expanduser().resolve() if args.baseline_json else None),
        baseline_key=str(args.baseline_key),
        centre=str(args.centre),
        bounds=str(args.bounds),
        log_x=bool(args.log_x),
        title=args.title,
        out=(Path(args.out).expanduser().resolve() if args.out else None),
    )
