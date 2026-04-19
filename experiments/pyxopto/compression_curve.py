"""Plot compression curves (metric vs latent_dim/n_components) from PyXOpto sweep outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from ._style import use_paper_style


def _latent_dim(variant_id: str) -> int | None:
    m = re.search(r"(?:^|,)\s*(?:n_components|z|latent_dim)\s*=\s*(\d+)\s*(?:,|$)", variant_id)
    return int(m.group(1)) if m else None


def _parse_variant_id(variant_id: str) -> dict[str, str]:
    if not variant_id:
        return {}
    out: dict[str, str] = {}
    for part in [p.strip() for p in variant_id.split(",") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"Could not parse variant_id part {part!r} in {variant_id!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _strip_latent(variant_id: str) -> str:
    d = _parse_variant_id(variant_id)
    for k in ("n_components", "latent_dim", "z"):
        d.pop(k, None)
    return ",".join(f"{k}={d[k]}" for k in sorted(d.keys()))


def _metric_stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float).reshape(-1)
    if a.size == 0:
        raise ValueError("Cannot compute stats on empty list")
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {
        "mean": float(a.mean()),
        "std": std,
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _metric_value(metric_payload: Any, *, centre: str | None = None) -> float:
    if isinstance(metric_payload, dict):
        if centre is not None and centre in metric_payload:
            return float(metric_payload[centre])
        if "median" in metric_payload:
            return float(metric_payload["median"])
        if "mean" in metric_payload:
            return float(metric_payload["mean"])
    return float(metric_payload)


def _resolve_run_dir(metrics_json: Path) -> Path:
    p = metrics_json.expanduser().resolve()
    if p.name == "metrics.json":
        run_dir = p.parent
    elif p.name.endswith(".metrics.json"):
        run_dir = p.parent / p.name[: -len(".metrics.json")]
    elif p.name.endswith("_metrics.json"):
        run_dir = p.parent / p.name[: -len("_metrics.json")]
    else:
        raise ValueError(f"Unsupported metrics path format: {p}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found for metrics file: {p}")
    return run_dir


def _row_matches_run(row: dict[str, Any], run: dict[str, Any]) -> bool:
    n_row = row.get("n_train")
    n_run = run.get("n_train")
    if n_row is None or n_run is None or int(n_row) != int(n_run):
        return False
    row_vid = str(row.get("variant_id", ""))
    run_vid = str(run.get("variant_id", ""))
    if row_vid and not row_vid.startswith("z=") and run_vid == row_vid:
        return True
    z_row = row.get("_z")
    z_run = _latent_dim(run_vid)
    return z_row is not None and z_run is not None and int(z_row) == int(z_run)


def _series_seed_traces(
    *,
    source_json: Path,
    rs: list[dict[str, Any]],
    metric: str,
    centre: str,
) -> dict[int, dict[int, float]]:
    run_dir = _resolve_run_dir(source_json)
    runs_path = run_dir / "runs.json"
    if not runs_path.exists():
        raise FileNotFoundError(f"{runs_path} not found; cannot draw seed traces")
    payload = json.loads(runs_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{runs_path} must contain a list of runs")

    traces: dict[int, dict[int, float]] = {}
    for row in rs:
        x = int(row["_z"])
        for run in payload:
            if not isinstance(run, dict) or not _row_matches_run(row, run):
                continue
            met = (run.get("metrics") or {}).get(metric)
            seed_raw = run.get("seed", run.get("data_seed"))
            if met is None or seed_raw is None:
                continue
            traces.setdefault(int(seed_raw), {})[x] = _metric_value(met, centre=centre)
    if not traces:
        raise ValueError(f"No seed traces extracted from {runs_path} for metric={metric!r}")
    return {seed: traces[seed] for seed in sorted(traces)}


def _series_style(series: str) -> tuple[str | None, str | None]:
    s = series.lower()
    if "oracle baseline" in s or "oracle" in s:
        return "tab:green", "D"
    if "pca-mlp" in s or "pca_mlp" in s or "pca mlp" in s:
        return "tab:purple", "v"
    return None, None


def _normalize_legend_label(label: str) -> str:
    return "Oracle" if label.strip().lower() == "oracle baseline" else label


def _apply_extra_logy_ticks(ax: plt.Axes, extra_ticks: list[float]) -> None:
    y0, y1 = ax.get_ylim()
    ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
    req = sorted({float(t) for t in extra_ticks if float(t) > 0.0})
    if req:
        ymin = min(ymin, req[0])
        ymax = max(ymax, req[-1])
    base = [float(t) for t in ax.yaxis.get_majorticklocs() if ymin <= float(t) <= ymax and float(t) > 0.0]
    ticks = sorted(set(base + req))
    if y0 <= y1:
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(ymax, ymin)
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))

    tick_set = tuple(ticks)

    def _fmt(val: float, _pos: int) -> str:
        if val <= 0.0:
            return ""
        if not any(math.isclose(val, t, rel_tol=1e-10, abs_tol=0.0) for t in tick_set):
            return ""
        exp = int(math.floor(math.log10(val)))
        mant = val / (10 ** exp)
        if math.isclose(mant, 1.0, rel_tol=1e-10):
            return rf"$10^{{{exp}}}$"
        mant_txt = str(int(round(mant))) if math.isclose(mant, round(mant), rel_tol=1e-10) else f"{mant:g}"
        return rf"${mant_txt}\times 10^{{{exp}}}$"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt))


def _summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for r in runs:
        vid = str(r.get("variant_id", ""))
        key = (str(r["emulator"]), int(r["n_train"]), vid)
        g = grouped.setdefault(key, {"seeds": set(), "metrics": {}})
        g["seeds"].add(int(r.get("seed", r.get("data_seed", 0))))
        for m, v in (r.get("metrics") or {}).items():
            g["metrics"].setdefault(str(m), []).append(float(v))

    out: list[dict[str, Any]] = []
    for (emu, n_train, vid), g in grouped.items():
        entry: dict[str, Any] = {
            "emulator": emu,
            "n_train": n_train,
            "n_seeds": len(g["seeds"]),
            "metrics": {m: _metric_stats(vs) for m, vs in g["metrics"].items()},
        }
        if vid:
            entry["variant_id"] = vid
        out.append(entry)

    out.sort(key=lambda d: (d["emulator"], d["n_train"], d.get("variant_id", "")))
    return out


def _load_variants(json_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("variants"), list):
        return list(payload["variants"])
    if isinstance(payload, dict) and isinstance(payload.get("summary"), list):  # PHOENIX-style
        return list(payload["summary"])
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return _summarize_runs(payload["runs"])
    if isinstance(payload, list):  # runs.json OR already-aggregated list
        if payload and isinstance(payload[0], dict) and isinstance(payload[0].get("metrics"), dict):
            m0 = next(iter(payload[0]["metrics"].values()), None)
            if isinstance(m0, dict) and "mean" in m0:
                return list(payload)
        return _summarize_runs(payload)
    raise ValueError(f"{json_path} is not a recognized sweep/run JSON (expected variants/summary/runs).")


def plot_compression_curve(
    jsons: str | Path | list[str | Path],
    *,
    metric: str = "mae",
    title: str | None = None,
    save_path: str | Path | None = None,
    centre: str = "mean",
    bounds: str = "std",
    log_x: bool = True,
    log_y: bool = False,
    emulators: list[str] | None = None,
    variant_regex: str | None = None,
    n_train: list[int] | None = None,
    best_per_z: bool = False,
    skip_z: list[int] | None = None,
    skip_first: bool = False,
    skip_last: bool = False,
    suffix: str | None = None,
    labels: list[str] | None = None,
    line_width: float = 1.5,
    style_path: str | Path | None = None,
    fig_size: tuple[float, float] | None = None,
    legend_inside: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    use_paper_style()
    if style_path is not None:
        plt.style.use(str(style_path))

    paths = [jsons] if isinstance(jsons, (str, Path)) else jsons
    paths = [Path(p) for p in paths]
    multi = len(paths) > 1

    labels_norm = [_normalize_legend_label(lbl.strip()) for lbl in labels] if labels is not None else None
    if labels_norm is not None and len(labels_norm) != len(paths):
        raise ValueError(f"labels ({len(labels)}) must match number of jsons ({len(paths)})")

    rows: list[dict[str, Any]] = []
    source_to_path: dict[str, Path] = {}
    for i, p in enumerate(paths):
        src = labels_norm[i] if labels_norm else p.resolve().parent.name
        source_to_path[src] = p.resolve()
        rows.extend([{**r, "_source": src} for r in _load_variants(p)])

    if emulators:
        keep = {e.lower() for e in emulators}
        rows = [r for r in rows if str(r.get("emulator", "")).lower() in keep]

    if n_train is not None:
        keep_n = {int(v) for v in n_train}
        rows = [r for r in rows if int(r["n_train"]) in keep_n]

    if variant_regex is not None:
        pat = re.compile(variant_regex)
        rows = [r for r in rows if pat.search(str(r.get("variant_id", "")))]

    rows = [{**r, "_z": _latent_dim(str(r.get("variant_id", "")))} for r in rows]
    rows = [r for r in rows if r["_z"] is not None]
    if not rows:
        raise ValueError("No rows with a parsable latent dimension (expected n_components=... or latent_dim=... in variant_id).")

    if best_per_z:
        keyed: dict[tuple[str, int, int, str], dict[str, Any]] = {}
        for r in rows:
            key = (str(r["emulator"]), int(r["n_train"]), int(r["_z"]), str(r["_source"]))
            v = float(r["metrics"][metric][centre])
            if key not in keyed or v < float(keyed[key]["metrics"][metric][centre]):
                keyed[key] = {**r, "variant_id": f"z={int(r['_z'])}"}
        rows = list(keyed.values())

    n_trains = sorted({int(r["n_train"]) for r in rows})
    by_series: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if labels:
            series = r["_source"]
        else:
            emu, nt = str(r["emulator"]), int(r["n_train"])
            vid = str(r.get("variant_id", ""))
            other = _strip_latent(vid)
            base = f"{emu} ({other})" if other else emu
            series = f"{base} (n={nt})" if len(n_trains) > 1 else base
            if multi:
                series = f"{series} [{r['_source']}]"
        by_series.setdefault(series, []).append(r)

    all_z = sorted({int(r["_z"]) for rs in by_series.values() for r in rs})
    to_skip = set(skip_z or [])
    if skip_first and all_z:
        to_skip.add(all_z[0])
    if skip_last and all_z:
        to_skip.add(all_z[-1])

    if to_skip:
        for s in by_series:
            by_series[s] = [r for r in by_series[s] if r["_z"] not in to_skip]
        all_z = sorted({int(r["_z"]) for rs in by_series.values() for r in rs})

    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "d", "p", "8"]
    metric_label = {"rmse": "RMSE", "mae": "MAE", "maqe_0.95": "MAQE_0.95"}.get(metric, metric.upper())

    series_items = (
        [(lbl, by_series[lbl]) for lbl in labels_norm if lbl in by_series] if labels_norm else sorted(by_series.items())
    )

    fig, ax = plt.subplots(figsize=(fig_size if fig_size is not None else ((8.5, 4.5) if multi else (7.0, 4.5))))
    for i, (series, rs) in enumerate(series_items):
        rs = sorted(rs, key=lambda d: int(d["_z"]))
        x = np.asarray([int(r["_z"]) for r in rs])
        y = np.asarray([float(r["metrics"][metric][centre]) for r in rs])
        if bounds == "none":
            lo = hi = None
        elif bounds == "std":
            std = np.asarray([float(r["metrics"][metric].get("std", 0.0)) for r in rs])
            lo, hi = y - std, y + std
        elif bounds == "iqr":
            lo = np.asarray([float(r["metrics"][metric]["q25"]) for r in rs])
            hi = np.asarray([float(r["metrics"][metric]["q75"]) for r in rs])
        elif bounds == "minmax":
            lo = np.asarray([float(r["metrics"][metric]["min"]) for r in rs])
            hi = np.asarray([float(r["metrics"][metric]["max"]) for r in rs])
        else:
            raise ValueError(f"Unknown bounds={bounds!r}; expected std|iqr|minmax|none")

        color, marker = _series_style(series)
        line = ax.plot(
            x,
            y,
            marker=(marker or markers[i % len(markers)]),
            linewidth=line_width,
            label=series,
            zorder=3,
            **({"color": color} if color else {}),
        )[0]
        if source_json := source_to_path.get(series):
            seed_traces = _series_seed_traces(
                source_json=source_json,
                rs=rs,
                metric=metric,
                centre=centre,
            )
            for trace in seed_traces.values():
                y_seed = [trace.get(int(z), float("nan")) for z in x]
                ax.plot(x, y_seed, color=line.get_color(), alpha=0.18, linewidth=line_width, zorder=1)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
        _apply_extra_logy_ticks(ax, [2e-2, 5e-3])
    ax.set_xlabel("Latent dimensionality")
    ax.set_ylabel(metric_label)
    if all_z:
        ax.set_xticks(all_z)
        ax.set_xticklabels([str(v) for v in all_z])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)
    if multi:
        if legend_inside:
            ax.legend(loc="upper right")
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
            fig.subplots_adjust(right=0.78)
    else:
        ax.legend()
        fig.tight_layout()

    if save_path is None:
        name = f"compression_curve_{metric}_{centre}_{bounds}_{'logx' if log_x else 'linx'}_{'logy' if log_y else 'liny'}"
        if to_skip:
            name += f"_skipz{'-'.join(map(str, sorted(to_skip)))}"
        if suffix:
            name += f"_{suffix}"
        save_path = paths[0].resolve().parent / f"{name}.png"
    else:
        save_path = Path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[pyxopto.compression_curve] wrote {save_path}")
    return fig, ax


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot PyXOpto compression curves from sweep metrics.json and/or runs.json")
    p.add_argument("jsons", type=str, nargs="+", help="Path(s) to sweep metrics.json (variants) or runs.json")
    p.add_argument("--metric", type=str, default="rmse", help="Metric to plot (e.g. rmse, mae, maqe_0.95)")
    p.add_argument("--out", type=str, default=None, help="Optional output image path (default: alongside first input JSON)")
    p.add_argument("--centre", type=str, default="median", choices=["mean", "median"], help="Central tendency to plot")
    p.add_argument("--bounds", type=str, default="iqr", choices=["std", "iqr", "minmax", "none"], help="Error band type")
    p.add_argument("--linear-x", action="store_true", help="Use a linear x-axis (default: log x-axis).")
    p.add_argument("--log-y", action="store_true", help="Use a log y-axis (default: linear y-axis).")
    p.add_argument("--title", type=str, default=None, help="Optional plot title")
    p.add_argument("--emulators", type=str, default=None, help="Comma-separated emulator filter (e.g. pca_icm,gplfr)")
    p.add_argument("--variant-regex", type=str, default=None, help="Regex filter applied to variant_id")
    p.add_argument("--n-train", type=str, default=None, help="Comma-separated n_train filter (e.g. 200 or 200,400)")
    p.add_argument("--best-per-z", action="store_true", help="For each (emulator, n_train, latent_dim), keep only the best variant.")
    p.add_argument("--skip-z", type=str, default=None, help="Comma-separated latent dimensions to skip")
    p.add_argument("--skip-first", action="store_true", help="Skip the smallest latent dimension")
    p.add_argument("--skip-last", action="store_true", help="Skip the largest latent dimension")
    p.add_argument("--suffix", type=str, default=None, help="Optional suffix to append to the filename")
    p.add_argument("--labels", type=str, default=None, help="Comma-separated legend labels for each input JSON (must match number of jsons)")
    p.add_argument("--line-width", type=float, default=1.5, help="Line width for plotted series")
    p.add_argument("--style-path", type=str, default=None, help="Optional matplotlib style file path to apply after paper style.")
    p.add_argument("--fig-width", type=float, default=None, help="Optional figure width in inches.")
    p.add_argument("--fig-height", type=float, default=None, help="Optional figure height in inches.")
    p.add_argument("--legend-inside", action="store_true", help="Place legend inside plot area.")
    args = p.parse_args()
    if (args.fig_width is None) ^ (args.fig_height is None):
        raise ValueError("--fig-width and --fig-height must be provided together")

    plot_compression_curve(
        args.jsons,
        metric=args.metric,
        save_path=args.out,
        centre=args.centre,
        bounds=args.bounds,
        log_x=not args.linear_x,
        log_y=args.log_y,
        title=args.title,
        emulators=(args.emulators.split(",") if args.emulators else None),
        variant_regex=args.variant_regex,
        n_train=([int(v) for v in args.n_train.split(",")] if args.n_train else None),
        best_per_z=args.best_per_z,
        skip_z=([int(v.strip()) for v in args.skip_z.split(",")] if args.skip_z else None),
        skip_first=args.skip_first,
        skip_last=args.skip_last,
        suffix=args.suffix,
        labels=([lbl.strip() for lbl in args.labels.split(",")] if args.labels else None),
        line_width=float(args.line_width),
        style_path=(Path(args.style_path).expanduser().resolve() if args.style_path else None),
        fig_size=((float(args.fig_width), float(args.fig_height)) if args.fig_width is not None else None),
        legend_inside=bool(args.legend_inside),
    )
