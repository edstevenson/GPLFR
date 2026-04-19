"""Plot toy-benchmark learning curves (metric vs n_train).

This is a thin wrapper around the (more up-to-date) PyXOpto plotting utilities:
`gplfr.applications.pyxopto.learning_curve`.

It is compatible with toy sweep outputs produced by:
- `scripts/toy_merge_parallel_seeds.py` (writes `metrics.json` with `variants`)
- raw `runs.json` produced by the toy sweep runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
from ._learning_curve_core import plot_learning_curve

_STYLE_PATH = Path(__file__).resolve().parents[1] / "uai_shared_small.mplstyle"


def _use_uai_style() -> None:
    plt.style.use(str(_STYLE_PATH))


def _find_baselines_json(first_json: str) -> Path | None:
    p = Path(first_json).expanduser().resolve()
    for parent in [p.parent, *p.parents]:
        cand = parent / "baselines.json"
        if cand.exists():
            return cand
    return None


def _find_trainmean_sweep_json(first_json: str) -> Path | None:
    p = Path(first_json).expanduser().resolve()
    for parent in [p.parent, *p.parents]:
        cands = sorted(parent.glob("trainmean_baseline_sweep_*.json"))
        if cands:
            return cands[0]
    return None


def _baseline_value(*, baselines_json: Path, metric: str) -> float:
    d = json.loads(baselines_json.read_text(encoding="utf-8"))
    key = "rmse_sig_trainmeanY" if metric == "rmse_sig" else ("rmse_obs_trainmeanY" if metric == "rmse_obs" else None)
    if key is None:
        raise ValueError(f"No default baseline key for metric={metric!r}; pass --baseline-key explicitly.")
    return float(d[key])


def _oracle_baseline_value(*, baselines_json: Path, metric: str) -> float:
    d = json.loads(baselines_json.read_text(encoding="utf-8"))
    key = "rmse_sig_oracle_ysig" if metric == "rmse_sig" else ("rmse_obs_oracle_ysig" if metric == "rmse_obs" else None)
    if key is None:
        raise ValueError(f"No default oracle baseline key for metric={metric!r}; pass --oracle-baseline-key explicitly.")
    return float(d[key])


def _latent_dim(variant_id: str) -> int | None:
    m = re.search(r"(?:^|,)\s*(?:n_components|z|latent_dim)\s*=\s*(\d+)\s*(?:,|$)", variant_id)
    return int(m.group(1)) if m else None


def _unique_z(json_path: str) -> set[int]:
    p = Path(json_path).expanduser()
    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("variants"), list):
        return {z for v in payload["variants"] if (z := _latent_dim(str(v.get("variant_id", "")))) is not None}
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return {z for r in payload["runs"] if (z := _latent_dim(str(r.get("variant_id", "")))) is not None}
    if isinstance(payload, list):
        return {z for r in payload if isinstance(r, dict) and (z := _latent_dim(str(r.get("variant_id", "")))) is not None}
    return set()


def _combine_regex(a: str | None, b: str) -> str:
    return b if a is None else rf"(?s)(?=.*{a})(?=.*{b}).*"


def _unique_z_filtered(json_path: str, *, variant_regex: str | None) -> set[int]:
    if variant_regex is None:
        return _unique_z(json_path)
    pat = re.compile(variant_regex)
    p = Path(json_path).expanduser()
    payload = json.loads(p.read_text(encoding="utf-8"))
    rows = payload.get("variants") if isinstance(payload, dict) else (payload.get("runs") if isinstance(payload, dict) else payload)
    if not isinstance(rows, list):
        return set()
    out: set[int] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        vid = str(r.get("variant_id", ""))
        if not pat.search(vid):
            continue
        z = _latent_dim(vid)
        if z is not None:
            out.add(int(z))
    return out


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
    elif p.name.endswith("_metrics.json"):
        run_dir = p.parent / p.name[: -len("_metrics.json")]
    else:
        raise ValueError(f"Unsupported metrics path format: {p}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found for metrics file: {p}")
    return run_dir


def _load_variants(metrics_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("variants"), list):
        return [v for v in payload["variants"] if isinstance(v, dict)]
    return []


def _selected_variant_ids(
    metrics_json: Path,
    *,
    metric: str,
    centre: str,
    variant_regex: str | None,
) -> dict[int, str]:
    rows = _load_variants(metrics_json)
    pat = re.compile(variant_regex) if variant_regex is not None else None
    best: dict[int, tuple[float, str]] = {}
    for row in rows:
        n_train = row.get("n_train")
        if n_train is None:
            continue
        vid = str(row.get("variant_id", ""))
        if pat is not None and not pat.search(vid):
            continue
        met = (row.get("metrics") or {}).get(metric)
        if met is None:
            continue
        v = _metric_value(met, centre=centre)
        n = int(n_train)
        if n not in best or v < best[n][0]:
            best[n] = (v, vid)
    return {n: vid for n, (_, vid) in best.items()}


def _seed_traces(
    metrics_json: Path,
    *,
    metric: str,
    selected_variant_ids: dict[int, str],
) -> dict[int, dict[int, float]]:
    if not selected_variant_ids:
        return {}

    run_dir = _resolve_run_dir(metrics_json)
    traces: dict[int, dict[int, float]] = {}

    runs_json = run_dir / "runs.json"
    if not runs_json.exists():
        raise FileNotFoundError(f"{runs_json} not found; cannot build per-seed traces")
    payload = json.loads(runs_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{runs_json} must contain a list of runs")
    for run in payload:
        if not isinstance(run, dict):
            continue
        n_raw = run.get("n_train")
        if n_raw is None:
            continue
        n_train = int(n_raw)
        if selected_variant_ids.get(n_train) != str(run.get("variant_id", "")):
            continue
        met = (run.get("metrics") or {}).get(metric)
        if met is None:
            continue
        seed_raw = run.get("seed", run.get("data_seed"))
        if seed_raw is None:
            continue
        seed = int(seed_raw)
        traces.setdefault(seed, {})[n_train] = _metric_value(met)
    if not traces:
        raise ValueError(f"No per-seed metric rows found in {runs_json} for metric={metric!r}")
    return {seed: traces[seed] for seed in sorted(traces)}


def _trainmean_curve_from_sweep(
    *,
    sweep_json: Path,
    metric: str,
    centre: str,
) -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    payload = json.loads(sweep_json.read_text(encoding="utf-8"))
    summary = payload.get("summary", [])
    runs = payload.get("runs", [])

    curve: dict[int, float] = {}
    for row in summary:
        if not isinstance(row, dict):
            continue
        n_raw = row.get("n_train")
        met = (row.get("metrics") or {}).get(metric)
        if n_raw is None or met is None:
            continue
        curve[int(n_raw)] = _metric_value(met, centre=centre)

    traces: dict[int, dict[int, float]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        n_raw = run.get("n_train")
        seed_raw = run.get("seed", run.get("data_seed"))
        met = (run.get("metrics") or {}).get(metric)
        if n_raw is None or seed_raw is None or met is None:
            continue
        traces.setdefault(int(seed_raw), {})[int(n_raw)] = _metric_value(met)
    return curve, {seed: traces[seed] for seed in sorted(traces)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot toy learning curves from sweep metrics.json and/or runs.json")
    p.add_argument("jsons", type=str, nargs="+", help="Path(s) to sweep metrics.json (variants) or runs.json")
    p.add_argument("--metric", type=str, default="rmse_sig", help="Metric to plot (default: rmse_sig)")
    p.add_argument("--out", type=str, default=None, help="Optional output image path (default: alongside first input JSON)")
    p.add_argument("--centre", type=str, default="median", choices=["mean", "median"], help="Central tendency to plot")
    p.add_argument("--bounds", type=str, default="iqr", choices=["std", "iqr", "minmax", "none"], help="Error band type")
    p.add_argument("--log-y", action="store_true", help="Use log scale on y-axis")
    p.add_argument("--title", type=str, default=None, help="Optional plot title")
    p.add_argument("--emulators", type=str, default=None, help="Comma-separated emulator filter (e.g. pca_gp,gplfr)")
    p.add_argument("--variant-regex", type=str, default=None, help="Regex filter applied to variant_id")
    p.add_argument("--z", type=int, default=None, help="Filter to a single latent dim / n_components (adds an extra variant_id regex).")
    p.add_argument("--suffix", type=str, default=None, help="Optional suffix to append to the default filename")
    p.add_argument("--all-variants", action="store_true", help="Plot each variant separately (default: best per emulator).")
    p.add_argument("--best-by-latent", action="store_true", help="Keep best per (emulator, n_train, latent dim) instead.")
    p.add_argument("--skip-n", type=str, default=None, help="Comma-separated training set sizes (n_train) to skip")
    p.add_argument("--skip-first", action="store_true", help="Skip the smallest training set size")
    p.add_argument("--skip-last", action="store_true", help="Skip the largest training set size")
    p.add_argument("--labels", type=str, default=None, help="Comma-separated legend labels for each input JSON (must match number of jsons)")
    p.add_argument("--no-baseline", action="store_true", help="Disable train-mean baseline line (default: enabled).")
    p.add_argument("--baseline-json", type=str, default=None, help="Path to baselines.json (default: search upward from first JSON).")
    p.add_argument("--baseline-sweep-json", type=str, default=None, help="Path to train-mean sweep JSON with runs/summary over n_train.")
    p.add_argument("--baseline-key", type=str, default=None, help="Override baseline key in baselines.json (default: metric-specific).")
    p.add_argument("--oracle-baseline", action="store_true", help="Add oracle baseline line (default: disabled).")
    p.add_argument("--oracle-baseline-key", type=str, default=None, help="Override oracle baseline key in baselines.json (default: metric-specific).")
    p.add_argument("--pdf", action="store_true", help="Also write a PDF copy alongside the PNG.")
    args = p.parse_args()

    z_pat = None if args.z is None else rf"(?:^|,)\s*(?:n_components|z|latent_dim)\s*=\s*{int(args.z)}\s*(?:,|$)"
    variant_regex = _combine_regex(args.variant_regex, z_pat) if z_pat is not None else args.variant_regex

    tag = "allvariants" if args.all_variants else ("bestbylatent" if args.best_by_latent else "bestbyemu")
    z_vals = {int(args.z)} if args.z is not None else _unique_z_filtered(args.jsons[0], variant_regex=variant_regex)
    z_tag = f"_z{next(iter(z_vals))}" if len(z_vals) == 1 and args.best_by_latent is False else ""

    out = (
        Path(args.out).expanduser().resolve()
        if args.out
        else Path(args.jsons[0]).expanduser().resolve().parent
        / f"learning_curve_{args.metric}_{args.centre}_{args.bounds}{'_logy' if args.log_y else ''}_{tag}{z_tag}{'' if not args.suffix else '_' + args.suffix}.png"
    )

    fig, ax = plot_learning_curve(
        args.jsons,
        metric=args.metric,
        save_path=out,
        centre=args.centre,
        bounds="none",
        log_y=bool(args.log_y),
        title=args.title,
        emulators=(args.emulators.split(",") if args.emulators else None),
        variant_regex=variant_regex,
        suffix=args.suffix,
        best_by_latent=bool(args.best_by_latent),
        best_by_emulator=(False if args.all_variants or args.best_by_latent else True),
        skip_n=([int(v.strip()) for v in args.skip_n.split(",")] if args.skip_n else None),
        skip_first=args.skip_first,
        skip_last=args.skip_last,
        labels=([lbl.strip() for lbl in args.labels.split(",")] if args.labels else None),
        line_width=1.0,
    )
    # pyxopto.plot_learning_curve applies paper style internally; re-apply shared compact style for final formatting.
    _use_uai_style()
    fig.set_size_inches(3.25, 2.4)

    series_labels = [lbl.strip() for lbl in args.labels.split(",")] if args.labels else [
        Path(j).expanduser().resolve().parent.name for j in args.jsons
    ]
    marker_cycle = ["o", "s", "D", "^", "v", "<", ">"]
    label_to_color: dict[str, str] = {}
    for i, series in enumerate(series_labels):
        line = next((ln for ln in ax.lines if ln.get_label() == series), None)
        if line is None:
            continue
        color = "tab:blue" if i == 0 else ("tab:orange" if i == 1 else line.get_color())
        marker = "o" if i == 0 else ("s" if i == 1 else marker_cycle[i % len(marker_cycle)])
        line.set_color(color)
        line.set_marker(marker)
        line.set_markersize(2)
        line.set_linewidth(1.0)
        label_to_color[series] = color

    if not args.all_variants:
        for i, json_path in enumerate(args.jsons):
            series = series_labels[i]
            color = label_to_color.get(series)
            if color is None:
                continue
            selected = _selected_variant_ids(
                Path(json_path),
                metric=args.metric,
                centre=args.centre,
                variant_regex=variant_regex,
            )
            seed_traces = _seed_traces(
                Path(json_path),
                metric=args.metric,
                selected_variant_ids=selected,
            )
            x_vals = sorted(selected)
            if not x_vals:
                continue
            for trace in seed_traces.values():
                y_vals = [trace.get(n, float("nan")) for n in x_vals]
                ax.plot(x_vals, y_vals, color=color, alpha=0.18, linewidth=1.0, zorder=1)

    baseline_y = None
    baseline_curve: dict[int, float] | None = None
    baseline_seed_traces: dict[int, dict[int, float]] = {}
    oracle_y = None
    if not args.no_baseline or args.oracle_baseline:
        bj = Path(args.baseline_json).expanduser().resolve() if args.baseline_json else _find_baselines_json(args.jsons[0])
        if not args.no_baseline:
            sweep_json = (
                Path(args.baseline_sweep_json).expanduser().resolve()
                if args.baseline_sweep_json
                else _find_trainmean_sweep_json(args.jsons[0])
            )
            if sweep_json is not None and sweep_json.exists():
                baseline_curve, baseline_seed_traces = _trainmean_curve_from_sweep(
                    sweep_json=sweep_json,
                    metric=args.metric,
                    centre=args.centre,
                )
            else:
                if bj is None:
                    raise FileNotFoundError(
                        "Could not locate trainmean sweep or baselines.json; pass --baseline-sweep-json/--baseline-json or disable with --no-baseline."
                    )
                d = json.loads(bj.read_text(encoding="utf-8"))
                baseline_y = float(d[args.baseline_key]) if args.baseline_key else _baseline_value(baselines_json=bj, metric=args.metric)
        if args.oracle_baseline:
            if bj is None:
                raise FileNotFoundError("Could not locate baselines.json; pass --baseline-json when using --oracle-baseline.")
            d = json.loads(bj.read_text(encoding="utf-8"))
            oracle_y = (
                float(d[args.oracle_baseline_key])
                if args.oracle_baseline_key
                else _oracle_baseline_value(baselines_json=bj, metric=args.metric)
            )

    x_for_baseline = sorted({int(x) for ln in ax.lines if ln.get_label() in series_labels for x in ln.get_xdata()})
    if baseline_curve and x_for_baseline:
        xb = [x for x in x_for_baseline if x in baseline_curve]
        for trace in baseline_seed_traces.values():
            y_vals = [trace.get(n, float("nan")) for n in xb]
            ax.plot(xb, y_vals, color="k", alpha=0.18, linewidth=1.0, zorder=1)
        ax.plot(
            xb,
            [baseline_curve[n] for n in xb],
            color="k",
            linestyle="--",
            marker="^",
            linewidth=1.0,
            markersize=2,
            label="Train-Mean",
            zorder=4,
        )
    if baseline_y is not None and x_for_baseline:
        ax.plot(
            x_for_baseline,
            [baseline_y] * len(x_for_baseline),
            color="k",
            linestyle="--",
            marker="^",
            linewidth=1.0,
            markersize=2,
            label="Train-Mean",
            zorder=4,
        )
    if oracle_y is not None and x_for_baseline:
        ax.plot(
            x_for_baseline,
            [oracle_y] * len(x_for_baseline),
            color="tab:green",
            linestyle="--",
            linewidth=1.0,
            label="oracle baseline",
            zorder=4,
        )

    if args.metric == "rmse_sig":
        ax.set_ylabel(r"$\mathrm{RMSE}_{\mathrm{sig}}$", fontsize=9)
    elif args.metric == "rmse_obs":
        ax.set_ylabel(r"$\mathrm{RMSE}_{\mathrm{obs}}$", fontsize=9)

    ax.set_xlabel("Training set size", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.minorticks_off()
    ax.grid(False, which="minor")
    ax.grid(alpha=0.25, which="major")
    if args.title is not None:
        ax.set_title(args.title, fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            fontsize=8,
            loc="upper right",
            frameon=True,
        )

    fig.savefig(out, dpi=300, bbox_inches="tight")
    if args.pdf and out.suffix.lower() != ".pdf":
        out_pdf = out.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"[toy.learning_curve] wrote {out_pdf}")
