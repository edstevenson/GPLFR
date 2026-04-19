"""Split-axis compression-curve plotter for toy benchmark sweeps.

Usage example:
  micromamba run -n xapm python -m gplfr.applications.toy.plot_split_axis_compression \
    --left-json experiments/toy1e/gplfr-homotopy-n800-z2-4-6-8-10-20-30-40-50-s0--4-log50-pvc9/metrics.json \
    --right-json experiments/toy1e/pcagp-n800-z2-4-6-8-10-20-30-40-50-s0--4-es10-log50-icelake/metrics.json \
    --baseline-json experiments/toy1e/baselines.json \
    --label-left GPLFR \
    --label-right PCA-GP \
    --out-dir experiments/toy1e/_figures \
    --tag gplfr-pvc9_vs_pcagp-icelake_z2-4-6-8-10-20-30-40-50
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

_STYLE_PATH = Path(__file__).resolve().parents[1] / "uai_shared_small.mplstyle"


def _use_uai_style() -> None:
    plt.style.use(str(_STYLE_PATH))


def _rows(metrics_json: Path, metric: str) -> list[tuple[int, float, float, float]]:
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    variants = payload.get("variants", [])
    out: list[tuple[int, float, float, float]] = []
    for v in variants:
        vid = str(v.get("variant_id", ""))
        m = re.search(r"(?:latent_dim|n_components)=([0-9]+)", vid)
        if m is None:
            continue
        z = int(m.group(1))
        met = (v.get("metrics") or {}).get(metric)
        if met is None:
            continue
        out.append((z, float(met["median"]), float(met["q25"]), float(met["q75"])))
    return sorted(out, key=lambda t: t[0])


def _latent_dim(variant_id: str) -> int | None:
    m = re.search(r"(?:latent_dim|n_components)=([0-9]+)", variant_id)
    return int(m.group(1)) if m else None


def _metric_value(metric_payload: Any) -> float:
    if isinstance(metric_payload, dict):
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


def _seed_traces(metrics_json: Path, metric: str) -> dict[int, dict[int, float]]:
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
        z = _latent_dim(str(run.get("variant_id", "")))
        met = (run.get("metrics") or {}).get(metric)
        if z is None or met is None:
            continue
        seed_raw = run.get("seed", run.get("data_seed"))
        if seed_raw is None:
            continue
        seed = int(seed_raw)
        traces.setdefault(seed, {})[int(z)] = _metric_value(met)
    if not traces:
        raise ValueError(f"No per-seed metric rows found in {runs_json} for metric={metric!r}")
    return {seed: traces[seed] for seed in sorted(traces)}


def _trainmean_stats_from_sweep(
    *,
    baseline_sweep_json: Path,
    n_train: int,
    metric: str,
) -> tuple[float, float, float]:
    payload = json.loads(baseline_sweep_json.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = payload.get("summary", [])
    for row in rows:
        if int(row.get("n_train", -1)) != int(n_train):
            continue
        stats = (row.get("metrics") or {}).get(metric)
        if stats is None:
            raise ValueError(f"{baseline_sweep_json} missing metric={metric!r} for n_train={n_train}")
        return float(stats["median"]), float(stats["q25"]), float(stats["q75"])
    raise ValueError(f"{baseline_sweep_json} has no summary row for n_train={n_train}")


def _plot_one(
    *,
    left_rows: list[tuple[int, float, float, float]],
    right_rows: list[tuple[int, float, float, float]],
    left_seed_traces: dict[int, dict[int, float]],
    right_seed_traces: dict[int, dict[int, float]],
    left_label: str,
    right_label: str,
    baseline_median: float,
    baseline_q25: float | None,
    baseline_q75: float | None,
    ylabel: str,
    out_path: Path,
) -> None:
    _use_uai_style()
    x_label_size = 9
    y_label_size = 9
    tick_size = 8
    legend_size = 8
    text_size = 8

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(3.25, 2.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.06},
    )

    series = [
        (left_rows, left_seed_traces, left_label, "#1f77b4", "o"),
        (right_rows, right_seed_traces, right_label, "#ff7f0e", "s"),
    ]
    for rows, seed_traces, label, color, marker in series:
        z = [x for x, _, _, _ in rows]
        med = [m for _, m, _, _ in rows]
        low_idx = [i for i, x in enumerate(z) if x <= 10]
        high_idx = [i for i, x in enumerate(z) if x >= 10]  # keep z=10 on both panels
        x_low = [z[i] for i in low_idx]
        x_high = [z[i] for i in high_idx]

        for trace in seed_traces.values():
            y_seed = [trace.get(int(x), float("nan")) for x in z]
            ax1.plot(x_low, [y_seed[i] for i in low_idx], color=color, alpha=0.18, lw=1.0, zorder=1)
            ax2.plot(x_high, [y_seed[i] for i in high_idx], color=color, alpha=0.18, lw=1.0, zorder=1)

        ax1.plot(x_low, [med[i] for i in low_idx], marker=marker, lw=1.0, ms=2, color=color, label=label, zorder=3)
        ax2.plot(x_high, [med[i] for i in high_idx], marker=marker, lw=1.0, ms=2, color=color, zorder=3)

    ax1.set_xlim(1.5, 10.5)
    ax2.set_xlim(9.5, 52)
    ax2.set_xticks([10, 20, 30, 40, 50])
    ax2.set_xticklabels(["10", "20", "30", "40", "50"])

    z_all = sorted({z for z, *_ in left_rows} | {z for z, *_ in right_rows})
    low_z = [z for z in z_all if z <= 10]
    high_z = [z for z in z_all if z >= 10]
    ax1.plot(
        low_z,
        [baseline_median] * len(low_z),
        color="k",
        linestyle="--",
        linewidth=1.0,
        zorder=4,
        label="Train-Mean",
    )
    ax2.plot(
        high_z,
        [baseline_median] * len(high_z),
        color="k",
        linestyle="--",
        linewidth=1.0,
        zorder=4,
    )

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(labelleft=False, left=False)

    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, lw=1.0)
    ax1.plot((1 - 0.015, 1 + 0.015), (-0.02, +0.02), **kwargs)
    ax1.plot((1 - 0.015, 1 + 0.015), (1 - 0.02, 1 + 0.02), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-0.015, +0.015), (-0.02, +0.02), **kwargs)
    ax2.plot((-0.015, +0.015), (1 - 0.02, 1 + 0.02), **kwargs)

    ax1.set_xlabel("")
    ax2.set_xlabel("")
    fig.subplots_adjust(bottom=0.18)
    fig.supxlabel("Latent dimensionality", y=0.07, fontsize=x_label_size)
    ax1.set_ylabel(ylabel, fontsize=y_label_size)
    ax1.tick_params(labelsize=tick_size)
    ax2.tick_params(labelsize=tick_size)
    ax1.minorticks_off()
    ax2.minorticks_off()
    ax1.grid(alpha=0.25)
    ax2.grid(alpha=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(
        by_label.values(),
        by_label.keys(),
        frameon=True,
        loc="upper right",
        labelcolor="linecolor",
        fontsize=legend_size,
    )

    for tick in ax1.get_xticklabels() + ax2.get_xticklabels():
        tick.set_fontsize(tick_size)
    for tick in ax1.get_yticklabels() + ax2.get_yticklabels():
        tick.set_fontsize(tick_size)
    for text in [ax1.title, ax2.title]:
        text.set_fontsize(text_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def _plot_one_logx(
    *,
    left_rows: list[tuple[int, float, float, float]],
    right_rows: list[tuple[int, float, float, float]],
    left_seed_traces: dict[int, dict[int, float]],
    right_seed_traces: dict[int, dict[int, float]],
    left_label: str,
    right_label: str,
    baseline_median: float,
    ylabel: str,
    out_path: Path,
) -> None:
    _use_uai_style()
    fig, ax = plt.subplots(figsize=(3.25, 2.6))

    series = [
        (left_rows, left_seed_traces, left_label, "#1f77b4", "o"),
        (right_rows, right_seed_traces, right_label, "#ff7f0e", "s"),
    ]
    for rows, seed_traces, label, color, marker in series:
        z = [x for x, _, _, _ in rows]
        med = [m for _, m, _, _ in rows]
        for trace in seed_traces.values():
            y_seed = [trace.get(int(x), float("nan")) for x in z]
            ax.plot(z, y_seed, color=color, alpha=0.18, lw=1.0, zorder=1)
        ax.plot(z, med, marker=marker, lw=1.0, ms=2, color=color, label=label, zorder=3)

    z_all = sorted({z for z, *_ in left_rows} | {z for z, *_ in right_rows})
    ax.plot(
        z_all,
        [baseline_median] * len(z_all),
        color="k",
        linestyle="--",
        linewidth=1.0,
        zorder=4,
        label="Train-Mean",
    )

    ax.set_xscale("log")
    ax.set_xlim(min(z_all) * 0.95, max(z_all) * 1.05)
    ax.set_xticks(z_all)
    ax.set_xticklabels([str(z) for z in z_all])
    ax.set_xlabel("Latent dimensionality")
    ax.set_ylabel(ylabel)
    ax.minorticks_off()
    ax.grid(alpha=0.25, which="major")
    ax.legend(
        frameon=True,
        loc="upper right",
        labelcolor="linecolor",
        bbox_to_anchor=(0.97, 0.955),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def main() -> None:
    p = argparse.ArgumentParser(description="Create split-axis toy compression plots (rmse_sig + rmse_obs).")
    p.add_argument("--left-json", required=True, type=str, help="metrics.json for first curve (e.g. GPLFR).")
    p.add_argument("--right-json", required=True, type=str, help="metrics.json for second curve (e.g. PCA-GP).")
    p.add_argument("--baseline-json", default=None, type=str, help="baselines.json containing scalar train-mean baselines.")
    p.add_argument("--baseline-sweep-json", default=None, type=str, help="JSON from train-mean sweep (contains median/q25/q75 per n_train).")
    p.add_argument("--baseline-n-train", default=None, type=int, help="n_train to use from --baseline-sweep-json (e.g. 400 or 800).")
    p.add_argument("--metrics", default="rmse_sig", type=str, help="Comma-separated metrics to plot (rmse_sig,rmse_obs).")
    p.add_argument("--label-left", required=True, type=str, help="Legend label for left-json curve.")
    p.add_argument("--label-right", required=True, type=str, help="Legend label for right-json curve.")
    p.add_argument("--out-dir", required=True, type=str, help="Output directory for generated PDFs.")
    p.add_argument("--tag", required=True, type=str, help="Filename suffix/tag after ..._brokenx_.")
    p.add_argument("--add-logx-copy", action="store_true", help="Also emit single-axis log-x copy (no broken axis).")
    args = p.parse_args()

    left_json = Path(args.left_json).expanduser().resolve()
    right_json = Path(args.right_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    metrics = {m.strip() for m in str(args.metrics).split(",") if m.strip()}
    valid_metrics = {"rmse_sig", "rmse_obs"}
    if not metrics:
        raise ValueError("--metrics produced an empty set")
    if not metrics.issubset(valid_metrics):
        raise ValueError(f"--metrics must be subset of {sorted(valid_metrics)}, got {sorted(metrics)}")

    if args.baseline_sweep_json:
        if args.baseline_n_train is None:
            raise ValueError("--baseline-n-train is required when --baseline-sweep-json is provided")
        baseline_sweep_json = Path(args.baseline_sweep_json).expanduser().resolve()
        baseline_sig_median, baseline_sig_q25, baseline_sig_q75 = _trainmean_stats_from_sweep(
            baseline_sweep_json=baseline_sweep_json,
            n_train=int(args.baseline_n_train),
            metric="rmse_sig",
        )
        baseline_obs_median, baseline_obs_q25, baseline_obs_q75 = _trainmean_stats_from_sweep(
            baseline_sweep_json=baseline_sweep_json,
            n_train=int(args.baseline_n_train),
            metric="rmse_obs",
        )
    else:
        if not args.baseline_json:
            raise ValueError("Provide --baseline-json or --baseline-sweep-json")
        baseline_json = Path(args.baseline_json).expanduser().resolve()
        baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
        baseline_sig_median = float(baseline["rmse_sig_trainmeanY"])
        baseline_obs_median = float(baseline["rmse_obs_trainmeanY"])
        baseline_sig_q25 = None
        baseline_sig_q75 = None
        baseline_obs_q25 = None
        baseline_obs_q75 = None

    if "rmse_sig" in metrics:
        left_sig = _rows(left_json, "rmse_sig")
        right_sig = _rows(right_json, "rmse_sig")
        left_seed_sig = _seed_traces(left_json, "rmse_sig")
        right_seed_sig = _seed_traces(right_json, "rmse_sig")
        _plot_one(
            left_rows=left_sig,
            right_rows=right_sig,
            left_seed_traces=left_seed_sig,
            right_seed_traces=right_seed_sig,
            left_label=args.label_left,
            right_label=args.label_right,
            baseline_median=baseline_sig_median,
            baseline_q25=baseline_sig_q25,
            baseline_q75=baseline_sig_q75,
            ylabel=r"RMSE$_{\mathrm{sig}}$",
            out_path=out_dir / f"compression_curve_rmse_sig_median_iqr_n800_brokenx_{args.tag}.pdf",
        )
        if args.add_logx_copy:
            _plot_one_logx(
                left_rows=left_sig,
                right_rows=right_sig,
                left_seed_traces=left_seed_sig,
                right_seed_traces=right_seed_sig,
                left_label=args.label_left,
                right_label=args.label_right,
                baseline_median=baseline_sig_median,
                ylabel=r"RMSE$_{\mathrm{sig}}$",
                out_path=out_dir / f"compression_curve_rmse_sig_median_iqr_n800_logx_{args.tag}.pdf",
            )

    if "rmse_obs" in metrics:
        left_obs = _rows(left_json, "rmse_obs")
        right_obs = _rows(right_json, "rmse_obs")
        left_seed_obs = _seed_traces(left_json, "rmse_obs")
        right_seed_obs = _seed_traces(right_json, "rmse_obs")
        _plot_one(
            left_rows=left_obs,
            right_rows=right_obs,
            left_seed_traces=left_seed_obs,
            right_seed_traces=right_seed_obs,
            left_label=args.label_left,
            right_label=args.label_right,
            baseline_median=baseline_obs_median,
            baseline_q25=baseline_obs_q25,
            baseline_q75=baseline_obs_q75,
            ylabel="RMSE_obs",
            out_path=out_dir / f"compression_curve_rmse_obs_median_iqr_n800_brokenx_{args.tag}.pdf",
        )


if __name__ == "__main__":
    main()
