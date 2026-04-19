"""Plot toy-benchmark compression curves (metric vs latent_dim / n_components).

This is a thin wrapper around the (more up-to-date) PyXOpto plotting utilities:
`gplfr.applications.pyxopto.compression_curve`.

It is compatible with toy sweep outputs produced by:
- `scripts/toy_merge_parallel_seeds.py` (writes `metrics.json` with `variants`)
- raw `runs.json` produced by the toy sweep runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._compression_curve_core import plot_compression_curve


def _unique_n_train(json_path: str) -> set[int]:
    p = Path(json_path).expanduser()
    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("variants"), list):
        return {int(v["n_train"]) for v in payload["variants"]}
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return {int(r["n_train"]) for r in payload["runs"]}
    if isinstance(payload, list):
        return {int(r["n_train"]) for r in payload if isinstance(r, dict) and "n_train" in r}
    return set()


def _find_baselines_json(first_json: str) -> Path | None:
    p = Path(first_json).expanduser().resolve()
    for parent in [p.parent, *p.parents]:
        cand = parent / "baselines.json"
        if cand.exists():
            return cand
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot toy compression curves from sweep metrics.json and/or runs.json")
    p.add_argument("jsons", type=str, nargs="+", help="Path(s) to sweep metrics.json (variants) or runs.json")
    p.add_argument("--metric", type=str, default="rmse_sig", help="Metric to plot (default: rmse_sig)")
    p.add_argument("--out", type=str, default=None, help="Optional output image path (default: alongside first input JSON)")
    p.add_argument("--centre", type=str, default="median", choices=["mean", "median"], help="Central tendency to plot")
    p.add_argument("--bounds", type=str, default="iqr", choices=["std", "iqr", "minmax", "none"], help="Error band type")
    p.add_argument("--log-x", action="store_true", help="Use a log x-axis (default: linear x-axis).")
    p.add_argument("--log-y", action="store_true", help="Use a log y-axis (default: linear y-axis).")
    p.add_argument("--title", type=str, default=None, help="Optional plot title")
    p.add_argument("--emulators", type=str, default=None, help="Comma-separated emulator filter (e.g. pca_gp,gplfr)")
    p.add_argument("--variant-regex", type=str, default=None, help="Regex filter applied to variant_id")
    p.add_argument("--n-train", type=str, default=None, help="Comma-separated n_train filter (e.g. 200 or 200,400)")
    p.add_argument("--all-variants", action="store_true", help="Plot each variant separately (default: best per latent dim).")
    p.add_argument("--skip-z", type=str, default=None, help="Comma-separated latent dimensions to skip")
    p.add_argument("--skip-first", action="store_true", help="Skip the smallest latent dimension")
    p.add_argument("--skip-last", action="store_true", help="Skip the largest latent dimension")
    p.add_argument("--suffix", type=str, default=None, help="Optional suffix to append to the filename")
    p.add_argument("--labels", type=str, default=None, help="Comma-separated legend labels for each input JSON (must match number of jsons)")
    p.add_argument("--no-baseline", action="store_true", help="Disable train-mean baseline line (default: enabled).")
    p.add_argument("--baseline-json", type=str, default=None, help="Path to baselines.json (default: search upward from first JSON).")
    p.add_argument("--baseline-key", type=str, default=None, help="Override baseline key in baselines.json (default: metric-specific).")
    p.add_argument("--oracle-baseline", action="store_true", help="Add oracle baseline line (default: disabled).")
    p.add_argument("--oracle-baseline-key", type=str, default=None, help="Override oracle baseline key in baselines.json (default: metric-specific).")
    p.add_argument("--pdf", action="store_true", help="Also write a PDF copy alongside the PNG.")
    args = p.parse_args()

    if args.n_train is None:
        n_trains = sorted(_unique_n_train(args.jsons[0]))
    else:
        n_trains = [int(v) for v in args.n_train.split(",")]
    if len(n_trains) != 1:
        raise ValueError(f"Compression curves require a single fixed n_train; got {n_trains}.")
    n_train = int(n_trains[0])

    out = (
        Path(args.out).expanduser().resolve()
        if args.out
        else Path(args.jsons[0]).expanduser().resolve().parent
        / f"compression_curve_{args.metric}_{args.centre}_{args.bounds}_n{n_train}_{'logx' if args.log_x else 'linx'}_{'logy' if args.log_y else 'liny'}{'' if not args.suffix else '_' + args.suffix}.png"
    )

    fig, ax = plot_compression_curve(
        args.jsons,
        metric=args.metric,
        save_path=out,
        centre=args.centre,
        bounds=args.bounds,
        log_x=bool(args.log_x),
        log_y=args.log_y,
        title=args.title,
        emulators=(args.emulators.split(",") if args.emulators else None),
        variant_regex=args.variant_regex,
        n_train=[n_train],
        best_per_z=not bool(args.all_variants),
        skip_z=([int(v.strip()) for v in args.skip_z.split(",")] if args.skip_z else None),
        skip_first=args.skip_first,
        skip_last=args.skip_last,
        suffix=args.suffix,
        labels=([lbl.strip() for lbl in args.labels.split(",")] if args.labels else None),
    )

    if args.metric == "rmse_sig":
        ax.set_ylabel("RMSE$_{sig}$")
    elif args.metric == "rmse_obs":
        ax.set_ylabel("RMSE$_{obs}$")

    if not args.no_baseline or args.oracle_baseline:
        bj = Path(args.baseline_json).expanduser().resolve() if args.baseline_json else _find_baselines_json(args.jsons[0])
        if bj is None:
            raise FileNotFoundError("Could not locate baselines.json; pass --baseline-json or disable with --no-baseline.")
        d = json.loads(bj.read_text(encoding="utf-8"))
        if not args.no_baseline:
            y = float(d[args.baseline_key]) if args.baseline_key else _baseline_value(baselines_json=bj, metric=args.metric)
            ax.axhline(y, color="k", linestyle="--", linewidth=1.3, label="Train-Mean Baseline")
        if args.oracle_baseline:
            y = (
                float(d[args.oracle_baseline_key])
                if args.oracle_baseline_key
                else _oracle_baseline_value(baselines_json=bj, metric=args.metric)
            )
            ax.axhline(y, color="g", linestyle="--", linewidth=1.3, label="oracle baseline")
        ax.legend(fontsize=("medium" if args.labels else "small"), loc="upper left" if len(args.jsons) > 1 else "best", bbox_to_anchor=(1.02, 1) if len(args.jsons) > 1 else None)

    fig.savefig(out, dpi=300, bbox_inches="tight")
    if args.pdf and out.suffix.lower() != ".pdf":
        out_pdf = out.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"[toy.compression_curve] wrote {out_pdf}")
