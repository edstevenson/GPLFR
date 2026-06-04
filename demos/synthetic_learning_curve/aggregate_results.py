from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from run_learning_curve import build_tasks


HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    cfg = json.loads((HERE / p.parse_args().config).read_text())
    runs = [json.loads(p.read_text()) for p in sorted((HERE / "results").glob("n_train_*/seed_*/result.json"))]
    summary = summarize(runs)
    write_plot_inputs(runs, summary)
    expected = {(t["n_train"], t["seed"]) for t in build_tasks(cfg)}
    completed = {(r["config"]["n_train"], r["config"]["seed"]) for r in runs}
    payload = {
        "expected_runs": len(expected),
        "completed_runs": len(completed),
        "missing_runs": [{"n_train": n, "seed": s} for n, s in sorted(expected - completed)],
        "summary": summary,
        "runs": runs,
        "model_py_sha256": sha256(HERE / "../../model.py"),
    }
    (HERE / "aggregate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "n_train": n_train,
            "n": len(rs),
            "rmse_sig": stats([r["test_metrics"]["rmse_sig"] for r in rs]),
            "rmse_obs": stats([r["test_metrics"]["rmse_obs"] for r in rs]),
            "fit_seconds": stats([r["fit_seconds"] for r in rs]),
        }
        for n_train, rs in grouped(runs).items()
    ]


def write_plot_inputs(runs: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    rows = [
        {"n_train": r["config"]["n_train"], "seed": r["config"]["seed"], "rmse_sig": r["test_metrics"]["rmse_sig"], "rmse_obs": r["test_metrics"]["rmse_obs"], "fit_seconds": r["fit_seconds"]}
        for r in sorted(runs, key=lambda r: (r["config"]["n_train"], r["config"]["seed"]))
    ]
    with (HERE / "plot_inputs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "plot_inputs.json").write_text(json.dumps({"summary": summary, "runs": rows}, indent=2, sort_keys=True) + "\n")


def grouped(runs: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        groups[int(r["config"]["n_train"])].append(r)
    return dict(sorted(groups.items()))


def stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"median": float(median(values)), "mean": float(a.mean()), "q25": float(np.quantile(a, 0.25)), "q75": float(np.quantile(a, 0.75)), "min": float(a.min()), "max": float(a.max())}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
