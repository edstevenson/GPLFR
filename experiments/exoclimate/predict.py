"""Benchmark-facing exoclimate prediction wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import exoworldsbench as ewb

from .weighting import build_group_report

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[2]


def parse_config_and_overrides(argv: list[str], default_name: str = "ewb-baseline.yaml") -> tuple[Path, dict[str, Any]]:
    cfg_path = APP_ROOT / "configs" / default_name
    overrides: dict[str, Any] = {}
    for arg in argv:
        if "=" not in arg:
            cfg_path = Path(arg).expanduser()
            continue
        key, value = arg.split("=", 1)
        if key == "config":
            cfg_path = Path(value).expanduser()
            continue
        overrides[key] = yaml.safe_load(value)
    return cfg_path, overrides


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return path


def load_config(config: str | Path | dict[str, Any] | None, *, argv: list[str] | None = None, default_name: str = "ewb-baseline.yaml") -> dict[str, Any]:
    cfg_path, overrides = parse_config_and_overrides([] if argv is None else argv, default_name=default_name)
    if config is None:
        cfg_path = cfg_path.resolve()
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    elif isinstance(config, dict):
        cfg = dict(config)
        cfg_path = Path(cfg.get("config_path", APP_ROOT / "configs" / default_name)).expanduser()
    else:
        cfg_path = Path(config).expanduser().resolve()
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg.update(overrides)
    cfg.setdefault("name", cfg_path.stem)
    cfg.setdefault("subset", "multi-GCM_complete-obs-only")
    cfg.setdefault("protocol", "standard")
    cfg.setdefault("include_standard_test_unique", False)
    cfg.setdefault("data_dir", "exoworldsbench/dataset")
    cfg.setdefault("out_dir", f"gplfr/experiments/exoclimate/runs/{cfg['name']}")
    cfg.setdefault("model_path", f"{cfg['out_dir']}/model.npz")
    cfg.setdefault("prediction_path", f"{cfg['out_dir']}/predictions.npz")
    cfg["config_path"] = cfg_path
    for key in ("config_path", "data_dir", "out_dir", "model_path", "prediction_path"):
        cfg[key] = _resolve_path(cfg[key])
    return cfg


def save_submission(path: str | Path, predictions: np.ndarray, bundle: ewb.DataBundle) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        predictions=np.asarray(predictions, dtype=np.float32),
        simulation_id=np.asarray(bundle.test_ids, dtype=np.int32),
        field_names=np.asarray(bundle.field_names),
    )
    return path


def predict(config: str | Path | dict[str, Any] | None = None, *, argv: list[str] | None = None) -> dict[str, Any]:
    cfg = load_config(config, argv=argv)
    with np.load(cfg["model_path"], allow_pickle=False) as npz:
        mean = np.asarray(npz["mean"], dtype=np.float32)
        field_names = np.asarray(npz["field_names"]).tolist()
        subset = str(np.asarray(npz["subset"]).item())
        protocol = str(np.asarray(npz["protocol"]).item())
    bundle = ewb.load(subset, protocol, data_dir=cfg["data_dir"], space="grid")
    stats = ewb.load_stats(subset, cfg["data_dir"])
    predictions = ewb.inverse_preprocess_outputs_grid(
        np.broadcast_to(mean, (len(bundle.X_test), *mean.shape)).copy(),
        field_names,
        stats,
        X=bundle.X_test,
    )[None]
    pred_path = save_submission(cfg["prediction_path"], predictions, bundle)
    metrics = ewb.evaluate.score_submission(pred_path, data_dir=cfg["data_dir"], subset=subset, protocol=protocol)
    out_dir = Path(cfg["out_dir"])
    metrics_path = write_json(out_dir / "metrics.json", metrics)
    group_report = build_group_report(metrics, field_names=bundle.field_names)
    group_report_path = write_json(out_dir / "group_report.json", group_report)
    return {
        "subset": subset,
        "protocol": protocol,
        "prediction_path": pred_path,
        "metrics_path": metrics_path,
        "group_report_path": group_report_path,
        "rmse_per_group": group_report["metrics"]["rmse"]["per_field_group"],
        "energy_score_per_group": group_report["metrics"]["energy_score"]["per_field_group"],
    }


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(predict(argv=sys.argv[1:] if argv is None else argv), indent=2, default=str))


if __name__ == "__main__":
    main()
