"""Benchmark-facing exoclimate train-mean wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import exoworldsbench as ewb

from .weighting import field_group_counts

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


def masked_mean_grid(Y: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(field_mask, dtype=np.float32)[:, :, None, None]
    total = np.where(mask.astype(bool), np.asarray(Y, dtype=np.float32), 0.0).sum(axis=0)
    count = mask.sum(axis=0)
    return np.where(count > 0, total / np.clip(count, 1.0, None), 0.0).astype(np.float32)


def _concat_train_unique(cfg: dict[str, Any], bundle: ewb.DataBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not cfg["include_standard_test_unique"]:
        return bundle.X_train, bundle.Y_train, bundle.field_mask_train, bundle.train_ids
    standard = ewb.load(cfg["subset"], "standard", data_dir=cfg["data_dir"], space="grid")
    keep = ~np.isin(standard.test_ids, bundle.test_ids)
    return (
        np.concatenate([bundle.X_train, standard.X_test[keep]], axis=0),
        np.concatenate([bundle.Y_train, standard.Y_test[keep]], axis=0),
        np.concatenate([bundle.field_mask_train, standard.field_mask_test[keep]], axis=0),
        np.concatenate([bundle.train_ids, standard.test_ids[keep]], axis=0),
    )


def train(config: str | Path | dict[str, Any] | None = None, *, argv: list[str] | None = None) -> dict[str, Any]:
    cfg = load_config(config, argv=argv)
    bundle = ewb.load(cfg["subset"], cfg["protocol"], data_dir=cfg["data_dir"], space="grid")
    stats = ewb.load_stats(cfg["subset"], cfg["data_dir"])
    X_train, Y_train, field_mask, train_ids = _concat_train_unique(cfg, bundle)
    mean = masked_mean_grid(ewb.preprocess_outputs_grid(Y_train, bundle.field_names, stats, X=X_train), field_mask)
    model_path = Path(cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        mean=mean,
        field_names=np.asarray(bundle.field_names),
        input_names=np.asarray(bundle.input_names),
        subset=np.asarray(cfg["subset"]),
        protocol=np.asarray(cfg["protocol"]),
        include_standard_test_unique=np.asarray(bool(cfg["include_standard_test_unique"])),
        train_ids=np.asarray(train_ids, dtype=np.int32),
    )
    summary = {
        "name": cfg["name"],
        "subset": cfg["subset"],
        "protocol": cfg["protocol"],
        "include_standard_test_unique": bool(cfg["include_standard_test_unique"]),
        "train_examples": int(len(train_ids)),
        "test_examples": int(len(bundle.test_ids)),
        "field_count": int(len(bundle.field_names)),
        "field_group_counts": field_group_counts(bundle.field_names),
        "model_path": model_path,
        "config_path": cfg["config_path"],
        "data_dir": cfg["data_dir"],
    }
    write_json(Path(cfg["out_dir"]) / "train_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(train(argv=sys.argv[1:] if argv is None else argv), indent=2, default=str))


if __name__ == "__main__":
    main()
