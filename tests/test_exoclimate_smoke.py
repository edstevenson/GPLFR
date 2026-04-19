from __future__ import annotations

import json

import numpy as np

import exoworldsbench as ewb
from gplfr.experiments.exoclimate import FIELD_GROUP_NAMES, predict, train


def _bundle(subset: str, protocol: str, *, field_names: list[str], input_names: list[str], space: str) -> ewb.DataBundle:
    n_train, n_test = 3, 2
    Y_train = np.arange(n_train * len(field_names) * 4, dtype=np.float32).reshape(n_train, len(field_names), 2, 2)
    Y_test = np.arange(n_test * len(field_names) * 4, dtype=np.float32).reshape(n_test, len(field_names), 2, 2)
    return ewb.DataBundle(
        X_train=np.ones((n_train, len(input_names)), dtype=np.float32),
        X_test=np.full((n_test, len(input_names)), 2.0, dtype=np.float32),
        Y_train=Y_train,
        Y_test=Y_test,
        field_mask_train=np.ones((n_train, len(field_names)), dtype=bool),
        field_mask_test=np.ones((n_test, len(field_names)), dtype=bool),
        train_ids=np.arange(n_train, dtype=np.int32),
        test_ids=np.arange(10, 10 + n_test, dtype=np.int32),
        field_names=field_names,
        input_names=input_names,
        meta_train=None,
        meta_test=None,
        space=space,
        subset=subset,
        protocol=protocol,
    )


def test_exoclimate_train_predict_smoke_with_public_ewb_api(monkeypatch, tmp_path) -> None:
    subset = "multi-GCM_complete-obs-only"
    field_names = ewb.canonical_field_names(subset)[:4]
    input_names = list(ewb.CANONICAL_INPUT_NAMES)

    monkeypatch.setattr(
        ewb,
        "load",
        lambda subset, protocol="standard", *, data_dir, space="grid": _bundle(
            subset, protocol, field_names=field_names, input_names=input_names, space=space
        ),
    )
    monkeypatch.setattr(ewb, "load_stats", lambda subset, data_dir: object())
    monkeypatch.setattr(ewb, "preprocess_outputs_grid", lambda Y, field_names, stats, X=None: np.asarray(Y, dtype=np.float32))
    monkeypatch.setattr(
        ewb,
        "inverse_preprocess_outputs_grid",
        lambda Y, field_names, stats, X=None: np.asarray(Y, dtype=np.float32),
    )
    monkeypatch.setattr(
        ewb.evaluate,
        "score_submission",
        lambda path, *, data_dir, subset, protocol: {
            "rmse": {"per_field": {name: i + 1 for i, name in enumerate(field_names)}},
            "energy_score": {"per_field": {name: 0.5 * (i + 1) for i, name in enumerate(field_names)}},
        },
    )

    cfg = {
        "name": "smoke",
        "subset": subset,
        "protocol": "standard",
        "data_dir": tmp_path,
        "out_dir": tmp_path,
        "model_path": tmp_path / "model.npz",
        "prediction_path": tmp_path / "predictions.npz",
    }
    train_summary = train(cfg)
    predict_summary = predict(cfg)

    assert train_summary["field_count"] == len(field_names)
    assert sum(train_summary["field_group_counts"].values()) == len(field_names)
    assert predict_summary["prediction_path"].exists()
    assert predict_summary["metrics_path"].exists()
    assert predict_summary["group_report_path"].exists()

    with np.load(predict_summary["prediction_path"], allow_pickle=False) as npz:
        assert npz["predictions"].shape == (1, 2, len(field_names), 2, 2)
        assert npz["field_names"].tolist() == field_names

    group_report = json.loads(predict_summary["group_report_path"].read_text())
    assert set(group_report["field_group_counts"]) <= set(FIELD_GROUP_NAMES)
    assert group_report["metrics"]["rmse"]["per_field_group"]
