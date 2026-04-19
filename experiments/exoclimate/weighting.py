from __future__ import annotations

from typing import Any

import numpy as np

FIELD_GROUP_NAMES: list[str] = [
    "temperature",
    "surface_temperature",
    "specific_humidity",
    "wind",
    "surface_pressure",
    "cloud_fraction",
    "olr_asr",
]


def _strip_level_suffix(field_name: str) -> tuple[str, int | None]:
    field_name = field_name.removesuffix("_dex")
    parts = field_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return field_name, None


def retrieve_field_group_index(base_name: str) -> int:
    lower = base_name.lower()
    if lower == "temperature":
        return 0
    if lower == "surface_temperature":
        return 1
    if lower == "specific_humidity":
        return 2
    if lower in ("u", "v", "streamfunction", "velocity_potential"):
        return 3
    if lower in ("surface_pressure", "surface_pressure_frac_dev"):
        return 4
    if lower in ("cloud_fraction", "cloud_propensity"):
        return 5
    if lower in ("olr_cloudy", "asr_cloudy"):
        return 6
    raise NotImplementedError(f"Unknown field group for '{base_name}'")


def field_group_name(field_name: str) -> str:
    group_idx = retrieve_field_group_index(_strip_level_suffix(field_name)[0])
    return FIELD_GROUP_NAMES[group_idx] if group_idx < len(FIELD_GROUP_NAMES) else str(group_idx)


def field_group_counts(field_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for field_name in field_names:
        group_name = field_group_name(field_name)
        counts[group_name] = counts.get(group_name, 0) + 1
    return dict(sorted(counts.items()))


def group_metric_from_per_field(per_field_metric: dict[str, Any]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for field_name, value in per_field_metric.items():
        value = float(value)
        if not np.isfinite(value):
            continue
        grouped.setdefault(field_group_name(field_name), []).append(value)
    return {group_name: float(np.mean(values)) for group_name, values in sorted(grouped.items())}


def build_group_report(metrics: dict[str, Any], *, field_names: list[str]) -> dict[str, Any]:
    return {
        "field_group_counts": field_group_counts(field_names),
        "metrics": {
            metric_name: {
                "per_field_group": group_metric_from_per_field(section.get("per_field") or {}),
            }
            for metric_name, section in metrics.items()
            if isinstance(section, dict) and "per_field" in section
        },
    }
