from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def make_generator(ref: Tensor, rng_seed: int | None) -> torch.Generator | None:
    if rng_seed is None:
        return None
    generator = torch.Generator(device=ref.device) if ref.device.type == "cuda" else torch.Generator()
    generator.manual_seed(rng_seed)
    return generator


def sample_randn(ref: Tensor, shape: tuple[int, ...], generator: torch.Generator | None = None) -> Tensor:
    if generator is None:
        return torch.randn(shape, device=ref.device, dtype=ref.dtype)
    if ref.device.type == "xpu":
        return torch.randn(shape, generator=generator, dtype=ref.dtype).to(device=ref.device)
    return torch.randn(shape, generator=generator, device=ref.device, dtype=ref.dtype)


def sample_randint(
    high: int,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.int64,
    replace: bool = True,
) -> Tensor:
    if replace:
        if generator is None:
            return torch.randint(high, shape, device=device, dtype=dtype)
        if device.type == "xpu":
            return torch.randint(high, shape, generator=generator, dtype=dtype).to(device=device)
        return torch.randint(high, shape, generator=generator, device=device, dtype=dtype)

    total = 1
    for dim in shape:
        total *= dim
    if total > high:
        raise ValueError(f"Cannot draw {total} unique integers in [0, {high}) without replacement.")
    if generator is None:
        perm = torch.randperm(high, device=device)
    elif device.type == "xpu":
        perm = torch.randperm(high, generator=generator).to(device=device)
    else:
        perm = torch.randperm(high, device=device, generator=generator)
    return perm.to(dtype=dtype)[:total].view(shape)


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


_PRECISION_ALIAS_TO_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "single": torch.float32,
    "double": torch.float64,
}


def resolve_precision_dtype(value: str | torch.dtype | None, *, default: torch.dtype = torch.float64) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    dtype = _PRECISION_ALIAS_TO_DTYPE.get(str(value).strip().lower())
    if dtype is None:
        raise ValueError(f"Unsupported precision '{value}'. Expected one of: {', '.join(sorted(_PRECISION_ALIAS_TO_DTYPE))}.")
    return dtype
