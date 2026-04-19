from __future__ import annotations
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np


# Parse paths
zip_path = Path(sys.argv[1]).expanduser()
out_path = Path(sys.argv[2]).expanduser()


# Collect HG entries
HG_PATTERN = re.compile(r"/hg/g-(?P<g>[0-9_]+)/mua-(?P<mua>[0-9_]+)-musr-(?P<musr>[0-9_]+)-invcm\.npz$")


def _to_float(value: str) -> float:
    return float(value.replace("_", "."))


with zipfile.ZipFile(zip_path) as zip_file:
    entries: list[tuple[float, float, float, str]] = []
    for entry_name in zip_file.namelist():
        entry_match = HG_PATTERN.search(entry_name)
        if entry_match is None:
            continue
        g_value = _to_float(entry_match.group("g"))
        mua_value = _to_float(entry_match.group("mua"))
        musr_value = _to_float(entry_match.group("musr"))
        entries.append((g_value, mua_value, musr_value, entry_name))

    entries.sort()

    targets_list: list[np.ndarray] = []
    for entry in entries:
        entry_name = entry[3]
        with zip_file.open(entry_name) as entry_file:
            data = np.load(io.BytesIO(entry_file.read()), allow_pickle=True)
            targets_list.append(data["reflectance"])


# Build arrays
inputs = np.array([entry[:3] for entry in entries], dtype=np.float32)
targets = np.stack(targets_list, axis=0).astype(np.float32)


# Save dataset
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(out_path, X=inputs, Y=targets)
