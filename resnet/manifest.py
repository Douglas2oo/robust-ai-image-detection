"""Manifest validation and deterministic image datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image


REQUIRED_COLUMNS = {
    "image_id",
    "relative_path",
    "label",
    "generator",
    "semantic_class",
    "split",
    "duplicate_group",
}


def read_manifest(path: Path) -> List[Dict[str, str]]:
    """Read and validate one frozen UTF-8 CSV manifest."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"Manifest {path} is missing required columns: {sorted(missing)}"
            )
        records = list(reader)

    seen = set()
    for index, record in enumerate(records, start=2):
        image_id = (record.get("image_id") or "").strip()
        relative_path = (record.get("relative_path") or "").strip()
        if not image_id or not relative_path:
            raise ValueError(f"Manifest {path}:{index} has an empty image_id/path")
        if image_id in seen:
            raise ValueError(f"Manifest {path}:{index} repeats image_id={image_id!r}")
        image_relative_path = Path(relative_path)
        if image_relative_path.is_absolute() or ".." in image_relative_path.parts:
            raise ValueError(f"Manifest {path}:{index} relative_path must stay under data.root")
        seen.add(image_id)
        try:
            label = int(record["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Manifest {path}:{index} has a non-integer label") from exc
        if label not in (0, 1):
            raise ValueError(f"Manifest {path}:{index} label must be 0 or 1")
        record["image_id"] = image_id
        record["relative_path"] = relative_path
        record["label"] = str(label)
    if not records:
        raise ValueError(f"Manifest {path} is empty")
    return records
