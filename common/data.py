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


def load_rgb_image(data_root: Path, record: Dict[str, str]) -> Image.Image:
    """Load one manifest image as a detached RGB PIL image."""

    image_path = Path(data_root) / record["relative_path"]
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image for {record['image_id']} does not exist: {image_path}"
        )
    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


def _view_seed(base_seed: int, image_id: str, view_index: int) -> int:
    # Python's hash is process-randomized, so use a stable integer seed instead.
    import hashlib

    digest = hashlib.sha256(
        f"{base_seed}:{image_id}:{view_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class ManifestImageDataset:
    """Torch-compatible dataset without importing torch at module import time."""

    def __init__(
        self,
        records: Sequence[Dict[str, str]],
        data_root: Path,
        preprocess: Callable,
        views_per_image: int = 1,
        base_seed: int = 0,
        image_transform: Optional[Callable[[Image.Image, int], Image.Image]] = None,
    ) -> None:
        if views_per_image < 1:
            raise ValueError("views_per_image must be at least 1")
        self.records = list(records)
        self.data_root = Path(data_root)
        self.preprocess = preprocess
        self.views_per_image = int(views_per_image)
        self.base_seed = int(base_seed)
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.records) * self.views_per_image

    def __getitem__(self, index: int) -> Tuple[object, int, str, int]:
        record_index, view_index = divmod(index, self.views_per_image)
        record = self.records[record_index]
        image = load_rgb_image(self.data_root, record)
        seed = _view_seed(self.base_seed, record["image_id"], view_index)
        if self.image_transform is not None:
            image = self.image_transform(image, seed)
        tensor = self.preprocess(image)
        return tensor, int(record["label"]), record["image_id"], view_index
