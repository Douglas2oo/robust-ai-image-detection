"""Deterministic image-space augmentation used before CLIP preprocessing."""

from __future__ import annotations

import io
import random
from typing import Any, Dict, List

from PIL import Image, ImageFilter


def _resampling_bicubic():
    return getattr(Image, "Resampling", Image).BICUBIC


class ClipAugmenter:
    """Apply independently sampled JPEG, resize, and blur transformations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.order: List[str] = list(config.get("order", ["jpeg", "resize", "gaussian_blur"]))
        allowed = {"jpeg", "resize", "gaussian_blur"}
        if set(self.order) != allowed or len(self.order) != len(allowed):
            raise ValueError("augmentation.order must contain jpeg, resize, gaussian_blur once each")

    @staticmethod
    def _should_apply(section: Dict[str, Any], rng: random.Random) -> bool:
        return rng.random() < float(section["probability"])

    @staticmethod
    def _jpeg(image: Image.Image, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(quality), optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()

    @staticmethod
    def _resize_roundtrip(image: Image.Image, scale: float) -> Image.Image:
        width, height = image.size
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=_resampling_bicubic(),
        )
        return resized.resize((width, height), resample=_resampling_bicubic())

    def __call__(self, image: Image.Image, seed: int) -> Image.Image:
        rng = random.Random(int(seed))
        output = image.convert("RGB")
        for name in self.order:
            section = self.config[name]
            if not self._should_apply(section, rng):
                continue
            if name == "jpeg":
                low, high = section["quality_range"]
                output = self._jpeg(output, rng.randint(int(low), int(high)))
            elif name == "resize":
                low, high = section["scale_range"]
                output = self._resize_roundtrip(output, rng.uniform(float(low), float(high)))
            else:
                low, high = section["sigma_range"]
                output = output.filter(ImageFilter.GaussianBlur(rng.uniform(float(low), float(high))))
        return output
