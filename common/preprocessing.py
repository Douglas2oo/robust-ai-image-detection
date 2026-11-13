"""Small helpers shared by model-specific preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


CLIP_PREPROCESS_VERSION = "openai_clip_preprocess_v1"


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB image while keeping preprocessing separate from augmentation."""

    return image if image.mode == "RGB" else image.convert("RGB")


def preprocess_clip_image(image: Image.Image, clip_preprocess):
    """Apply the transform returned by ``clip.load`` to an RGB image."""

    return clip_preprocess(ensure_rgb(image))


@dataclass
class ClipImagePreprocessor:
    """Picklable wrapper suitable for DataLoader worker processes."""

    transform: object

    def __call__(self, image: Image.Image):
        return preprocess_clip_image(image, self.transform)
