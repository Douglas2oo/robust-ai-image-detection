"""Configuration, reproducibility, and cache identity helpers for CLIP runs."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    validate_config(config)
    return config


def _null_paths(value: Any, prefix: str = "") -> Iterable[str]:
    if value is None:
        yield prefix or "<root>"
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _null_paths(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _null_paths(child, f"{prefix}[{index}]")


def validate_config(config: Dict[str, Any]) -> None:
    nulls = list(_null_paths(config))
    if nulls:
        raise ValueError("Configuration contains unset values: " + ", ".join(nulls))
    for key in ("experiment", "seed", "model", "data", "training", "augmentation", "features", "paths"):
        if key not in config:
            raise ValueError(f"Configuration is missing required section: {key}")
    model = config["model"]
    if model.get("provider") != "openai_clip":
        raise ValueError("Only model.provider=openai_clip is supported")
    if model.get("name") != "ViT-L/14":
        raise ValueError("This implementation is configured for CLIP ViT-L/14")
    if model.get("output_dim") != 1:
        raise ValueError("model.output_dim must be 1")
    if int(model.get("feature_dim", 0)) < 1:
        raise ValueError("model.feature_dim must be a positive integer")
    if model.get("normalize_features") is not True:
        raise ValueError("model.normalize_features must be true for both CLIP experiments")
    if not model.get("freeze_encoder", False):
        raise ValueError("The CLIP encoder must be frozen")
    if config["features"].get("mode") not in {"cached_original", "cached_augmented_views"}:
        raise ValueError("features.mode must be cached_original or cached_augmented_views")
    if config["features"].get("views_per_image", 0) < 1:
        raise ValueError("features.views_per_image must be at least 1")
    if not config["augmentation"].get("enabled") and config["features"].get("views_per_image") != 1:
        raise ValueError("Unaugmented feature caches must use exactly one view per image")
    if config["data"].get("image_size") != 224:
        raise ValueError("CLIP ViT-L/14 configuration must use image_size=224")
    if config["data"].get("color_mode") != "RGB":
        raise ValueError("CLIP preprocessing requires data.color_mode=RGB")
    if config["training"].get("loss") != "binary_cross_entropy_with_logits":
        raise ValueError("Only binary_cross_entropy_with_logits is supported")
    if bool(config["augmentation"].get("enabled")) != (
        config["features"].get("mode") == "cached_augmented_views"
    ):
        raise ValueError("augmentation.enabled and features.mode disagree")
    if config["augmentation"].get("enabled") and config["features"].get("views_per_image") != 2:
        raise ValueError("The configured augmented experiment must use exactly K=2 views")
    if not isinstance(config["augmentation"].get("enabled"), bool):
        raise ValueError("augmentation.enabled must be a boolean")
    training = config["training"]
    for key in ("batch_size", "epochs", "learning_rate", "weight_decay", "patience"):
        if key not in training:
            raise ValueError(f"training.{key} is required")
    if training["batch_size"] < 1 or training["epochs"] < 1 or training["patience"] < 1:
        raise ValueError("training batch_size, epochs, and patience must be positive")
    if training["learning_rate"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("training learning_rate must be positive and weight_decay non-negative")
    features = config["features"]
    if features.get("batch_size", 0) < 1 or features.get("num_workers", -1) < 0:
        raise ValueError("features.batch_size must be positive and num_workers non-negative")
    if int(features.get("cache_chunk_size", 512)) < 1:
        raise ValueError("features.cache_chunk_size must be a positive integer")
    for key in ("root", "train_manifest", "val_manifest"):
        if not isinstance(config["data"].get(key), str):
            raise ValueError(f"data.{key} must be a path string")
    if not isinstance(features.get("cache_dir"), str):
        raise ValueError("features.cache_dir must be a path string")
    if config["augmentation"].get("enabled"):
        augmentation = config["augmentation"]
        expected_order = {"jpeg", "resize", "gaussian_blur"}
        order = augmentation.get("order", [])
        if len(order) != 3 or set(order) != expected_order:
            raise ValueError("augmentation.order must list jpeg, resize, gaussian_blur once each")
        specs = (
            ("jpeg", "quality_range", 1, 100),
            ("resize", "scale_range", 0, 1),
            ("gaussian_blur", "sigma_range", 0, None),
        )
        for transform, range_key, low_limit, high_limit in specs:
            section = augmentation.get(transform, {})
            probability = section.get("probability")
            bounds = section.get(range_key)
            if probability is None or not 0 <= float(probability) <= 1:
                raise ValueError(f"augmentation.{transform}.probability must be in [0, 1]")
            if not isinstance(bounds, list) or len(bounds) != 2 or float(bounds[0]) > float(bounds[1]):
                raise ValueError(f"augmentation.{transform}.{range_key} must be an ordered pair")
            if float(bounds[0]) < low_limit:
                raise ValueError(f"augmentation.{transform}.{range_key} is below its valid range")
            if high_limit is not None and float(bounds[1]) > high_limit:
                raise ValueError(f"augmentation.{transform}.{range_key} is above its valid range")


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_checkpoint_identity(model_name: str) -> str:
    """Return OpenAI CLIP's official model URL, which embeds the weight hash."""

    try:
        import clip
    except ImportError:
        # Keeps pure configuration/cache unit tests usable without the model runtime.
        return f"openai-clip-unavailable:{model_name}"
    models = getattr(clip, "_MODELS", None)
    if models is None:
        try:
            from clip.clip import _MODELS as models
        except ImportError:
            models = {}
    model_url = models.get(model_name)
    if not model_url:
        raise ValueError(f"No official OpenAI CLIP checkpoint is registered for {model_name}")
    return str(model_url)


def cache_signature(config: Dict[str, Any], manifest_path: Path, split: str) -> str:
    """Hash only inputs that change encoded features, excluding training seed/output paths."""

    augmentation = config["augmentation"] if split == "train" else {"enabled": False}
    payload = {
        "cache_schema": 2,
        "split": split,
        "manifest_sha256": file_sha256(manifest_path),
        "model": {
            "provider": config["model"]["provider"],
            "name": config["model"]["name"],
            "weights_identity": clip_checkpoint_identity(config["model"]["name"]),
            "feature_dim": config["model"]["feature_dim"],
            "normalize_features": bool(config["model"]["normalize_features"]),
        },
        "image_size": config["data"]["image_size"],
        "color_mode": config["data"]["color_mode"],
        "preprocess": "openai_clip_preprocess_v1",
        "views_per_image": config["features"]["views_per_image"] if split == "train" else 1,
        "augmentation": augmentation,
        "augmentation_implementation": "clip_image_augmentation_v1" if augmentation.get("enabled") else None,
        "augmentation_seed": config["augmentation"].get("seed") if split == "train" else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
