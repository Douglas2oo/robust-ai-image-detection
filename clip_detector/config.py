"""CLIP 实验的配置读取、路径解析与随机种子
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ("data_root", "manifest_dir", "cache_dir", "checkpoint_dir", "output_dir")
DEFAULT_PATHS = dict(data_root="data", manifest_dir="manifests", cache_dir="cache/clip",
                     checkpoint_dir="checkpoints", output_dir="outputs")


def resolve_path(value) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    return path if path.is_absolute() else PROJECT_ROOT / path


def add_path_arguments(parser) -> None:
    for key in PATH_KEYS:
        parser.add_argument(f"--{key}", default=None, help=f"覆盖配置中的 paths.{key}")


def path_overrides(args) -> Dict[str, str]:
    return {k: getattr(args, k) for k in PATH_KEYS if getattr(args, k, None)}


def validate_config(config: Dict[str, Any]) -> None:
    for key in ("experiment", "model", "augmentation", "features", "training"):
        if key not in config:
            raise ValueError(f"Configuration is missing required section: {key}")
    if not config["model"].get("freeze_encoder", False):
        raise ValueError("The CLIP encoder must be frozen")
    aug = config["augmentation"]
    if not isinstance(aug.get("enabled"), bool):
        raise ValueError("augmentation.enabled must be a boolean")
    if aug["enabled"]:
        for name, range_key in (("resize", "scale_range"), ("gaussian_blur", "sigma_range"),
                                ("jpeg", "quality_range")):
            sec = aug.get(name) or {}
            if not 0 <= float(sec.get("probability", -1)) <= 1:
                raise ValueError(f"augmentation.{name}.probability must be in [0, 1]")
            rng = sec.get(range_key)
            if not isinstance(rng, list) or len(rng) != 2 or rng[0] > rng[1]:
                raise ValueError(f"augmentation.{name}.{range_key} must be an ordered pair")
    if int(config["features"].get("views_per_image", 0)) < 1:
        raise ValueError("features.views_per_image must be at least 1")
    tr = config["training"]
    for key in ("seeds", "learning_rates", "weight_decays", "epochs", "patience", "batch_size"):
        if tr.get(key) in (None, []):
            raise ValueError(f"training.{key} is required")


def load_config(path, overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    validate_config(config)
    paths = dict(DEFAULT_PATHS)
    paths.update(config.get("paths") or {})
    paths.update(overrides or {})
    config["paths"] = {k: str(resolve_path(v)) for k, v in paths.items()}
    return config


def train_mode(config: Dict[str, Any]) -> str:
    return "train_aug" if config["augmentation"]["enabled"] else "train_clean"


def train_cache_path(config: Dict[str, Any]) -> Path:
    """训练特征缓存：无增强 -> train_clean.pt，有增强 -> train_aug.pt。"""
    return Path(config["paths"]["cache_dir"]) / f"{train_mode(config)}.pt"


def degrade_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把 YAML 的增强参数转成 common.corruptions.random_degrade 使用的格式。"""
    aug = config["augmentation"]
    if not aug["enabled"]:
        return None
    return dict(
        p_resize=float(aug["resize"]["probability"]), resize_range=tuple(aug["resize"]["scale_range"]),
        p_blur=float(aug["gaussian_blur"]["probability"]), blur_range=tuple(aug["gaussian_blur"]["sigma_range"]),
        p_jpeg=float(aug["jpeg"]["probability"]), jpeg_range=tuple(int(q) for q in aug["jpeg"]["quality_range"]),
    )


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
