"""Frozen OpenAI CLIP ViT-L/14 with a trainable linear binary head."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def load_clip_encoder(
    model_name: str = "ViT-L/14",
    device: str = "cpu",
    download_root: Optional[str] = None,
) -> Tuple[nn.Module, object]:
    """Load the official OpenAI CLIP model and its preprocessing transform."""

    try:
        import clip
    except ImportError as exc:  # pragma: no cover - depends on the runtime environment
        raise ImportError(
            "OpenAI CLIP is required. Install requirements.txt before extracting features."
        ) from exc
    kwargs = {"device": device, "jit": False}
    if download_root is not None:
        kwargs["download_root"] = download_root
    return clip.load(model_name, **kwargs)


def clip_feature_dim(encoder: nn.Module) -> int:
    visual = getattr(encoder, "visual", None)
    output_dim = getattr(visual, "output_dim", None)
    if output_dim is None:
        projection = getattr(visual, "proj", None)
        if projection is not None and getattr(projection, "shape", None) is not None:
            output_dim = projection.shape[-1]
    if output_dim is None:
        output_dim = getattr(encoder, "embed_dim", None)
    if output_dim is None:
        raise ValueError("The CLIP encoder does not expose visual.output_dim")
    return int(output_dim)


def encode_image_features(
    encoder: nn.Module,
    images: torch.Tensor,
    normalize_features: bool = True,
) -> torch.Tensor:
    """Encode images without gradients and return float32 features."""

    with torch.no_grad():
        features = encoder.encode_image(images)
    features = features.float()
    if normalize_features:
        features = F.normalize(features, dim=-1)
    return features


class LinearBinaryHead(nn.Module):
    """A single linear logit over frozen CLIP features."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(feature_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features.float()).squeeze(-1)


class FrozenCLIPClassifier(nn.Module):
    """Frozen encoder plus trainable head for image-space inference."""

    def __init__(
        self,
        encoder: nn.Module,
        normalize_features: bool = True,
        feature_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.normalize_features = bool(normalize_features)
        self.feature_dim = int(feature_dim or clip_feature_dim(encoder))
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.head = LinearBinaryHead(self.feature_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        return encode_image_features(self.encoder, images, self.normalize_features)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return self.head(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_features(images))


def load_linear_head_checkpoint(
    checkpoint_path: Path,
    map_location: str = "cpu",
) -> Tuple[LinearBinaryHead, dict]:
    """Restore the trained linear head and return it with checkpoint metadata."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    if checkpoint.get("model_type") != "frozen_clip_vit_l14_linear_head":
        raise ValueError(f"Unsupported checkpoint model type: {checkpoint.get('model_type')}")
    head = LinearBinaryHead(int(checkpoint["feature_dim"]))
    head.load_state_dict(checkpoint["head_state_dict"])
    head.to(map_location)
    head.eval()
    return head, checkpoint
