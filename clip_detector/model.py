"""冻结的 OpenAI CLIP ViT-L/14 图像编码器 + 可训练的线性二分类头

只加载通用预训练权重，不加载任何已发布的检测头。标签约定：0 = Real，1 = AI-Generated
"""

from __future__ import annotations

import torch

DEFAULT_MODEL = "openai/clip-vit-large-patch14"


def load_clip_encoder(name: str = DEFAULT_MODEL, device: str = "cuda"):
    """返回冻结的 CLIP 视觉塔；输出为 768 维投影图像特征"""
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    if name == "tiny-random":
        cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                               num_attention_heads=2, image_size=224, patch_size=32,
                               projection_dim=16)
        model = CLIPVisionModelWithProjection(cfg)
    else:
        model = CLIPVisionModelWithProjection.from_pretrained(name)
    model.eval().requires_grad_(False).to(device)
    return model


VIT_MODEL = "torchvision/vit_b_16-imagenet1k_v1"


def load_encoder(name: str, device: str):
    """Return a frozen encoder compatible with the shared feature cache."""
    if name != VIT_MODEL:
        return load_clip_encoder(name, device)

    from torchvision.models import ViT_B_16_Weights, vit_b_16

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = torch.nn.Identity()
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def encode(model, x: torch.Tensor, device: str, normalize: bool = False,
           model_name: str = DEFAULT_MODEL) -> torch.Tensor:
    x = x.to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        f = model(x) if model_name == VIT_MODEL else model(pixel_values=x).image_embeds
    f = f.float()
    if normalize:
        f = torch.nn.functional.normalize(f, dim=-1)
    return f


class LinearHead(torch.nn.Module):
    """冻结 CLIP 特征上的单个线性层，输出 P(AI) 的 logit"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(dim, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.fc(f).squeeze(-1)


def load_head(path, device: str = "cpu"):
    """读取 head.pt，返回 (线性头, 检查点字典)"""
    ck = torch.load(path, map_location=device, weights_only=False)
    head = LinearHead(ck["dim"]).to(device)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, ck
