"""所有模型共用的 224 像素预处理

几何处理对所有模型一致：短边 bicubic 缩放到 224 -> 中心裁剪 224 -> ToTensor。
归一化使用各自预训练权重的均值/方差（CLIP 或 ImageNet）。
训练增强和测试扰动都在这一步之前、在原始分辨率的 PIL 图片上进行
"""

from __future__ import annotations

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def build_preprocess(mean, std, size: int = IMAGE_SIZE):
    from torchvision import transforms as T

    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def clip_preprocess():
    """CLIP ViT-L/14 使用的预处理"""
    return build_preprocess(CLIP_MEAN, CLIP_STD)


def imagenet_preprocess():
    """ResNet-50 等 ImageNet 预训练模型使用的预处理"""
    return build_preprocess(IMAGENET_MEAN, IMAGENET_STD)
