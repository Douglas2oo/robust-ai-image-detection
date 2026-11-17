"""所有模型共用的图像退化：测试扰动（固定强度）和训练增强（随机强度）

全部作用于原始分辨率的 RGB PIL 图片，在 224 预处理和归一化之前执行；对真实图和生成图使用完全相同的条件；测试时每次只施加一种扰动

测试扰动（Proposal §3 Evaluation）：
- jpegQ   ：以 quality=Q 存为 JPEG 再读回，Q ∈ {90, 70, 50}
- downS   ：双线性缩小到 S 倍再双线性放大回原尺寸，S ∈ {0.5, 0.25}
- blurσ   ：PIL GaussianBlur(radius=σ)，σ ∈ {0.5, 1.0}
"""

from __future__ import annotations

import io

from PIL import Image, ImageFilter

EVAL_PERTURBS = ["clean", "jpeg90", "jpeg70", "jpeg50", "down0.5", "down0.25", "blur0.5", "blur1.0"]

DEFAULT_TRAIN_AUG = dict(p_resize=0.3, resize_range=(0.25, 1.0),
                         p_blur=0.3, blur_range=(0.1, 1.5),
                         p_jpeg=0.5, jpeg_range=(50, 100))


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


def down_restore(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def apply_eval_perturb(img: Image.Image, name: str) -> Image.Image:
    """施加一种固定强度的测试扰动；name 取 EVAL_PERTURBS 中的值"""
    if name == "clean":
        return img
    if name.startswith("jpeg"):
        return jpeg(img, int(name[4:]))
    if name.startswith("down"):
        return down_restore(img, float(name[4:]))
    if name.startswith("blur"):
        return blur(img, float(name[4:]))
    raise ValueError(f"unknown perturbation {name}")


def random_degrade(img: Image.Image, rng, cfg=None) -> Image.Image:
    """训练增强：三种退化独立抽样、可叠加，顺序固定为 缩放 -> 模糊 -> JPEG
    """
    cfg = cfg or DEFAULT_TRAIN_AUG
    if rng.random() < cfg["p_resize"]:
        img = down_restore(img, rng.uniform(*cfg["resize_range"]))
    if rng.random() < cfg["p_blur"]:
        img = blur(img, rng.uniform(*cfg["blur_range"]))
    if rng.random() < cfg["p_jpeg"]:
        img = jpeg(img, rng.randint(*cfg["jpeg_range"]))
    return img
