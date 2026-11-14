"""CLIP 训练增强：随机水平翻转 + 对称的随机退化（缩放 -> 模糊 -> JPEG）

退化本身在 common.corruptions.random_degrade 中实现，供其他模型复用；
连接YAML配置，提供一个独立可调用的增强器
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

from PIL import Image

from common.corruptions import random_degrade


class ClipAugmenter:
    """image, seed 增强后的图片"""

    def __init__(self, degrade_cfg: Optional[Dict[str, Any]], hflip: float = 0.5) -> None:
        self.degrade_cfg = degrade_cfg
        self.hflip = float(hflip)

    def __call__(self, image: Image.Image, seed: int) -> Image.Image:
        rng = random.Random(int(seed))
        output = image.convert("RGB")
        if rng.random() < self.hflip:
            output = output.transpose(Image.FLIP_LEFT_RIGHT)
        if self.degrade_cfg is not None:
            output = random_degrade(output, rng, self.degrade_cfg)
        return output
