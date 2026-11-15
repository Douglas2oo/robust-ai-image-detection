"""读取固定的数据清单，并按清单顺序提供图片

训练/评测只用两列：
  relpath  相对于数据根目录的路径
  label    0 = Real, 1 = AI-Generated
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image

from common.corruptions import apply_eval_perturb, random_degrade

REQUIRED_COLUMNS = ("relpath", "label")


def read_manifest(path) -> pd.DataFrame:
    """读取并校验一份清单；不做任何重新抽样或划分。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} is missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError(f"Manifest {path} is empty")
    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"Manifest {path}: label must be 0 (Real) or 1 (AI)")
    rel = df["relpath"].astype(str)
    if rel.map(lambda p: os.path.isabs(p) or ".." in Path(p).parts).any():
        raise ValueError(f"Manifest {path}: relpath must stay under the data root")
    if rel.duplicated().any():
        raise ValueError(f"Manifest {path}: duplicated relpath {rel[rel.duplicated()].iloc[0]!r}")
    if "source" not in df.columns:
        df["source"] = df["label"].map({0: "real", 1: "ai"})
    return df


def load_rgb(path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def stable_seed(*parts) -> int:
    """由 (种子, 图片序号, 视图序号) 得到确定的随机种子，与 DataLoader 的调度无关。"""
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


class ManifestImageDataset:
    """按清单顺序返回每张图片的 K 个视图

    mode="train_clean"：随机水平翻转；
    mode="train_aug"  ：随机水平翻转 + common.corruptions.random_degrade；
    mode="eval"       ：perturbs 中每种测试扰动各一个视图

    返回 (views [K, 3, 224, 224], label, ok)。ok=False 表示图片读取失败
    preprocess 传 common.preprocessing.clip_preprocess() 或 imagenet_preprocess()。
    """

    def __init__(self, df, data_root, mode, preprocess, views=1, perturbs=("clean",),
                 seed=0, aug_cfg=None, hflip=0.5):
        if mode not in ("train_clean", "train_aug", "eval"):
            raise ValueError(f"unknown mode {mode}")
        self.paths = [os.path.join(str(data_root), p) for p in df["relpath"].tolist()]
        self.labels = df["label"].astype(int).tolist()
        self.mode, self.views, self.perturbs, self.seed = mode, int(views), list(perturbs), int(seed)
        self.pre, self.aug_cfg, self.hflip = preprocess, aug_cfg, float(hflip)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch

        try:
            img = load_rgb(self.paths[i])
            ok = True
        except Exception as e:
            print(f"[warn] cannot read {self.paths[i]}: {e}", flush=True)
            img, ok = Image.new("RGB", (224, 224)), False
        out = []
        if self.mode == "eval":
            for p in self.perturbs:
                out.append(self.pre(apply_eval_perturb(img, p)))
        else:
            for v in range(self.views):
                rng = random.Random(stable_seed(self.seed, i, v))
                x = img
                if rng.random() < self.hflip:
                    x = x.transpose(Image.FLIP_LEFT_RIGHT)
                if self.mode == "train_aug":
                    x = random_degrade(x, rng, self.aug_cfg)
                out.append(self.pre(x))
        return torch.stack(out), self.labels[i], ok
