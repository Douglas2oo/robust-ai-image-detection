"""抽取并缓存冻结 CLIP 的特征（编码器冻结，特征可重复使用）

  python -m clip_detector.cache_features --config clip_detector/configs/baseline.yaml
  python -m clip_detector.cache_features --config clip_detector/configs/augmented.yaml

缓存目录结构（两组实验共用验证集和测试集的缓存）：
  <cache_dir>/train_clean.pt   train.csv，随机翻转，K 个视图           [N, K, D]
  <cache_dir>/train_aug.pt     train.csv，随机翻转 + 退化增强，K 个视图 [N, K, D]
  <cache_dir>/val_eval.pt      val.csv，每种测试扰动一个视图            [N, 8, D]
  <cache_dir>/test/test_*.pt   每个 test_*.csv，同上                    [N, 8, D]
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from clip_detector.config import (add_path_arguments, degrade_config, load_config, path_overrides,
                                  set_seed, train_cache_path, train_mode)
from clip_detector.model import VIT_MODEL, encode, load_encoder
from common.data import ManifestImageDataset, read_manifest
from common.preprocessing import clip_preprocess, imagenet_preprocess


def _num_workers(value) -> int:
    if value in (None, "auto"):
        return max(0, int(os.environ.get("SLURM_CPUS_ON_NODE", 8)) - 2)
    return int(value)


def extract(config, manifest: Path, out: Path, mode: str, model_name: str) -> None:
    """对一份清单抽特征，保存 feats [N, V, D] (float16)、labels、ok、relpath、source、meta"""
    if out.exists():
        print(f"[skip] {out} exists")
        return
    feat_cfg = config["features"]
    seed = int(feat_cfg.get("view_seed", 0))
    views = int(feat_cfg["views_per_image"])
    perturbs = list(config["evaluation"]["perturbations"])
    normalize = bool(config["model"].get("normalize_features", False))
    workers = _num_workers(feat_cfg.get("num_workers"))
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = read_manifest(manifest)
    preprocess = imagenet_preprocess() if model_name == VIT_MODEL else clip_preprocess()
    ds = ManifestImageDataset(df, config["paths"]["data_root"], mode, preprocess, views=views,
                              perturbs=perturbs, seed=seed, aug_cfg=degrade_config(config),
                              hflip=float(config["augmentation"].get("hflip", 0.5)))
    dl = torch.utils.data.DataLoader(ds, batch_size=int(feat_cfg["batch_size"]), shuffle=False,
                                     num_workers=workers, pin_memory=(device == "cuda"),
                                     persistent_workers=workers > 0,
                                     prefetch_factor=4 if workers > 0 else None)
    model = load_encoder(model_name, device)
    print(f"[info] {len(ds)} images, mode={mode}, views={perturbs if mode == 'eval' else views}, "
          f"device={device}, workers={workers}")

    feats, labels, oks = [], [], []
    t0 = time.time()
    for x, y, ok in tqdm(dl, mininterval=30):
        b, v = x.shape[:2]
        f = encode(model, x.flatten(0, 1), device, normalize, model_name).view(b, v, -1)
        feats.append(f.half().cpu()); labels.append(y); oks.append(ok)
    dt = time.time() - t0
    feats = torch.cat(feats)
    payload = dict(feats=feats, labels=torch.cat(labels), ok=torch.cat(oks),
                   relpath=df["relpath"].tolist(), source=df["source"].tolist(),
                   meta=dict(mode=mode, views=views, perturbs=perturbs if mode == "eval" else None,
                             model=model_name, manifest=str(Path(manifest).resolve()), seed=seed,
                             normalize_features=normalize,
                             augmentation=config["augmentation"] if mode == "train_aug" else None,
                             seconds=dt, img_per_s=feats.shape[0] * feats.shape[1] / max(dt, 1e-9)))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)
    print(f"[done] {out}: feats {tuple(feats.shape)}, {dt:.0f}s, "
          f"{payload['meta']['img_per_s']:.0f} img/s, unreadable={int((~payload['ok']).sum())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--only", choices=["all", "train", "val", "test"], default="all")
    ap.add_argument("--model", default=None, help="覆盖 model.name（测试用：tiny-random）")
    add_path_arguments(ap)
    a = ap.parse_args()
    config = load_config(a.config, path_overrides(a))
    model_name = a.model or config["model"]["name"]
    manifests = Path(config["paths"]["manifest_dir"])
    cache = Path(config["paths"]["cache_dir"])

    if a.only in ("all", "train"):
        extract(config, manifests / "train.csv", train_cache_path(config), train_mode(config), model_name)
    if a.only in ("all", "val"):
        extract(config, manifests / "val.csv", cache / "val_eval.pt", "eval", model_name)
    if a.only in ("all", "test"):
        tests = sorted(glob.glob(str(manifests / "test_*.csv")))
        if not tests:
            print(f"[warn] no test_*.csv in {manifests}")
        for m in tests:
            extract(config, Path(m), cache / "test" / f"{Path(m).stem}.pt", "eval", model_name)


if __name__ == "__main__":
    main()
