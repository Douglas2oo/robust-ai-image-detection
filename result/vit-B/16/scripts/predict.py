"""单图推理演示：JPEG/PNG 图片 -> Real / AI-Generated + 校准后的置信度 + 延迟

  python -m scripts.predict --checkpoint checkpoints/clip_aug/seed0/head.pt img1.jpg img2.png
  python -m scripts.predict --checkpoint checkpoints/clip_aug/seed0/head.pt --benchmark 200 some.jpg

"""

from __future__ import annotations

import argparse
import time

import torch

from clip_detector.model import encode, load_clip_encoder, load_head
from common.data import load_rgb
from common.preprocessing import clip_preprocess


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="JPEG 或 PNG 图片")
    ap.add_argument("--checkpoint", required=True, help="clip_detector.train 输出的 head.pt")
    ap.add_argument("--model", default=None, help="默认使用检查点里记录的编码器")
    ap.add_argument("--benchmark", type=int, default=0, help="重复 N 次前向以测量延迟")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head, ck = load_head(a.checkpoint, device)
    model = load_clip_encoder(a.model or ck["model"], device)
    normalize = bool(ck.get("train_meta", {}).get("normalize_features", False))
    pre = clip_preprocess()
    T, thr = ck["temperature"], ck["threshold"]

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for path in a.images:
        t0 = time.perf_counter()
        x = pre(load_rgb(path)).unsqueeze(0)
        with torch.no_grad():
            p_ai = torch.sigmoid(head(encode(model, x, device, normalize)) / T).item()
        sync()
        ms = (time.perf_counter() - t0) * 1000
        label = "AI-Generated" if p_ai >= thr else "Real"
        conf = p_ai if label == "AI-Generated" else 1 - p_ai
        print(f"{path}: {label}  confidence={conf:.3f}  P(AI)={p_ai:.3f}  P(Real)={1 - p_ai:.3f}  ({ms:.1f} ms)")

    if a.benchmark:
        x = pre(load_rgb(a.images[0])).unsqueeze(0)
        for _ in range(10):  # 预热
            with torch.no_grad():
                head(encode(model, x, device, normalize))
        sync()
        t0 = time.perf_counter()
        for _ in range(a.benchmark):
            with torch.no_grad():
                head(encode(model, x, device, normalize))
        sync()
        print(f"[latency] model forward, batch 1: {(time.perf_counter() - t0) / a.benchmark * 1000:.2f} ms/img on {device}")


if __name__ == "__main__":
    main()
