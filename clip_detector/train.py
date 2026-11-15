"""在缓存的 CLIP 特征上训练线性二分类头（BCE）

  python -m clip_detector.train --config clip_detector/configs/baseline.yaml
  python -m clip_detector.train --config clip_detector/configs/augmented.yaml

对每个种子：在 lr × weight decay 网格上训练，按验证集 BCE 选最优并早停；
然后只用验证集拟合温度 T、选定判定阈值。测试集不参与。
输出：<checkpoint_dir>/<experiment>/seed<k>/head.pt 和 train_log.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from clip_detector.config import (add_path_arguments, load_config, path_overrides, set_seed,
                                  train_cache_path, train_mode)
from clip_detector.model import LinearHead
from common.calibration import fit_temperature, pick_threshold
from common.metrics import binary_metrics


def load_feats(path, clean_only: bool = False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache is missing: {path}. Run clip_detector.cache_features first.")
    d = torch.load(path, weights_only=False)
    keep = d["ok"].bool()
    x = d["feats"][keep].float()
    if clean_only:
        x = x[:, d["meta"]["perturbs"].index("clean")]
    return x, d["labels"][keep].float(), d["meta"]


@torch.no_grad()
def logits_of(head, x, bs: int = 8192):
    return torch.cat([head(x[i:i + bs]) for i in range(0, len(x), bs)])


def train_one(xtr, ytr, xva, yva, lr, wd, seed, epochs, patience, bs, device):
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n, v, dim = xtr.shape
    head = LinearHead(dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    best = (float("inf"), None, -1)
    hist = []
    bad = 0
    for ep in range(epochs):
        head.train()
        view = torch.randint(0, v, (n,), generator=g).to(xtr.device)
        xe = xtr[torch.arange(n, device=xtr.device), view]
        perm = torch.randperm(n, generator=g).to(xtr.device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = xe[idx], ytr[idx]
            loss = F.binary_cross_entropy_with_logits(head(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        head.eval()
        vl = F.binary_cross_entropy_with_logits(logits_of(head, xva), yva).item()
        hist.append(dict(epoch=ep, train_loss=tot / n, val_loss=vl))
        if vl < best[0] - 1e-5:
            best = (vl, {k: t.detach().clone() for k, t in head.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best[1])
    return head, best[0], best[2], hist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="覆盖 training.seeds")
    ap.add_argument("--overwrite", action="store_true", help="重新训练已存在的种子（默认跳过）")
    add_path_arguments(ap)
    a = ap.parse_args()
    config = load_config(a.config, path_overrides(a))
    tr = config["training"]
    seeds = a.seeds or [int(s) for s in tr["seeds"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_root = Path(config["paths"]["checkpoint_dir"]) / config["experiment"]
    todo = [s for s in seeds if a.overwrite or not (run_root / f"seed{s}" / "head.pt").is_file()]
    for s in sorted(set(seeds) - set(todo)):
        print(f"[skip] {run_root / f'seed{s}' / 'head.pt'} exists (use --overwrite to retrain)")
    if not todo:
        return

    xtr, ytr, mtr = load_feats(train_cache_path(config))
    if mtr["mode"] != train_mode(config):
        raise ValueError(f"Train cache mode {mtr['mode']} does not match config ({train_mode(config)})")
    xva, yva, _ = load_feats(Path(config["paths"]["cache_dir"]) / "val_eval.pt", clean_only=True)
    xtr, ytr, xva, yva = xtr.to(device), ytr.to(device), xva.to(device), yva.to(device)
    print(f"[data] train {tuple(xtr.shape)} (mode={mtr['mode']}), val {tuple(xva.shape)}, device={device}")

    for seed in todo:
        rd = run_root / f"seed{seed}"
        rd.mkdir(parents=True, exist_ok=True)
        grid = []
        best = None
        for lr, wd in itertools.product([float(v) for v in tr["learning_rates"]],
                                        [float(v) for v in tr["weight_decays"]]):
            head, vl, ep, hist = train_one(xtr, ytr, xva, yva, lr, wd, seed, int(tr["epochs"]),
                                           int(tr["patience"]), int(tr["batch_size"]), device)
            grid.append(dict(lr=lr, wd=wd, val_loss=vl, best_epoch=ep, epochs_run=len(hist)))
            print(f"  seed={seed} lr={lr:g} wd={wd:g}: val_bce={vl:.4f} @ep{ep}")
            if best is None or vl < best[1]:
                best = (head, vl, lr, wd, ep, hist)
        head, vl, lr, wd, ep, hist = best

        lv = logits_of(head, xva).cpu()
        yv = yva.cpu()
        T = fit_temperature(lv, yv)
        p_raw = torch.sigmoid(lv).numpy()
        p_cal = torch.sigmoid(lv / T).numpy()
        thr, _ = pick_threshold(p_cal, yv.numpy())
        val_m = binary_metrics(yv.numpy(), p_cal, thr)
        val_m_raw = binary_metrics(yv.numpy(), p_raw, 0.5)

        torch.save(dict(state_dict={k: t.cpu() for k, t in head.state_dict().items()},
                        dim=xtr.shape[-1], temperature=T, threshold=thr,
                        lr=lr, wd=wd, best_epoch=ep, seed=seed, train_meta=mtr,
                        model=mtr["model"], experiment=config["experiment"]), rd / "head.pt")
        with open(rd / "train_log.json", "w") as fh:
            json.dump(dict(experiment=config["experiment"], grid=grid,
                           chosen=dict(lr=lr, wd=wd, best_epoch=ep, val_bce=vl),
                           temperature=T, threshold=thr, val_metrics_calibrated=val_m,
                           val_metrics_raw=val_m_raw, history=hist), fh, indent=2)
        print(f"[seed {seed}] chosen lr={lr:g} wd={wd:g} ep={ep}  T={T:.3f} thr={thr:.3f}  "
              f"val bal_acc={val_m['bal_acc']:.4f} auroc={val_m['auroc']:.4f} "
              f"ECE raw={val_m_raw['ece']:.4f} -> cal={val_m['ece']:.4f}")


if __name__ == "__main__":
    main()
