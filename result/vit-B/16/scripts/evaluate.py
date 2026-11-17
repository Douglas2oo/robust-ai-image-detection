"""各模型共用的评测入口：所有测试集 × 所有扰动，按种子汇总。

两种输入可以单独或同时使用（同时使用时所有模型进入同一张汇总表）：

1) CLIP：训练好的线性头 + 缓存的测试特征
   python -m scripts.evaluate --runs checkpoints/clip_noaug checkpoints/clip_aug \\
       --feat_dir cache/clip/test --out_dir outputs/eval --save_preds

2) 任意模型（如 ResNet）：逐图预测 CSV
   python -m scripts.evaluate --preds outputs/resnet/preds_resnet_seed0.csv.gz --out_dir outputs/eval
   必需列：config, seed, test, perturb, label, p_ai, threshold
     config     模型/实验名（如 resnet）      seed       训练种子
     test       sd14 / sd15 / midjourney / adm / biggan / ext_*
     perturb    common.corruptions.EVAL_PERTURBS 中的名字
     label      0 = Real, 1 = AI               p_ai       校准后的 P(AI)
     threshold  验证集上确定的判定阈值（同一 config+seed 下为常数）
   可选列：p_ai_uncal（校准前的 P(AI)，用于 ece_uncal）、relpath、source

输出：
  metrics_per_run.csv   每个 (config, seed, test, perturb) 一行
  summary.csv           各种子的均值/标准差，外加未见生成器的宏平均 unseen_macro
  preds_<config>_seed<k>.csv.gz   （--save_preds）CLIP 的逐图预测，格式同上
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from common.metrics import METRIC_COLUMNS, binary_metrics


def rows_from_caches(runs, feat_dir, out_dir, save_preds):
    import torch

    from clip_detector.model import load_head

    tests = {}
    for f in sorted(glob.glob(os.path.join(feat_dir, "test_*.pt"))):
        d = torch.load(f, weights_only=False)
        name = os.path.basename(f)[5:-3]
        keep = d["ok"].bool()
        tests[name] = dict(x=d["feats"][keep].float(), y=d["labels"][keep].numpy(),
                           perturbs=d["meta"]["perturbs"],
                           relpath=np.array(d["relpath"])[keep.numpy()],
                           source=np.array(d["source"])[keep.numpy()])
        print(f"[test] {name}: {tuple(tests[name]['x'].shape)}")
    if not tests:
        raise FileNotFoundError(f"No test_*.pt feature caches in {feat_dir}")

    rows = []
    for cfg_dir in runs:
        cfg = os.path.basename(os.path.normpath(cfg_dir))
        heads = sorted(glob.glob(os.path.join(cfg_dir, "seed*", "head.pt")))
        if not heads:
            raise FileNotFoundError(f"No seed*/head.pt under {cfg_dir}")
        for hp in heads:
            head, ck = load_head(hp)
            seed, T, thr = ck["seed"], ck["temperature"], ck["threshold"]
            preds = []
            for name, t in tests.items():
                with torch.no_grad():
                    logit = head(t["x"].flatten(0, 1)).view(t["x"].shape[0], -1)
                for j, pert in enumerate(t["perturbs"]):
                    p = torch.sigmoid(logit[:, j] / T).numpy()
                    p_raw = torch.sigmoid(logit[:, j]).numpy()
                    m = binary_metrics(t["y"], p, thr)
                    m_raw = binary_metrics(t["y"], p_raw, thr)
                    rows.append(dict(config=cfg, seed=seed, test=name, perturb=pert,
                                     ece_uncal=m_raw["ece"], **m))
                    if save_preds:
                        preds.append(pd.DataFrame(dict(config=cfg, seed=seed, test=name, perturb=pert,
                                                       relpath=t["relpath"], source=t["source"],
                                                       label=t["y"], p_ai=p, p_ai_uncal=p_raw,
                                                       threshold=thr)))
            if save_preds and preds:
                pd.concat(preds).to_csv(os.path.join(out_dir, f"preds_{cfg}_seed{seed}.csv.gz"), index=False)
            print(f"[eval] {cfg} seed{seed} done (T={T:.3f}, thr={thr:.3f})")
    return rows


def rows_from_preds(files):
    need = ["config", "seed", "test", "perturb", "label", "p_ai", "threshold"]
    rows = []
    for f in files:
        d = pd.read_csv(f)
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise ValueError(f"{f} is missing columns: {missing}")
        for (cfg, seed), run in d.groupby(["config", "seed"], sort=False):
            thrs = run["threshold"].unique()
            if len(thrs) != 1:
                raise ValueError(f"{f}: threshold must be constant within config={cfg} seed={seed}")
            thr = float(thrs[0])
            for (test, pert), g in run.groupby(["test", "perturb"], sort=False):
                m = binary_metrics(g["label"].values, g["p_ai"].values, thr)
                ece_uncal = (binary_metrics(g["label"].values, g["p_ai_uncal"].values, thr)["ece"]
                             if "p_ai_uncal" in g.columns else float("nan"))
                rows.append(dict(config=cfg, seed=seed, test=test, perturb=pert, ece_uncal=ece_uncal, **m))
            print(f"[eval] {cfg} seed{seed} from {os.path.basename(f)} (thr={thr:.3f})")
    return rows


def summarize(rows, out_dir, seen, unseen):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "metrics_per_run.csv"), index=False)

    # 未见生成器的宏平均：按 (config, seed, perturb) 对各生成器取平均
    un = df[df.test.isin(unseen)]
    if len(un):
        macro = un.groupby(["config", "seed", "perturb"])[METRIC_COLUMNS].mean().reset_index().assign(test="unseen_macro")
        df = pd.concat([df, macro], ignore_index=True)

    summ = df.groupby(["config", "test", "perturb"])[METRIC_COLUMNS].agg(["mean", "std"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    fmt = dict(index=False, float_format=lambda v: f"{v:.4f}")
    show = ["config", "test", "acc_mean", "bal_acc_mean", "auroc_mean", "ap_mean", "f1_mean", "ece_mean", "bal_acc_std"]
    print("\n=== clean ===")
    print(summ[summ.perturb == "clean"][show].to_string(**fmt))
    rob = summ[summ.test.isin([seen, "unseen_macro"])]
    print("\n=== robustness (seen / unseen macro) ===")
    print(rob[["config", "test", "perturb", "bal_acc_mean", "auroc_mean", "ece_mean"]].to_string(**fmt))
    return summ


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="CLIP 实验目录（包含 seed*/head.pt）")
    ap.add_argument("--feat_dir", default=None, help="test_<name>.pt 测试特征缓存所在目录")
    ap.add_argument("--preds", nargs="*", default=[], help="任意模型的逐图预测 CSV（可为 .csv.gz）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seen", default="sd14")
    ap.add_argument("--unseen", default="sd15,adm,midjourney,biggan")
    ap.add_argument("--save_preds", action="store_true")
    a = ap.parse_args()
    if not a.runs and not a.preds:
        ap.error("give --runs (with --feat_dir) and/or --preds")
    if a.runs and not a.feat_dir:
        ap.error("--runs requires --feat_dir")
    os.makedirs(a.out_dir, exist_ok=True)

    rows = []
    if a.runs:
        rows += rows_from_caches(a.runs, a.feat_dir, a.out_dir, a.save_preds)
    if a.preds:
        rows += rows_from_preds(a.preds)
    summarize(rows, a.out_dir, a.seen, a.unseen.split(","))


if __name__ == "__main__":
    main()
