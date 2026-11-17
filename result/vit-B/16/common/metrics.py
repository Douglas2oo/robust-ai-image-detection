"""统一的二分类指标。AI-Generated 为正类

阈值类指标（acc / bal_acc / precision / recall / F1）使用验证集确定的固定阈值；
AUROC / AP 使用连续得分。ECE 按预测类别的置信度计算（15 个区间）
"""

from __future__ import annotations

import numpy as np

METRIC_COLUMNS = ["acc", "bal_acc", "precision", "recall", "f1", "auroc", "ap", "ece", "ece_uncal", "brier"]


def ece_score(y, p, n_bins: int = 15) -> float:
    """预测类别置信度的期望校准误差。"""
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def binary_metrics(y, p, thr) -> dict:
    """y: 标签 (0/1)；p: P(AI)；thr: 判定阈值（p >= thr 判为 AI）。"""
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 f1_score, precision_score, recall_score, roc_auc_score)

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)
    two = len(np.unique(y)) == 2
    return dict(
        n=len(y), n_real=int((y == 0).sum()), n_ai=int((y == 1).sum()),
        acc=accuracy_score(y, yhat),
        bal_acc=balanced_accuracy_score(y, yhat) if two else float("nan"),
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        f1=f1_score(y, yhat, zero_division=0),
        auroc=roc_auc_score(y, p) if two else float("nan"),
        ap=average_precision_score(y, p) if two else float("nan"),
        real_acc=float((yhat[y == 0] == 0).mean()) if (y == 0).any() else float("nan"),
        ai_acc=float((yhat[y == 1] == 1).mean()) if (y == 1).any() else float("nan"),
        ece=ece_score(y, p),
        brier=float(np.mean((p - y) ** 2)),
    )
