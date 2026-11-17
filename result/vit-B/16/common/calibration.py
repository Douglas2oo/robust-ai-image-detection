"""验证集置信度校准与阈值选择。只能用验证集调用；测试集只应用冻结后的 T 和阈值

P(AI) = sigmoid(logit / T)，P(Real) = 1 - P(AI)
"""

from __future__ import annotations

import numpy as np


def fit_temperature(logits, labels) -> float:
    """用 LBFGS 最小化验证集 BCE 来拟合温度 T。logits / labels 为 CPU 上的一维 float 张量"""
    import torch
    import torch.nn.functional as F

    logt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / logt.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logt.exp().clamp(0.05, 100).item())


def calibrated_prob(logits, temperature: float):
    """logits: torch 张量 -> 校准后的 P(AI)"""
    import torch

    return torch.sigmoid(logits / temperature)


def pick_threshold(p, y):
    """在验证集上选平衡准确率最高的阈值；并列时取最接近 0.5 的。返回 (阈值, 平衡准确率)"""
    from sklearn.metrics import balanced_accuracy_score

    cands = np.unique(np.concatenate([np.quantile(p, np.linspace(0, 1, 401)), [0.5]]))
    scores = np.array([balanced_accuracy_score(y, (p >= t).astype(int)) for t in cands])
    best = scores.max()
    ties = cands[np.isclose(scores, best)]
    return float(ties[np.argmin(np.abs(ties - 0.5))]), float(best)
