"""LiZAD 风格损失函数：FocalLoss（2 类）+ BinaryDiceLoss。

用于在正常训练图上训练轻量适配器（零样本设定下，训练掩码全为 0，
即"把正常区域的异常概率压到 0"，从而让偏离正常的缺陷区域自然获得高异常分）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """2 类 Focal Loss（支持平滑标签）。target 为整数类别索引 [N] 或 [N,H,W]。"""

    def __init__(self, apply_nonlin=None, alpha=0.75, gamma=2, smooth=1e-5, size_average=True):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1).view(-1, 1).long()
        alpha = torch.ones(num_class, 1, device=logit.device) * (1 - self.alpha)
        alpha[0] = self.alpha  # 类别 0（正常/背景）权重更高
        idx = target
        one_hot = torch.zeros(target.size(0), num_class, device=logit.device).scatter_(1, idx, 1)
        if self.smooth:
            one_hot = one_hot.clamp(self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot * logit.softmax(dim=1)).sum(1) + self.smooth
        logpt = pt.log()
        alpha = alpha[idx].squeeze(1)
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt
        return loss.mean() if self.size_average else loss.sum()


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        n = targets.size(0)
        inp = inputs.view(n, -1)
        tgt = targets.view(n, -1)
        inter = (inp * tgt).sum(1)
        union = inp.sum(1) + tgt.sum(1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
