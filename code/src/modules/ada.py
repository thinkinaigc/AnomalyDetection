"""ADA：自适应缺陷感知多尺度对齐（创新点 1）。

给定三个尺度特征 {P3, P4, P5}：
1. 计算每个尺度的高频能量占比 r_s（梯度幅值均值 / 特征幅值均值），
   反映该尺度上"细节/缺陷"信号的强弱；
2. 用 softmax 把 r 转成图像自适应的尺度权重 w（A3）；
   消融变体 A2 为固定权重（等权拼接）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def high_freq_ratio(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """高频能量占比 r（逐样本标量）。

    r = mean(|∇F|) / (mean(|F|) + eps)，在通道维取均值后再在空间维取均值。
    """
    gx = torch.abs(feat[:, :, :, 1:] - feat[:, :, :, :-1])
    gy = torch.abs(feat[:, :, 1:, :] - feat[:, :, :-1, :])
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    grad = torch.sqrt(gx * gx + gy * gy + eps)
    num = grad.mean(dim=(1, 2, 3))
    den = feat.abs().mean(dim=(1, 2, 3)) + eps
    return num / den


class AdaptiveScaleWeight(nn.Module):
    """图像自适应尺度权重。

    可训练参数仅 1 个标量 temperature（默认固定为常量，仍可随投影头一起优化）。
    """

    def __init__(self, num_scales: int = 3, temperature: float = 4.0, learnable: bool = True):
        super().__init__()
        self.num_scales = num_scales
        if learnable:
            self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, ratios: torch.Tensor) -> torch.Tensor:
        """ratios: [B, S] -> weights: [B, S]，沿尺度维 softmax。"""
        z = (ratios - ratios.mean(dim=1, keepdim=True)) / (ratios.std(dim=1, keepdim=True) + 1e-6)
        return torch.softmax(self.temperature * z, dim=1)

    def uniform(self, batch: int, device) -> torch.Tensor:
        return torch.full((batch, self.num_scales), 1.0 / self.num_scales, device=device)


def fuse_scales(
    feats: dict[str, torch.Tensor],
    weights: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """按权重融合多尺度特征（先插值到同一分辨率再加权求和）。

    feats:   {'P3': [B,C,H,W], 'P4':..., 'P5':...}
    weights: [B, S]
    """
    names = sorted(feats.keys())
    stack = [F.interpolate(feats[n], size=target_size, mode="bilinear", align_corners=False) for n in names]
    stack = torch.stack(stack, dim=1)  # [B,S,C,H,W]
    w = weights.to(stack.dtype).view(weights.shape[0], weights.shape[1], 1, 1, 1)
    return (stack * w).sum(dim=1)
