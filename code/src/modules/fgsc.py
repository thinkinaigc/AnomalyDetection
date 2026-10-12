"""FGSC：频域引导的分数空间校准（创新点 2）。

动机
----
逐位置(PaDiM式)异常打分在高频处存在**孤立假阳性噪声**（相邻像素分数跳变），
这些噪声显著拉低像素级 AUROC/AP；而在**高频纹理区域**噪声更集中、误报更严重。

机制（频域低通 / 高斯核在频域等价低通）
------------------------------------
对异常分数图做各向同性高斯低通，并让低通强度由**局部纹理能量 T** 引导：

    α = a0 + a1·T            (a0: 全局基础降噪强度, a1: 纹理引导的额外强度)
    S_cal = (1 - α)·S + α · LowPass_K(S)

- 平滑区域(T≈0)：α≈a0，保留基础降噪（抑制高频孤立噪声）同时不过度模糊边界；
- 纹理区域(T≈1)：α≈a0+a1，低通更强，抑制纹理驱动的误报。

该形式把"固定降噪"与"纹理自适应"统一为一个连续谱（a1=0 即固定低通）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_kernel(k: int, device, dtype=torch.float32) -> torch.Tensor:
    ax = torch.arange(k, dtype=dtype, device=device) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2))
    return (g / g.sum()).view(1, 1, 1, k)


def gaussian_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    g = gaussian_kernel(k, x.device, x.dtype)
    x = F.conv2d(x, g, padding=(0, k // 2))
    x = F.conv2d(x, g.transpose(2, 3), padding=(k // 2, 0))
    return x


@torch.no_grad()
def calibrate(
    score: torch.Tensor,
    texture: torch.Tensor,
    lam: float = 0.8,
    tau: float = 0.4,
    mode: str = "adaptive",
    kernel: int = 33,
) -> torch.Tensor:
    """对异常分数图做纹理引导的频域低通校准。

    参数
    ----
    score:   [B,1,H,W] 异常概率图 ∈[0,1]
    texture: [B,1,H,W] 纹理能量图 T ∈[0,1]
    lam:     a0 基础降噪强度 ∈[0,1]
    tau:     a1 纹理引导的额外强度 ∈[0,1]
    mode:    'off' | 'lowpass'(固定低通,忽略纹理) | 'adaptive'(纹理引导)
    kernel:  低通核大小 K
    """
    if mode == "off" or kernel <= 1 or lam <= 0:
        return score
    if texture.shape[-2:] != score.shape[-2:]:
        texture = F.interpolate(texture, size=score.shape[-2:], mode="bilinear", align_corners=False)
    low = gaussian_blur(score, kernel)
    if mode == "lowpass":
        a1 = 0.0
    else:  # 'adaptive'
        a1 = tau
    alpha = (lam + a1 * texture).clamp(0.0, 1.0)
    return (1 - alpha) * score + alpha * low


@torch.no_grad()
def image_score(score_map: torch.Tensor, topk: float = 0.01) -> torch.Tensor:
    """由像素分数图得到图像级分数：取 top-k 比例的像素均值。"""
    b = score_map.shape[0]
    flat = score_map.reshape(b, -1)
    k = max(int(flat.shape[1] * topk), 1)
    top = torch.topk(flat, k, dim=1).values
    return top.mean(dim=1)