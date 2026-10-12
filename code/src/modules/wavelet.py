"""Haar 小波与纹理能量图（FGSC 的阶段 A）。

用 2D 卷积实现单级 Haar 分解，四个子带：
    LL(低频) / LH(水平高频) / HL(垂直高频) / HH(对角高频)
实现为 torch 算子，可直接跑在 GPU 上，无需额外依赖（不依赖 PyWavelets）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _haar_kernels(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """返回 [4,1,2,2] 卷积核，顺序 LL / LH / HL / HH。"""
    k = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],   # LL
            [[1.0, -1.0], [1.0, -1.0]],  # LH（沿 x 方向差分）
            [[1.0, 1.0], [-1.0, -1.0]],  # HL（沿 y 方向差分）
            [[1.0, -1.0], [-1.0, 1.0]],  # HH
        ],
        dtype=dtype,
        device=device,
    ) * 0.5
    return k.unsqueeze(1)


@torch.no_grad()
def haar_dwt2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """单级 Haar 分解。

    参数
    ----
    x: [B, 1, H, W] 灰度图（浮点，任意量程）

    返回
    ----
    LL, LH, HL, HH: 均为 [B, 1, H/2, W/2]
    """
    b, c, h, w = x.shape
    k = _haar_kernels(x.dtype, x.device).repeat(c, 1, 1, 1)  # [4c,1,2,2]
    y = F.conv2d(x, k, stride=2, groups=c)  # [B, 4c, H/2, W/2]
    ll = y[:, 0:c]
    lh = y[:, c:2 * c]
    hl = y[:, 2 * c:3 * c]
    hh = y[:, 3 * c:]
    return ll, lh, hl, hh


@torch.no_grad()
def texture_confidence(
    image: torch.Tensor,
    win: int = 9,
    out_size: tuple[int, int] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """纹理置信度图 T(x,y) ∈ [0,1]，值越大表示局部纹理越密集。

    步骤
    ----
    1. 灰度化 + 单级 Haar，取三个高频子带
    2. 高频幅值 A = sqrt(LH^2 + HL^2 + HH^2)
    3. 局部方差 E = Var_win(A)（用均值-平方差实现，等价于局部方差）
    4. 上采样回原分辨率并做 min-max 归一化

    参数
    ----
    image: [B, 3, H, W] 或 [B, 1, H, W] 图像（浮点）
    win: 局部窗口边长（奇数）
    out_size: 可选，输出分辨率（默认与输入一致）
    """
    if image.dim() != 4:
        raise ValueError("image 需为 [B,C,H,W]")
    if image.shape[1] == 3:
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    else:
        gray = image
    _, lh, hl, hh = haar_dwt2(gray)
    amp = torch.sqrt(lh * lh + hl * hl + hh * hh + eps)  # [B,1,H/2,W/2]

    pad = win // 2
    mean_a = F.avg_pool2d(amp, win, stride=1, padding=pad)
    mean_a2 = F.avg_pool2d(amp * amp, win, stride=1, padding=pad)
    var = (mean_a2 - mean_a * mean_a).clamp(min=0.0)

    if out_size is None:
        out_size = (image.shape[-2], image.shape[-1])
    var = F.interpolate(var, size=out_size, mode="bilinear", align_corners=False)

    b = var.shape[0]
    flat = var.reshape(b, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    t = (flat - lo) / (hi - lo + eps)
    return t.reshape_as(var)
