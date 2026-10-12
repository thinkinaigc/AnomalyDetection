"""轻量适配器（LiZAD 式）：仅对视觉 patch 特征做适配，文本方向保持冻结。

关键设计（与 LiZAD 一致，避免训练坍塌）：
    文本 normal/abnormal 嵌入由冻结文本编码器给出，作为固定的跨模态语义锚点；
    只训练每个尺度的 patch 适配器，把视觉特征映射到该语义空间，使正常区域对齐
    "正常"文本、偏离正常的缺陷区域自然获得高"异常"相似度。
结构：Linear(+Bottleneck) -> LeakyReLU（last_activation）。可训练参数 < 1M。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int | None = None, bottleneck: int | None = None, last_activation: bool = True):
        super().__init__()
        output_dim = output_dim or input_dim
        if bottleneck is not None:
            self.net = nn.Sequential(
                nn.Linear(input_dim, bottleneck),
                nn.LeakyReLU(),
                nn.Linear(bottleneck, output_dim),
                nn.LeakyReLU() if last_activation else nn.Identity(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LeakyReLU() if last_activation else nn.Identity(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScaleAdapters(nn.Module):
    """每个尺度一份独立的 patch 适配器（文本方向冻结，仅做归一化）。"""

    def __init__(self, in_dims: dict[str, int], out_dim: int = 512, bottleneck: int = 256):
        super().__init__()
        self.scale_names = list(in_dims.keys())
        self.out_dim = out_dim
        self.patch = nn.ModuleDict(
            {name: Adapter(c, out_dim, bottleneck=bottleneck) for name, c in in_dims.items()}
        )

    def patch_forward(self, feat_map: torch.Tensor, scale: str) -> torch.Tensor:
        """feat_map: [B,C,H,W] -> [B,out_dim,H,W]，并 L2 归一化。"""
        b, c, h, w = feat_map.shape
        y = self.patch[scale](feat_map.permute(0, 2, 3, 1).reshape(-1, c))
        y = y.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return torch.nn.functional.normalize(y.float(), dim=1)

    def text_forward(self, tn: torch.Tensor, ta: torch.Tensor):
        """文本方向冻结：仅归一化，不进入可训练适配器。"""
        tn_a = torch.nn.functional.normalize(tn.float(), dim=-1)
        ta_a = torch.nn.functional.normalize(ta.float(), dim=-1)
        return tn_a, ta_a
