"""冻结文本编码器：MobileCLIP-S1（仅文本塔），带提示词嵌入缓存。

权重来自 weights/mobileclip_s1/open_clip_model.safetensors（Apple 官方 open_clip 格式），
分词器复用 open_clip 自带的 CLIP BPE（vocab 49408，与 MobileCLIP 训练时一致）。
"""

from __future__ import annotations

import os
from typing import Sequence

import torch
import torch.nn as nn

from ..utils.common import path


def _load_safetensors(file: str) -> dict:
    from safetensors.torch import load_file

    return load_file(file)


class MobileCLIPTextEncoder(nn.Module):
    """MobileCLIP-S1 文本塔（冻结）。"""

    def __init__(
        self,
        weight_path: str | None = None,
        device: torch.device | str = "cuda",
        context_length: int = 77,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        from open_clip.model import TextTransformer

        weight_path = weight_path or path("weights", "mobileclip_s1", "open_clip_model.safetensors")
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"未找到 MobileCLIP 权重: {weight_path}，请先运行 code/scripts/download_weights.py")

        self.context_length = context_length
        self.embed_dim = 512

        self.text = TextTransformer(
            context_length=context_length,
            vocab_size=49408,
            width=512,
            heads=8,
            layers=12,
            no_causal_mask=True,
            output_dim=512,
        )

        sd = _load_safetensors(weight_path)
        text_sd = {}
        for k, v in sd.items():
            if k.startswith("text."):
                text_sd[k[len("text."):]] = v
        if not text_sd:
            raise RuntimeError(f"{weight_path} 中未找到 text.* 权重")
        missing, unexpected = self.text.load_state_dict(text_sd, strict=False)
        if missing:
            raise RuntimeError(f"MobileCLIP 文本塔缺少必需键: {missing[:5]} ...")

        self.text.eval()
        for p in self.text.parameters():
            p.requires_grad_(False)

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.text.to(self.device).to(dtype)

        self._cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], use_cache: bool = True) -> torch.Tensor:
        """返回 L2 归一化后的文本嵌入 [N, 512]。"""
        keys = tuple(prompts)
        if use_cache and keys in self._cache:
            return self._cache[keys]
        from open_clip import tokenize

        tokens = tokenize(list(prompts), context_length=self.context_length).to(self.device)
        feats = self.text(tokens).float()
        feats = torch.nn.functional.normalize(feats, dim=-1)
        if use_cache:
            self._cache[keys] = feats
        return feats

    def __repr__(self) -> str:  # pragma: no cover
        return f"MobileCLIPTextEncoder(embed_dim={self.embed_dim}, ctx={self.context_length})"
