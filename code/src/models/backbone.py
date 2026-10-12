"""冻结视觉主干：MobileCLIP-S1 的视觉塔（FastViT）。

选择理由（与 LiZAD 一致）：
    MobileCLIP 的视觉塔与文本塔是同族 CLIP 对，天然对齐；而 DINOv2 视觉特征
    与 CLIP 文本不在同一空间，需训练大映射才能对齐。为获得可靠的零样本异常
    信号，本文采用 MobileCLIP-S1 视觉塔（亦符合"轻量 + 8G 可训"的卖点）。

提供多尺度（实际为多分辨率 + 多深度）patch 特征：FastViT 的 4 个 stage 输出
分辨率依次为 56 / 28 / 14 / 7。默认取 stage (1,2,3) 对应 P3/P4/P5。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileCLIPBackbone(nn.Module):
    def __init__(
        self,
        model_id: str = "MobileCLIP-S1",
        weights: str | None = None,
        device: str | torch.device = "cuda",
        stages: tuple[int, ...] = (1, 2, 3),
        scale_names: tuple[str, ...] = ("P3", "P4", "P5"),
    ):
        super().__init__()
        from open_clip import create_model_and_transforms

        self.device = torch.device(device)
        weights = weights or "weights/mobileclip_s1/open_clip_model.safetensors"
        model, _, _ = create_model_and_transforms(model_id, pretrained=weights)
        self.visual = model.visual.to(self.device)
        self.trunk = self.visual.trunk
        self.stages = list(stages)
        self.scale_names = list(scale_names)

        self._feats: dict[int, torch.Tensor] = {}
        self._handles = []
        for s in self.stages:
            self._handles.append(self.trunk.stages[s].register_forward_hook(self._make_hook(s)))

        self.visual.eval()
        for p in self.visual.parameters():
            p.requires_grad_(False)

        self.image_mean = tuple(float(x) for x in model.visual.image_mean)
        self.image_std = tuple(float(x) for x in model.visual.image_std)
        self.image_size = int(model.visual.image_size[0]) if isinstance(model.visual.image_size, (tuple, list)) else int(model.visual.image_size)

    def _make_hook(self, idx: int):
        def hook(_mod, _inp, out):
            self._feats[idx] = out.detach()

        return hook

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """image: [B,3,H,W]（已按 MobileCLIP 统计归一化）。返回 {scale: [B,C,H,W]}。"""
        self._feats = {}
        _ = self.trunk(image.to(self.device))
        feats = {}
        for s, name in zip(self.stages, self.scale_names):
            f = self._feats[s]
            if f.dim() == 3:  # [B, N, C] -> 空间化（FastViT stage 输出为 [B,C,H,W]，通常不走此分支）
                b, n, c = f.shape
                side = int(n ** 0.5)
                f = f.transpose(1, 2).reshape(b, c, side, side)
            feats[name] = f
        return feats

    def out_channels(self) -> dict[str, int]:
        sd = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)
        with torch.no_grad():
            feats = self.forward(sd)
        return {k: v.shape[1] for k, v in feats.items()}

    def trainable_params(self) -> int:
        return 0


class DINOv2Backbone(nn.Module):
    """冻结 DINOv2 视觉骨干（facebookresearch/dinov2, torch.hub 已缓存）。

    提取 patch token 网格作为密集特征。支持**多尺度**（不同输入分辨率 -> 不同 patch 网格，
    如 P3=392(28x28) + P4=224(16x16)），用于多尺度融合/ADA 消融。
    每个尺度经独立的随机 1x1 固定投影降维（降噪/控协方差开销）。
    归一化统计用 ImageNet（DINOv2 训练分布）。
    """

    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_id: str = "dinov2_vitb14_reg",
        scales: dict = None,
        proj_dim: int = 256,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        # scales: {name: input_size}；默认单尺度 P3=392
        self.scales = dict(scales) if scales else {"P3": 392}
        self.scale_names = tuple(self.scales.keys())
        self.image_size = int(max(self.scales.values()))  # 数据集按最大尺寸 resize，骨干内再插值到各尺度
        self.image_mean = self.IMAGE_MEAN
        self.image_std = self.IMAGE_STD
        self.proj_dim = int(proj_dim)

        model = torch.hub.load("facebookresearch/dinov2:main", model_id, pretrained=True)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 每个尺度独立构建投影（C -> proj_dim）
        self.projs = nn.ModuleDict()
        self.model_out_c = {}
        with torch.no_grad():
            for name, sz in self.scales.items():
                sd = torch.randn(1, 3, sz, sz, device=self.device)
                ff = self.model.forward_features(sd)
                c = self._feat_grid(ff).shape[1]
                self.model_out_c[name] = c
                if self.proj_dim and self.proj_dim != c:
                    conv = nn.Conv2d(c, self.proj_dim, 1, bias=False).to(self.device)
                    nn.init.normal_(conv.weight, 0, 1.0)
                    for pp in conv.parameters():
                        pp.requires_grad_(False)
                    self.projs[name] = conv
                else:
                    self.projs[name] = None
                if name == self.image_level_scale:
                    gdim = self._global_token(ff).shape[-1]
                    if self.proj_dim and self.proj_dim != gdim:
                        gconv = nn.Conv2d(gdim, self.proj_dim, 1, bias=False).to(self.device)
                        nn.init.normal_(gconv.weight, 0, 1.0)
                        for pp in gconv.parameters():
                            pp.requires_grad_(False)
                        self.proj_global = gconv
                    else:
                        self.proj_global = None

    # 取第一个尺度用于提取全局 token（图像级全局描述子）
    image_level_scale = property(lambda self: next(iter(self.scales)))

    @staticmethod
    def _feat_grid(feat):
        # dinov2 forward_features 返回 dict；取 patch token 序列 [B, N, C] -> [B, C, h, w]
        if isinstance(feat, dict):
            for k in ("x_norm_patchtokens", "x_norm_patchtokens", "patch_tokens"):
                if k in feat:
                    feat = feat[k]
                    break
            else:
                feat = list(feat.values())[0]
        b, n, c = feat.shape
        side = int(n ** 0.5)
        patch = feat.transpose(1, 2).reshape(b, c, side, side)
        return patch

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """image: [B,3,H,W]（已按 ImageNet 统计归一化，尺寸=image_size）。
        骨干内按各尺度尺寸双线性插值后提取 patch 网格；同时在默认尺度附加全局
        token（CLS + register tokens 拼接后均值池化）作为图像级全局描述子。
        返回 {scale: [B,C',h,w], 'GLOBAL': [B, C']}。"""
        out = {}
        for name, sz in self.scales.items():
            x = F.interpolate(image.to(self.device), size=(sz, sz), mode="bilinear", align_corners=False)
            ff = self.model.forward_features(x)
            grid = self._feat_grid(ff)  # [B, C, h, w]
            proj = self.projs[name]
            if proj is not None:
                grid = proj(grid)
            out[name] = grid
            if name == self.image_level_scale:
                g = self._global_token(ff)  # [B, C']
                if self.proj_global is not None:
                    g = self.proj_global(g.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
                out["GLOBAL"] = g
        return out

    @staticmethod
    def _global_token(ff: dict) -> torch.Tensor:
        """取 CLS 与 register token（若存在）拼接后均值池化，得到 [B, C] 全局描述子。"""
        toks = []
        for k in ("x_norm_clstoken", "x_norm_regtokens"):
            if isinstance(ff, dict) and k in ff:
                t = ff[k]
                if t.dim() == 3:  # [B, R, C] -> 池化为 [B, C]
                    t = t.mean(dim=1)
                toks.append(t)
        if not toks:  # 退化：用 patch 均值
            g = DINOv2Backbone._feat_grid(ff)
            return g.mean(dim=(-1, -2))
        return torch.cat(toks, dim=-1)  # [B, C']

    def out_channels(self) -> dict[str, int]:
        return {
            n: (self.proj_dim if self.projs[n] is not None else self.model_out_c[n])
            for n in self.scales
        }

    def trainable_params(self) -> int:
        return 0
