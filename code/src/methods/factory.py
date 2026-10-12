"""构建方法实例：选择视觉骨干（MobileCLIP-S1 / DINOv2）+ 可选文本塔 + 轻量适配器 + 消融开关。

视觉骨干只用于提取冻结密集特征；评分由 methods/base.py 的逐位置高斯正常参考完成。
"""

from __future__ import annotations

import torch

from ..models.backbone import DINOv2Backbone, MobileCLIPBackbone
from ..utils.common import get_device
from .fgsc_ada import ABLATION, METHOD_ALIAS, FGSC_ADA, LiZAD


def build_method(cfg: dict, variant: str = "A1", device: torch.device | str = "cuda", override_method: dict = None):
    """cfg 为 configs/default.yaml 中的完整配置字典。override_method 中的键优先于 ABLATION 变体默认。"""
    device = get_device(device) if isinstance(device, str) else device

    mcfg = dict(cfg.get("model", {}))
    bb_name = str(mcfg.get("backbone", "mobileclip")).lower()

    if bb_name == "dinov2":
        backbone = DINOv2Backbone(
            model_id=mcfg.get("dinov2_model_id", "dinov2_vitb14_reg"),
            scales=dict(mcfg.get("dinov2_scales", {"P3": 392})),
            proj_dim=int(mcfg.get("dinov2_proj_dim", 256)),
            device=device,
        )
        text_encoder = None
        adapters = None
        default_cov = "diag"
    else:
        backbone = MobileCLIPBackbone(
            model_id=mcfg.get("mobileclip_model_id", "MobileCLIP-S1"),
            weights=mcfg.get("weights"),
            device=device,
            stages=tuple(mcfg.get("stages", (1, 2, 3))),
            scale_names=tuple(mcfg.get("scale_names", ("P3", "P4", "P5"))),
        )
        from ..models.projection import ScaleAdapters
        from ..models.text_encoder import MobileCLIPTextEncoder

        text_encoder = MobileCLIPTextEncoder(device=device)
        in_dims = backbone.out_channels()
        adapters = ScaleAdapters(
            in_dims,
            out_dim=int(mcfg.get("out_dim", 512)),
            bottleneck=int(mcfg.get("adapter_bottleneck", 256)),
        ).to(device)
        default_cov = "full"

    key = METHOD_ALIAS.get(variant.lower(), variant.upper())
    base = dict(cfg.get("method", {}))
    base.update(ABLATION.get(key, {}))
    if override_method:
        base.update(override_method)
    base["img_size"] = int(mcfg.get("img_size", 256))
    base["map_size"] = int(cfg.get("eval", {}).get("map_size", 256))
    base["out_dim"] = int(mcfg.get("out_dim", 512))
    base["cov_mode"] = base.get("cov_mode", default_cov)
    base["scale_names"] = backbone.scale_names

    cls = LiZAD if key == "A1" else FGSC_ADA
    method = cls(backbone, text_encoder, adapters, cfg=base, device=device)
    method.eval()
    return method
