"""方法变体定义（消融实验 A1–A7 与主实验）。

打分骨干：冻结 MobileCLIP-S1 多尺度特征 + 逐位置(PaDiM式)正常参考马氏距离。
A1 单尺度基线(P3)   : fusion=single,  fgsc=off
A2 固定多尺度均值    : fusion=mean,    fgsc=off
A3 ADA 自适应多尺度  : fusion=ada,     fgsc=off
A4 + FGSC 固定低通    : fusion=single,  fgsc=lowpass
A5 + FGSC 纹理引导低通 : fusion=single,  fgsc=adaptive
A6 ADA + FGSC(纹理引导): fusion=ada,     fgsc=adaptive
A7 + FGSC + DPA      : fusion=single,  fgsc=adaptive, dpa=True
"""

from __future__ import annotations

from .base import BaseZSAD

ABLATION = {
    "A1": dict(fusion="single", single_scale="P3", fgsc="off", dpa=False),
    "A2": dict(fusion="mean", fgsc="off", dpa=False),
    "A3": dict(fusion="ada", fgsc="off", dpa=False),
    "A4": dict(fusion="single", single_scale="P3", fgsc="lowpass", dpa=False),
    "A5": dict(fusion="single", single_scale="P3", fgsc="adaptive", dpa=False),
    "A6": dict(fusion="ada", fgsc="adaptive", dpa=False),
    "A7": dict(fusion="single", single_scale="P3", fgsc="adaptive", dpa=True),
}

METHOD_ALIAS = {
    "lizad": "A1",
    "winclip_fixed_multiscale": "A2",
    "ada": "A3",
    "fgsc": "A4",
    "fgsc_global": "A5",
    "fgsc_ada": "A6",
    "ours": "A7",
    "full": "A7",
}


class LiZAD(BaseZSAD):
    """轻量单尺度对齐基线（本文直接改进对象）。"""

    def __init__(self, *args, cfg: dict | None = None, **kwargs):
        cfg = dict(cfg or {})
        cfg.setdefault("fusion", "single")
        cfg.setdefault("single_scale", "P3")
        cfg.setdefault("fgsc", "off")
        cfg.setdefault("dpa", False)
        super().__init__(*args, cfg=cfg, **kwargs)


class FGSC_ADA(BaseZSAD):
    """本文方法，可通过 cfg 开关退化成任意消融变体。"""

    def __init__(self, *args, cfg: dict | None = None, **kwargs):
        cfg = dict(cfg or {})
        cfg.setdefault("fusion", "single")
        cfg.setdefault("single_scale", "P3")
        cfg.setdefault("fgsc", "adaptive")
        cfg.setdefault("dpa", True)
        super().__init__(*args, cfg=cfg, **kwargs)


def resolve_variant(name: str) -> dict:
    """把方法名/消融编号解析成配置字典。"""
    key = METHOD_ALIAS.get(name.lower(), name.upper())
    if key not in ABLATION:
        raise KeyError(f"未知方法名: {name}（可选: {sorted(METHOD_ALIAS)} / {sorted(ABLATION)}）")
    return dict(ABLATION[key])
