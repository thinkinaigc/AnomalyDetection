"""DPA：工业领域分层提示词适配（支撑组件）。

三层提示词：
    L0 通用层  : a photo of a normal/defective {object}
    L1 表面层  : 表面缺陷相关描述（scratch / crack / stain ...）
    L2 类别层  : 具体类别的典型缺陷名（由类别字典给出）

路由规则：按图像纹理复杂度（纹理置信度图均值）动态选择提示词子集——
纹理越复杂 → 越需要细粒度缺陷提示词，否则通用提示词已足够。
"""

from __future__ import annotations

from typing import Sequence

# 各数据集类别的显示名（用于填充模板），未登记类别用 object 兜底
OBJECT_NAMES = {
    # MVTec-AD
    "bottle": "bottle", "cable": "cable", "capsule": "capsule", "carpet": "carpet",
    "grid": "grid", "hazelnut": "hazelnut", "leather": "leather", "metal_nut": "metal nut",
    "pill": "pill", "screw": "screw", "tile": "tile", "toothbrush": "toothbrush",
    "transistor": "transistor", "wood": "wood", "zipper": "zipper",
    # PKU-Market-PCB
    "missing_hole": "printed circuit board", "mouse_bite": "printed circuit board",
    "open_circuit": "printed circuit board", "short": "printed circuit board",
    "spur": "printed circuit board", "spurious_copper": "printed circuit board",
    # NEU-DET
    "crazing": "steel surface", "inclusion": "steel surface", "patches": "steel surface",
    "pitted_surface": "steel surface", "rolled-in_scale": "steel surface",
    "rolled_in_scale": "steel surface", "scratches": "steel surface",
}

# L2：类别特有的典型缺陷描述（仅列常见类别，缺失时用 L1 兜底）
SPECIFIC_DEFECTS = {
    "bottle": ["broken bottle", "cracked bottle", "contaminated bottle"],
    "cable": ["bent cable", "cut cable insulation", "missing cable"],
    "capsule": ["cracked capsule", "faulty capsule imprint", "contaminated capsule"],
    "carpet": ["cut carpet", "stained carpet", "torn carpet"],
    "grid": ["broken grid", "bent grid", "thread on grid"],
    "hazelnut": ["cracked hazelnut", "cut hazelnut", "hole on hazelnut"],
    "leather": ["cut leather", "stained leather", "scratched leather"],
    "metal_nut": ["scratched metal nut", "flipped metal nut", "bent metal nut"],
    "pill": ["cracked pill", "contaminated pill", "faulty pill imprint"],
    "screw": ["scratched screw head", "manipulated screw front", "damaged screw thread"],
    "tile": ["cracked tile", "stained tile", "oil spot on tile"],
    "toothbrush": ["defective toothbrush", "broken toothbrush bristle"],
    "transistor": ["bent transistor lead", "cut transistor lead", "damaged transistor case"],
    "wood": ["scratched wood", "stained wood", "hole in wood"],
    "zipper": ["broken zipper teeth", "split zipper teeth", "rough zipper"],
    "open_circuit": ["open circuit on printed circuit board", "broken copper trace"],
    "short": ["short circuit on printed circuit board", "unwanted copper bridge"],
    "mouse_bite": ["mouse bite on printed circuit board", "notched copper edge"],
    "spur": ["spur on printed circuit board", "unexpected copper spur"],
    "missing_hole": ["missing hole on printed circuit board", "absent via hole"],
    "spurious_copper": ["spurious copper on printed circuit board", "stray copper island"],
    "crazing": ["crazing on steel surface", "network of fine cracks"],
    "inclusion": ["inclusion on steel surface", "embedded impurity"],
    "patches": ["patches on steel surface", "large surface patch"],
    "pitted_surface": ["pitted steel surface", "dense pits"],
    "rolled-in_scale": ["rolled-in scale on steel surface", "pressed oxide scale"],
    "rolled_in_scale": ["rolled-in scale on steel surface", "pressed oxide scale"],
    "scratches": ["scratches on steel surface", "thin scratch line"],
}

L1_SURFACE = [
    "a photo of a {} with a surface defect",
    "a photo of a damaged {}",
    "a photo of a {} with a scratch",
    "a photo of a {} with a crack",
    "a photo of a {} with a stain",
]

L0_NORMAL = [
    "a photo of a normal {}",
    "a perfect {}",
    "a flawless {}",
    "a photo of a good {}",
]

L0_ABNORMAL = [
    "a photo of an anomalous {}",
    "a photo of a defective {}",
    "a damaged {}",
    "an abnormal {}",
]


def object_name(category: str) -> str:
    return OBJECT_NAMES.get(category, category.replace("_", " "))


def build_prompts(category: str, level: str = "L0+L1+L2") -> tuple[list[str], list[str]]:
    """返回 (normal_prompts, abnormal_prompts)。"""
    name = object_name(category)
    normal = [t.format(name) for t in L0_NORMAL]
    abnormal = [t.format(name) for t in L0_ABNORMAL]
    if level in ("L0+L1", "L0+L1+L2"):
        abnormal += [t.format(name) for t in L1_SURFACE]
    if level == "L0+L1+L2":
        abnormal += SPECIFIC_DEFECTS.get(category, [])
    return normal, abnormal


def route_level(texture_complexity: float, t1: float = 0.25, t2: float = 0.45) -> str:
    """按纹理复杂度路由提示词层级。

    complexity < t1 -> L0；t1~t2 -> L0+L1；>= t2 -> L0+L1+L2
    """
    if texture_complexity < t1:
        return "L0"
    if texture_complexity < t2:
        return "L0+L1"
    return "L0+L1+L2"


def dedup(prompts: Sequence[str]) -> list[str]:
    seen, out = set(), []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out
