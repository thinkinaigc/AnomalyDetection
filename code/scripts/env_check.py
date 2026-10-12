"""环境检查：依赖 / CUDA / 权重是否就绪。被 script/gpu_start.bat 调用。"""

from __future__ import annotations

import importlib.util as u
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODS = ["torch", "torchvision", "open_clip", "timm", "sklearn", "scipy",
        "cv2", "pandas", "numpy", "yaml", "safetensors"]


def main() -> int:
    print("============ 环境检查 ============")
    for m in MODS:
        print(f"  {m:12s} {'OK' if u.find_spec(m) else '缺失'}")
    try:
        import torch
        print(f"  torch       {torch.__version__}  CUDA={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU         {torch.cuda.get_device_name(0)}  "
                  f"显存={torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
    except Exception as e:  # noqa
        print("  torch 导入失败:", e)

    print("\n============ 权重检查 ============")
    for name, path in (
        ("DINOv2", os.path.join(ROOT, "weights", "dinov2_vits14_pretrain.pth")),
        ("MobileCLIP", os.path.join(ROOT, "weights", "mobileclip_s1", "open_clip_model.safetensors")),
    ):
        if os.path.isfile(path):
            print(f"  {name:10s} OK  {path}")
        else:
            print(f"  {name:10s} [缺失] -> 运行 code/scripts/download_weights.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
