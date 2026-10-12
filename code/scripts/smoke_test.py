"""冒烟测试：验证权重加载 / 主干 / 文本编码器 / 各模块 / 全流程前向与显存。

用法:
    python code/scripts/smoke_test.py [--device cuda] [--img-size 224] [--batch 2]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AnomalyDataset  # noqa: E402
from src.methods.factory import build_method  # noqa: E402
from src.modules.wavelet import haar_dwt2, texture_confidence  # noqa: E402
from src.utils.common import ROOT, get_device, get_logger, load_yaml  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--category", default="bottle")
    args = ap.parse_args()

    logger = get_logger("smoke")
    device = get_device(args.device)
    logger.info(f"device={device} torch={torch.__version__}")

    cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
    cfg["model"]["img_size"] = args.img_size
    cfg["eval"]["map_size"] = args.img_size

    method = build_method(cfg, "A7", device=device)
    logger.info(f"method: {method}")
    logger.info(f"backbone: {method.backbone}")
    logger.info(f"text_encoder: {method.text_encoder}")
    logger.info(f"可训练参数: {method.trainable_params()}")

    method.set_category(args.category)
    ds = AnomalyDataset(os.path.join(ROOT, "data", "mvtec_ad"), args.category, "test",
                        img_size=args.img_size, mask_size=args.img_size)
    imgs = torch.stack([ds[i]["image"] for i in range(min(args.batch, len(ds)))]).to(device)
    logger.info(f"images {tuple(imgs.shape)}")

    # 1) Haar + 纹理图
    tex = texture_confidence(imgs[:, :3] * 0 + imgs[:, :3], win=9, out_size=(args.img_size // 8,) * 2)
    logger.info(f"texture_map {tuple(tex.shape)} range=[{tex.min():.3f}, {tex.max():.3f}]")
    ll, lh, hl, hh = haar_dwt2(imgs[:, :1])
    logger.info(f"haar subbands: {tuple(ll.shape)} {tuple(lh.shape)} {tuple(hl.shape)} {tuple(hh.shape)}")

    # 2) 主干多尺度特征
    feats = method.extract(imgs)
    for k, v in feats.items():
        logger.info(f"feat {k}: {tuple(v.shape)}")

    # 3) 全流程前向（各变体）
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    for variant in ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]:
        m = build_method(cfg, variant, device=device)
        m.set_category(args.category)
        out = m.predict(imgs)
        logger.info(
            f"{variant}: map={tuple(out['anomaly_map'].shape)} "
            f"range=[{out['anomaly_map'].min():.3f},{out['anomaly_map'].max():.3f}] "
            f"img_score={[round(float(x), 4) for x in out['image_score']]}"
        )
    if device.type == "cuda":
        logger.info(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024 ** 2:.0f} MB")

    # 4) 训练一步（投影头）
    m = build_method(cfg, "A6", device=device)
    m.set_category(args.category)
    m.fit([{"image": imgs[i:i + 1]} for i in range(imgs.shape[0])], epochs=1, logger=logger)
    logger.info(f"fit 后 score_min={float(m.score_min):.3f} score_max={float(m.score_max):.3f}")
    logger.info("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
