"""图像级打分策略实验：从缓存的异常图上比较不同聚合方式（不重跑主干）。

同时输出每个单尺度的像素 AUROC，用于判断多尺度融合是否值得。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))

from src.data.dataset import AnomalyDataset  # noqa: E402
from src.methods.factory import build_method  # noqa: E402


def gaussian_blur(x: torch.Tensor, k: int = 7) -> torch.Tensor:
    if k <= 1:
        return x
    ax = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2))
    g = (g / g.sum()).view(1, 1, 1, k)
    x = F.conv2d(x, g, padding=(0, k // 2))
    x = F.conv2d(x, g.transpose(2, 3), padding=(k // 2, 0))
    return x


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="bottle,cable,grid,metal_nut,pill,screw,toothbrush,transistor,zipper")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "code/configs/default.yaml"), encoding="utf-8"))
    method = build_method(cfg, "A1", device="cuda")
    method.eval()

    from torch.utils.data import DataLoader

    strategies = [
        ("top1%", 0, 0.01), ("top1%", 5, 0.01), ("top1%", 9, 0.01),
        ("top3%", 5, 0.03), ("top5%", 5, 0.05), ("top10%", 5, 0.10),
        ("top5%", 9, 0.05), ("top1%", 15, 0.01),
    ]
    acc = {s: [] for s in strategies}
    per_scale_pix = {}

    for cat in args.categories.split(","):
        method.set_category(cat)
        td = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "train", 256, 256)
        method.fit(DataLoader(td, batch_size=8, shuffle=False, num_workers=0))
        ds = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "test", 256, 256)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

        maps, labels, masks = [], [], []
        scale_maps = {n: [] for n in method.scale_names}
        for b in dl:
            img = b["image"].cuda()
            feats = method.extract(img)
            smaps = method._score_maps(feats)
            raw01 = {n: method._to01(smaps[n]) for n in smaps}
            fused = method._fuse_maps(raw01, feats)
            maps.append(fused.cpu())
            for n in raw01:
                up = F.interpolate(raw01[n], size=fused.shape[-2:], mode="bilinear", align_corners=False)
                scale_maps[n].append(up.cpu())
            labels.append(b["label"])
            masks.append(b["mask"])
        M = torch.cat(maps)
        L = torch.cat(labels)
        MK = torch.cat(masks)
        ms = {n: torch.cat(v) for n, v in scale_maps.items()}

        # 每尺度像素 AUROC（全图协议）
        for n, S in ms.items():
            pm = MK.ravel().numpy() > 0.5
            ps = S[:, 0].reshape(-1).numpy()
            per_scale_pix.setdefault(n, []).append(roc_auc_score(pm, ps))

        for s in strategies:
            _, ksz, tk = s
            sc = []
            for i in range(M.shape[0]):
                m = gaussian_blur(M[i:i + 1], ksz)
                flat = m.reshape(-1)
                k = max(int(flat.shape[0] * tk), 1)
                sc.append(float(torch.topk(flat, k).values.mean()))
            acc[s].append(roc_auc_score(L.numpy(), sc))

    print("=== per-scale pixel AUROC (full protocol) ===")
    for n, v in per_scale_pix.items():
        print(f"  {n}: {np.mean(v):.4f}")
    print("=== image AUROC by strategy (mean over cats) ===")
    for s in strategies:
        name, k, tk = s
        print(f"  blur{k}/top{tk*100:.0f}%: {np.mean(acc[s]):.4f}")


if __name__ == "__main__":
    main()