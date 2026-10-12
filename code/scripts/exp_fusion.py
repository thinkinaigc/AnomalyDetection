"""多尺度融合策略实验：收集每尺度异常图一次，比较不同融合方式（像素/图像 AUROC）。"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))

from src.data.dataset import AnomalyDataset  # noqa: E402
from src.methods.factory import build_method  # noqa: E402
from src.modules.ada import high_freq_ratio  # noqa: E402


def img_score(amap, topk=0.01, blur=11):
    if blur > 1:
        ax = torch.arange(blur, dtype=torch.float32) - (blur - 1) / 2
        g = torch.exp(-(ax ** 2) / (2 * (blur / 4.0) ** 2)); g = (g / g.sum()).view(1, 1, 1, blur)
        amap = F.conv2d(amap, g, padding=(0, blur // 2))
        amap = F.conv2d(amap, g.transpose(2, 3), padding=(blur // 2, 0))
    b = amap.shape[0]; flat = amap.reshape(b, -1)
    k = max(int(flat.shape[1] * topk), 1)
    return torch.topk(flat, k, dim=1).values.mean(dim=1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A1")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "code/configs/default.yaml"), encoding="utf-8"))
    method = build_method(cfg, args.variant, device="cuda"); method.eval()
    cats = cfg["data"]["mvtec_categories"]
    names = list(method.scale_names)

    results = {}
    per_cat_best = {}
    for cat in cats:
        method.set_category(cat)
        td = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "train", 256, 256)
        method.fit(DataLoader(td, batch_size=8, shuffle=False, num_workers=0))
        ds = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "test", 256, 256)
        smaps, labels, masks = {n: [] for n in names}, [], []
        for b in DataLoader(ds, batch_size=4, shuffle=False, num_workers=0):
            feats = method.extract(b["image"].cuda())
            sc = method._score_maps(feats)
            for n in sc:
                smaps[n].append(method._to01(sc[n]).cpu())
            labels.append(b["label"]); masks.append(b["mask"])
        S = {n: torch.cat(v) for n, v in smaps.items()}
        L = torch.cat(labels); MK = torch.cat(masks)
        pos = MK.numpy().ravel() > 0.5
        tgt = MK.shape[-2:]  # 掩码分辨率

        def up(x):
            return F.interpolate(x, size=tgt, mode="bilinear", align_corners=False)

        variants = {
            "single_P3": up(S["P3"]),
            "single_P4": up(S["P4"]),
            "mean_P3P4": 0.5 * (up(S["P3"]) + up(S["P4"])),
            "mean_P3P4P5": (up(S["P3"]) + up(S["P4"]) + up(S["P5"])) / 3,
            "max_P3P4": torch.maximum(up(S["P3"]), up(S["P4"])),
            "max_P3P4P5": torch.maximum(torch.maximum(up(S["P3"]), up(S["P4"])), up(S["P5"])),
            "w2_1_1": (2 * up(S["P3"]) + up(S["P4"]) + up(S["P5"])) / 4,
        }
        for key, amap in variants.items():
            pix = roc_auc_score(pos, amap[:, 0].reshape(-1).numpy())
            img = roc_auc_score(L.numpy(), img_score(amap).numpy())
            results.setdefault(key, []).append((pix, img))
            per_cat_best.setdefault(cat, {})[key] = (pix, img)

    print(f"{'variant':14s}{'pixAUR':>9s}{'imgAUR':>9s}")
    for key, v in results.items():
        p = np.mean([x[0] for x in v]); i = np.mean([x[1] for x in v])
        print(f"{key:14s}{p:9.4f}{i:9.4f}")
    print("\nper-cat pixel AUROC:")
    for cat in cats:
        row = " ".join(f"{k}={per_cat_best[cat][k][0]:.3f}" for k in ["single_P3", "mean_P3P4", "max_P3P4", "mean_P3P4P5"])
        print(f"  {cat:12s}{row}")


if __name__ == "__main__":
    main()