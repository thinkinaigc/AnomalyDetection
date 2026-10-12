"""测试：对异常图做频域(高斯低通)平滑是否提升像素 AUROC，作为 FGSC 的可用机制。"""
from __future__ import annotations

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


def blur(x, k):
    if k <= 1:
        return x
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2)); g = (g / g.sum()).view(1, 1, 1, k)
    x = F.conv2d(x, g, padding=(0, k // 2))
    x = F.conv2d(x, g.transpose(2, 3), padding=(k // 2, 0))
    return x


@torch.no_grad()
def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "code/configs/default.yaml"), encoding="utf-8"))
    method = build_method(cfg, "A1", device="cuda"); method.eval()
    cats = cfg["data"]["mvtec_categories"]
    ks = [1, 3, 5, 7, 9, 13]
    acc = {k: [] for k in ks}
    for cat in cats:
        method.set_category(cat)
        td = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "train", 256, 256)
        method.fit(DataLoader(td, batch_size=8, shuffle=False, num_workers=0))
        ds = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "test", 256, 256)
        maps, MK = [], []
        for b in DataLoader(ds, batch_size=4, shuffle=False, num_workers=0):
            m = method.raw_score(b["image"].cuda())[0]
            maps.append(m.cpu()); MK.append(b["mask"])
        M = torch.cat(maps); MK = torch.cat(MK)
        pos = MK.numpy().ravel() > 0.5
        for k in ks:
            mb = blur(M, k)
            acc[k].append(roc_auc_score(pos, mb[:, 0].reshape(-1).numpy()))
    print(f"{'blurK':>6s}{'pixAUR':>9s}")
    for k in ks:
        print(f"{k:6d}{np.mean(acc[k]):9.4f}")


if __name__ == "__main__":
    main()