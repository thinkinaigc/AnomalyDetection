"""P4：图像级打分策略对比，试图修复 grid/screw 图像 AUROC<0.5。

候选：
  peak   : 当前方法 —— 高斯模糊后对 top-1% 像素取均值（峰值敏感，错位类失效）
  area   : 异常面积占比 —— 标准化分数图中超过阈值 t 的像素比例（对孤立假峰鲁棒）
逐类别 fit 一次，比较 peak vs area(t) 的 图像AUROC / 像素AUROC（全图协议）。
"""
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
    ax = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2)); g = (g / g.sum()).view(1, 1, 1, k)
    x = F.conv2d(x, g, padding=(0, k // 2)); x = F.conv2d(x, g.transpose(2, 3), padding=(k // 2, 0))
    return x


@torch.no_grad()
def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "code/configs/default.yaml"), encoding="utf-8"))
    cfg["method"]["ref_smooth"] = 7
    cats = cfg["data"]["mvtec_categories"]
    # 候选图像级打分
    topks = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
    t = 0.7
    acc = {f"topk{k}": [] for k in topks}
    acc["max_pa"] = []; acc["area07"] = []
    per_cat = {}
    for cat in cats:
        method = build_method(cfg, "A1", device="cuda"); method.eval()
        method.set_category(cat)
        td = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "train", 256, 256)
        method.fit(DataLoader(td, batch_size=8, shuffle=False, num_workers=0))
        ds = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "test", 256, 256)
        M, L, MK = [], [], []
        for b in DataLoader(ds, batch_size=4, shuffle=False, num_workers=0):
            m = method.raw_score(b["image"].cuda())[0]
            M.append(F.interpolate(m, size=(256, 256), mode="bilinear", align_corners=False).cpu())
            L.append(b["label"]); MK.append(b["mask"])
        M = torch.cat(M); L = torch.cat(L); MK = torch.cat(MK)
        pos = MK.numpy().ravel() > 0.5
        pix = roc_auc_score(pos, M[:, 0].reshape(-1).numpy())
        flat = M[:, 0].reshape(M.shape[0], -1)
        sm = blur(M, 11)
        smf = sm[:, 0].reshape(sm.shape[0], -1)
        pk = {k: torch.topk(smf, max(int(smf.shape[1] * k), 1), dim=1).values.mean(1) for k in topks}
        area = (M[:, 0] > t).float().reshape(M.shape[0], -1).mean(1)
        # 逐图像 max(peak0.01, area) 与 area 单独
        max_pa = torch.max(pk[0.01], area)
        for k in topks:
            acc[f"topk{k}"].append(roc_auc_score(L.numpy(), pk[k].numpy()))
        acc["max_pa"].append(roc_auc_score(L.numpy(), max_pa.numpy()))
        acc["area07"].append(roc_auc_score(L.numpy(), area.numpy()))
        per_cat[cat] = {k: acc[k][-1] for k in acc}
        print(f"  {cat:12s} pix={pix:.3f} " + " ".join(f"{k}={per_cat[cat][k]:.3f}" for k in acc), flush=True)
    print("\n=== AVG image AUROC ===")
    for k in acc:
        print(f"  {k:8s} {np.mean(acc[k]):.4f}")


if __name__ == "__main__":
    main()