"""P4：正常参考均值的位置空间平滑（ref_smooth）能否修复 grid/screw 图像 AUROC。

逐类别 fit（仅正常图），测 ref_smooth ∈ {0,3,5,7} 的 像素AUROC / 图像AUROC（全图协议）。
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


def img_score(amap, topk=0.01, blur=11):
    if blur > 1:
        ax = torch.arange(blur, dtype=torch.float32) - (blur - 1) / 2
        g = torch.exp(-(ax ** 2) / (2 * (blur / 4.0) ** 2)); g = (g / g.sum()).view(1, 1, 1, blur)
        amap = F.conv2d(amap, g, padding=(0, blur // 2))
        amap = F.conv2d(amap, g.transpose(2, 3), padding=(blur // 2, 0))
    b = amap.shape[0]; flat = amap.reshape(b, -1)
    kk = max(int(flat.shape[1] * topk), 1)
    return torch.topk(flat, kk, dim=1).values.mean(dim=1)


@torch.no_grad()
def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "code/configs/default.yaml"), encoding="utf-8"))
    cats = cfg["data"]["mvtec_categories"]
    Ks = [0, 3, 5, 7]
    results = {k: {"pix": [], "img": []} for k in Ks}
    for cat in cats:
        for k in Ks:
            method = build_method(cfg, "A1", device="cuda"); method.eval()
            method.ref_smooth = k
            method.set_category(cat)
            td = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "train", 256, 256)
            method.fit(DataLoader(td, batch_size=8, shuffle=False, num_workers=0))
            ds = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "test", 256, 256)
            maps, L, MK = [], [], []
            for b in DataLoader(ds, batch_size=4, shuffle=False, num_workers=0):
                m = method.raw_score(b["image"].cuda())[0]
                maps.append(F.interpolate(m, size=(256, 256), mode="bilinear", align_corners=False).cpu())
                L.append(b["label"]); MK.append(b["mask"])
            M = torch.cat(maps); L = torch.cat(L); MK = torch.cat(MK)
            pos = MK.numpy().ravel() > 0.5
            pix = roc_auc_score(pos, M[:, 0].reshape(-1).numpy())
            img = roc_auc_score(L.numpy(), img_score(M).numpy())
            results[k]["pix"].append(pix); results[k]["img"].append(img)
            print(f"  {cat:12s} ref={k:2d} pix={pix:.3f} img={img:.3f}", flush=True)
    print("\n=== AVG (MVTec 9类) ===")
    for k in Ks:
        print(f"  ref={k:2d}  pix={np.mean(results[k]['pix']):.4f}  img={np.mean(results[k]['img']):.4f}")


if __name__ == "__main__":
    main()