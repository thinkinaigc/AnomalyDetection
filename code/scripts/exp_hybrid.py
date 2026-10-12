"""对比：像素图(per-position) + 图像级分数(per-position vs kNN 位置无关) 的 AUROC。

动机：per-position 在错位/纹理类(grid/screw)图像级<0.5（模型失配导致正常图假高分）。
用位置无关的 kNN 记忆库分数做图像级判定，检验能否修复这些类别。
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
    k = max(int(flat.shape[1] * topk), 1)
    return torch.topk(flat, k, dim=1).values.mean(dim=1)


@torch.no_grad()
def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "code/configs/default.yaml"), encoding="utf-8"))
    method = build_method(cfg, "A1", device="cuda"); method.eval()
    cats = cfg["data"]["mvtec_categories"]
    P = "P3"
    perpos_img, knn_img, pix_auc, blend_img = [], [], [], []
    for cat in cats:
        method.set_category(cat)
        # per-position stats
        td = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "train", 256, 256)
        method.fit(DataLoader(td, batch_size=8, shuffle=False, num_workers=0))
        perpos_stats = {k: {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} for k, v in method._stats.items()}
        # knn bank
        feats_buf = []
        for b in DataLoader(td, batch_size=8, shuffle=False, num_workers=0):
            f = method.extract(b["image"].cuda())[P]
            feats_buf.append(F.normalize(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).float(), dim=1).cpu())
        bank = torch.cat(feats_buf).to(method.device)[:4000]
        with torch.no_grad():
            d = 1.0 - (bank @ bank.T)
            nd = torch.topk(d, 4, dim=1, largest=False).values[:, 1:].mean(1)
        knn_stat = {"d_mean": float(nd.mean()), "d_std": float(nd.std().clamp(min=1e-6))}

        ds = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "test", 256, 256)
        pp, kn, L, MK = [], [], [], []
        for b in DataLoader(ds, batch_size=4, shuffle=False, num_workers=0):
            fmap = method.extract(b["image"].cuda())[P]
            pp.append(F.interpolate(method._to01(method._scale_dist(fmap, perpos_stats[P])), size=(256, 256), mode="bilinear", align_corners=False).cpu())
            # inline kNN dist
            x = F.normalize(fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]).float(), dim=1)
            dmat = 1.0 - (x @ bank.T)
            ndmap = torch.topk(dmat, min(3, bank.shape[0]), dim=1, largest=False).values.mean(1)
            ndmap = (ndmap - knn_stat["d_mean"]) / knn_stat["d_std"]
            kn.append(F.interpolate(torch.sigmoid(ndmap).reshape(fmap.shape[0], fmap.shape[2], fmap.shape[3]).unsqueeze(1), size=(256, 256), mode="bilinear", align_corners=False).cpu())
            L.append(b["label"]); MK.append(b["mask"])
        PP = torch.cat(pp); KN = torch.cat(kn); L = torch.cat(L); MK = torch.cat(MK)
        pos = MK.numpy().ravel() > 0.5
        pix = roc_auc_score(pos, PP[:, 0].reshape(-1).numpy())
        i_pp = roc_auc_score(L.numpy(), img_score(PP).numpy())
        i_kn = roc_auc_score(L.numpy(), img_score(KN).numpy())
        blend = 0.5 * (img_score(PP).numpy() + img_score(KN).numpy())
        i_bl = roc_auc_score(L.numpy(), blend)
        perpos_img.append(i_pp); knn_img.append(i_kn); blend_img.append(i_bl); pix_auc.append(pix)
        print(f"  {cat:12s} pix(perpos)={pix:.3f} img_perpos={i_pp:.3f} img_knn={i_kn:.3f} img_blend={i_bl:.3f}")
    print(f"\nAVG pix(perpos)={np.mean(pix_auc):.4f} img_perpos={np.mean(perpos_img):.4f} img_knn={np.mean(knn_img):.4f} img_blend={np.mean(blend_img):.4f}")


if __name__ == "__main__":
    main()