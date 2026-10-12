"""FGSC(v2)：纹理引导的频域自适应低通校准。

机制：per-position 打分的异常图在高频处存在孤立假阳性噪声；对"高频纹理区域"做
更强的频域低通（把该处与邻域平滑混合），而平滑区域保留原始锐度以免损伤微小缺陷边界：

    S_cal = (1 - a*T) * S  +  (a*T) * LowPass_K(S)

其中 T 为 Haar 纹理能量图(归一化到[0,1])，a 为最大平滑强度。K 为低通核大小。
对比：无平滑基线 / 固定低通 / 纹理自适应低通。
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
from src.modules.wavelet import texture_confidence  # noqa: E402


def blur(x, k):
    if k <= 1:
        return x
    ax = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2)); g = (g / g.sum()).view(1, 1, 1, k)
    x = F.conv2d(x, g, padding=(0, k // 2))
    x = F.conv2d(x, g.transpose(2, 3), padding=(k // 2, 0))
    return x


def img_score(amap, topk=0.01, iblur=11):
    amap = blur(amap, iblur)
    b = amap.shape[0]; flat = amap.reshape(b, -1)
    k = max(int(flat.shape[1] * topk), 1)
    return torch.topk(flat, k, dim=1).values.mean(dim=1)


@torch.no_grad()
def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "code/configs/default.yaml"), encoding="utf-8"))
    method = build_method(cfg, "A1", device="cuda"); method.eval()
    cats = cfg["data"]["mvtec_categories"]

    configs = [("base", 0.0, 0.0, 1)]
    for a0 in (0.4, 0.6, 0.8, 1.0):
        for a1 in (0.0, 0.4):
            for K in (21, 25, 33):
                configs.append((f"a0{a0}_a1{a1}_K{K}", a0, a1, K))

    acc = {}
    for cat in cats:
        method.set_category(cat)
        td = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "train", 256, 256)
        method.fit(DataLoader(td, batch_size=8, shuffle=False, num_workers=0))
        ds = AnomalyDataset(os.path.join(ROOT, "data/mvtec_ad"), cat, "test", 256, 256)
        S, T, L, MK = [], [], [], []
        for b in DataLoader(ds, batch_size=4, shuffle=False, num_workers=0):
            img = b["image"].cuda()
            s, tex = method.raw_score(img)
            S.append(s.cpu()); T.append(tex.cpu()); L.append(b["label"]); MK.append(b["mask"])
        S = torch.cat(S); T = torch.cat(T); L = torch.cat(L); MK = torch.cat(MK)
        pos = MK.numpy().ravel() > 0.5
        from sklearn.metrics import average_precision_score
        for name, a0, a1, K in configs:
            if a0 == 0.0 and a1 == 0.0:
                cal = S
            else:
                alpha = (a0 + a1 * T).clamp(0, 1)  # [B,1,H,W]
                cal = (1 - alpha) * S + alpha * blur(S, K)
            pix = roc_auc_score(pos, cal.reshape(-1).numpy())
            pap = average_precision_score(pos, cal.reshape(-1).numpy())
            img = roc_auc_score(L.numpy(), img_score(cal).numpy())
            acc.setdefault(name, []).append((pix, img, pap))

    print(f"{'cfg':18s}{'pixAUR':>9s}{'imgAUR':>9s}{'pixAP':>9s}")
    rows = []
    for name, a0, a1, K in configs:
        p = np.mean([x[0] for x in acc[name]]); i = np.mean([x[1] for x in acc[name]]); ap = np.mean([x[2] for x in acc[name]])
        rows.append((p, i, name, ap))
    for p, i, name, ap in sorted(rows, reverse=True):
        print(f"{name:20s}{p:9.4f}{i:9.4f}{ap:9.4f}")


if __name__ == "__main__":
    main()