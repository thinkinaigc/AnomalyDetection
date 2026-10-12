# -*- coding: utf-8 -*-
"""PSC 平滑算子与逐图归一化的公共实现（供缓存实验复用）。"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

CACHE = r"d:\cvs\pythoncode\AnomalyDetection\exp\sensitivity"
CATS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_C = {}


def load(cat):
    if cat not in _C:
        z = np.load(os.path.join(CACHE, f"cache_{cat}_s0.10.npz"))
        _C[cat] = (torch.from_numpy(z["raw"]).float(),
                   torch.from_numpy(z["tex"]).float(),
                   z["label"], z["mask"].reshape(-1) > 0.5)
    return _C[cat]


def gauss(x, k):
    if k <= 1:
        return x
    ax = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2))
    g = (g / g.sum()).view(1, 1, 1, k)
    y = F.conv2d(x, g, padding=(0, k // 2))
    return F.conv2d(y, g.transpose(2, 3), padding=(k // 2, 0))


def median(x, r):
    """半径 r 的 2D 中值滤波：对孤立尖峰是最优的稳健算子。"""
    if r <= 0:
        return x
    d = 2 * r + 1
    p = F.pad(x, (r, r, r, r), mode="reflect")
    b = x.shape[0]
    out = torch.empty_like(x)
    st = max(1, 64 // (d * d))
    for i in range(0, b, st):
        j = min(b, i + st)
        u = F.unfold(p[i:j], kernel_size=d).view(j - i, d * d, -1)
        out[i:j] = u.median(dim=1).values.view(j - i, 1, x.shape[2], x.shape[3])
    return out


def guided(x, g, r, eps):
    """Guided Filter (He et al., CVPR'10/TPAMI'11)，引导图 g 为标量图（如纹理图 T）。"""
    if r <= 0:
        return x
    k = 2 * r + 1
    I = F.pad(g, (r, r, r, r), mode="reflect")
    P = F.pad(x, (r, r, r, r), mode="reflect")
    mI, mP = F.avg_pool2d(I, k, 1), F.avg_pool2d(P, k, 1)
    cII = F.avg_pool2d(I * I, k, 1) - mI * mI
    cIP = F.avg_pool2d(I * P, k, 1) - mI * mP
    a = cIP / (cII + eps)
    b = mP - a * mI
    # mI/mP/cII/cIP 已由无 padding 的 avg_pool2d 降到有效尺寸，故 a、b 已是 [B,C,H,W]；
    # 末次 box 滤波在有效区上做（count_include_pad=False 保证边界不偏置）
    ma = F.avg_pool2d(a, k, 1, padding=r, count_include_pad=False)
    mb = F.avg_pool2d(b, k, 1, padding=r, count_include_pad=False)
    return ma * g + mb


def pnorm(s, mode):
    """逐图归一化：像素指标拼接全图分数后统一计算，尺度大的图会主导跨图排序。"""
    if mode == "none":
        return s
    b = s.shape[0]
    f = s.reshape(b, -1)
    if mode == "minmax":
        lo = f.min(1, keepdim=True).values
        hi = f.max(1, keepdim=True).values
        o = (f - lo) / (hi - lo + 1e-8)
    elif mode == "zscore":
        o = (f - f.mean(1, keepdim=True)) / (f.std(1, keepdim=True) + 1e-8)
    else:
        o = f.argsort(1).argsort(1).float() / (f.shape[1] - 1)
    return o.view_as(s)


def topk(x, q):
    b = x.shape[0]
    f = x.reshape(b, -1)
    k = max(int(f.shape[1] * q), 1)
    return torch.topk(f, k, dim=1).values.mean(dim=1)


def run(K=0, r=0, gr=0, geps=1e-3, norm="none", q=0.01, blur=11, a0=0.6, px=True):
    """在缓存上评估一个配置。返回 (image, pixel, AP) 的 15 类宏平均。"""
    ia, pa, ap = [], [], []
    for c in CATS:
        raw, tex, lab, msk = load(c)
        n = raw.shape[0]
        isc, ps = [], []
        for i in range(0, n, 32):
            m = min(32, n - i)
            S = raw[i:i + m].unsqueeze(1).to(DEV)
            T = tex[i:i + m].unsqueeze(1).to(DEV)
            if T.shape[-2:] != S.shape[-2:]:
                T = F.interpolate(T, size=S.shape[-2:], mode="bilinear", align_corners=False)
            if gr > 0:
                low = guided(S, T, gr, geps)
            elif r > 0:
                low = median(S, r)
            else:
                low = gauss(S, K)
            s = (1.0 - a0) * S + a0 * low
            if blur > 1:
                s = gauss(s, blur)
            s = pnorm(s, norm)
            isc.append(topk(s, q).cpu().numpy())
            if px:
                ps.append(s[:, 0].reshape(m, -1).cpu().numpy().reshape(-1))
            del S, T, low, s
        ia.append(roc_auc_score(lab, np.concatenate(isc)))
        if px:
            p = np.concatenate(ps)
            pa.append(roc_auc_score(msk, p))
            ap.append(average_precision_score(msk, p))
    return float(np.mean(ia)), float(np.mean(pa)), float(np.mean(ap))


def dom(a, b):
    """a 是否帕累托支配 b（三个指标都不差且至少一个更好）"""
    return all(x >= y - 1e-9 for x, y in zip(a, b)) and any(x > y + 1e-9 for x, y in zip(a, b))


def fmt(tag, t, base=None):
    s = f"  {tag:30s} image {t[0]:.4f}  pixel {t[1]:.4f}  AP {t[2]:.4f}"
    if base is not None:
        s += f"   dImg {t[0]-base[0]:+.4f} dPix {t[1]-base[1]:+.4f} dAP {t[2]-base[2]:+.4f}"
    if base is not None and dom(t, base):
        s += "  *"
    return s
