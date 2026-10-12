# -*- coding: utf-8 -*-
"""在同一基准(同一批缓存/同一次 fit)上测量四行校准消融，替换论文中基准不一致的旧表。

旧表的问题: 论文 tab:fgsc 的三行来自 _exp_fgsc_ablation.py 的另一次 fit，
且缺少"生产配置"(常数 alpha=0.8)这一行 —— 而头条数字恰恰由它产生。
本脚本在 exp/sensitivity 的 15 类缓存上统一测量:
  none      : 不做任何校准
  full blur : alpha=1 (完全替换为高斯平滑)
  constant  : alpha=0.8 (生产配置, a1=0)
  routed    : alpha=0.8+0.4T (纹理门控, 论文中已声明关闭)
绝对值与正文头条表有约 -0.005 的缓存偏移, 故只用于行间比较。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = r"d:\cvs\pythoncode\AnomalyDetection"
sys.path.insert(0, os.path.join(ROOT, "code"))
from src.modules.fgsc import calibrate, gaussian_blur  # noqa: E402

CACHE = os.path.join(ROOT, "exp", "sensitivity")
CATS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
        "wood", "zipper"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, BLUR, Q = 33, 11, 0.01


def topk_score(x, q):
    b = x.shape[0]
    flat = x.reshape(b, -1)
    k = max(int(flat.shape[1] * q), 1)
    return torch.topk(flat, k, dim=1).values.mean(dim=1)


_CACHE = {}


def load(cat):
    if cat not in _CACHE:
        z = np.load(os.path.join(CACHE, f"cache_{cat}_s0.10.npz"))
        _CACHE[cat] = (torch.from_numpy(z["raw"]).float(),
                       torch.from_numpy(z["tex"]).float(),
                       z["label"], z["mask"].reshape(-1) > 0.5)
    return _CACHE[cat]


def measure(fn):
    ia, pa, ap = [], [], []
    for c in CATS:
        raw, tex, lab, msk = load(c)
        n = raw.shape[0]
        isc, ps = [], []
        for i in range(0, n, 32):
            m = min(32, n - i)
            r = raw[i:i + m].unsqueeze(1).to(DEV)
            t = tex[i:i + m].unsqueeze(1).to(DEV)
            s = fn(r, t)
            s = gaussian_blur(s, BLUR) if BLUR > 1 else s
            isc.append(topk_score(s, Q).cpu().numpy())
            ps.append(s[:, 0].reshape(m, -1).cpu().numpy().reshape(-1))
            del r, t, s
        ia.append(roc_auc_score(lab, np.concatenate(isc)))
        p = np.concatenate(ps)
        pa.append(roc_auc_score(msk, p))
        ap.append(average_precision_score(msk, p))
    return float(np.mean(pa)), float(np.mean(ap)), float(np.mean(ia))


ROWS = [
    ("none (A2, no calibration)", lambda r, t: r),
    ("full blur (alpha=1)", lambda r, t: gaussian_blur(r, K)),
    ("constant alpha=0.8 (deployed)", lambda r, t: calibrate(r, t, lam=0.8, tau=0.0, mode="lowpass", kernel=K)),
    ("routed alpha=0.8+0.4T (off)", lambda r, t: calibrate(r, t, lam=0.8, tau=0.4, mode="adaptive", kernel=K)),
]


def main():
    print(f"cache basis, {len(CATS)} categories, K={K} blur={BLUR} q={Q}, device={DEV}")
    print(f"{'configuration':34s} {'Pixel':>9s} {'AP':>9s} {'Image':>9s}")
    base = None
    for name, fn in ROWS:
        p, a, i = measure(fn)
        if base is None:
            base = (p, a, i)
        print(f"{name:34s} {p:9.4f} {a:9.4f} {i:9.4f}"
              f"   (dPix {p - base[0]:+.4f}, dAP {a - base[1]:+.4f}, dImg {i - base[2]:+.4f})")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
