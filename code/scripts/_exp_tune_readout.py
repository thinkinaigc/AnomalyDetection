# -*- coding: utf-8 -*-
"""在已有 15 类缓存(shrink=0.10)上枚举读出/校准参数，零 GPU 成本。

缓存来自 _exp_sensitivity.py：raw(PSC off 的原始分数图) + tex(纹理图) + mask + label。
本脚本对同一批缓存枚举：
  mode  : lowpass(生产配置, a1=0) / adaptive(带纹理门控)
  a0    : 全局低通强度
  q     : 图像级 top-k 比例
  blur  : 图像级打分前平滑核
输出图像 AUROC(15类宏平均)网格，并给出生产配置与候选配置的对照。
注意：缓存为 fp16 且仅一次 fit，绝对值与生产头条表有系统偏移(约 -0.007 图像)，
      故只用于比较"相对差值"，最终结论必须用全流程复跑验证。
"""
from __future__ import annotations

import os
import sys
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = r"d:\cvs\pythoncode\AnomalyDetection"
sys.path.insert(0, os.path.join(ROOT, "code"))
from src.modules.fgsc import calibrate  # noqa: E402

CACHE = os.path.join(ROOT, "exp", "sensitivity")
CATS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
        "wood", "zipper"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# 生产配置(default.yaml): a0=0.8 a1=0.4 K=33 q=0.01 blur=11 shrink=0.1
PROD = dict(a0=0.8, a1=0.4, K=33, q=0.01, blur=11)


def gblur(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    ax = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2))
    g = (g / g.sum()).view(1, 1, 1, k)
    x = F.conv2d(x, g, padding=(0, k // 2))
    return F.conv2d(x, g.transpose(2, 3), padding=(k // 2, 0))


def topk_score(x: torch.Tensor, q: float) -> torch.Tensor:
    b = x.shape[0]
    flat = x.reshape(b, -1)
    k = max(int(flat.shape[1] * q), 1)
    return torch.topk(flat, k, dim=1).values.mean(dim=1)


_CACHE = {}


def load(cat: str):
    """惰性加载并常驻显存内的原始张量(fp32 复用)，避免重复 IO。"""
    if cat in _CACHE:
        return _CACHE[cat]
    z = np.load(os.path.join(CACHE, f"cache_{cat}_s0.10.npz"))
    raw = torch.from_numpy(z["raw"]).float()
    tex = torch.from_numpy(z["tex"]).float()
    lab = z["label"]
    msk = z["mask"].reshape(-1) > 0.5
    _CACHE[cat] = (raw, tex, lab, msk)
    return _CACHE[cat]


def eval_cfg(mode: str, a0: float, a1: float, K: int, q: float, blur: int,
             want_pixel: bool = True):
    ia, pa, ap = [], [], []
    for c in CATS:
        raw, tex, lab, msk = load(c)
        n = raw.shape[0]
        isc, ps = [], []
        for i in range(0, n, 32):
            m = min(32, n - i)
            r = raw[i:i + m].unsqueeze(1).to(DEV)
            t = tex[i:i + m].unsqueeze(1).to(DEV)
            if mode == "off":
                s = r
            else:
                s = calibrate(r, t, lam=a0, tau=a1, mode=mode, kernel=K)
            s = gblur(s, blur)
            isc.append(topk_score(s, q).cpu().numpy())
            if want_pixel:
                ps.append(s[:, 0].reshape(m, -1).cpu().numpy().reshape(-1))
            del r, t, s
        ia.append(roc_auc_score(lab, np.concatenate(isc)))
        if want_pixel:
            p = np.concatenate(ps)
            pa.append(roc_auc_score(msk, p))
            ap.append(average_precision_score(msk, p))
    return (float(np.mean(ia)), float(np.mean(pa)), float(np.mean(ap)))


def main():
    torch.manual_seed(0)
    print(f"cache dir : {CACHE}")
    print(f"categories: {len(CATS)}   device: {DEV}\n")

    # ---------- 1) 生产配置 vs 带门控 ----------
    print("=" * 78)
    print("1) 生产配置(lowpass, a1=0) vs 纹理门控(adaptive)  —— 15 类缓存, fp16")
    print("=" * 78)
    for mode, a1 in (("lowpass", 0.0), ("adaptive", PROD["a1"])):
        i, p, a = eval_cfg(mode, PROD["a0"], a1, PROD["K"], PROD["q"], PROD["blur"])
        tag = "生产(无门控)" if mode == "lowpass" else "带纹理门控"
        print(f"  {mode:9s} {tag:14s} img={i:.4f}  pix={p:.4f}  AP={a:.4f}")
    print()

    # ---------- 2) mode x a0 网格(像素/图像权衡) ----------
    print("=" * 78)
    print("2) lowpass 下扫描 a0 (K=33, q=0.01, blur=11)")
    print("=" * 78)
    print(f"  {'a0':>6s} {'image':>9s} {'pixel':>9s} {'AP':>9s}")
    for a0 in (0.4, 0.6, 0.7, 0.8, 0.9, 1.0):
        i, p, a = eval_cfg("lowpass", a0, 0.0, PROD["K"], PROD["q"], PROD["blur"])
        print(f"  {a0:6.2f} {i:9.4f} {p:9.4f} {a:9.4f}")
    print()

    # ---------- 3) q 扫描(只影响图像级) ----------
    print("=" * 78)
    print("3) lowpass 下扫描 q (a0=0.8, K=33, blur=11) —— q 只改图像级读出")
    print("=" * 78)
    print(f"  {'q':>8s} {'image':>9s} {'pixel':>9s} {'AP':>9s}")
    for q in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        i, p, a = eval_cfg("lowpass", PROD["a0"], 0.0, PROD["K"], q, PROD["blur"],
                           want_pixel=(q == 0.01))
        print(f"  {q:8.3f} {i:9.4f} " + (f"{p:9.4f} {a:9.4f}" if q == 0.01 else ""))
    print()

    # ---------- 4) blur 扫描 ----------
    print("=" * 78)
    print("4) lowpass 下扫描 blur (a0=0.8, K=33, q=0.01)")
    print("=" * 78)
    print(f"  {'blur':>6s} {'image':>9s} {'pixel':>9s} {'AP':>9s}")
    for b in (1, 3, 5, 7, 11, 15, 21):
        i, p, a = eval_cfg("lowpass", PROD["a0"], 0.0, PROD["K"], PROD["q"], b)
        print(f"  {b:6d} {i:9.4f} {p:9.4f} {a:9.4f}")
    print()

    # ---------- 5) K 扫描 ----------
    print("=" * 78)
    print("5) lowpass 下扫描 K (a0=0.8, q=0.01, blur=11) —— K 越大越平滑")
    print("=" * 78)
    print(f"  {'K':>6s} {'image':>9s} {'pixel':>9s} {'AP':>9s}")
    for K in (5, 9, 17, 25, 33, 49, 65):
        i, p, a = eval_cfg("lowpass", PROD["a0"], 0.0, K, PROD["q"], PROD["blur"])
        print(f"  {K:6d} {i:9.4f} {p:9.4f} {a:9.4f}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
