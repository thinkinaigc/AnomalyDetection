# -*- coding: utf-8 -*-
"""在缓存上做配对的组合搜索，寻找比 a0=0.6 更优的 PSC/读出配置。

为什么用缓存：缓存是同一批固定分数图，配置之间的比较是"配对"的，方向可靠；
而每个配置各跑一次全流程会有 ~0.001-0.002 的运行间抖动。最终候选再用全流程复跑确认。

分三阶段收敛，避免组合爆炸：
  A) a0 x blur 图像级粗筛
  B) 候选的像素/AP 精算 + Pareto 前沿
  C) 最优解附近细扫 K 与 q
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = r"d:\cvs\pythoncode\AnomalyDetection"
sys.path.insert(0, os.path.join(ROOT, "code"))
from src.modules.fgsc import calibrate  # noqa: E402

CACHE = os.path.join(ROOT, "exp", "sensitivity")
CATS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
        "wood", "zipper"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
A0_REF, BLUR_REF, K_REF, Q_REF = 0.6, 11, 33, 0.01

_C = {}


def load(cat):
    if cat not in _C:
        z = np.load(os.path.join(CACHE, f"cache_{cat}_s0.10.npz"))
        _C[cat] = (torch.from_numpy(z["raw"]).float(),
                   torch.from_numpy(z["tex"]).float(),
                   z["label"], z["mask"].reshape(-1) > 0.5)
    return _C[cat]


def gblur(x, k):
    if k <= 1:
        return x
    ax = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2))
    g = (g / g.sum()).view(1, 1, 1, k)
    x = torch.nn.functional.conv2d(x, g, padding=(0, k // 2))
    return torch.nn.functional.conv2d(x, g.transpose(2, 3), padding=(k // 2, 0))


def topk(x, q):
    b = x.shape[0]
    f = x.reshape(b, -1)
    k = max(int(f.shape[1] * q), 1)
    return torch.topk(f, k, dim=1).values.mean(dim=1)


def run(a0, blur, K, q, pixel=True):
    ia, pa, ap = [], [], []
    for c in CATS:
        raw, tex, lab, msk = load(c)
        n = raw.shape[0]
        isc, ps = [], []
        for i in range(0, n, 32):
            m = min(32, n - i)
            r = raw[i:i + m].unsqueeze(1).to(DEV)
            t = tex[i:i + m].unsqueeze(1).to(DEV)
            s = calibrate(r, t, lam=a0, tau=0.0, mode="lowpass", kernel=K)
            s = gblur(s, blur)
            isc.append(topk(s, q).cpu().numpy())
            if pixel:
                ps.append(s[:, 0].reshape(m, -1).cpu().numpy().reshape(-1))
            del r, t, s
        ia.append(roc_auc_score(lab, np.concatenate(isc)))
        if pixel:
            p = np.concatenate(ps)
            pa.append(roc_auc_score(msk, p))
            ap.append(average_precision_score(msk, p))
    return (float(np.mean(ia)), float(np.mean(pa)), float(np.mean(ap)))


def dominates(a, b):
    return all(x >= y - 1e-9 for x, y in zip(a, b)) and any(x > y + 1e-9 for x, y in zip(a, b))


def report(rows, ref, title):
    print(f"\n{title}")
    for cfg, (i, p, a) in sorted(rows, key=lambda t: -t[1][0]):
        mark = " *" if dominates((i, p, a), ref) else ""
        print(f"  a0={cfg[0]:.1f} blur={cfg[1]:2d} K={cfg[2]:3d} q={cfg[3]:.3f} | "
              f"image {i:.4f}  pixel {p:.4f}  AP {a:.4f}{mark}")


def main():
    torch.manual_seed(0)
    ref = run(A0_REF, BLUR_REF, K_REF, Q_REF, pixel=True)
    print(f"参考(已采纳) a0={A0_REF} blur={BLUR_REF} K={K_REF} q={Q_REF}: "
          f"image={ref[0]:.4f} pixel={ref[1]:.4f} AP={ref[2]:.4f}")

    A0S = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    BLURS = [1, 3, 5, 7, 11]
    print("\n" + "=" * 78)
    print("A) a0 x blur 图像级粗筛 (K=33, q=1%)")
    print("=" * 78)
    print(f"{'a0':>5s}" + "".join(f"{b:>9d}" for b in BLURS) + "     (行=a0, 列=blur)")
    grid = {}
    for a0 in A0S:
        row = []
        for b in BLURS:
            v = run(a0, b, K_REF, Q_REF, pixel=False)[0]
            grid[(a0, b, K_REF, Q_REF)] = v
            row.append(v)
        print(f"{a0:5.1f}" + "".join(f"{v:9.4f}" for v in row), flush=True)
    best_img = max(grid, key=grid.get)
    print(f"图像级最优: a0={best_img[0]} blur={best_img[1]} -> {grid[best_img]:.4f}")

    cands = set()
    for a0 in A0S:
        bs = max(BLURS, key=lambda b: grid[(a0, b, K_REF, Q_REF)])
        cands.add((a0, bs, K_REF, Q_REF))
    for a0 in (0.4, 0.5, 0.6, 0.7):
        cands.add((a0, BLUR_REF, K_REF, Q_REF))
    cands.add(best_img)
    cands.add((A0_REF, BLUR_REF, K_REF, Q_REF))

    print("\n" + "=" * 78)
    print("B) 候选精算 (K=33, q=1%)   * = 帕累托支配已采纳配置")
    print("=" * 78)
    print(f"{'a0':>5s}{'blur':>6s}{'image':>10s}{'pixel':>10s}{'AP':>10s}"
          f"{'dImg':>9s}{'dPix':>9s}{'dAP':>9s}")
    rows = []
    for cfg in sorted(cands):
        i, p, a = run(*cfg, pixel=True)
        rows.append((cfg, (i, p, a)))
        mark = " *" if dominates((i, p, a), ref) else ""
        print(f"{cfg[0]:5.1f}{cfg[1]:6d}{i:10.4f}{p:10.4f}{a:10.4f}"
              f"{i-ref[0]:+9.4f}{p-ref[1]:+9.4f}{a-ref[2]:+9.4f}{mark}", flush=True)
    front = [r for r in rows if not any(dominates(o[1], r[1]) for o in rows if o is not r)]
    report(front, ref, "Pareto 前沿:")

    bi = max(front, key=lambda t: t[1][0])[0] if front else (A0_REF, BLUR_REF)
    print("\n" + "=" * 78)
    print(f"C) 在 a0={bi[0]} blur={bi[1]} 附近细扫 K 与 q   * = 帕累托支配已采纳配置")
    print("=" * 78)
    print(f"{'K':>5s}{'q':>8s}{'image':>10s}{'pixel':>10s}{'AP':>10s}")
    fine = []
    for K, q in itertools.product([17, 25, 33, 41], [0.002, 0.005, 0.01, 0.02]):
        i, p, a = run(bi[0], bi[1], K, q, pixel=True)
        fine.append(((bi[0], bi[1], K, q), (i, p, a)))
        mark = " *" if dominates((i, p, a), ref) else ""
        print(f"{K:5d}{q:8.3f}{i:10.4f}{p:10.4f}{a:10.4f}{mark}", flush=True)
    ff = [r for r in fine if not any(dominates(o[1], r[1]) for o in fine if o is not r)]
    report(ff, ref, "\n细化后 Pareto 前沿:")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
