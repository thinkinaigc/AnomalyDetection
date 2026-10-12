# -*- coding: utf-8 -*-
"""实验 1：平滑算子对比 + 逐图归一化 + a0 x K 交互（缓存配对比较）。

依据：论文自身已证明增益来自"移除正常表面的孤立分数尖峰"，故测试对孤立尖峰更稳健/
边缘保持的算子。median=中值(稳健统计最优)；guided=导向滤波(He et al. CVPR'10)以纹理
图 T 为引导；gauss=现状基线。另测工业 AD 常见但本文未用的逐图归一化。
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _psc_ops import run, dom, fmt

KR, AR = 33, 0.6


def main():
    torch.manual_seed(0)
    nopsc = run(K=1, a0=0.0)
    base = run(K=KR, a0=AR)
    print(f"ref  noPSC a0=0         image {nopsc[0]:.4f}  pixel {nopsc[1]:.4f}  AP {nopsc[2]:.4f}")
    print(f"base gauss K={KR} a0={AR}     image {base[0]:.4f}  pixel {base[1]:.4f}  AP {base[2]:.4f}\n")

    print("=" * 96)
    print("1) smoothing operator   (* = Pareto-dominates baseline)")
    print("=" * 96)
    rows = [(f"gauss K={KR} base", base)]
    print(fmt(f"gauss K={KR} base", base, base), flush=True)
    for K in (17, 25, 41):
        t = run(K=K, a0=AR)
        rows.append((f"gauss K={K}", t))
        print(fmt(f"gauss K={K}", t, base), flush=True)
    for r in (1, 2, 3):
        for a in (0.6, 1.0):
            t = run(r=r, a0=a)
            rows.append((f"median r={r} a={a}", t))
            print(fmt(f"median r={r} a={a}", t, base), flush=True)
    for gr in (2, 4, 8):
        for a in (0.6, 1.0):
            t = run(gr=gr, a0=a)
            rows.append((f"guided r={gr} a={a}", t))
            print(fmt(f"guided r={gr} a={a}", t, base), flush=True)
    win = [x for x in rows if dom(x[1], base)]
    print("\n  dominating baseline: %d" % len(win))
    for tag, t in win:
        print(fmt(tag, t))

    print("\n" + "=" * 96)
    print("2) per-image score normalisation")
    print("=" * 96)
    for nm in ("none", "minmax", "zscore", "rank"):
        print(fmt("norm=" + nm, run(K=KR, a0=AR, norm=nm), base), flush=True)

    print("\n" + "=" * 96)
    print("3) a0 x K grid (image level)")
    print("=" * 96)
    KS = [17, 25, 33, 41]
    print("%5s" % "a0" + "".join("%10d" % K for K in KS))
    for a in (0.4, 0.5, 0.6, 0.7, 0.8):
        print("%5.1f" % a + "".join("%10.4f" % run(K=K, a0=a, px=False)[0] for K in KS), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
