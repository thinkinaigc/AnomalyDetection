# -*- coding: utf-8 -*-
"""对比两次全量 15 类运行（头条 a0=0.8 vs 候选 a0=0.6）。"""
from __future__ import annotations

import csv
import os
import sys

ROOT = r"d:\cvs\pythoncode\AnomalyDetection"
METRICS = ("image_auroc", "pixel_auroc", "pixel_ap", "latency_ms", "peak_mem_mb")


def load(d):
    p = os.path.join(ROOT, "outputs", d, "summary.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    out = {}
    for k in METRICS:
        v = [float(r[k]) for r in rows if r.get(k) not in ("", None)]
        out[k] = sum(v) / len(v)
    return out, len(rows), {r["category"]: r for r in rows}


def main():
    dirs = sys.argv[1:] or ["mvtec15_A2fgscL", "mvtec15_a06"]
    res = [load(d) for d in dirs]
    base = res[0][0]
    hdr = f"{'metric':14s}" + "".join(f"{d:>20s}" for d in dirs)
    if len(res) > 1:
        hdr += f"{'delta':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for k in METRICS:
        line = f"{k:14s}" + "".join(f"{m[k]:20.4f}" for m, _, _ in res)
        if len(res) > 1:
            line += f"{res[-1][0][k] - base[k]:+10.4f}"
        print(line)
    print("n categories: " + ", ".join(f"{d}={n}" for d, (_, n, _) in zip(dirs, res)))

    if len(res) > 1:
        print("\nper-category delta (candidate - baseline), sorted by |d image|:")
        cats = sorted(res[0][2])
        print(f"{'category':14s}{'d image':>10s}{'d pixel':>10s}{'d AP':>10s}")
        rows = []
        for c in cats:
            r0, r1 = res[0][2][c], res[-1][2][c]
            rows.append((c,
                         float(r1["image_auroc"]) - float(r0["image_auroc"]),
                         float(r1["pixel_auroc"]) - float(r0["pixel_auroc"]),
                         float(r1["pixel_ap"]) - float(r0["pixel_ap"])))
        for c, di, dp, da in sorted(rows, key=lambda t: -abs(t[1])):
            print(f"{c:14s}{di:+10.4f}{dp:+10.4f}{da:+10.4f}")


if __name__ == "__main__":
    main()
