# -*- coding: utf-8 -*-
"""从生产运行的 summary.csv 重建 paper/figures/data.json。

背景
----
paper/figures/data.json 早期由一次更早的实验导出，其逐类数值与最终生产运行
(outputs/mvtec15_A2fgscL / neu_ours / pvel_ours) 不一致，导致图 fig_percat /
fig_cross 的柱状值与正文表格对不上（15 类图像宏平均 0.9168 vs 正文 0.9219）。
本脚本以生产运行的 summary.csv 为唯一事实来源重建 data.json，
其中 "agg"（图像级聚合上限）沿用原值，因其来自另一实验(同图同 fit 的读出策略搜索)。

用法: python code/scripts/_rebuild_data_json.py
"""
from __future__ import annotations

import csv
import json
import os

ROOT = r"d:\cvs\pythoncode\AnomalyDetection"
OUT = os.path.join(ROOT, "paper", "figures", "data.json")

# 论文主结果统一为 DINOv2 + v2（seed 42 作为逐类代表性运行）：
#   v2 = DINOv2 + diag + fusion=mean + FGSC 固定低通 a0=0.6/K=33
#        + shrinkage 0.02 + ref_smooth 0 + image_score_topk 0.005
SRC = {
    "mvtec": os.path.join(ROOT, "outputs", "mvtec15_v2_s42", "summary.csv"),
    "neu": os.path.join(ROOT, "outputs", "neu_v2", "summary.csv"),
    "pvel": os.path.join(ROOT, "outputs", "pvel_v2", "summary.csv"),
}
# 回退到同骨干旧运行（仅当 v2 运行缺失时）
SRC_FALLBACK = {
    "mvtec": os.path.join(ROOT, "outputs", "mvtec15_a06", "summary.csv"),
    "neu": os.path.join(ROOT, "outputs", "neu_a06", "summary.csv"),
    "pvel": os.path.join(ROOT, "outputs", "pvel_a06", "summary.csv"),
}
KEYS = (("img", "image_auroc"), ("pix", "pixel_auroc"), ("pap", "pixel_ap"))


def load(path: str) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"空文件: {path}")
    out = {}
    for r in rows:
        cat = r["category"]
        out[cat] = {}
        for short, col in KEYS:
            v = r.get(col, "")
            out[cat][short] = float(v) if v not in ("", None) else None
    return out


def macro(d: dict, key: str) -> float:
    vals = [v[key] for v in d.values() if v.get(key) is not None]
    return sum(vals) / len(vals)


def main():
    old = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            old = json.load(fh)

    data = {}
    print("重建 per-category 数据（来源：生产运行 summary.csv）")
    for name, path in SRC.items():
        if not os.path.exists(path):
            fb = SRC_FALLBACK[name]
            print(f"  [warn] {name}: 未找到 a0=0.6 运行，回退到 {os.path.basename(os.path.dirname(fb))}")
            path = fb
        if not os.path.exists(path):
            raise SystemExit(f"缺少源文件: {path}")
        d = load(path)
        data[name] = d
        n = len(d)
        print(f"  {name:6s} {n:2d} 类  image={macro(d,'img'):.4f} "
              f"pixel={macro(d,'pix'):.4f}  AP={macro(d,'pap'):.4f}   <- {path}")

    # 聚合上限来自独立的读出策略搜索实验，沿用既有值
    data["agg"] = old.get("agg", {})
    if not data["agg"]:
        raise SystemExit("原 data.json 缺少 agg 段，无法保留聚合上限数据")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    # 与旧值对比，报告变动
    if old:
        print("\n与旧 data.json 的 15 类宏平均对比：")
        for name in ("mvtec", "neu", "pvel"):
            if name not in old or name not in data:
                continue
            for short, label in (("img", "image"), ("pix", "pixel"), ("pap", "AP")):
                try:
                    o = sum(v[short] for v in old[name].values()) / len(old[name])
                except (KeyError, TypeError, ZeroDivisionError):
                    continue
                nw = macro(data[name], short)
                flag = "" if abs(o - nw) < 5e-4 else "   <-- 变动"
                print(f"  {name:6s} {label:6s} 旧={o:.4f}  新={nw:.4f}  "
                      f"Δ={nw - o:+.4f}{flag}")

    print(f"\n已写入 {OUT}")
    print("下一步：运行 _restyle_figs.py 重绘图，再执行 paper/_package.py 编译。")


if __name__ == "__main__":
    main()
