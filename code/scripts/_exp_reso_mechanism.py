# -*- coding: utf-8 -*-
"""验证"升分辨率无效 micro-defect"的机制归因是否受PSC 污染。

背景（领域审稿人 P0-3）：论文把 screw/hazelnut 的 image AUROC 在 392→518px 下
不改善归因于"position-wise 假设"。但部署配置里 PSC 用 K=33 在 28×28 patch 网格
上做低通——核宽(σ=33/4≈8.25 patch)已覆盖整个 28 宽网格，即PSC 近似全局平均。
若真是全局平均，则缺陷定位精度被PSC 本身抹平，此时"分辨率无关"可能是 PSC 的
性质而非 position-wise 假设的性质。

本实验在 screw / hazelnut 上做 2×2 对照：
    factor ∈ {PSC on (K=33), PSC off}
    resolution ∈ {392, 448, 518}
报告 image AUROC / pixel AUROC / pixel AP。若PSC off 时分辨率提升能恢复
image AUROC，则论文的机制归因必须改写。

产出：outputs/exp_reso_mechanism.json
"""
from __future__ import annotations
import os
import sys
import json
import argparse

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AnomalyDataset  # noqa: E402
from src.methods.factory import build_method  # noqa: E402
from src.utils.common import ROOT, load_yaml, set_seed  # noqa: E402

FROZEN = {
    "fusion": "mean", "cov_mode": "diag",
    "fgsc_lambda": 0.6, "fgsc_kernel": 33,
    "shrinkage": 0.02, "ref_smooth": 0,
    "image_score_topk": 0.005, "image_blur_k": 11, "projection_seed": 42,
}
RESOS = [392, 448, 518]
# 数据集按 image_size 缩放后再插值到各尺度，故 image_size 取能被 14 整除且为偶数的值
IMG_SIZE = {392: 392, 448: 448, 518: 518}


@torch.no_grad()
def run(cat, reso, psc, data_root, device, img_size, ms, mean, std, seed=42):
    set_seed(seed)
    cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
    cfg.setdefault("model", {})["backbone"] = "dinov2"
    ov = dict(FROZEN)
    ov["fgsc"] = "lowpass" if psc else "off"
    # 分辨率覆盖：P3 分支输入尺寸（config 字段 dinov2_scales）
    # P4 必须被 patch 14 整除：reso//2 对 518 得 259（不整除），故向下对齐
    p4 = max(224, (reso // 2 // 14) * 14)
    cfg["model"]["dinov2_scales"] = {"P3": reso, "P4": p4}
    m = build_method(cfg, "A2", device=device, override_method=ov)
    m.set_category(cat)
    tr = AnomalyDataset(data_root, cat, "train", img_size, ms, mean, std, only_good=True)
    m.fit(DataLoader(tr, batch_size=8, shuffle=False, num_workers=0))
    m.eval()
    te = AnomalyDataset(data_root, cat, "test", img_size, ms, mean, std)
    pm, ps, yy, IS = [], [], [], []
    for b in DataLoader(te, batch_size=8, shuffle=False, num_workers=0):
        out = m.predict(b["image"].to(device))
        amap = out["anomaly_map"].float().cpu().numpy()[:, 0]
        mask = b["mask"].numpy()
        for i in range(amap.shape[0]):
            mm, s = mask[i], amap[i]
            if mm.shape != s.shape:
                import cv2
                s = cv2.resize(s, (mm.shape[1], mm.shape[0]), interpolation=cv2.INTER_LINEAR)
            pm.append(mm.ravel()); ps.append(s.ravel())
        yy.append(b["label"].numpy())
        IS.append(out["image_score"].float().cpu().numpy())
    pm = np.concatenate(pm); ps = np.concatenate(ps)
    yy = np.concatenate(yy); IS = np.concatenate(IS)
    pos = pm > 0.5
    out = {
        "image": float(roc_auc_score(yy, IS)) if len(np.unique(yy)) > 1 else float("nan"),
        "pixel": float(roc_auc_score(pos, ps)),
        "pap": float(average_precision_score(pos, ps)),
        "patch_grid": int(reso // 14),
    }
    del m
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/mvtec_ad")
    ap.add_argument("--cats", nargs="*", default=["screw", "hazelnut"])
    ap.add_argument("--resos", nargs="*", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/exp_reso_mechanism.json")
    args = ap.parse_args()

    cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
    img_size = 256
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    ms = int(cfg["eval"]["map_size"])
    data_root = os.path.join(ROOT, args.root) if not os.path.isabs(args.root) else args.root

    resos = args.resos or RESOS
    res = {"resos": resos, "cats": args.cats, "runs": {}}
    for cat in args.cats:
        for psc in (True, False):
            for r in resos:
                key = f"{cat}|{'pscon' if psc else 'pscoff'}|{r}"
                try:
                    # 数据集按 img_size 缩放，须与 backbone 最大尺度一致，
                    # 否则骨干内插值后网格尺寸不匹配 patch 14
                    v = run(cat, r, psc, data_root, args.device, r, ms, mean, std)
                except Exception as e:  # 分辨率覆盖可能不被支持
                    v = {"error": f"{type(e).__name__}: {e}"}
                res["runs"][key] = v
                print(key, "->", v, flush=True)
                with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
    print("saved -> " + args.out)


if __name__ == "__main__":
    main()