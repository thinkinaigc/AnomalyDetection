"""FGSC 超参网格搜索（适配 per-position 打分流水线）。

每个类别只跑一次主干（fit + 收集测试的原始异常图与纹理图），随后在内存里枚举
(mode, λ, τ) 做频域校准并算指标（像素/图像，全图协议），选出跨类别通用最优配置。
"""
from __future__ import annotations

import argparse
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


def apply_fgsc(score: torch.Tensor, tex: torch.Tensor, lam: float, tau: float, mode: str) -> torch.Tensor:
    if lam == 0.0 or mode == "off":
        return score
    tex = F.interpolate(tex, size=score.shape[-2:], mode="bilinear", align_corners=False)
    if mode == "global":
        return (score - lam * tex).clamp(0, 1)
    if mode == "conditional":
        return torch.where(score < tau, score - lam * tex, score).clamp(0, 1)
    if mode == "mul_conditional":
        return torch.where(score < tau, score * (1.0 - lam * tex), score).clamp(0, 1)
    raise ValueError(mode)


def image_score(amap: torch.Tensor, topk=0.01, blur=11):
    if blur > 1:
        ax = torch.arange(blur, dtype=torch.float32, device=amap.device) - (blur - 1) / 2
        g = torch.exp(-(ax ** 2) / (2 * (blur / 4.0) ** 2)); g = (g / g.sum()).view(1, 1, 1, blur)
        amap = F.conv2d(amap, g, padding=(0, blur // 2))
        amap = F.conv2d(amap, g.transpose(2, 3), padding=(blur // 2, 0))
    b = amap.shape[0]
    flat = amap.reshape(b, -1)
    k = max(int(flat.shape[1] * topk), 1)
    return torch.topk(flat, k, dim=1).values.mean(dim=1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mvtec")
    ap.add_argument("--variant", default="A1")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "code/configs/default.yaml"), encoding="utf-8"))
    method = build_method(cfg, args.variant, device="cuda")
    method.eval()
    ds_root = {"mvtec": "data/mvtec_ad", "pcb": "data/processed/pcb", "neu": "data/processed/neu"}[args.dataset]
    cats = cfg["data"][f"{args.dataset}_categories"]

    # 预取每个类别测试的原始图 + 纹理 + 标签 + 掩码
    per_cat = {}
    for cat in cats:
        method.set_category(cat)
        td = AnomalyDataset(os.path.join(ROOT, ds_root), cat, "train", 256, 256)
        method.fit(DataLoader(td, batch_size=8, shuffle=False, num_workers=0))
        ds = AnomalyDataset(os.path.join(ROOT, ds_root), cat, "test", 256, 256)
        raws, texs, labels, masks = [], [], [], []
        for b in DataLoader(ds, batch_size=4, shuffle=False, num_workers=0):
            raw, tex = method.raw_score(b["image"].cuda())
            raws.append(raw.cpu()); texs.append(tex.cpu())
            labels.append(b["label"]); masks.append(b["mask"])
        per_cat[cat] = (torch.cat(raws), torch.cat(texs), torch.cat(labels), torch.cat(masks))
        print(f"  [{cat}] n={len(labels[-1]) if False else per_cat[cat][2].shape[0]}", flush=True)

    lams = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    taus = [0.3, 0.4, 0.5, 0.6]
    modes = ["conditional", "global", "mul_conditional"]
    rows = []
    for mode in modes:
        for lam in lams:
            for tau in taus:
                pix, img = [], []
                for cat in cats:
                    raw, tex, labels, masks = per_cat[cat]
                    cal = apply_fgsc(raw, tex, lam, tau, mode)
                    m = masks.numpy().ravel() > 0.5
                    s = cal[:, 0].reshape(-1).numpy()
                    pix.append(roc_auc_score(m, s))
                    isc = image_score(cal).numpy()
                    img.append(roc_auc_score(labels.numpy(), isc))
                rows.append((mode, lam, tau, float(np.mean(pix)), float(np.mean(img)), pix, img))

    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"\n=== FGSC 网格 ({args.dataset}/{args.variant}) 按 平均像素AUROC 降序 ===")
    print(f"{'mode':15s}{'lam':>5s}{'tau':>5s}{'pixAUR':>9s}{'imgAUR':>9s}")
    for r in rows[:12]:
        print(f"{r[0]:15s}{r[1]:5.2f}{r[2]:5.2f}{r[3]:9.4f}{r[4]:9.4f}")
    best = rows[0]
    print(f"\nBEST mode={best[0]} lam={best[1]} tau={best[2]}")
    print("per-cat pix:", " ".join(f"{c}={v:.3f}" for c, v in zip(cats, best[5])))
    base = [r for r in rows if r[1] == 0.0][0]
    print(f"(baseline lam=0 avg pix={base[3]:.4f} avg img={base[4]:.4f})")


if __name__ == "__main__":
    main()