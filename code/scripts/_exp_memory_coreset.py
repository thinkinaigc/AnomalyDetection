# -*- coding: utf-8 -*-
"""显存对照实验：本文方法 vs PatchCore 标准 1% coreset kNN bank（同 backbone）。

回应方法学审稿意见：原文 "1.9x less memory" 的对照基线是"存全部训练 patch"，
而 PatchCore 的标准协议是 1% coreset 子采样。本实验按标准协议重测峰值显存，
使 memory 对照公平。latency 显式 cuda.synchronize() 计时。

bank_size=0 即存全部 patch（本文原对照基线）；bank_size=1%N 为 PatchCore 标准协议。
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AnomalyDataset  # noqa: E402
from src.eval.metrics import aupro  # noqa: E402,F401  (确保 metrics 模块可导入)
from src.methods.factory import build_method  # noqa: E402
from src.utils.common import ROOT, load_yaml, set_seed  # noqa: E402

# Table I 冻结配置（不得改动）
FROZEN = {
    "fusion": "mean",
    "cov_mode": "diag",
    "fgsc": "lowpass",
    "fgsc_lambda": 0.6,
    "fgsc_kernel": 33,
    "shrinkage": 0.02,
    "ref_smooth": 0,
    "image_score_topk": 0.005,
    "image_blur_k": 11,
}

MVTEC_CATS = ["bottle", "screw", "hazelnut", "transistor", "capsule"]


def build(seed: int, device: str, extra: dict | None = None):
    set_seed(seed)
    cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
    cfg.setdefault("model", {})["backbone"] = "dinov2"
    ov = dict(FROZEN)
    if extra:
        ov.update(extra)
    return build_method(cfg, "A2", device=device, override_method=ov)


@torch.no_grad()
def measure(model, images, device, warmup=10, iters=50):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for i in range(warmup):
        model.predict(images[i % len(images)])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ts = []
    for i in range(iters):
        x = images[i % len(images)]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model.predict(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
    peak = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
    return peak, float(np.mean(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/mvtec_ad")
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/exp_memory_coreset.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
    img_size = int(getattr(cfg["model"], "image_size", 256)) if "image_size" in cfg.get("model", {}) else 256
    bb_cfg = cfg.get("model", {})
    img_size = 256
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    ms = int(cfg["eval"]["map_size"])

    data_root = os.path.join(ROOT, args.root) if not os.path.isabs(args.root) else args.root
    cats = args.cats or [c for c in MVTEC_CATS
                         if os.path.isdir(os.path.join(data_root, c))]
    print(f"device={device} cats={cats} frac={args.frac}", flush=True)

    res = {"frac": args.frac, "device": str(device), "cats": cats, "per_cat": {}}

    for cat in cats:
        tr = AnomalyDataset(data_root, cat, "train", img_size, ms, mean, std, only_good=True)
        te = AnomalyDataset(data_root, cat, "test", img_size, ms, mean, std)
        loader = DataLoader(tr, batch_size=8, shuffle=False, num_workers=0)
        images = []
        for i in range(min(10, len(te))):
            images.append(te[i]["image"].unsqueeze(0).to(device))

        # ---------- 探针：只用无 bank 的 fit 统计 patch 数（避免 O(M^2) 全量 kNN） ----------
        mp = build(args.seed, device)
        mp.set_category(cat)
        mp.fit(loader)
        n_patch_per_img = 0
        state_mb = 0.0
        for n, st in mp._stats.items():
            hw = st.get("hw")
            if hw is not None:
                n_patch_per_img += int(hw[0]) * int(hw[1])
            mu = st.get("mu")
            if mu is not None:
                state_mb += mu.numel() * mu.element_size() / 1024.0 / 1024.0
        # 与kNN 分支一致：所有尺度的 patch 合起来存进一个 bank
        full_patches = n_patch_per_img * len(tr)
        del mp
        torch.cuda.empty_cache()

        # ---------- Ours: 无 bank（复用探针模型的显存基线需重测，故单独测） ----------
        m0 = build(args.seed, device)
        m0.set_category(cat)
        m0.fit(loader)
        m0.eval()
        smb = 0.0
        for n, st in m0._stats.items():
            mu = st.get("mu")
            if mu is not None:
                smb += mu.numel() * mu.element_size() / 1024.0 / 1024.0
        p0, l0 = measure(m0, images, device)
        state_mb = smb
        del m0
        torch.cuda.empty_cache()
        bank_size = max(1, int(round(full_patches * args.frac)))
        m1 = build(args.seed, device, {"knn": True, "bank_size": bank_size})
        m1.set_category(cat)
        m1.fit(loader)
        m1.eval()
        bank_mb = 0.0
        for n, st in m1._stats.items():
            b = st.get("bank")
            if b is not None:
                bank_mb += b.numel() * b.element_size() / 1024.0 / 1024.0
        p1, l1 = measure(m1, images, device)
        del m1
        torch.cuda.empty_cache()

        # ---------- 全量bank 的存储量：解析计算 ----------
        # 全量 kNN 的 fit需 O(M^2) 距离矩阵（M 可达数十万），实测不现实；
        # 但它的存储量是确定的：full_patches × d × 4 bytes。
        d_dim = 256
        bank_full_mb = full_patches * d_dim * 4 / 1024.0 / 1024.0

        # 全量bank 的峰值显存 = 本文峰值（骨干/参考部分）+ 全量 bank 存储
        # （推理时bank 常驻，实测的 Ours 峰值已含同等骨干激活）
        p2 = p0 + bank_full_mb
        l2 = float("nan")

        res["per_cat"][cat] = {
            "n_train": len(tr), "full_patches": full_patches,
            "bank_size_1pct": bank_size, "bank_mb_1pct": bank_mb,
            "bank_mb_full": bank_full_mb,
            "ours_peak_mb": p0, "ours_latency_ms": l0, "ours_state_mb": state_mb,
            "knn1_peak_mb": p1, "knn1_latency_ms": l1,
            "knnfull_peak_mb": p2, "knnfull_latency_ms": l2,
            "ratio_knn1_over_ours": p1 / max(p0, 1e-9),
            "ratio_knnfull_over_ours": p2 / max(p0, 1e-9),
        }
        print(f"[{cat}] patches={full_patches} bank1%={bank_size} ({bank_mb:.0f}MB) "
              f"bankfull={bank_full_mb:.0f}MB | ours {p0:.0f}MB/{l0:.0f}ms  "
              f"knn1% {p1:.0f}MB/{l1:.0f}ms (r={p1/max(p0,1e-9):.3f})  "
              f"knnfull {p2:.0f}MB (r={p2/max(p0,1e-9):.3f})", flush=True)

    P = res["per_cat"]
    o = np.mean([v["ours_peak_mb"] for v in P.values()])
    k1 = np.mean([v["knn1_peak_mb"] for v in P.values()])
    kf = np.mean([v["knnfull_peak_mb"] for v in P.values()])
    res["summary"] = {
        "ours_peak_mean_mb": float(o),
        "knn1_peak_mean_mb": float(k1),
        "knnfull_peak_mean_mb": float(kf),
        "ratio_knn1_over_ours": float(k1 / max(o, 1e-9)),
        "ratio_knnfull_over_ours": float(kf / max(o, 1e-9)),
        "ours_latency_mean_ms": float(np.mean([v["ours_latency_ms"] for v in P.values()])),
        "knn1_latency_mean_ms": float(np.mean([v["knn1_latency_ms"] for v in P.values()])),
        "bank_mb_1pct_mean": float(np.mean([v["bank_mb_1pct"] for v in P.values()])),
        "bank_mb_full_mean": float(np.mean([v["bank_mb_full"] for v in P.values()])),
        "ours_state_mb_mean": float(np.mean([v["ours_state_mb"] for v in P.values()])),
    }
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\nSUMMARY\n" + json.dumps(res["summary"], indent=2))
    print("saved -> " + args.out)


if __name__ == "__main__":
    main()