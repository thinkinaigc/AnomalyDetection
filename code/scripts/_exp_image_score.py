"""图像级打分策略搜索（与生产 fit 完全一致：全量训练集 + set_seed(42)）。

目的：在固定异常图（同一 fit）上比较多种图像级聚合，找出对所有类一致生效的更强打分。
注意：图像 AUROC 对随机投影种子敏感，故必须用与生产相同的 fit 才能比较（否则基线对不上）。
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import AnomalyDataset, list_categories
from src.methods.factory import build_method
from src.utils.common import ROOT, load_yaml, set_seed

CFG = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
DEVICE = "cuda"
SEED = int(os.environ.get("SEED", "42"))
KS = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
TS = [0.5, 0.6, 0.7, 0.8]


def prims(M: np.ndarray) -> dict:
    flat = np.sort(M.reshape(-1))[::-1]
    out = {"max": float(flat[0])}
    n = flat.size
    for k in KS:
        out[f"topk_{k}"] = float(flat[: max(int(n * k), 1)].mean())
    for t in TS:
        out[f"area_{t}"] = float((M > t).mean())
    return out


def build_scorers():
    s = {}
    for k in KS:
        s[f"sm_topk_{k}"] = lambda a, b, k=k: a[f"topk_{k}"]
    for t in TS:
        s[f"sm_area_{t}"] = lambda a, b, t=t: a[f"area_{t}"]
    s["sm_max"] = lambda a, b: a["max"]
    for k in KS:
        s[f"sc_topk_{k}"] = lambda a, b, k=k: b[f"topk_{k}"]
    s["prod_topk_0.01"] = lambda a, b: a["topk_0.01"]  # 生产基线
    # 组合：多尺度 top-k 取 max（自适应缺陷尺度）
    s["maxTopk(0.003,0.01,0.05)"] = lambda a, b: max(a["topk_0.003"], a["topk_0.01"], a["topk_0.05"])
    s["maxTopk(0.005,0.02,0.1)"] = lambda a, b: max(a["topk_0.005"], a["topk_0.02"], a["topk_0.1"])
    s["maxTopk+area(0.01,0.05,0.2,a0.6)"] = lambda a, b: max(a["topk_0.01"], a["topk_0.05"], a["topk_0.2"], a["area_0.6"])
    s["maxTopk+area(0.005,0.02,0.1,a0.7)"] = lambda a, b: max(a["topk_0.005"], a["topk_0.02"], a["topk_0.1"], a["area_0.7"])
    s["max(sc_topk0.01,area0.6)"] = lambda a, b: max(b["topk_0.01"], a["area_0.6"])
    s["max(sc_topk0.01,sm_topk0.01,area0.6)"] = lambda a, b: max(b["topk_0.01"], a["topk_0.01"], a["area_0.6"])
    s["max(sm_topk0.01,sm_area0.6,sm_area0.7)"] = lambda a, b: max(a["topk_0.01"], a["area_0.6"], a["area_0.7"])
    s["0.5*topk0.01+0.5*maxTopk(0.005,0.05)"] = lambda a, b: 0.5 * a["topk_0.01"] + 0.5 * max(a["topk_0.005"], a["topk_0.05"])
    return s


def main():
    set_seed(SEED)
    cfg = dict(CFG)
    cfg.setdefault("model", {})["backbone"] = "dinov2"  # 与生产一致：DINOv2 ViT-B
    method = build_method(cfg, "A2", device=DEVICE,
                         override_method={"fusion": "mean", "cov_mode": "diag", "fgsc": "lowpass"})
    data_root = os.path.join(ROOT, cfg["data"]["mvtec_root"])
    cats = list_categories(data_root)
    bb = method.backbone
    img_size = int(getattr(bb, "image_size", 256))
    mean = getattr(bb, "image_mean", (0.48145466, 0.4578275, 0.40821073))
    std = getattr(bb, "image_std", (0.26862954, 0.26130258, 0.27577711))
    map_size = int(cfg["eval"]["map_size"])
    scorers = build_scorers()
    names = list(scorers)
    store = {c: {n: {"s": [], "y": []} for n in names} for c in cats}

    for cat in cats:
        method.set_category(cat)
        tr = AnomalyDataset(data_root, cat, "train", img_size, map_size, mean, std, only_good=True)
        method.fit(DataLoader(tr, batch_size=8, shuffle=False, num_workers=0), max_samples=0)  # 全量，同生产
        method.eval()
        te = AnomalyDataset(data_root, cat, "test", img_size, map_size, mean, std)
        for batch in DataLoader(te, batch_size=8, shuffle=False, num_workers=0):
            img = batch["image"].to(DEVICE)
            lab = batch["label"].numpy()
            out = method.predict(img)
            sc = out["anomaly_map"].detach()
            sm = method._gaussian_blur(sc, method.image_blur_k).cpu().numpy()[:, 0]
            sc = sc.cpu().numpy()[:, 0]
            for i in range(img.shape[0]):
                pa = prims(sm[i]); pb = prims(sc[i])
                for n in names:
                    store[cat][n]["s"].append(float(scorers[n](pa, pb)))
                    store[cat][n]["y"].append(int(lab[i]))
        print(f"[{cat}] done", flush=True)

    # per-class AUROC
    per_class = {}
    for n in names:
        per_class[n] = {}
        for cat in cats:
            y = np.array(store[cat][n]["y"]); s = np.array(store[cat][n]["s"])
            per_class[n][cat] = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
    rows = sorted(((float(np.nanmean(list(per_class[n].values()))), n) for n in names), reverse=True)
    print(f"\n=== 图像级打分候选（SEED={SEED}, 15类 mean AUROC，降序）===")
    for m, n in rows:
        print(f"  {m:.4f}  {n}")
    # 生产基线 & top3 的逐类对比
    show = ["prod_topk_0.01"] + [n for _, n in rows[:3]]
    print(f"\n=== 逐类 image AUROC（生产基线 vs top3）===")
    print(f"{'class':12s} " + " ".join(f"{n[:18]:>18s}" for n in show))
    for cat in cats:
        print(f"{cat:12s} " + " ".join(f"{per_class[n][cat]:18.4f}" for n in show))


if __name__ == "__main__":
    main()
